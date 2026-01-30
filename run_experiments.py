import asyncio
import logging
import argparse
import json
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any
import numpy as np

# Ensure the project root is in the Python path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    get_all_game_configs,
    get_challenger_models,
    get_defender_model,
    get_model_display_name,
    get_experiments_dir,
    get_simulation_count,
    get_data_dir,
    load_config
)
from games import create_game
from agents import create_agent, AgentResponse
from metrics.metric_utils import GameResult, create_game_result, MetricCalculator

utils = MetricCalculator()

class Competition:
    """
    Orchestrates game simulations with robust Per-Model Concurrency Control.
    """

    def __init__(self, challenger_models: List[str], defender_model: str, mock_mode: bool = False):
        self.challenger_models = challenger_models
        self.defender_model = defender_model
        self.mock_mode = mock_mode
        self.output_dir = get_experiments_dir()
        self.logger = logging.getLogger(self.__class__.__name__)

        # --- LAYER 1: PER-MODEL CONCURRENCY CONTROL ---
        self.max_concurrent_per_model = int(os.getenv("MAX_CONCURRENT_REQUESTS", 200))
        self.model_semaphores = {}

        self.logger.info(f"🚀 Concurrency initialized: Max {self.max_concurrent_per_model} requests PER MODEL.")

        # Load Datasets
        try:
            dataset_path = get_data_dir() / "master_datasets.json"
            with open(dataset_path, 'r') as f:
                self.master_datasets = json.load(f)
            self.logger.info("Successfully loaded master datasets.")
        except FileNotFoundError:
            self.logger.critical(f"❌ CRITICAL ERROR: Input file not found at '{dataset_path}'.")
            sys.exit(1)

    def _get_semaphore(self, model_name: str) -> asyncio.Semaphore:
        if model_name not in self.model_semaphores:
            self.model_semaphores[model_name] = asyncio.Semaphore(self.max_concurrent_per_model)
        return self.model_semaphores[model_name]

    async def _throttled_get_response(self, agent, prompt, call_id, config):
        sem = self._get_semaphore(agent.model_name)
        async with sem:
            return await agent.get_response(prompt, call_id, config)

    async def run_all_experiments(self):
        self.logger.info("=" * 80)
        self.logger.info("STARTING EXPERIMENT SUITE")
        if self.mock_mode: self.logger.info("🎭 RUNNING IN MOCK MODE 🎭")
        self.logger.info("=" * 80)

        all_game_names = list(load_config().get('game_configs', {}).keys())
        
        for game_name in all_game_names:
            game_configs = get_all_game_configs(game_name)
            for config in game_configs:
                self.logger.info(f"--- Preparing: [{game_name}]-[{config.condition_name}] ---")

                for challenger_model in self.challenger_models:
                    await self._run_and_save_model_simulations(challenger_model, config)
                
                self.logger.info(f"--- Completed: [{game_name}]-[{config.condition_name}] ---")

    async def _run_and_save_model_simulations(self, challenger_model: str, config):
        """Runs all simulations for a single model in parallel (throttled by model limits)."""
        display_name = get_model_display_name(challenger_model)
        
        # --- FIX: Pass config.game_name instead of config.experiment_type ---
        num_simulations = get_simulation_count(config.game_name)
        # --------------------------------------------------------------------

        self.logger.info(f"  -> Launching {num_simulations} simulations for: [{display_name}]")
        
        tasks = [
            self.run_single_simulation(challenger_model, config, sim_id) 
            for sim_id in range(num_simulations)
        ]
        
        simulation_results = await asyncio.gather(*tasks)
        self._save_competition_result(challenger_model, config, simulation_results)

    async def run_single_simulation(self, challenger_model: str, config, sim_id: int):
        game = create_game(config.game_name)
        game_state = game.initialize_game_state(config, sim_id)
        game_state['simulation_id'] = sim_id

        # Load reproducible data
        dataset_key = f"{config.game_name}_{config.experiment_type}_{config.condition_name}"
        if dataset_key in self.master_datasets:
            dataset = self.master_datasets[dataset_key]
            
            if config.game_name == 'spulber' and 'player_private_costs' in dataset:
                costs_data = dataset.get('player_private_costs', {})
                sim_costs = {}
                for player, costs_list in costs_data.items():
                    if isinstance(costs_list, list) and len(costs_list) > sim_id:
                        sim_costs[player] = costs_list[sim_id]
                    else:
                        sim_costs[player] = costs_list
                game_state['player_private_costs'] = sim_costs
            
            elif config.game_name == 'athey_bagwell' and 'player_true_costs' in dataset:
                costs_data = dataset.get('player_true_costs', {})
                sim_costs = {}
                for player, costs_matrix in costs_data.items():
                    if len(costs_matrix) > sim_id:
                        sim_costs[player] = costs_matrix[sim_id]
                game_state['cost_sequences'] = sim_costs

            elif 'demand_shocks' in dataset:
                shocks_matrix = dataset['demand_shocks']
                if len(shocks_matrix) > sim_id:
                    game_state['demand_shocks'] = shocks_matrix[sim_id]

        agents = {
            'challenger': create_agent(challenger_model, 'challenger', mock_mode=self.mock_mode),
            **{f'defender_{i+1}': create_agent(self.defender_model, f'defender_{i+1}', mock_mode=self.mock_mode)
               for i in range(config.constants['number_of_players'] - 1)}
        }

        if hasattr(game, 'update_game_state'):
            return await self._run_dynamic_game(game, agents, config, game_state, challenger_model)
        else:
            return await self._run_static_game(game, agents, config, game_state, challenger_model)

    async def _get_all_actions(self, game, agents, config, game_state, stage: int = 1):
        call_id = f"{config.game_name}-{game_state.get('simulation_id', 'N/A')}"
        prompts = {pid: game.generate_player_prompt(pid, game_state, config) for pid in agents}
        
        tasks = {
            pid: self._throttled_get_response(agent, prompts[pid], call_id, config) 
            for pid, agent in agents.items()
        }
        
        responses = await asyncio.gather(*tasks.values())
        response_map = dict(zip(agents.keys(), responses))
        
        actions = {
            pid: game.parse_llm_response(resp.content, pid, call_id, stage=stage) or {} 
            for pid, resp in response_map.items()
        }

        return actions, response_map

    async def _run_static_game(self, game, agents, config, game_state, challenger_model: str):
        actions, responses = await self._get_all_actions(game, agents, config, game_state)
        payoffs = game.calculate_payoffs(actions, config, game_state)
        game_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
        game_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}

        return create_game_result(
            game_state['simulation_id'], config.game_name, config.experiment_type, 
            config.condition_name, challenger_model, list(agents.keys()), 
            actions, payoffs, game_data
        )

    async def _run_dynamic_game(self, game, agents, config, game_state, challenger_model: str):
        if config.game_name == 'athey_bagwell':
            return await self._run_athey_bagwell_game(game, agents, config, game_state, challenger_model)
        if config.game_name == 'green_porter':
            return await self._run_green_porter_game(game, agents, config, game_state, challenger_model)
        return None 

    async def _run_green_porter_game(self, game, agents, config, game_state, challenger_model: str):
        time_horizon = config.constants.get('time_horizon', 50)
        all_rounds_data = []

        while game_state['current_period'] <= time_horizon:
            if game_state['market_state'] == 'Collusive':
                actions, responses = await self._get_all_actions(game, agents, config, game_state)
            else:
                cournot_quantity = config.constants.get('cournot_quantity')
                actions = {pid: {'quantity': cournot_quantity} for pid in agents}
                responses = {}
            
            payoffs = game.calculate_payoffs(actions, config, game_state)
            round_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
            if responses:
                round_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}
            all_rounds_data.append(round_data)
            game_state = game.update_game_state(game_state, actions, config, payoffs)

        profit_streams = defaultdict(list)
        for round_data in all_rounds_data:
            for p_id, profit in round_data.get('payoffs', {}).items():
                profit_streams[p_id].append(profit)
        
        final_npvs = {p: utils.calculate_npv(s, config.constants.get('discount_factor', 0.9)) for p, s in profit_streams.items()}
        
        game_data = {
            "constants": config.constants, "rounds": all_rounds_data, 
            "final_npvs": final_npvs, "state_history": game_state.get('state_history', [])
        }

        return create_game_result(
            game_state['simulation_id'], config.game_name, config.experiment_type, 
            config.condition_name, challenger_model, list(agents.keys()), {}, final_npvs, game_data
        )

    async def _run_athey_bagwell_game(self, game, agents, config, game_state, challenger_model: str):
        time_horizon = config.constants.get('time_horizon', 50)
        all_rounds_data = []

        for period in range(1, time_horizon + 1):
            game_state['current_period'] = period
            
            if period % 2 != 0: # Odd Period (Strategic)
                # 1. Get reports from agents
                reports, responses = await self._get_all_actions(game, agents, config, game_state, stage=1)
                
                # 2. Calculate payoffs
                # FIX: We pass the 'reports' directly to the game class. 
                # The game class extracts 'low'/'high' reports and handles market share logic internally.
                payoffs = game.calculate_payoffs(reports, config, game_state)
                
                # 3. Log data
                round_data = game.get_game_data_for_logging(reports, payoffs, config, game_state)
                round_data['llm_metadata'] = {pid: resp.__dict__ for pid, resp in responses.items()}
                all_rounds_data.append(round_data)

                # 4. Update Game State
                # FIX: Explicitly call update_game_state so that 'report_history' is populated.
                game_state = game.update_game_state(game_state, reports, config, payoffs)

            else: # Even Period (Enforcement)
                # 1. No actions required from agents in this phase
                actions = {}
                
                # 2. Calculate payoffs
                # The game class uses the history from the previous ODD period to determine enforcement allocations.
                payoffs = game.calculate_payoffs(actions, config, game_state)
                
                # 3. Log data
                round_data = game.get_game_data_for_logging(actions, payoffs, config, game_state)
                all_rounds_data.append(round_data)
                
                # 4. Update Game State
                # Records 'enforcement' placeholder in history
                game_state = game.update_game_state(game_state, actions, config, payoffs)

        profit_streams = defaultdict(list)
        for round_data in all_rounds_data:
            for p_id, profit in round_data.get('payoffs', {}).items():
                profit_streams[p_id].append(profit)

        final_npvs = {p: utils.calculate_npv(s, config.constants.get('discount_factor', 0.9)) for p, s in profit_streams.items()}
        game_data = { "constants": config.constants, "rounds": all_rounds_data, "final_npvs": final_npvs }
        
        return create_game_result(
            game_state['simulation_id'], config.game_name, config.experiment_type, 
            config.condition_name, challenger_model, list(agents.keys()), {}, final_npvs, game_data
        )

    def _save_competition_result(self, challenger, config, results):
        challenger_dir = self.output_dir / config.game_name / challenger
        challenger_dir.mkdir(parents=True, exist_ok=True)
        filepath = challenger_dir / f"{config.condition_name}_competition_result.json"
        
        dict_results = [res.__dict__ for res in results]
        output_data = {
            "game_name": config.game_name, "experiment_type": config.experiment_type,
            "condition_name": config.condition_name, "challenger_model": challenger,
            "simulation_results": dict_results
        }

        with open(filepath, 'w') as f:
            json.dump(output_data, f, indent=2)
        self.logger.info(f"Saved results to {filepath}")

def setup_logging(verbose: bool, mock_mode: bool):
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"experiment_{timestamp}{'_mock' if mock_mode else ''}.log"
    
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.info(f"Logging to: {log_file}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM Game Theory Experiment Runner")
    parser.add_argument('--mock', action='store_true', help="Run in mock mode.")
    parser.add_argument('-v', '--verbose', action='store_true', help="Enable verbose logging.")
    return parser.parse_args()

async def main():
    args = parse_arguments()
    setup_logging(args.verbose, args.mock)
    competition = Competition(get_challenger_models(), get_defender_model(), mock_mode=args.mock)
    await competition.run_all_experiments()

if __name__ == "__main__":
    asyncio.run(main())