import numpy as np
from typing import Dict, Any, Optional

from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, ReportParsingMixin, QuantityParsingMixin

class AtheyBagwellGame(DynamicGame, ReportParsingMixin, QuantityParsingMixin):
    """
    Implements the Athey & Bagwell (2008) game of collusion with persistent cost shocks.
    """

    def __init__(self):
        super().__init__("athey_bagwell")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes the game state."""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        num_players = constants.get('number_of_players', 3)
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

        # Use simulation_id to ensure reproducibility
        np.random.seed(simulation_id)
        
        cost_sequences = {}
        persistence = constants.get('persistence_probability', 0.7)
        
        for player_id in player_ids:
            current_cost = 'high' if np.random.rand() < 0.5 else 'low'
            costs = [current_cost]
            
            for _ in range(1, time_horizon):
                if np.random.rand() >= persistence:
                    current_cost = 'low' if current_cost == 'high' else 'high'
                costs.append(current_cost)
            cost_sequences[player_id] = costs
            
        return {
            'current_period': 1,
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids},
            'last_market_shares': {} 
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        current_period = game_state['current_period']
        if current_period % 2 == 0:
             return "" 

        true_cost = game_state['cost_sequences'][player_id][current_period - 1]
        
        report_history = game_state.get('report_history', {})
        your_history = report_history.get(player_id, [])
        
        # Clean history: remove 'enforcement' placeholders
        clean_your_history = [r for r in your_history if r != 'enforcement']
        
        other_history_lines = []
        other_pids = [p for p in report_history if p != player_id]
        
        if other_pids:
            # Reconstruct history lines for previous ODD periods
            max_len = len(report_history[other_pids[0]])
            # Step by 2 to get odd periods (indices 0, 2, 4...)
            for i in range(0, max_len, 2): 
                round_num = i + 1
                reports_str = ", ".join([f"{pid}: {report_history[pid][i]}" for pid in other_pids])
                other_history_lines.append(f"Period {round_num}: {reports_str}")

        other_history_str = "\n".join(other_history_lines) if other_history_lines else "No previous reports."
        your_history_str = ", ".join(clean_your_history) if clean_your_history else "None"

        variables = get_prompt_variables(
            game_config, player_id=player_id, current_round=current_period,
            your_cost_type=true_cost,
            your_reports_history_detailed=your_history_str,
            all_other_reports_history_detailed=other_history_str
        )
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        return self.parse_report_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        constants = game_config.constants
        costs = constants.get('cost_types', {'low': 15, 'high': 25})
        market_price = constants.get('market_price', 40)
        market_size = constants.get('market_size', 100)
        
        current_period = game_state['current_period']
        player_ids = list(game_state['cost_sequences'].keys())
        market_shares = {}

        if current_period % 2 != 0:
            # --- ODD PERIOD (Strategic) ---
            reports = {}
            for pid in player_ids:
                # Safely extract report, defaulting to 'high' (yield)
                act = actions.get(pid, {})
                if act and 'report' in act:
                    reports[pid] = str(act['report']).lower()
                else:
                    reports[pid] = 'high'

            low_reporters = [pid for pid, r in reports.items() if r == 'low']
            
            if not low_reporters:
                share = 1.0 / len(player_ids)
                for pid in player_ids: market_shares[pid] = share
            else:
                share = 1.0 / len(low_reporters)
                for pid in player_ids:
                    market_shares[pid] = share if pid in low_reporters else 0.0
                    
        else:
            # --- EVEN PERIOD (Enforcement) ---
            # Use history to determine shares based on previous odd period
            prev_odd_index = current_period - 2
            
            prev_reports = {}
            report_history = game_state.get('report_history', {})
            
            for pid in player_ids:
                phist = report_history.get(pid, [])
                if 0 <= prev_odd_index < len(phist):
                    prev_reports[pid] = phist[prev_odd_index]
                else:
                    prev_reports[pid] = 'high'

            prev_low_reporters = [pid for pid, r in prev_reports.items() if r == 'low']
            
            eligible_players = [pid for pid in player_ids if pid not in prev_low_reporters]
            
            if not eligible_players:
                # Everyone claimed low previously -> Reset, share equally
                share = 1.0 / len(player_ids)
                for pid in player_ids: market_shares[pid] = share
            else:
                share = 1.0 / len(eligible_players)
                for pid in player_ids:
                    market_shares[pid] = share if pid in eligible_players else 0.0

        payoffs = {}
        for pid in player_ids:
            cost_type = game_state['cost_sequences'][pid][current_period - 1]
            unit_cost = costs.get(cost_type, 25)
            
            qty = market_shares.get(pid, 0.0) * market_size
            profit = (market_price - unit_cost) * qty
            payoffs[pid] = profit
            
        game_state['last_market_shares'] = market_shares
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig, payoffs: Dict[str, float]) -> Dict:
        current_period = game_state['current_period']
        player_ids = list(game_state['cost_sequences'].keys())

        if current_period % 2 != 0:
            # ODD Period: Record actual reports
            for pid in player_ids:
                act = actions.get(pid, {})
                report = 'high'
                if act and 'report' in act:
                    report = str(act['report']).lower()
                game_state['report_history'][pid].append(report)
        else:
            # EVEN Period: Record placeholder
            for pid in player_ids:
                game_state['report_history'][pid].append('enforcement')

        game_state['current_period'] += 1
        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        current_period = game_state.get('current_period', 1)
        cost_idx = current_period - 1
        
        return {
            "period": current_period,
            "actions": actions, 
            "payoffs": payoffs,
            "player_true_costs": {
                pid: seq[cost_idx] for pid, seq in game_state.get('cost_sequences', {}).items() 
                if cost_idx < len(seq)
            },
            "market_shares": game_state.get('last_market_shares', {})
        }