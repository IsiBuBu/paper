import numpy as np
from typing import Dict, Any, Optional, List
from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, ReportParsingMixin

class AtheyBagwellGame(DynamicGame, ReportParsingMixin):
    """
    Implements the Athey & Bagwell (2001/2008) game of collusion with persistent private information.
    """

    def __init__(self):
        super().__init__("athey_bagwell")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes game state with pre-generated cost sequences for reproducibility."""
        constants = game_config.constants
        num_players = constants.get('number_of_players', 3)
        time_horizon = constants.get('time_horizon', 25)
        persistence = constants.get('persistence_probability', 0.7)
        
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]
        
        # Generate persistent cost sequences
        np.random.seed(simulation_id)
        cost_sequences = {}
        
        for pid in player_ids:
            sequence = []
            current = 'low' if np.random.random() > 0.5 else 'high'
            sequence.append(current)
            for _ in range(time_horizon - 1):
                if np.random.random() >= persistence:
                    current = 'high' if current == 'low' else 'low'
                sequence.append(current)
            cost_sequences[pid] = sequence

        return {
            'current_period': 1,
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids},
            'profit_history': {pid: [] for pid in player_ids},
            'allocation_history': [] 
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        constants = game_config.constants
        market_size = constants.get('market_size', 100)
        price = constants.get('market_price', 40)
        cost_low = constants.get('cost_types', {}).get('low', 10)
        cost_high = constants.get('cost_types', {}).get('high', 30)
        discount = constants.get('discount_factor', 0.83)
        persistence = constants.get('persistence_probability', 0.7)
        
        current_round_idx = game_state['current_period'] - 1
        current_cost_type = game_state['cost_sequences'][player_id][current_round_idx]
        current_cost_val = cost_low if current_cost_type == 'low' else cost_high
        
        # 1. Calculate "Claim" Profit (Immediate)
        profit_claim_today = (price - current_cost_val) * market_size
        
        # 2. Calculate "Yield" Profit (Future Expected Value)
        if current_cost_type == 'low':
            prob_next_low = persistence
            prob_next_high = 1.0 - persistence
        else:
            prob_next_high = persistence
            prob_next_low = 1.0 - persistence
            
        future_profit_if_low = (price - cost_low) * market_size
        future_profit_if_high = (price - cost_high) * market_size
        
        expected_future_value = discount * (
            (prob_next_low * future_profit_if_low) + 
            (prob_next_high * future_profit_if_high)
        )

        history_table_vars = self._format_history_tables(player_id, game_state)
        variables = get_prompt_variables(game_config, player_id=player_id, current_round=game_state['current_period'])
        
        variables.update({
            "market_size": market_size,
            "market_price": price,
            "low_cost": cost_low,
            "high_cost": cost_high,
            "persistence_probability": persistence,
            "discount_factor": discount,
            "your_cost_type": current_cost_type.capitalize(),
            "your_true_cost_value": f"{current_cost_val}",
            "profit_claim_today": f"{profit_claim_today:.2f}",
            "expected_future_value": f"{expected_future_value:.2f}",
            "low_cost_margin": f"{price - cost_low}",
            "high_cost_margin": f"{price - cost_high}",
            "your_reports_history_detailed": history_table_vars['your_history'],
            "all_other_reports_history_detailed": history_table_vars['others_history']
        })

        return self.prompt_template.format(**variables)

    def _format_history_tables(self, player_id: str, game_state: Dict) -> Dict[str, str]:
        current_period = game_state['current_period']
        if current_period == 1:
            return {"your_history": "No previous reports.", "others_history": "No previous reports."}
            
        your_reports = game_state['report_history'][player_id]
        your_hist_lines = [f"Period {i+1}: {report}" for i, report in enumerate(your_reports)]
        
        others_hist_lines = []
        for pid, reports in game_state['report_history'].items():
            if pid != player_id:
                report_str = ", ".join(reports)
                others_hist_lines.append(f"{pid}: [{report_str}]")
                
        return {
            "your_history": "\n".join(your_hist_lines) if your_hist_lines else "None",
            "others_history": "\n".join(others_hist_lines) if others_hist_lines else "None"
        }

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        return self.parse_report_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        constants = game_config.constants
        market_size = constants.get('market_size', 100)
        price = constants.get('market_price', 40)
        current_period = game_state['current_period']
        cost_sequences = game_state['cost_sequences']
        
        is_odd_period = (current_period % 2 != 0)
        winners = []
        
        if is_odd_period:
            for pid, action in actions.items():
                if action and str(action.get('report')).lower() == 'low':
                    winners.append(pid)
            if not winners: winners = list(actions.keys())
        else:
            prev_reports = {pid: reports[-1] for pid, reports in game_state['report_history'].items()}
            for pid, report in prev_reports.items():
                if str(report).lower() == 'high':
                    winners.append(pid)
            if not winners: winners = list(actions.keys())

        payoffs = {}
        share = market_size / len(winners) if winners else 0
        
        for pid in actions.keys():
            cost_type = cost_sequences[pid][current_period - 1]
            cost_val = constants['cost_types'].get(cost_type, 20)
            profit = (price - cost_val) * share if pid in winners else 0.0
            payoffs[pid] = profit
            
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig, payoffs: Dict[str, float]) -> Dict:
        for pid, action in actions.items():
            report = action.get('report', 'high') 
            game_state['report_history'][pid].append(report)
            game_state['profit_history'][pid].append(payoffs.get(pid, 0.0))
        game_state['current_period'] += 1
        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """
        CRITICAL FIX: Explicitly logs the period and true costs for this specific round.
        This allows the Metrics Calculator to verify Truthfulness/Rationality without
        relying on the mutable game_state reference.
        """
        current_period = game_state['current_period']
        
        # Snapshot costs for this period
        period_costs = {}
        if 'cost_sequences' in game_state:
            for pid, sequence in game_state['cost_sequences'].items():
                idx = current_period - 1
                if 0 <= idx < len(sequence):
                    period_costs[pid] = sequence[idx]
        
        return {
            "period": current_period,
            "actions": actions,
            "payoffs": payoffs,
            "player_true_costs": period_costs
        }