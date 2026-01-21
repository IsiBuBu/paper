# games/athey_bagwell.py

import numpy as np
from typing import Dict, Any, Optional

from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, ReportParsingMixin, QuantityParsingMixin

class AtheyBagwellGame(DynamicGame, ReportParsingMixin, QuantityParsingMixin):
    """
    Implements the Athey & Bagwell (2008) game of collusion with persistent cost shocks,
    using a two-stage, odd-even scheme to manage incentives.
    """

    def __init__(self):
        super().__init__("athey_bagwell")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes the game state for the two-stage, odd-even scheme."""
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        num_players = constants.get('number_of_players', 3)
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

        # Pre-generate the persistent cost streams for all players
        np.random.seed(simulation_id)
        cost_sequences = {}
        for player_id in player_ids:
            costs = ['high' if np.random.rand() < 0.5 else 'low']
            persistence = constants.get('persistence_probability', 0.7)
            for _ in range(1, time_horizon):
                next_cost = costs[-1] if np.random.rand() < persistence else ('low' if costs[-1] == 'high' else 'high')
                costs.append(next_cost)
            cost_sequences[player_id] = costs
            
        return {
            'current_period': 1,
            'cost_sequences': cost_sequences,
            'report_history': {pid: [] for pid in player_ids},
            'last_market_shares': {} 
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """
        Generates a prompt for a player ONLY during an Odd period,
        which is the only time a strategic decision is made.
        """
        current_period = game_state['current_period']
        # Even periods are deterministic enforcement phases; no LLM decision needed.
        if current_period % 2 == 0:
             raise ValueError("Prompts should not be generated for Even periods.")

        true_cost = game_state['cost_sequences'][player_id][current_period - 1]
        
        # Format history for the prompt
        report_history = game_state.get('report_history', {})
        your_history = report_history.get(player_id, [])
        other_history = {pid: reports for pid, reports in report_history.items() if pid != player_id}
        
        # Format history strings
        your_history_str = ", ".join(your_history) if your_history else "N/A"
        
        other_history_lines = []
        # History is only available up to the previous period
        if other_history:
            # Assuming all histories have same length
            hist_len = len(next(iter(other_history.values())))
            for i in range(hist_len):
                line = f"Period {i+1}: " + ", ".join([f"{pid}: {reports[i]}" for pid, reports in other_history.items()])
                other_history_lines.append(line)
        other_history_str = "; ".join(other_history_lines) if other_history_lines else "No other player reports yet."

        variables = get_prompt_variables(
            game_config, player_id=player_id, current_round=current_period,
            your_cost_type=true_cost,
            your_reports_history_detailed=your_history_str,
            all_other_reports_history_detailed=other_history_str
        )
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        """Parses the LLM's report decision."""
        # This game only cares about 'report', not 'quantity'
        return self.parse_report_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates payoffs based on market allocation rules (Odd/Even scheme)."""
        constants = game_config.constants
        costs = constants.get('cost_types', {'low': 15, 'high': 25})
        market_price = constants.get('market_price', 30)
        market_size = constants.get('market_size', 100)
        
        current_period = game_state['current_period']
        # Use player IDs from the actions dict (assuming it contains all players) 
        # or from game_state keys if actions might be partial (though usually full).
        player_ids = list(game_state['cost_sequences'].keys())
        
        market_shares = {}

        if current_period % 2 != 0:
            # --- ODD PERIOD (Strategic) ---
            # 1. Extract reports from actions (default to 'high' if missing)
            reports = {pid: actions.get(pid, {}).get('report', 'high').lower() for pid in player_ids}
            
            # 2. Identify firms reporting 'low'
            low_reporters = [pid for pid, r in reports.items() if r == 'low']
            
            # 3. Allocate Market Share
            if not low_reporters:
                # Case A: Everyone reported 'high' -> Split equally
                share = 1.0 / len(player_ids)
                for pid in player_ids: market_shares[pid] = share
            else:
                # Case B: Some reported 'low' -> Split 100% among low reporters
                share = 1.0 / len(low_reporters)
                for pid in player_ids:
                    market_shares[pid] = share if pid in low_reporters else 0.0
                    
        else:
            # --- EVEN PERIOD (Enforcement) ---
            # Market allocation depends on the reports from the PREVIOUS (Odd) period.
            # Period indices: Current is even (e.g., 2). Previous is 1.
            # History list index: period 1 is at index 0. So look at current_period - 2.
            
            hist_idx = current_period - 2
            report_history = game_state.get('report_history', {})
            
            # Reconstruct who reported what in the previous round
            prev_reports = {}
            for pid in player_ids:
                phist = report_history.get(pid, [])
                # Safety check: if history is missing, assume 'high' (safe play)
                prev_reports[pid] = phist[hist_idx] if len(phist) > hist_idx else 'high'

            # Identify who claimed the market previously (the 'low' reporters)
            prev_low_reporters = [pid for pid, r in prev_reports.items() if r == 'low']
            
            # Enforcement Rule:
            # If you reported Low in Odd, you get restricted share in Even (punishment/balancing).
            # If everyone was the same, no imbalance to correct -> Split equally.
            if not prev_low_reporters or len(prev_low_reporters) == len(player_ids):
                # Symmetric previous actions -> Symmetric allocation
                share = 1.0 / len(player_ids)
                for pid in player_ids: market_shares[pid] = share
            else:
                # Asymmetric: Those who reported 'low' get 0. Those who reported 'high' split the rest.
                # This incentivizes truth-telling by making 'low' reports costly in the future.
                high_reporters = [pid for pid in player_ids if pid not in prev_low_reporters]
                share = 1.0 / len(high_reporters)
                for pid in player_ids:
                    market_shares[pid] = share if pid in high_reporters else 0.0

        # --- Calculate Profit ---
        payoffs = {}
        for pid in player_ids:
            # Retrieve true cost for the CURRENT period
            true_cost_type = game_state['cost_sequences'][pid][current_period - 1]
            true_cost = costs[true_cost_type]
            
            # Profit = Margin * Quantity
            quantity = market_shares.get(pid, 0) * market_size
            profit = (market_price - true_cost) * quantity
            payoffs[pid] = profit
            
        game_state['last_market_shares'] = market_shares
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig, payoffs: Dict[str, float]) -> Dict:
        """Updates the game state by recording reports (only in Odd periods)."""
        current_period = game_state['current_period']
        
        # Only record reports if we actually received them (Odd periods)
        # In Even periods, actions might be empty or dummy, so we don't append to report_history.
        if current_period % 2 != 0:
            for pid in actions.keys(): # Iterate over keys in actions
                 action = actions[pid]
                 # Ensure we record 'high' if parse failed or key missing
                 report = action.get('report', 'high') 
                 game_state['report_history'][pid].append(report)
        else:
            # For Even periods, we might append 'N/A' or just nothing depending on how 
            # history is indexed. Based on the logic above, we prefer report_history 
            # to only contain the strategic decisions (Odd periods). 
            # However, prompt generation iterates by i*2+1, implying history might need 
            # to align. Let's append placeholders if needed, but the current generate_prompt
            # logic seems to handle sparse history well. 
            # To keep indices simple (index 0 = Period 1, index 1 = Period 2?), 
            # let's append 'enforcement' to keep lists aligned with periods.
            for pid in game_state['report_history']:
                game_state['report_history'][pid].append('enforcement')

        return game_state

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers round-specific outcomes for detailed logging."""
        period = game_state.get('current_period', 1)
        period_type = "Odd" if period % 2 != 0 else "Even"

        return {
            "period": period,
            "period_type": period_type,
            "actions": actions, 
            "payoffs": payoffs,
            "player_true_costs": {pid: seq[period-1] for pid, seq in game_state.get('cost_sequences', {}).items() if period-1 < len(seq)},
            "game_outcomes": {
                "player_market_shares": game_state.get('last_market_shares', {}),
                "player_profits": payoffs
            }
        }