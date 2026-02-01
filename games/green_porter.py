# games/green_porter.py

import numpy as np
from typing import Dict, Any, Optional, List

from config.config import GameConfig, get_prompt_variables
from games.base_game import DynamicGame, QuantityParsingMixin

class GreenPorterGame(DynamicGame, QuantityParsingMixin):
    """
    Implements the Green & Porter (1984) dynamic oligopoly game.

    This class manages a multi-round Cournot competition with demand uncertainty.
    It uses a State Transition Algorithm triggered by a price threshold to switch
    between "Collusive" and "Reversionary" (punishment) phases.
    """

    def __init__(self):
        super().__init__("green_porter")
        self.game_config: Optional[GameConfig] = None

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes game state with pre-generated demand shocks for the simulation."""
        self.game_config = game_config  # Store config for the parser to use
        constants = game_config.constants
        time_horizon = constants.get('time_horizon', 50)
        demand_shock_std = constants.get('demand_shock_std', 5)
        demand_shock_mean = constants.get('demand_shock_mean', 0)
        num_players = constants.get('number_of_players', 3)
        player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

        # Pre-generate demand shocks for reproducibility
        np.random.seed(simulation_id)
        demand_shocks = np.random.normal(demand_shock_mean, demand_shock_std, time_horizon).tolist()
        
        return {
            'current_period': 1,
            'market_state': 'Collusive',
            'punishment_timer': 0,
            'demand_shocks': demand_shocks,
            'price_history': [],
            'state_history': [],
            'quantity_history': {pid: [] for pid in player_ids},
            'profit_history': {pid: [] for pid in player_ids}
        }

    
    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a player with current market conditions."""
        current_period = game_state.get('current_period', 1)
        price_history = game_state.get('price_history', [])
        state_history = game_state.get('state_history', [])
        
        # --- 1. Format History Table ---
        history_limit = 10
        start_index = max(0, len(price_history) - history_limit)
        
        recent_prices = price_history[start_index:]
        recent_states = state_history[start_index:]
        
        history_lines = []
        for i, (price, state) in enumerate(zip(recent_prices, recent_states)):
            period_num = start_index + i + 1
            history_lines.append(f"Period {period_num} | {state} | ${price:.2f}")
            
        formatted_history_table = "\n".join(history_lines) if history_lines else "No history yet."

        # --- 2. Calculate Economic Variables for Prompt ---
        constants = game_config.constants
        base_demand = constants.get('base_demand', 120)
        slope = constants.get('demand_slope', 1)
        mc = constants.get('marginal_cost', 20)
        q_cournot = constants.get('cournot_quantity', 25)
        num_players = constants.get('number_of_players', 3) 
        q_collusive = constants.get('collusive_quantity', 17)

        # A. Expected Price War Profit (All Cournot)
        total_q_war = q_cournot * num_players
        price_war_price = max(0, base_demand - (slope * total_q_war))
        price_war_profit = (price_war_price - mc) * q_cournot

        # B. Immediate Defect Profit (You Cournot, Others Collusive)
        total_q_defect = ((num_players - 1) * q_collusive) + q_cournot
        defect_price = max(0, base_demand - (slope * total_q_defect))
        immediate_defect_profit = (defect_price - mc) * q_cournot
        
        # C. Expected Cooperate Profit (All Collusive) - NEW CALCULATION
        total_q_coop = q_collusive * num_players
        expected_price_coop = max(0, base_demand - (slope * total_q_coop))
        expected_cooperate_profit = (expected_price_coop - mc) * q_collusive

        variables = get_prompt_variables(
            game_config,
            player_id=player_id,
            current_round=current_period,
            current_market_state=game_state.get('market_state', 'Collusive')
        )
        
        # Inject variables for the prompt formatter
        variables.update({
            'formatted_history_table': formatted_history_table,
            'expected_price_war_price': f"{price_war_price:.2f}",
            'expected_price_war_profit': f"{price_war_profit:.2f}",
            'immediate_defect_profit': f"{immediate_defect_profit:.2f}",
            'expected_cooperate_profit': f"{expected_cooperate_profit:.2f}"  # ADD THIS LINE
        })
        
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        """
        Parses the LLM's 'Cooperate' or 'Defect' decision and returns the corresponding quantity.
        """
        if not self.game_config:
            self.logger.error("Game config not initialized. Cannot map action to quantity.")
            return None

        constants = self.game_config.constants
        collusive_quantity = constants.get('collusive_quantity')
        cournot_quantity = constants.get('cournot_quantity')

        # First, try parsing a structured JSON response
        json_action = self.robust_json_parse(response)
        if json_action and 'action' in json_action and isinstance(json_action['action'], str):
            action_str = json_action['action'].lower()
            if action_str == 'cooperate':
                return {'quantity': collusive_quantity}
            if action_str == 'defect':
                return {'quantity': cournot_quantity}

        # Fallback to simple keyword search
        text_lower = response.lower()
        if 'cooperate' in text_lower:
            return {'quantity': collusive_quantity}
        if 'defect' in text_lower:
            return {'quantity': cournot_quantity}

        self.logger.warning(f"[{call_id}] Could not parse a valid action for {player_id}. Defaulting to cooperation.")
        return {'quantity': collusive_quantity}


    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates player payoffs based on total quantity and the current demand shock."""
        constants = game_config.constants
        base_demand = constants.get('base_demand', 120)
        demand_slope = constants.get('demand_slope', 1)
        marginal_cost = constants.get('marginal_cost', 20)
        
        current_period = game_state['current_period']
        demand_shock = game_state['demand_shocks'][current_period - 1]

        quantities = {pid: action.get('quantity', constants.get('cournot_quantity', 25)) for pid, action in actions.items()}
        total_quantity = sum(quantities.values())

        market_price = max(0, base_demand - (demand_slope * total_quantity) + demand_shock)
        
        payoffs = {}
        for player_id, quantity in quantities.items():
            profit = (market_price - marginal_cost) * quantity
            payoffs[player_id] = profit
        
        game_state['current_market_price'] = market_price
        return payoffs

    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig, payoffs: Dict[str, float]) -> Dict:
        """Updates the game state using the State Transition Algorithm."""
        constants = game_config.constants
        trigger_price = constants.get('trigger_price', 66)
        punishment_periods = constants.get('punishment_duration', 2)
        market_price = game_state.get('current_market_price', 0)
        
        # Capture the state BEFORE updating it, to record what state generated this price
        current_state_label = game_state['market_state']

        # Update histories
        game_state['price_history'].append(market_price)
        game_state['state_history'].append(current_state_label)
        
        for pid, action in actions.items():
            game_state['quantity_history'][pid].append(action.get('quantity', constants.get('cournot_quantity', 25)))
            game_state['profit_history'][pid].append(payoffs.get(pid, 0.0))

        # State Transition Algorithm
        if game_state['market_state'] == 'Collusive':
            if market_price < trigger_price:
                game_state['market_state'] = 'Reversionary'
                game_state['punishment_timer'] = punishment_periods
        else: # Reversionary state
            game_state['punishment_timer'] -= 1
            if game_state['punishment_timer'] <= 0:
                game_state['market_state'] = 'Collusive'

        game_state['current_period'] += 1
        return game_state
    
    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers round-specific outcomes for detailed logging."""
        current_period_index = game_state.get('current_period', 1) - 1
        return {
            "period": current_period_index + 1,
            "market_state": game_state.get('market_state', 'Collusive'),
            "demand_shock": game_state.get('demand_shocks', [])[current_period_index],
            "market_price": game_state.get('current_market_price', 0),
            "actions": actions,
            "payoffs": payoffs
        }