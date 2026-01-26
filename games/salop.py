# games/salop.py

from typing import Dict, Any, Optional, List
from config.config import GameConfig, get_prompt_variables
from games.base_game import StaticGame, PriceParsingMixin

class SalopGame(StaticGame, PriceParsingMixin):
    """
    Implements the Salop (1979) spatial competition game.
    """

    def __init__(self):
        super().__init__("salop")

    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """
        Initializes the state for a new Salop game simulation.
        CRITICAL FIX: Calculates derived geometric variables like 'distance_to_neighbor'.
        """
        constants = game_config.constants.copy()
        
        # Calculate derived variables required by the prompt
        circumference = constants.get('circumference', 1.0)
        num_players = constants.get('number_of_players', 3)
        
        # Avoid division by zero
        if num_players > 0:
            constants['distance_to_neighbor'] = circumference / num_players
        else:
            constants['distance_to_neighbor'] = 0.33 # Fallback

        return {
            'game_type': 'static',
            'current_round': 1,
            'simulation_id': simulation_id,
            'constants': constants # Stores the updated constants with distance_to_neighbor
        }

    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a player using the game's template and configuration."""
        # We use the constants from game_state because they contain the calculated distance_to_neighbor
        current_constants = game_state.get('constants', {})
        
        # Merge state constants with game_config to ensure variables are found
        # (game_state variables take precedence if they exist)
        variables = get_prompt_variables(game_config, player_id=player_id, **game_state)
        
        # Explicitly update variables with the calculated constants to fix the KeyError
        variables.update(current_constants)
        
        return self.prompt_template.format(**variables)

    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        """Parses the LLM's price decision using the inherited mixin."""
        return self.parse_price_response(response, player_id, call_id)

    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates player payoffs using the Market Share Algorithm."""
        constants = game_state.get('constants', game_config.constants)
        
        marginal_cost = constants.get('marginal_cost', 8)
        fixed_cost = constants.get('fixed_cost', 100)
        transport_cost = constants.get('transport_cost', 1.5)
        market_size = constants.get('market_size', 1000)
        v = constants.get('reservation_price', 30)
        
        # Determine active players from actions to support partial lists if needed
        # But generally we assume N players in a circle
        num_firms = len(actions)
        if num_firms < 2: 
            return {pid: 0.0 for pid in actions}

        prices = {pid: action.get('price', v) for pid, action in actions.items() if isinstance(action, dict)}
        player_ids = sorted(prices.keys())
        
        payoffs = {}
        quantities = {}
        
        for i, player_id in enumerate(player_ids):
            p_i = prices[player_id]
            # Neighbors (Circular indices)
            p_left = prices[player_ids[(i - 1 + num_firms) % num_firms]]
            p_right = prices[player_ids[(i + 1) % num_firms]]

            # Competitive Boundary (Indifferent consumer location)
            # Standard Salop formula: x = (p_neighbor - p_i) / 2t + 1/2N
            segment_length = 1.0 / num_firms
            
            x_left = (p_left - p_i) / (2 * transport_cost) + (segment_length / 2)
            x_right = (p_right - p_i) / (2 * transport_cost) + (segment_length / 2)

            # Monopoly Boundary (Reservation price constraint)
            x_max = (v - p_i) / transport_cost

            # Effective Reach (Min of competitive vs monopoly)
            reach_left = max(0, min(x_left, x_max))
            reach_right = max(0, min(x_right, x_max))
            
            # Constrain total reach to physical availability (cannot overlap beyond neighbor)
            # Note: A full Salop implementation handles overlapping regions more dynamically,
            # but this approximation holds for the "Competitive vs Monopoly" testing.
            total_reach = min(1.0, reach_left + reach_right)
            
            quantity_sold = total_reach * market_size
            quantities[player_id] = quantity_sold
            
            profit = (p_i - marginal_cost) * quantity_sold - fixed_cost
            payoffs[player_id] = profit
            
        if game_state is not None:
            game_state['player_quantities'] = quantities
        return payoffs

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers all relevant data from the Salop game for detailed logging."""
        return {
            "constants": game_state.get('constants', game_config.constants),
            "player_quantities": game_state.get('player_quantities', {})
        }