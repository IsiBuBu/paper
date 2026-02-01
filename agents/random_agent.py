import random
import json
import time
from typing import Optional

from .base_agent import BaseLLMAgent, AgentResponse
from config.config import GameConfig

class RandomAgent(BaseLLMAgent):
    """
    A non-strategic baseline agent that selects valid actions uniformly at random.
    
    SCIENTIFIC NOTE: 
    To strictly compare against Temp=0 LLMs, this agent uses a localized 
    Random instance seeded per-call. This ensures that 'Simulation 5' 
    always produces the same 'Random' moves, regardless of async execution order.
    """

    def __init__(self, model_name: str, player_id: str, seed: int = None):
        super().__init__(model_name, player_id)
        # We do not seed the global random here to avoid side effects
        self.default_seed = seed

    def get_action(self, game_config: GameConfig, seed: Optional[int] = None) -> str:
        """
        Determines the game from the game_config and returns a random, valid action.
        Uses a local RNG instance for thread-safety and reproducibility.
        """
        # Create a local RNG instance to ensure thread safety in async loops
        rng = random.Random(seed if seed is not None else self.default_seed)
        
        action = {}
        game_name = game_config.game_name
        constants = game_config.constants

        if game_name == "athey_bagwell":
            # 50/50 split creates the baseline for the "Indifference Point"
            report = rng.choice(["high", "low"])
            action = {"report": report}

        elif game_name == "green_porter":
            # 50/50 split ensures they will eventually trigger the "Tight Trigger" (58)
            chosen_action = rng.choice(["Cooperate", "Defect"])
            action = {"action": chosen_action}

        elif game_name == "salop":
            # Random price between Cost and Max Willingness.
            # In "Hard Mode" (Reservation=20, MC=8), this tight range allows
            # us to see if they accidentally hit the optimal or drift.
            low = float(constants.get('marginal_cost', 8))
            high = float(constants.get('reservation_price', 30))
            price = rng.uniform(low, high)
            action = {"price": round(price, 2)}

        elif game_name == "spulber":
            # CRITICAL UPDATE for Spulber:
            # Random Agents do not know their private cost in this architecture (it's in game_state).
            # Therefore, they bid randomly across the entire valid domain [0, Demand Intercept].
            # This allows them to bid BELOW cost (Winner's Curse), which LLMs should avoid.
            low = 0.0 
            high = float(constants.get('demand_intercept', 100))
            price = rng.uniform(low, high)
            action = {"price": round(price, 2)}

        else:
            self.logger.warning(f"Unknown game '{game_name}' for RandomAgent. Defaulting to generic price.")
            action = {"price": rng.uniform(10, 50)}
            
        return json.dumps(action)

    async def get_response(self, prompt: str, call_id: str, game_config: GameConfig, seed: Optional[int] = None) -> AgentResponse:
        """
        Wraps get_action to provide a standardized AgentResponse object.
        Passes the simulation-specific seed down to the decision logic.
        """
        start_time = time.time()
        try:
            # Pass the seed to ensure reproducibility per simulation run
            content = self.get_action(game_config, seed=seed)
            duration = time.time() - start_time
            
            return AgentResponse(
                content=content,
                model=self.model_name,
                success=True,
                reasoning_content=None,
                tokens_used=0,
                output_tokens=0,
                thinking_tokens=0,
                reasoning_char_count=0,
                response_time=duration
            )
        except Exception as e:
            self.logger.error(f"[{call_id}] RandomAgent failed: {e}")
            return AgentResponse(
                content="",
                model=self.model_name,
                success=False,
                error=str(e),
                response_time=time.time() - start_time
            )