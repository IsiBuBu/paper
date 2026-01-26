import random
import json
import time
from typing import Optional

from .base_agent import BaseLLMAgent, AgentResponse
from config.config import GameConfig

class RandomAgent(BaseLLMAgent):
    """
    A non-strategic baseline agent that selects valid actions uniformly at random.
    It uses the game_config object for deterministic game identification.
    """

    def __init__(self, model_name: str, player_id: str, seed: int = None):
        super().__init__(model_name, player_id)
        if seed is not None:
            random.seed(seed)

    def get_action(self, game_config: GameConfig) -> str:
        """
        Determines the game from the game_config and returns a random, valid action.
        """
        action = {}
        game_name = game_config.game_name
        constants = game_config.constants

        if game_name == "athey_bagwell":
            report = random.choice(["high", "low"])
            action = {"report": report}
        elif game_name == "green_porter":
            chosen_action = random.choice(["Cooperate", "Defect"])
            action = {"action": chosen_action}
        elif game_name == "salop":
            # Chooses a random price between marginal cost and reservation price
            low = constants.get('marginal_cost', 8)
            high = constants.get('reservation_price', 30)
            price = random.uniform(low, high)
            action = {"price": round(price, 2)}
        elif game_name == "spulber":
            # Chooses a random price between own cost and demand intercept
            # Note: Random agent doesn't know private cost here easily without state,
            # so we use a safe fallback range from constants
            low = constants.get('your_cost', 10) # Fallback
            high = constants.get('demand_intercept', 100)
            price = random.uniform(low, high)
            action = {"price": round(price, 2)}
        else:
            self.logger.warning(f"Unknown game '{game_name}' for RandomAgent. Defaulting to generic price.")
            action = {"price": random.uniform(10, 50)}
            
        return json.dumps(action)

    async def get_response(self, prompt: str, call_id: str, game_config: GameConfig, seed: Optional[int] = None) -> AgentResponse:
        """
        Wraps get_action to provide a standardized AgentResponse object.
        Populates new metrics fields with zero/empty values.
        """
        start_time = time.time()
        try:
            content = self.get_action(game_config)
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