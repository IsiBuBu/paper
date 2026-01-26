# agents/base_agent.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# Import GameConfig for type hinting
from config.config import GameConfig

@dataclass
class AgentResponse:
    """
    A standardized container for the response from any agent.
    
    Attributes:
        content: The cleaned answer text (JSON ready, NO thinking tags).
        model: Model identifier.
        success: Boolean status.
        reasoning_content: The extracted reasoning/thinking text.
        error: Error message if failed.
        tokens_used: Total tokens (prompt + completion).
        output_tokens: Completion tokens (includes thinking if model outputs it there).
        thinking_tokens: Count of tokens used for reasoning (from API usage or estimated).
        reasoning_char_count: Character count of the reasoning content.
        response_time: Latency in seconds.
    """
    content: str
    model: str
    success: bool
    reasoning_content: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    reasoning_char_count: int = 0
    response_time: float = 0.0

class BaseLLMAgent(ABC):
    """
    An abstract base class that defines the common interface for all agents.
    """

    def __init__(self, model_name: str, player_id: str):
        """
        Initializes the agent with a model name and a player ID.
        """
        self.model_name = model_name
        self.player_id = player_id
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{player_id}")

    @abstractmethod
    async def get_response(self, prompt: str, call_id: str, game_config: GameConfig, seed: Optional[int] = None) -> AgentResponse:
        """
        Generates a response from the agent.
        """
        pass