import logging
from .base_agent import BaseLLMAgent, AgentResponse
from .experiment_agent import ExperimentAgent
from .random_agent import RandomAgent

def create_agent(model_name: str, player_id: str, mock_mode: bool = False, **kwargs) -> BaseLLMAgent:
    """
    Factory function to create an appropriate agent based on the model name.
    """
    logger = logging.getLogger(__name__)

    if mock_mode:
        logger.info(f"🎭 MOCK MODE: Creating RandomAgent for {model_name} as {player_id}")
        return RandomAgent(model_name="random_mock", player_id=player_id)

    # 1. Baseline Agents
    if model_name == 'random_agent':
        logger.info(f"🎲 Creating RandomAgent baseline for {model_name} as {player_id}")
        return RandomAgent(model_name=model_name, player_id=player_id)
    
    # 2. Experiment Agents (OpenAI-Compatible / vLLM)
    # (Qwen, Llama) go through the ExperimentAgent which uses the OpenAI SDK.
    else:
        logger.info(f"🤖 Creating ExperimentAgent for {model_name} as {player_id}")
        return ExperimentAgent(model_name=model_name, player_id=player_id)

__all__ = [
    "BaseLLMAgent",
    "AgentResponse",
    "ExperimentAgent",
    "RandomAgent",
    "create_agent"
]