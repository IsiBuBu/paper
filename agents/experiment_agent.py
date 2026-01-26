import asyncio
import logging
import time
import os
import re
import json
from typing import Dict, Any, Optional

# Reliability libraries
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

# OpenAI SDK
from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError, APIStatusError

from .base_agent import BaseLLMAgent, AgentResponse
from config.config import GameConfig, load_config

class ExperimentAgent(BaseLLMAgent):
    """
    An agent that uses the OpenAI-compatible API (e.g., vLLM, DeepSeek, OpenAI) to make decisions.
    """

    def __init__(self, model_name: str, player_id: str):
        super().__init__(model_name, player_id)
        
        # Load specific config for this model
        global_config = load_config()
        self.model_config = global_config.get('model_configs', {}).get(model_name, {})
        self.api_config = global_config.get('api_config', {})
        
        # Initialize Client
        self.client = self._initialize_client()

    def _initialize_client(self) -> AsyncOpenAI:
        """Initializes the AsyncOpenAI client using environment variables."""
        try:
            # Defaults to localhost for vLLM, or reads standard OPENAI env vars
            base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
            api_key = os.getenv("OPENAI_API_KEY", "EMPTY") 
            
            return AsyncOpenAI(api_key=api_key, base_url=base_url)
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    # --- LAYER 3: EXPONENTIAL BACKOFF ---
    @retry(
        retry=retry_if_exception_type((RateLimitError, APIConnectionError, APIStatusError)),
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6)
    )
    async def _make_api_call(self, **kwargs):
        """Executes the raw API call with auto-retries."""
        return await self.client.chat.completions.create(**kwargs)

    async def get_response(self, prompt: str, call_id: str, game_config: GameConfig, seed: Optional[int] = None) -> AgentResponse:
        """
        Main decision loop.
        """
        start_time = time.time()
        
        # 1. Prepare Arguments
        temperature = self.model_config.get('temperature', 0.0)
        
        api_args = {
            "model": self.model_config.get('model_name', self.model_name),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }

        # JSON Enforcement
        if self.model_config.get('forced_json_response', False):
            api_args["response_format"] = {"type": "json_object"}

        # --- FIX: Explicitly handle Reasoning Effort ---
        # We retrieve the value from config. If it exists and is not 'standard', we pass it.
        # This ensures 'none', 'low', 'medium', 'high' are all respected.
        reasoning_effort = self.model_config.get('reasoning_effort', 'standard')
        
        # vLLM and newer OpenAI endpoints accept 'reasoning_effort' at the top level
        # if the model supports it (like o1, o3-mini, or r1 distillations handled via specific servers).
        if reasoning_effort != 'standard':
            api_args["reasoning_effort"] = reasoning_effort

        try:
            # 2. Execute with Retry Logic
            response = await self._make_api_call(**api_args)
            
            duration = time.time() - start_time
            choice = response.choices[0]
            message = choice.message
            usage = response.usage
            
            raw_content = message.content or ""
            reasoning_content = None
            thinking_tokens = 0
            
            # 3. Deep Extraction Logic
            output_mode = self.model_config.get('reasoning_output', 'none')
            
            # Mode A: API-Native Reasoning Field (DeepSeek R1 / OpenAI o1)
            if output_mode == 'reasoning_tokens':
                if hasattr(message, 'reasoning_content') and message.reasoning_content:
                    reasoning_content = message.reasoning_content
                elif hasattr(usage, 'completion_tokens_details'):
                    # Fallback token count if text is hidden
                    thinking_tokens = getattr(usage.completion_tokens_details, 'reasoning_tokens', 0)
            
            # Mode B: Embedded XML Tags (<think>)
            elif output_mode == 'output_tokens':
                # Extract content inside <think> tags using DOTALL for newlines
                think_match = re.search(r'<think>(.*?)</think>', raw_content, flags=re.DOTALL)
                
                if think_match:
                    reasoning_content = think_match.group(1).strip()
                    # CRITICAL: Remove thinking from final content so JSON parse works
                    raw_content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
                    
                    # Estimate tokens if API didn't report them
                    if thinking_tokens == 0:
                        thinking_tokens = len(reasoning_content) // 4
            
            # 4. Final Metrics
            final_content = raw_content.strip()
            reasoning_char_count = len(reasoning_content) if reasoning_content else 0
            
            return AgentResponse(
                content=final_content,
                model=self.model_name,
                success=True,
                reasoning_content=reasoning_content,
                tokens_used=usage.total_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
                thinking_tokens=thinking_tokens,
                reasoning_char_count=reasoning_char_count,
                response_time=duration
            )

        except Exception as e:
            self.logger.error(f"[{call_id}] Failed after retries: {e}")
            return AgentResponse(
                content="", 
                model=self.model_name, 
                success=False, 
                error=str(e),
                response_time=time.time() - start_time
            )