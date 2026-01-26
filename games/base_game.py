# games/base_game.py

import json
import re
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional

from config.config import GameConfig

# --- Helper Function for Numeric Extraction ---

def extract_numeric_value(text: str, field_name: str) -> Optional[float]:
    """Extracts the first numeric value associated with a given field name in a string."""
    # Pattern to find "field": number, "field": "number", or field: number
    # Handles potential extra spacing or newlines
    pattern = rf'"{field_name}"\s*:\s*"?(\d+\.?\d*)"?'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return None

# --- Parsing Mixins ---

class PriceParsingMixin:
    """Mixin for games that require parsing a 'price' or 'bid' from LLM responses."""
    def parse_price_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Robustly parses a price or bid from an LLM's text response."""
        # 1. JSON Parse
        if (json_action := self.robust_json_parse(response)):
            for key in ['price', 'bid']:
                if key in json_action and isinstance(json_action[key], (int, float)):
                    return {key: float(json_action[key])}
                # Handle string numbers inside JSON
                elif key in json_action and isinstance(json_action[key], str):
                    try:
                        return {key: float(json_action[key])}
                    except ValueError:
                        pass

        # 2. Regex Fallback for key-value pairs
        for key in ['price', 'bid']:
            if (value := extract_numeric_value(response, key)) is not None:
                return {key: value}
        
        # 3. Final Fallback: First floating point number found
        # Be careful this doesn't pick up the round number or similar integers if possible
        # We look for something that looks explicitly like a price (digits.digits) or just digits
        if (match := re.search(r'(\d+\.?\d+)', response)):
            return {'price': float(match.group(1))}
            
        return None

class QuantityParsingMixin:
    """Mixin for games that require parsing a 'quantity'."""
    def parse_quantity_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Robustly parses a quantity from an LLM's text response."""
        if (json_action := self.robust_json_parse(response)):
            if 'quantity' in json_action:
                try:
                    return {'quantity': float(json_action['quantity'])}
                except (ValueError, TypeError):
                    pass

        if (value := extract_numeric_value(response, 'quantity')) is not None:
            return {'quantity': value}
            
        if (match := re.search(r'(\d+\.?\d+)', response)):
            return {'quantity': float(match.group(1))}

        return None

class ReportParsingMixin:
    """Mixin for games that require parsing a 'high' or 'low' report."""
    def parse_report_response(self, response: str, player_id: str, call_id: str) -> Optional[Dict[str, Any]]:
        """Robustly parses a 'high' or 'low' report from an LLM's text response."""
        # 1. JSON Parse
        if (json_action := self.robust_json_parse(response)):
            if 'report' in json_action and isinstance(json_action['report'], str):
                report = json_action['report'].lower()
                if 'low' in report: return {'report': 'low'}
                if 'high' in report: return {'report': 'high'}
        
        text_lower = response.lower()
        
        # 2. Key-Value Regex Lookahead
        # Looks for "report" followed closely by high or low
        if re.search(r'report.*?low', text_lower): return {'report': 'low'}
        if re.search(r'report.*?high', text_lower): return {'report': 'high'}
        
        # 3. Simple Keyword (Risky but necessary fallback)
        # Prioritize explicit words over substring matches if possible
        if 'low' in text_lower: return {'report': 'low'}
        if 'high' in text_lower: return {'report': 'high'}
            
        return None

# --- Abstract Base Classes for Games ---

class EconomicGame(ABC):
    """Abstract base class for all economic games."""
    
    def __init__(self, game_name: str):
        self.game_name = game_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.prompt_template = self._load_prompt_template(game_name)

    def _load_prompt_template(self, game_name: str) -> str:
        """Loads the prompt template for the game from the /prompts directory."""
        prompt_path = Path(f"prompts/{game_name}.md")
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt template not found for {game_name} at {prompt_path}")
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def robust_json_parse(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Finds and parses the first valid JSON object in a string.
        Explicitly removes <think> tags before parsing to handle reasoning leaks.
        """
        if not response:
            return None

        # 1. Strip <think> tags (redundant safety)
        clean_text = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        
        # 2. Find first JSON-like block { ... }
        # DOTALL allows . to match newlines
        match = re.search(r'\{.*\}', clean_text, re.DOTALL)
        
        if match:
            json_str = match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 3. Try to clean common JSON errors (like trailing commas) if simple parse fails
                # (Optional advanced logic can go here)
                pass
        return None

    @abstractmethod
    def generate_player_prompt(self, player_id: str, game_state: Dict, game_config: GameConfig) -> str:
        """Generates a prompt for a specific player based on the current game state."""
        pass

    @abstractmethod
    def parse_llm_response(self, response: str, player_id: str, call_id: str, stage: int = 1) -> Optional[Dict[str, Any]]:
        """
        Parses an LLM's string response into a structured action dictionary.
        The 'stage' parameter is included for compatibility with multi-stage games.
        """
        pass

    @abstractmethod
    def calculate_payoffs(self, actions: Dict[str, Any], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, float]:
        """Calculates the payoffs for all players based on their actions."""
        pass

    def get_game_data_for_logging(self, actions: Dict[str, Any], payoffs: Dict[str, float], game_config: GameConfig, game_state: Optional[Dict] = None) -> Dict[str, Any]:
        """Gathers all relevant data from a game round for logging and analysis."""
        return {
            'game_name': self.game_name,
            'actions': actions,
            'payoffs': payoffs,
            'constants': game_config.constants,
            'game_state': game_state
        }

class StaticGame(EconomicGame):
    """Base class for static (single-round) games like Salop and Spulber."""
    
    @abstractmethod
    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes the state for a new simulation."""
        pass

class DynamicGame(EconomicGame):
    """Base class for dynamic (multi-round) games like Green-Porter and Athey-Bagwell."""

    @abstractmethod
    def initialize_game_state(self, game_config: GameConfig, simulation_id: int = 0) -> Dict[str, Any]:
        """Initializes the state for a new simulation."""
        pass

    @abstractmethod
    def update_game_state(self, game_state: Dict, actions: Dict[str, Any], game_config: GameConfig) -> Dict:
        """Updates the game state after a round is completed."""
        pass