import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from functools import lru_cache

# --- Data Class for Game Configuration ---

@dataclass
class GameConfig:
    """Represents the fully resolved configuration for a single game condition."""
    game_name: str
    experiment_type: str
    condition_name: str
    constants: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Converts the GameConfig instance to a dictionary."""
        return {
            "game_name": self.game_name,
            "experiment_type": self.experiment_type,
            "condition_name": self.condition_name,
            "constants": self.constants
        }

# --- Core Configuration Loading (Cached) ---

@lru_cache(maxsize=1)
def load_config() -> Dict[str, Any]:
    """
    Loads and caches the main JSON configuration file using a path
    relative to this script, making it robust to where it's called from.
    """
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.json"
    
    if not config_path.exists():
        # Fallback to looking in current directory if run from config folder
        config_path = Path("config.json")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at '{config_path}' or '{base_dir / 'config.json'}'")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# --- Model Configuration Accessors ---

def get_challenger_models() -> List[str]:
    """Returns the list of challenger model names."""
    return load_config().get('models', {}).get('challenger_models', [])

def get_defender_model() -> str:
    """Returns the defender model name."""
    return load_config().get('models', {}).get('defender_model', '')

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Returns the configuration for a specific model."""
    configs = load_config().get('model_configs', {})
    # If explicit config exists, return it
    if model_name in configs:
        return configs[model_name]
    # Fallback: return a minimal default config if not found
    return {"model_name": model_name, "temperature": 0.0}

def get_model_display_name(model_name: str) -> str:
    """Returns the display name for a model, defaulting to the model name."""
    return get_model_config(model_name).get('display_name', model_name)

# --- Experiment and Game Configuration Accessors ---

def get_experiment_config() -> Dict[str, Any]:
    """Returns the main experiment configuration dictionary."""
    return load_config().get('experiment_config', {})

def get_simulation_count(game_name: str) -> int:
    """
    Gets the number of simulations for a specific game.
    Retrieves from game_configs -> [game_name] -> simulation_config -> num_simulations.
    """
    game_conf = load_config().get('game_configs', {}).get(game_name, {})
    sim_conf = game_conf.get('simulation_config', {})
    # Default to 50 if not specified
    return sim_conf.get('num_simulations', 50)

def get_all_game_configs(game_name: str) -> List[GameConfig]:
    """
    Generates a set of GameConfig instances for a given game based on
    'baseline' and 'structural_variations'.
    """
    configs = []
    all_game_data = load_config().get('game_configs', {})
    
    if game_name not in all_game_data:
        return []
        
    game_data = all_game_data[game_name]
    baseline_constants = game_data.get('baseline', {}).copy()
    
    # Merge challenger specific config if present (e.g. for Spulber)
    baseline_constants.update(game_data.get('challenger_config', {}))

    # 1. Baseline Config
    # If no structural variations exist, or just as a base, we always include baseline?
    # Usually in experiments we treat baseline as one condition and variations as others.
    # We'll create a 'baseline' condition first.
    configs.append(GameConfig(
        game_name=game_name,
        experiment_type='baseline',
        condition_name='baseline',
        constants=baseline_constants
    ))

    # 2. Structural Variations
    structural_variations = game_data.get('structural_variations', {})
    for struct_name, struct_params in structural_variations.items():
        # Copy baseline and override with variation params
        struct_constants = baseline_constants.copy()
        struct_constants.update(struct_params)
        
        configs.append(GameConfig(
            game_name=game_name,
            experiment_type='structural_variations',
            condition_name=struct_name,
            constants=struct_constants
        ))

    return configs

# --- Utility Function for Prompt Variables ---

def get_prompt_variables(game_config: GameConfig, player_id: str, **kwargs) -> Dict[str, Any]:
    """Consolidates all variables needed to format a game prompt."""
    if not isinstance(game_config, GameConfig):
        raise TypeError(f"Expected GameConfig, but got {type(game_config)}")
    variables = game_config.constants.copy()
    
    if 'cost_types' in variables and isinstance(variables['cost_types'], dict):
        variables['high_cost'] = variables['cost_types'].get('high')
        variables['low_cost'] = variables['cost_types'].get('low')

    variables.update({
        'player_id': player_id,
        'number_of_competitors': variables.get('number_of_players', 1) - 1,
        **kwargs
    })

    if 'price_history' in variables:
        variables['len_price_history'] = len(variables['price_history'])
    return variables

# --- Directory Path Accessors ---

def get_main_output_dir() -> Path:
    """Returns the main output directory Path object."""
    return Path(load_config().get('output', {}).get('main_output_dir', 'output'))

def get_data_dir() -> Path:
    """Returns the path for the data directory."""
    return get_main_output_dir() / load_config().get('output', {}).get('data_dir', 'data')

def get_experiments_dir() -> Path:
    """Returns the path for the experiments (raw results) directory."""
    return get_main_output_dir() / load_config().get('output', {}).get('experiments_dir', 'experiments')

def get_analysis_dir() -> Path:
    """Returns the path for the analysis output directory."""
    return get_main_output_dir() / load_config().get('output', {}).get('analysis_dir', 'analysis')