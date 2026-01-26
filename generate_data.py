import json
import numpy as np
import random
from pathlib import Path
import sys
import logging
from typing import Dict, Any

# Add the project root to the Python path to allow for package imports
sys.path.append(str(Path(__file__).resolve().parent))

from config.config import get_all_game_configs, get_experiment_config, GameConfig, get_data_dir

def generate_spulber_data(game_config: GameConfig, num_sims: int) -> Dict[str, Any]:
    """Generates private cost data for all defenders in the Spulber game."""
    constants = game_config.constants
    # We generate based on the config passed (usually the 5-player config)
    num_defenders = constants["number_of_players"] - 1
    
    defender_costs = {}
    for i in range(1, num_defenders + 1):
        costs = np.random.normal(
            constants["rival_cost_mean"],
            constants["rival_cost_std"],
            num_sims
        )
        defender_costs[f'defender_{i}'] = np.maximum(0, costs).round(2).tolist()

    # The challenger's cost is fixed in the config, not generated randomly here.
    private_costs = {"challenger": constants.get("your_cost"), **defender_costs}
    return {"player_private_costs": private_costs}

def generate_green_porter_data(game_config: GameConfig, num_sims: int) -> Dict[str, Any]:
    """Generates lists of random demand shocks for the Green & Porter game."""
    constants = game_config.constants
    time_horizon = constants["time_horizon"]
    
    shock_lists = [
        np.random.normal(
            constants.get("demand_shock_mean", 0),
            constants["demand_shock_std"],
            time_horizon
        ).round(2).tolist()
        for _ in range(num_sims)
    ]
    return {"demand_shocks": shock_lists}

def generate_athey_bagwell_data(game_config: GameConfig, num_sims: int) -> Dict[str, Any]:
    """Generates persistent true cost type streams for all players in the Athey & Bagwell game."""
    constants = game_config.constants
    num_players = constants["number_of_players"]
    time_horizon = constants["time_horizon"]
    persistence_prob = constants["persistence_probability"]
    player_ids = ['challenger'] + [f'defender_{i}' for i in range(1, num_players)]

    cost_streams = {}
    for player_id in player_ids:
        sim_streams = []
        for _ in range(num_sims):
            stream = ["low" if random.random() < 0.5 else "high"]
            for _ in range(1, time_horizon):
                next_cost = stream[-1] if random.random() < persistence_prob else ("low" if stream[-1] == "high" else "high")
                stream.append(next_cost)
            sim_streams.append(stream)
        cost_streams[player_id] = sim_streams
        
    return {"player_true_costs": cost_streams}

def main():
    """
    Generates and saves all master datasets required for the experiments,
    ensuring 3-player data is a consistent subset of 5-player data.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    exp_config = get_experiment_config()
    seed = exp_config.get("random_seed")
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    num_sims = exp_config.get("main_experiment_simulations", 50)
    source_datasets = {}
    final_datasets = {}
    
    game_data_generators = {
        "spulber": generate_spulber_data,
        "green_porter": generate_green_porter_data,
        "athey_bagwell": generate_athey_bagwell_data
    }
    
    all_games = ["green_porter", "spulber", "athey_bagwell"]
    
    # --- Stage 1: Generate the base source datasets (Max N=5) ---
    logger.info("--- Stage 1: Generating base source datasets ---")
    all_configs_map = {game: get_all_game_configs(game) for game in all_games}

    # 1. Green-Porter: One source for all (shocks are independent of player count)
    gp_config = next((c for c in all_configs_map['green_porter']), None)
    if gp_config:
        logger.info("Generating base dataset for Green & Porter...")
        source_datasets['green_porter_base'] = generate_green_porter_data(gp_config, num_sims)

    # 2. Spulber and Athey-Bagwell: Generate 5-player master data
    for game_name in ['spulber', 'athey_bagwell']:
        # Find the 5-player config
        base_5p_config = next((c for c in all_configs_map[game_name] if c.constants.get('number_of_players') == 5), None)
        
        if base_5p_config:
            logger.info(f"Generating 5-player base dataset for {game_name}...")
            source_datasets[f'{game_name}_5_player_base'] = game_data_generators[game_name](base_5p_config, num_sims)
        else:
            logger.warning(f"No 5-player config found for {game_name}. Cannot generate master subsettable data.")


    # --- Stage 2: Create final datasets by copying or subsetting ---
    logger.info("\n--- Stage 2: Deriving final datasets from base sources ---")
    
    for game_name in all_games:
        for config in all_configs_map[game_name]:
            dataset_key = f"{config.game_name}_{config.experiment_type}_{config.condition_name}"
            
            # Case A: Green-Porter (Shared Shocks)
            if game_name == 'green_porter':
                final_datasets[dataset_key] = source_datasets['green_porter_base']
                continue

            # Case B: Spulber & Athey-Bagwell (Subset Logic)
            source_key = f"{game_name}_5_player_base"
            
            if source_key not in source_datasets:
                logger.error(f"Skipping {dataset_key}: Source {source_key} missing.")
                continue

            source_data = source_datasets[source_key]
            target_num_players = config.constants.get('number_of_players', 3)
            
            if target_num_players == 5:
                # Use the full dataset
                logger.info(f"Assigning full 5-player data to '{dataset_key}'")
                final_datasets[dataset_key] = source_data
            
            elif target_num_players == 3:
                # Subset to Challenger + Defender 1 + Defender 2
                logger.info(f"Deriving 3-player subset for '{dataset_key}'")
                subset_data = {}
                
                # Iterate through top-level keys (e.g., "player_private_costs" or "player_true_costs")
                for data_type, content in source_data.items():
                    if isinstance(content, dict) and 'defender_1' in content:
                        # It's player-indexed data
                        player_subset = {}
                        # Always keep challenger
                        if 'challenger' in content:
                            player_subset['challenger'] = content['challenger']
                        # Keep defenders 1 to N-1
                        for i in range(1, target_num_players):
                            pid = f"defender_{i}"
                            if pid in content:
                                player_subset[pid] = content[pid]
                        subset_data[data_type] = player_subset
                    else:
                        # Global data (unlikely here but safe to keep)
                        subset_data[data_type] = content
                
                final_datasets[dataset_key] = subset_data
            
            else:
                logger.warning(f"Unsupported player count {target_num_players} for {dataset_key}. Skipping.")


    # --- Save to Disk ---
    output_dir = get_data_dir()
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = output_dir / "master_datasets.json"

    with open(output_path, 'w') as f:
        json.dump(final_datasets, f, indent=2)
        
    logger.info(f"\n✅ Successfully generated {len(final_datasets)} master datasets.")
    logger.info(f"   Saved to: {output_path}")

if __name__ == "__main__":
    main()