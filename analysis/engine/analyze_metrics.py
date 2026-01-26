import sys
import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# 1. Setup Python Path (Must be AFTER imports of sys/Path)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 2. Local Imports
from config.config import get_challenger_models, get_defender_model, get_experiments_dir, get_analysis_dir
from metrics import (
    GameResult,
    PlayerMetrics,
    ExperimentResults,
    PerformanceMetricsCalculator,
    MAgICMetricsCalculator
)

class MetricsAnalyzer:
    """
    Processes raw simulation results to calculate and save aggregate
    Performance and MAgIC metrics for each experimental condition.
    """

    def __init__(self):
        self.results_dir = get_experiments_dir()
        self.output_dir = get_analysis_dir()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Instantiate calculators
        self.perf_calc = PerformanceMetricsCalculator()
        self.magic_calc = MAgICMetricsCalculator()

    def analyze_all_games(self):
        """
        Analyzes all available game results and saves the final hierarchal JSON metrics.
        """
        self.logger.info("Starting comprehensive metrics analysis...")
        
        # Iterate over game directories
        # Using rglob to find any folder that looks like a game folder is risky
        # Better: iterate direct children of results_dir
        if not self.results_dir.exists():
             self.logger.error(f"Experiments dir not found: {self.results_dir}")
             return

        game_dirs = [d for d in self.results_dir.iterdir() if d.is_dir()]

        for game_dir in game_dirs:
            game_name = game_dir.name
            self.logger.info(f"--- Analyzing game: {game_name.upper()} ---")

            try:
                all_sim_results = self._load_simulation_results(game_dir)
                if not all_sim_results:
                    self.logger.warning(f"No valid simulation results found for {game_name}. Skipping.")
                    continue

                experiment_results = self._calculate_all_metrics_for_game(game_name, all_sim_results)
                self._save_experiment_results(experiment_results)

            except Exception as e:
                self.logger.error(f"Failed to analyze {game_name}: {e}", exc_info=True)

        self.logger.info("Comprehensive metrics analysis complete.")

    def _load_simulation_results(self, game_dir: Path) -> Dict[tuple, List[GameResult]]:
        """
        Loads raw results using RECURSIVE globbing to handle nested model folders.
        """
        grouped_results = defaultdict(list)

        # Use rglob to find files in any subdirectory depth
        for json_file in game_dir.rglob("*_competition_result*.json"):
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)

                key = (
                    data['challenger_model'],
                    data['experiment_type'],
                    data['condition_name']
                )
                
                for sim_data in data.get('simulation_results', []):
                    grouped_results[key].append(GameResult(**sim_data))
            except Exception as e:
                self.logger.error(f"Error loading {json_file}: {e}")

        self.logger.info(f"Loaded {sum(len(v) for v in grouped_results.values())} total simulations across {len(grouped_results)} conditions.")
        return grouped_results

    def _calculate_all_metrics_for_game(self, game_name: str, all_sim_results: Dict[tuple, List[GameResult]]) -> ExperimentResults:
        exp_results = ExperimentResults(
            game_name=game_name,
            challenger_models=get_challenger_models(),
            defender_model=get_defender_model()
        )

        for (challenger, exp_type, cond_name), sim_results in all_sim_results.items():
            player_metrics = PlayerMetrics(
                player_id=challenger,
                game_name=game_name,
                experiment_type=exp_type,
                condition_name=cond_name
            )

            # Calculate Aggregate Metrics
            player_metrics.performance_metrics = self.perf_calc.calculate_all_performance_metrics(sim_results, 'challenger')
            player_metrics.magic_metrics = self.magic_calc.calculate_all_magic_metrics(sim_results, 'challenger')
            
            if challenger not in exp_results.results:
                exp_results.results[challenger] = {}
            
            exp_results.results[challenger][cond_name] = player_metrics

        return exp_results

    def _save_experiment_results(self, exp_results: ExperimentResults):
        output_file = self.output_dir / f"{exp_results.game_name}_metrics_analysis.json"

        serializable_data = {
            "game_name": exp_results.game_name,
            "challenger_models": exp_results.challenger_models,
            "defender_model": exp_results.defender_model,
            "results": {
                challenger: {
                    cond: {
                        "performance_metrics": {k: v.to_dict() for k, v in metrics.performance_metrics.items()},
                        "magic_metrics": {k: v.to_dict() for k, v in metrics.magic_metrics.items()},
                        "dynamic_metrics": metrics.dynamic_metrics
                    } for cond, metrics in conditions.items()
                } for challenger, conditions in exp_results.results.items()
            }
        }

        with open(output_file, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        self.logger.info(f"Successfully saved analyzed metrics to {output_file}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    MetricsAnalyzer().analyze_all_games()