import sys
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from scipy import stats
import numpy as np

# 1. Setup Python Path (Must be AFTER imports of sys/Path)
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# 2. Local Imports
from config.config import get_experiments_dir
from metrics import (
    GameResult,
    PerformanceMetricsCalculator,
    MAgICMetricsCalculator,
)

# Helper function for confidence interval
def get_ci_half_width(data):
    if len(data) < 2:
        return 0.0
    sem = stats.sem(data)
    if sem == 0:
        return 0.0
    return sem * stats.t.ppf((1 + 0.95) / 2., len(data)-1)

class SummaryCreator:
    """
    Processes the raw experiment results into flat CSV files.
    """

    def __init__(self, analysis_dir: str = "output/analysis"):
        self.analysis_dir = Path(analysis_dir)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        self.experiments_dir = get_experiments_dir()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.perf_calc = PerformanceMetricsCalculator()
        self.magic_calc = MAgICMetricsCalculator()

    def create_all_summaries(self):
        self.logger.info("Creating summary CSV files from raw experiment results...")
        all_perf_records = []
        all_magic_records = []

        # Use rglob to find files in deep subdirectories
        json_files = list(self.experiments_dir.rglob("*_competition_result*.json"))
        
        if not json_files:
            self.logger.warning(f"No raw experiment result JSON files found in {self.experiments_dir}. Cannot create summary CSVs.")
            return

        self.logger.info(f"Found {len(json_files)} result files to process.")

        for result_file in json_files:
            try:
                with open(result_file, 'r') as f:
                    data = json.load(f)

                if not data.get('simulation_results'):
                    continue
                    
                challenger_model = data.get('challenger_model')
                condition_name = data.get('condition_name')
                game_name = data.get('game_name')

                game_results = [GameResult(**sim) for sim in data.get('simulation_results', [])]
                
                if not game_results:
                    continue

                for i, result in enumerate(game_results):
                    # A. Performance Metrics
                    game_specific_perf = self.perf_calc.calculate_all_performance_metrics([result], 'challenger')
                    for metric_name, metric_obj in game_specific_perf.items():
                        all_perf_records.append({
                            'game': game_name, 
                            'model': challenger_model, 
                            'condition': condition_name, 
                            'metric': metric_name, 
                            'value': metric_obj.value
                        })
                    
                    # B. MAgIC Metrics
                    magic_metrics = self.magic_calc.calculate_all_magic_metrics([result], 'challenger')
                    for metric_name, metric_obj in magic_metrics.items():
                        all_magic_records.append({
                            'game': game_name, 
                            'model': challenger_model, 
                            'condition': condition_name, 
                            'metric': metric_name, 
                            'value': metric_obj.value
                        })

            except Exception as e:
                self.logger.error(f"Error processing file {result_file}: {e}")
                continue

        # Aggregate and Save Performance Metrics
        perf_df = pd.DataFrame(all_perf_records)
        if not perf_df.empty:
            perf_agg_df = perf_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg(['mean', 'std', ('ci_95', get_ci_half_width)]).reset_index()
            self._save_to_csv(perf_agg_df, "performance_metrics.csv")
        else:
            self.logger.warning("No performance records found.")

        # Aggregate and Save MAgIC Metrics
        magic_df = pd.DataFrame(all_magic_records)
        if not magic_df.empty:
            magic_agg_df = magic_df.groupby(['game', 'model', 'condition', 'metric'])['value'].agg(['mean','std', ('ci_95', get_ci_half_width)]).reset_index()
            self._save_to_csv(magic_agg_df, "magic_behavioral_metrics.csv")
        else:
            self.logger.warning("No MAgIC records found.")
            
        self.logger.info("Successfully created and saved aggregated summary CSV files.")

    def _save_to_csv(self, df: pd.DataFrame, filename: str):
        if df.empty: return
        output_path = self.analysis_dir / filename
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved data to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    SummaryCreator().create_all_summaries()