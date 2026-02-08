"""
Data loading and preprocessing module for MAgIC analysis.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and preprocesses performance and MAgIC behavioral metrics."""
    
    def __init__(self, analysis_dir: Path, config_path: Optional[Path] = None, 
                 experiments_dir: Optional[Path] = None):
        self.analysis_dir = Path(analysis_dir)
        self.config_path = config_path
        self.experiments_dir = experiments_dir
        self.model_configs = {}
        self.display_names = {'random_agent': 'Random'}
        self.family_encoder = None
        
        # Load config if available
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                self.model_configs = config.get('model_configs', {})
                for name, cfg in self.model_configs.items():
                    self.display_names[name] = cfg.get('display_name', name)
    
    def load(self, include_random: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load performance and MAgIC metrics.
        
        Args:
            include_random: Whether to include random agent in results
            
        Returns:
            Tuple of (performance_df, magic_df)
        """
        perf_df = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
        magic_df = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
        
        perf_df = self._filter_models(perf_df, include_random)
        magic_df = self._filter_models(magic_df, include_random)
        
        logger.info(f"Loaded {len(perf_df)} perf, {len(magic_df)} MAgIC rows "
                   f"(random={'in' if include_random else 'ex'}cluded)")
        return perf_df, magic_df
    
    def load_token_data(self) -> pd.DataFrame:
        """Load reasoning token/character count data from experiment files.
        
        Returns:
            DataFrame with columns: model, game, condition, avg_reasoning_chars, std_reasoning_chars
        """
        if not self.experiments_dir or not self.experiments_dir.exists():
            return pd.DataFrame()
        
        records = []
        for exp_file in self.experiments_dir.rglob("*_competition_result*.json"):
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                
                model = data.get('challenger_model', '')
                game = data.get('game_name', '')
                condition = data.get('condition_name', '')
                
                if not model or not game or 'random' in model.lower() or 'gemma' in model.lower():
                    continue
                
                # Extract reasoning character counts
                chars = []
                for sim in data.get('simulation_results', []):
                    gd = sim.get('game_data', {})
                    # Check both game-level and round-level metadata
                    for loc in [gd.get('llm_metadata', {})] + \
                              [r.get('llm_metadata', {}) for r in gd.get('rounds', [])]:
                        for pid, meta in loc.items():
                            if 'challenger' in pid.lower() and isinstance(meta, dict):
                                chars.append(meta.get('reasoning_char_count', 0))
                
                if chars:
                    records.append({
                        'model': model,
                        'game': game,
                        'condition': condition,
                        'avg_reasoning_chars': round(np.mean(chars), 1),
                        'std_reasoning_chars': round(np.std(chars), 1)
                    })
            except Exception as e:
                logger.debug(f"Failed to load {exp_file}: {e}")
                continue
        
        return pd.DataFrame(records) if records else pd.DataFrame()
    
    def get_thinking_status(self, model: str) -> bool:
        """Check if model has thinking/reasoning mode enabled."""
        return self.model_configs.get(model, {}).get('reasoning_output', 'none') != 'none'
    
    def get_display_name(self, model: str) -> str:
        """Get human-readable display name for model."""
        return self.display_names.get(model, str(model).split('/')[-1])
    
    def _filter_models(self, df: pd.DataFrame, include_random: bool) -> pd.DataFrame:
        """Filter out gemma and optionally random agent."""
        return df[~df['model'].apply(
            lambda m: 'gemma' in str(m).lower() or 
                     (not include_random and 'random' in str(m).lower())
        )].copy()
