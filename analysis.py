#!/usr/bin/env python3
"""
MAgIC Analysis Pipeline - Publication Version

Usage:
    python analysis.py

RANDOM AGENT INCLUSION STRATEGY:
--------------------------------
RQ1 (Performance Tables): INCLUDE random agent as baseline
    - Shows LLMs beat random → validates strategic capability
    - Tables: T_perf_win_rate, T_perf_avg_profit, T_perf_game_specific

RQ2 (Behavioral Profiles): INCLUDE random agent for comparison
    - Random should cluster separately from strategic models
    - Tables: T_magic_{game}
    - Figures: F_similarity_{game}, F_similarity_3v5

RQ3 (Regressions): EXCLUDE random agent
    - Random has no meaningful features for regression
    - Tables: T_mlr_*, T5a/b_magic_to_perf, T6_pca_variance
    - Figures: F_pca_scree

RQ4 (Thinking Analysis): EXCLUDE random agent
    - Random is not a thinking model, would add noise
    - Tables: T8_thinking_*
    - Figures: F5_think_vs_inst

FEATURE MODELING:
-----------------
Model features used in MLR regression (Features → Performance):

1. thinking (binary: 0 or 1)
   - Derived from config.json: model_configs[model].reasoning_output
   - If reasoning_output in ['reasoning_tokens', 'output_tokens'] → thinking = 1
   - If reasoning_output == 'none' → thinking = 0
   
2. architecture_moe (binary: 0 or 1)
   - Derived from model name pattern (-A##B indicates MoE)
   
3. size_params (continuous)
   - Extracted from model name (e.g., 32B → 32.0)
   
4. family_encoded (categorical)
   - Model family: qwen, llama, gemma, etc.
   
5. version (continuous)
   - Model generation (e.g., Qwen3 → 3.0, Llama4 → 4.0)

TOKEN ANALYSIS (Table 8, Figure 5):
-----------------------------------
reasoning_char_count is extracted from experiment JSONs:
- Static games (Salop, Spulber): game_data.llm_metadata.challenger.reasoning_char_count
- Dynamic games (Green-Porter, Athey-Bagwell): game_data.rounds[].llm_metadata.challenger.reasoning_char_count

Table 8 shows average reasoning chars per THINKING model per game/condition.
Figure 5 shows bar chart of reasoning chars by game for thinking models.
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
import re
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

from analysis.engine.analyze_metrics import MetricsAnalyzer
from analysis.engine.create_summary_csvs import SummaryCreator
from config.config import get_experiments_dir, get_analysis_dir

try:
    import statsmodels.api as sm
    from scipy.stats import linregress, ttest_rel, pearsonr
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger("AnalysisPipeline")

# =============================================================================
# CONFIGURATION
# =============================================================================

GAME_CONFIGS = {
    'salop': {
        'performance_metrics': ['average_profit', 'win_rate', 'market_price'],
        'magic_metrics': ['rationality', 'reasoning', 'cooperation'],
        'game_specific_metric': 'market_price'
    },
    'spulber': {
        'performance_metrics': ['average_profit', 'win_rate', 'allocative_efficiency'],
        'magic_metrics': ['rationality', 'judgment', 'reasoning', 'self_awareness'],
        'game_specific_metric': 'allocative_efficiency'
    },
    'green_porter': {
        'performance_metrics': ['average_profit', 'win_rate', 'reversion_frequency'],
        'magic_metrics': ['cooperation', 'coordination'],
        'game_specific_metric': 'reversion_frequency'
    },
    'athey_bagwell': {
        'performance_metrics': ['average_profit', 'win_rate', 'productive_efficiency'],
        'magic_metrics': ['rationality', 'reasoning', 'deception', 'cooperation'],
        'game_specific_metric': 'productive_efficiency'
    }
}

METRIC_DIRECTION = {
    'average_profit': '↑', 'win_rate': '↑', 'market_price': '↑',
    'allocative_efficiency': '↑', 'productive_efficiency': '↑', 'reversion_frequency': '↓',
    'rationality': '↑', 'reasoning': '↑', 'cooperation': '↑', 'coordination': '↑',
    'judgment': '↑', 'self_awareness': '↑', 'deception': '↑',
}

MODEL_FEATURES = ['architecture_moe', 'size_params', 'family_encoded', 'version', 'thinking']
COLLINEARITY_THRESHOLD = 0.95


# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    def __init__(self, analysis_dir: Path, config_path: Optional[Path] = None, 
                 experiments_dir: Optional[Path] = None):
        self.analysis_dir = Path(analysis_dir)
        self.config_path = config_path
        self.experiments_dir = experiments_dir
        self.model_configs = {}
        self.display_names = {}
        self.family_encoder = None
        
        # Add display name for random agent
        self.display_names['random_agent'] = 'Random'
    
    def load(self, include_random: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load performance and MAgIC dataframes.
        
        Args:
            include_random: If True, include random_agent in results (for RQ1/RQ2).
                           If False, exclude random_agent (for RQ3/RQ4 regressions).
        """
        perf_path = self.analysis_dir / "performance_metrics.csv"
        magic_path = self.analysis_dir / "magic_behavioral_metrics.csv"
        
        if not perf_path.exists() or not magic_path.exists():
            raise FileNotFoundError(f"Required CSVs not found in {self.analysis_dir}")
        
        perf_df = pd.read_csv(perf_path)
        magic_df = pd.read_csv(magic_path)
        
        perf_df = self._filter_models(perf_df, include_random=include_random)
        magic_df = self._filter_models(magic_df, include_random=include_random)
        
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                self.model_configs = config.get('model_configs', {})
                for model_name, cfg in self.model_configs.items():
                    self.display_names[model_name] = cfg.get('display_name', model_name)
        
        random_status = "included" if include_random else "excluded"
        logger.info(f"Loaded {len(perf_df)} performance rows, {len(magic_df)} MAgIC rows (random agent {random_status})")
        return perf_df, magic_df
    
    def load_token_data(self) -> pd.DataFrame:
        """Extract reasoning_char_count from experiment JSON files.
        
        JSON Structure differs by game type:
        - Static (Salop, Spulber): simulation_results[].game_data.llm_metadata.challenger
        - Dynamic (Green-Porter, Athey-Bagwell): simulation_results[].game_data.rounds[].llm_metadata.challenger
        """
        if not self.experiments_dir or not self.experiments_dir.exists():
            logger.warning("Experiments directory not available for token extraction")
            return pd.DataFrame()
        
        records = []
        
        for exp_file in self.experiments_dir.rglob("*_competition_result*.json"):
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                
                model = data.get('challenger_model', '')
                game = data.get('game_name', '')
                condition = data.get('condition_name', '')
                
                if not model or not game:
                    continue
                
                # Skip random/defender
                if 'random' in model.lower() or 'gemma' in model.lower():
                    continue
                
                sim_results = data.get('simulation_results', [])
                all_reasoning_chars = []
                
                for sim in sim_results:
                    game_data = sim.get('game_data', {})
                    
                    # STATIC GAMES: llm_metadata at game_data level (no rounds key)
                    if 'llm_metadata' in game_data:
                        llm_meta = game_data['llm_metadata']
                        # Find challenger entry
                        for pid, meta in llm_meta.items():
                            if 'challenger' in pid.lower() and isinstance(meta, dict):
                                all_reasoning_chars.append(meta.get('reasoning_char_count', 0))
                                break
                    
                    # DYNAMIC GAMES: llm_metadata in each round
                    rounds = game_data.get('rounds', [])
                    for rnd in rounds:
                        llm_meta = rnd.get('llm_metadata', {})
                        for pid, meta in llm_meta.items():
                            if 'challenger' in pid.lower() and isinstance(meta, dict):
                                all_reasoning_chars.append(meta.get('reasoning_char_count', 0))
                                break
                
                if all_reasoning_chars:
                    records.append({
                        'model': model,
                        'game': game,
                        'condition': condition,
                        'avg_reasoning_chars': round(np.mean(all_reasoning_chars), 1),
                        'total_reasoning_chars': int(np.sum(all_reasoning_chars)),
                        'n_observations': len(all_reasoning_chars)
                    })
                    
            except Exception as e:
                logger.debug(f"Could not parse {exp_file}: {e}")
                continue
        
        if records:
            df = pd.DataFrame(records)
            logger.info(f"Loaded token data: {len(df)} model/game/condition combinations")
            return df
        
        logger.warning("No token data extracted from experiments")
        return pd.DataFrame()
    
    def get_thinking_status(self, model: str) -> bool:
        """Check if model has thinking enabled based on config."""
        cfg = self.model_configs.get(model, {})
        reasoning_output = cfg.get('reasoning_output', 'none')
        return reasoning_output != 'none'
    
    def get_display_name(self, model: str) -> str:
        """Get display name from config.json."""
        if model in self.display_names:
            return self.display_names[model]
        return str(model).split('/')[-1]
    
    def _filter_models(self, df: pd.DataFrame, include_random: bool = False) -> pd.DataFrame:
        """Filter models from dataframe.
        
        Args:
            df: Input dataframe with 'model' column
            include_random: If True, keep random_agent (for RQ1/RQ2 baseline comparison).
                           Always excludes defender (gemma).
        """
        def is_excluded(model: str) -> bool:
            m = str(model).lower()
            # Always exclude defender (gemma)
            if 'gemma' in m:
                return True
            # Conditionally exclude random agent
            if not include_random and 'random' in m:
                return True
            return False
        return df[~df['model'].apply(is_excluded)].copy()
    
    def extract_model_features(self, models: List[str]) -> pd.DataFrame:
        records = []
        for model in models:
            m_lower = str(model).lower()
            
            is_moe = bool(re.search(r'-a\d+b', m_lower)) or 'moe' in m_lower
            if 'maverick' in m_lower or 'scout' in m_lower:
                is_moe = True
            
            size = 0.0
            match = re.search(r'(?<!a)(\d+\.?\d*)b(?!-)', m_lower)
            if match:
                size = float(match.group(1))
            
            family = 'unknown'
            for fam in ['qwen', 'llama', 'gemma', 'mistral', 'gemini', 'gpt', 'claude']:
                if fam in m_lower:
                    family = fam
                    break
            
            version = 0.0
            for pattern in [r'qwen(\d+)', r'llama-?(\d+\.?\d*)', r'gemma-?(\d+)']:
                match = re.search(pattern, m_lower)
                if match:
                    version = float(match.group(1))
                    break
            
            thinking = 0
            if model in self.model_configs:
                ro = self.model_configs[model].get('reasoning_output', 'none')
                if ro in ['reasoning_tokens', 'output_tokens']:
                    thinking = 1
            elif 'thinking' in m_lower and 'off' not in m_lower:
                thinking = 1
            
            records.append({
                'model': model,
                'display_name': self.get_display_name(model),
                'architecture_moe': int(is_moe),
                'size_params': size,
                'family': family,
                'version': version,
                'thinking': thinking
            })
        
        df = pd.DataFrame(records)
        if SKLEARN_AVAILABLE and len(df) > 0:
            self.family_encoder = LabelEncoder()
            df['family_encoded'] = self.family_encoder.fit_transform(df['family'])
        else:
            df['family_encoded'] = 0
        return df


# =============================================================================
# TABLE GENERATOR
# =============================================================================

class TableGenerator:
    """Generate publication tables.
    
    RQ1 (Performance) and RQ2 (Behavioral) tables INCLUDE random agent as baseline.
    RQ3 (Regressions) and RQ4 (Thinking) tables EXCLUDE random agent.
    """
    
    def __init__(self, perf_df_with_random: pd.DataFrame, magic_df_with_random: pd.DataFrame,
                 perf_df_no_random: pd.DataFrame, magic_df_no_random: pd.DataFrame,
                 features_df: pd.DataFrame, output_dir: Path, loader: DataLoader):
        # Data WITH random agent (for RQ1 performance tables, RQ2 behavioral tables)
        self.perf_df = perf_df_with_random
        self.magic_df = magic_df_with_random
        
        # Data WITHOUT random agent (for RQ3 regressions, RQ4 thinking analysis)
        self.perf_df_no_random = perf_df_no_random
        self.magic_df_no_random = magic_df_no_random
        
        self.features_df = features_df
        self.output_dir = output_dir
        self.loader = loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self, token_df: pd.DataFrame = None):
        self.performance_win_rate_table()
        self.performance_avg_profit_table()
        self.performance_game_specific_table()
        self.performance_3v5_summary()
        
        for game in GAME_CONFIGS.keys():
            self.magic_per_game_table(game)
        
        self.mlr_features_to_performance()
        self.mlr_magic_to_performance()
        self.pca_variance_table()
        
        # Table 8: Cost-benefit analysis
        if token_df is not None and not token_df.empty:
            self.cost_benefit_table(token_df)
    
    def _display_name(self, model: str) -> str:
        return self.loader.get_display_name(model)
    
    def _format_mean_std(self, mean_val, std_val) -> str:
        if pd.isna(mean_val):
            return "N/A"
        if pd.isna(std_val) or std_val == 0:
            return f"{mean_val:.3f}"
        return f"{mean_val:.3f} ± {std_val:.3f}"
    
    @staticmethod
    def _sig_stars(p):
        if p is None or pd.isna(p):
            return ''
        if p < 0.001:
            return '***'
        if p < 0.01:
            return '**'
        if p < 0.05:
            return '*'
        return ''
    
    def _save_table_as_png(self, df: pd.DataFrame, filename: str, title: str, 
                           bold_best: bool = True, metric_cols: List[str] = None):
        if not PLOT_AVAILABLE or df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(max(14, len(df.columns) * 1.8), max(5, len(df) * 0.5)))
        ax.axis('off')
        
        cell_colors = [['white'] * len(df.columns) for _ in range(len(df))]
        
        if bold_best and metric_cols:
            for col_idx, col in enumerate(df.columns):
                if col in metric_cols:
                    direction = '↓' if '↓' in col else '↑'
                    try:
                        numeric_vals = pd.to_numeric(df[col].astype(str).str.split(' ±').str[0], errors='coerce')
                        best_idx = numeric_vals.idxmin() if direction == '↓' else numeric_vals.idxmax()
                        if pd.notna(best_idx):
                            row_idx = df.index.get_loc(best_idx)
                            cell_colors[row_idx][col_idx] = '#90EE90'
                    except:
                        pass
        
        table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center',
                        loc='center', cellColours=cell_colors)
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.5)
        
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _compute_3v5_pvalues(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Compute paired t-test p-values for 3P vs 5P."""
        results = []
        metric_df = df[df['metric'] == metric]
        
        for game in metric_df['game'].unique():
            game_df = metric_df[metric_df['game'] == game]
            
            cond_3 = game_df[game_df['condition'].str.contains('baseline|few|3', case=False, na=False)]
            cond_5 = game_df[game_df['condition'].str.contains('more|5', case=False, na=False)]
            
            if cond_3.empty or cond_5.empty:
                continue
            
            common_models = set(cond_3['model'].unique()) & set(cond_5['model'].unique())
            if len(common_models) < 3:
                continue
            
            vals_3, vals_5 = [], []
            for model in common_models:
                v3 = cond_3[cond_3['model'] == model]['mean'].values
                v5 = cond_5[cond_5['model'] == model]['mean'].values
                if len(v3) > 0 and len(v5) > 0:
                    vals_3.append(v3[0])
                    vals_5.append(v5[0])
            
            if len(vals_3) >= 3:
                try:
                    t_stat, p_value = ttest_rel(vals_3, vals_5)
                    results.append({
                        'game': game, 'metric': metric,
                        'mean_3P': round(np.mean(vals_3), 4),
                        'mean_5P': round(np.mean(vals_5), 4),
                        'mean_diff': round(np.mean(vals_3) - np.mean(vals_5), 4),
                        't_statistic': round(t_stat, 4),
                        'p_value': p_value,
                        'sig': self._sig_stars(p_value),
                        'n_models': len(vals_3)
                    })
                except:
                    pass
        
        return pd.DataFrame(results)
    
    def _build_metric_table_3v5(self, df: pd.DataFrame, metric: str) -> pd.DataFrame:
        """Build table with 3P and 5P columns for each game."""
        games = list(df['game'].unique())
        metric_df = df[df['metric'] == metric].copy()
        if metric_df.empty:
            return pd.DataFrame()
        
        records = []
        for model in metric_df['model'].unique():
            row = {'Model': self._display_name(model)}
            model_df = metric_df[metric_df['model'] == model]
            
            for game in games:
                game_df = model_df[model_df['game'] == game]
                
                cond_3 = game_df[game_df['condition'].str.contains('baseline|few|3', case=False, na=False)]
                cond_5 = game_df[game_df['condition'].str.contains('more|5', case=False, na=False)]
                
                if not cond_3.empty:
                    row[f'{game}_3P'] = self._format_mean_std(cond_3['mean'].iloc[0], 
                                                              cond_3['std'].iloc[0] if 'std' in cond_3 else 0)
                else:
                    row[f'{game}_3P'] = 'N/A'
                
                if not cond_5.empty:
                    row[f'{game}_5P'] = self._format_mean_std(cond_5['mean'].iloc[0],
                                                              cond_5['std'].iloc[0] if 'std' in cond_5 else 0)
                else:
                    row[f'{game}_5P'] = 'N/A'
            
            records.append(row)
        
        return pd.DataFrame(records)
    
    def performance_win_rate_table(self):
        """Win rate table for RQ1.
        
        Note: Uses perf_df which INCLUDES random agent as performance baseline.
        """
        logger.info("Generating Win Rate table (includes random agent)...")
        direction = METRIC_DIRECTION.get('win_rate', '↑')
        table_df = self._build_metric_table_3v5(self.perf_df, 'win_rate')
        
        if table_df.empty:
            return
        
        new_cols = {col: f"{col} {direction}" for col in table_df.columns if col != 'Model'}
        table_df = table_df.rename(columns=new_cols)
        
        table_df.to_csv(self.output_dir / "T_perf_win_rate.csv", index=False)
        metric_cols = [c for c in table_df.columns if c != 'Model']
        self._save_table_as_png(table_df, "T_perf_win_rate.png", 
                               f"Win Rate by Model (3P vs 5P) {direction}", metric_cols=metric_cols)
        
        pval_df = self._compute_3v5_pvalues(self.perf_df, 'win_rate')
        if not pval_df.empty:
            pval_df.to_csv(self.output_dir / "T_perf_win_rate_pvalues.csv", index=False)
    
    def performance_avg_profit_table(self):
        """Average profit table for RQ1.
        
        Note: Uses perf_df which INCLUDES random agent as performance baseline.
        """
        logger.info("Generating Average Profit table (includes random agent)...")
        direction = METRIC_DIRECTION.get('average_profit', '↑')
        table_df = self._build_metric_table_3v5(self.perf_df, 'average_profit')
        
        if table_df.empty:
            return
        
        new_cols = {col: f"{col} {direction}" for col in table_df.columns if col != 'Model'}
        table_df = table_df.rename(columns=new_cols)
        
        table_df.to_csv(self.output_dir / "T_perf_avg_profit.csv", index=False)
        metric_cols = [c for c in table_df.columns if c != 'Model']
        self._save_table_as_png(table_df, "T_perf_avg_profit.png",
                               f"Average Profit by Model (3P vs 5P) {direction}", metric_cols=metric_cols)
        
        pval_df = self._compute_3v5_pvalues(self.perf_df, 'average_profit')
        if not pval_df.empty:
            pval_df.to_csv(self.output_dir / "T_perf_avg_profit_pvalues.csv", index=False)
    
    def performance_game_specific_table(self):
        """Game-specific metrics table for RQ1 (3P vs 5P).
        
        Note: Uses perf_df which INCLUDES random agent as performance baseline.
        """
        logger.info("Generating Game-Specific metrics table (includes random agent)...")
        
        records = []
        for model in self.perf_df['model'].unique():
            row = {'Model': self._display_name(model)}
            model_df = self.perf_df[self.perf_df['model'] == model]
            
            for game, config in GAME_CONFIGS.items():
                metric = config['game_specific_metric']
                direction = METRIC_DIRECTION.get(metric, '↑')
                
                game_metric_df = model_df[(model_df['game'] == game) & (model_df['metric'] == metric)]
                
                # 3-player condition
                cond_3 = game_metric_df[game_metric_df['condition'].str.contains('baseline|few|3', case=False, na=False)]
                if not cond_3.empty:
                    row[f'{game}_{metric}_3P {direction}'] = self._format_mean_std(
                        cond_3['mean'].iloc[0], cond_3['std'].iloc[0] if 'std' in cond_3 else 0)
                else:
                    row[f'{game}_{metric}_3P {direction}'] = 'N/A'
                
                # 5-player condition
                cond_5 = game_metric_df[game_metric_df['condition'].str.contains('more|5', case=False, na=False)]
                if not cond_5.empty:
                    row[f'{game}_{metric}_5P {direction}'] = self._format_mean_std(
                        cond_5['mean'].iloc[0], cond_5['std'].iloc[0] if 'std' in cond_5 else 0)
                else:
                    row[f'{game}_{metric}_5P {direction}'] = 'N/A'
            
            records.append(row)
        
        table_df = pd.DataFrame(records)
        table_df.to_csv(self.output_dir / "T_perf_game_specific.csv", index=False)
        metric_cols = [c for c in table_df.columns if c != 'Model']
        self._save_table_as_png(table_df, "T_perf_game_specific.png",
                               "Game-Specific Metrics (3P vs 5P)", metric_cols=metric_cols)
        
        # Compute p-values for each game-specific metric
        all_pvals = []
        for game, config in GAME_CONFIGS.items():
            metric = config['game_specific_metric']
            pval_df = self._compute_3v5_pvalues(self.perf_df, metric)
            if not pval_df.empty:
                all_pvals.append(pval_df)
        
        if all_pvals:
            combined_pvals = pd.concat(all_pvals, ignore_index=True)
            combined_pvals.to_csv(self.output_dir / "T_perf_game_specific_pvalues.csv", index=False)
    
    def performance_3v5_summary(self):
        """Summary of all 3P vs 5P p-values."""
        logger.info("Generating 3P vs 5P summary...")
        
        all_pvals = []
        for metric in ['win_rate', 'average_profit']:
            pval_df = self._compute_3v5_pvalues(self.perf_df, metric)
            if not pval_df.empty:
                all_pvals.append(pval_df)
        
        for game, config in GAME_CONFIGS.items():
            metric = config['game_specific_metric']
            pval_df = self._compute_3v5_pvalues(self.perf_df, metric)
            if not pval_df.empty:
                all_pvals.append(pval_df)
        
        if all_pvals:
            combined = pd.concat(all_pvals, ignore_index=True)
            combined.to_csv(self.output_dir / "T_perf_3v5_pvalues_summary.csv", index=False)
            self._save_table_as_png(combined.round(4), "T_perf_3v5_pvalues_summary.png",
                                   "Performance: 3P vs 5P Statistical Comparison", bold_best=False)
    
    def magic_per_game_table(self, game: str):
        """MAgIC behavioral metrics table for RQ2.
        
        Note: Uses magic_df which INCLUDES random agent as behavioral baseline.
        Random agent should show distinctive (often lower) behavioral scores.
        """
        logger.info(f"Generating MAgIC table for {game} (includes random agent)...")
        
        game_df = self.magic_df[self.magic_df['game'] == game]
        if game_df.empty:
            return
        
        metrics = GAME_CONFIGS[game]['magic_metrics']
        
        records = []
        for model in game_df['model'].unique():
            row = {'Model': self._display_name(model)}
            model_df = game_df[game_df['model'] == model]
            
            for metric in metrics:
                direction = METRIC_DIRECTION.get(metric, '↑')
                metric_df = model_df[model_df['metric'] == metric]
                
                if not metric_df.empty:
                    row[f'{metric} {direction}'] = self._format_mean_std(
                        metric_df['mean'].mean(), metric_df['std'].mean() if 'std' in metric_df else 0)
                else:
                    row[f'{metric} {direction}'] = 'N/A'
            
            records.append(row)
        
        table_df = pd.DataFrame(records)
        table_df.to_csv(self.output_dir / f"T_magic_{game}.csv", index=False)
        metric_cols = [c for c in table_df.columns if c != 'Model']
        self._save_table_as_png(table_df, f"T_magic_{game}.png",
                               f"MAgIC Metrics: {game.replace('_', ' ').title()}", metric_cols=metric_cols)
    
    def _remove_collinear(self, df: pd.DataFrame, preds: List[str]) -> Tuple[List[str], List]:
        if len(preds) < 2:
            return preds, []
        
        dropped = []
        remaining = list(preds)
        
        while len(remaining) >= 2:
            valid = [p for p in remaining if p in df.columns and df[p].std() > 0]
            if len(valid) < 2:
                break
            
            corr = df[valid].corr().abs()
            found = False
            for i in range(len(valid)):
                for j in range(i + 1, len(valid)):
                    if corr.iloc[i, j] >= COLLINEARITY_THRESHOLD:
                        to_drop = valid[i] if df[valid[i]].var() <= df[valid[j]].var() else valid[j]
                        dropped.append(to_drop)
                        remaining.remove(to_drop)
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        
        return remaining, dropped
    
    def mlr_features_to_performance(self):
        """MLR: Model Features → Performance Metrics.
        
        Note: Uses perf_df_no_random to exclude random agent from regressions.
        """
        if not STATSMODELS_AVAILABLE:
            return
        
        logger.info("Generating MLR: Features → Performance (excluding random agent)...")
        
        perf_pivot = self.perf_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], columns='metric', values='mean'
        ).reset_index()
        
        merged = perf_pivot.merge(self.features_df, on='model', how='left')
        
        results = []
        
        for game in merged['game'].unique():
            game_df = merged[merged['game'] == game].copy()
            
            valid_preds = [p for p in MODEL_FEATURES if p in game_df.columns and game_df[p].std() > 0]
            valid_preds, _ = self._remove_collinear(game_df, valid_preds)
            
            if not valid_preds:
                continue
            
            perf_metrics = [c for c in perf_pivot.columns if c not in ['game', 'model', 'condition']]
            
            for target in perf_metrics:
                if target not in game_df.columns or game_df[target].std() == 0:
                    continue
                
                valid = game_df[valid_preds + [target]].dropna()
                if len(valid) < len(valid_preds) + 2:
                    continue
                
                X = sm.add_constant(valid[valid_preds])
                Y = valid[target]
                
                try:
                    model = sm.OLS(Y, X).fit()
                    
                    for pred in valid_preds:
                        results.append({
                            'game': game, 'target': target, 'predictor': pred,
                            'coef': round(model.params.get(pred, np.nan), 4),
                            'std_err': round(model.bse.get(pred, np.nan), 4),
                            'p_value': model.pvalues.get(pred, np.nan),
                            'sig': self._sig_stars(model.pvalues.get(pred, np.nan)),
                            'r_squared': round(model.rsquared, 4),
                            'n_obs': int(model.nobs)
                        })
                except Exception as e:
                    logger.warning(f"MLR failed: {e}")
        
        if results:
            df_out = pd.DataFrame(results)
            df_out.to_csv(self.output_dir / "T_mlr_features_to_performance.csv", index=False)
            
            sig_df = df_out[df_out['p_value'] < 0.05].copy()
            if not sig_df.empty:
                sig_df.to_csv(self.output_dir / "T_mlr_features_to_perf_significant.csv", index=False)
                self._save_table_as_png(sig_df.round(4), "T_mlr_features_to_perf_significant.png",
                                       "Features → Performance (p < 0.05)", bold_best=False)
            
            logger.info(f"  Saved MLR Features→Perf: {len(df_out)} rows, {len(sig_df)} significant")
    
    def mlr_magic_to_performance(self):
        """MLR: MAgIC → Performance (T5a, T5b).
        
        Note: Uses no_random dataframes to exclude random agent from regressions.
        """
        if not STATSMODELS_AVAILABLE:
            return
        
        logger.info("Generating T5: MAgIC → Performance (excluding random agent)...")
        
        magic_pivot = self.magic_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], columns='metric', values='mean'
        ).reset_index()
        
        perf_pivot = self.perf_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], columns='metric', values='mean'
        ).reset_index()
        
        merged = perf_pivot.merge(magic_pivot, on=['game', 'model', 'condition'], how='inner')
        
        if merged.empty:
            logger.warning("No merged data for T5")
            return
        
        perf_metrics = [c for c in perf_pivot.columns if c not in ['game', 'model', 'condition']]
        magic_metrics = [c for c in magic_pivot.columns if c not in ['game', 'model', 'condition']]
        
        results_coef = []
        results_hier = []
        
        for game in merged['game'].unique():
            game_df = merged[merged['game'] == game].copy()
            
            available_magic = [m for m in magic_metrics if m in game_df.columns and game_df[m].std() > 0]
            available_magic, _ = self._remove_collinear(game_df, available_magic)
            
            if not available_magic:
                continue
            
            for target in perf_metrics:
                if target not in game_df.columns or game_df[target].std() == 0:
                    continue
                
                valid = game_df[[target] + available_magic].dropna()
                if len(valid) < len(available_magic) + 2:
                    continue
                
                X = sm.add_constant(valid[available_magic])
                Y = valid[target]
                
                try:
                    model = sm.OLS(Y, X).fit()
                    
                    for magic_var in available_magic:
                        if magic_var in model.params.index:
                            results_coef.append({
                                'game': game, 'target': target, 'predictor': magic_var,
                                'coef': round(model.params[magic_var], 4),
                                'std_err': round(model.bse[magic_var], 4),
                                'p_value': model.pvalues[magic_var],
                                'sig': self._sig_stars(model.pvalues[magic_var])
                            })
                    
                    results_hier.append({
                        'game': game, 'target': target,
                        'R2': round(model.rsquared, 4),
                        'R2_adj': round(model.rsquared_adj, 4),
                        'n_predictors': len(available_magic),
                        'n_obs': int(model.nobs)
                    })
                except Exception as e:
                    logger.warning(f"T5 failed: {e}")
        
        if results_coef:
            df_coef = pd.DataFrame(results_coef)
            df_coef.to_csv(self.output_dir / "T5a_magic_to_perf_coef.csv", index=False)
            self._save_table_as_png(df_coef.round(4), "T5a_magic_to_perf_coef.png",
                                   "T5a: MAgIC → Performance Coefficients", bold_best=False)
        
        if results_hier:
            df_hier = pd.DataFrame(results_hier)
            df_hier.to_csv(self.output_dir / "T5b_magic_to_perf_summary.csv", index=False)
            self._save_table_as_png(df_hier, "T5b_magic_to_perf_summary.png",
                                   "T5b: MAgIC → Performance Model Summary", bold_best=False)
    
    def pca_variance_table(self):
        """PCA Variance analysis for RQ3 dimensionality validation.
        
        Note: Uses magic_df_no_random to exclude random agent.
        """
        if not SKLEARN_AVAILABLE:
            return
        
        logger.info("Generating T6: PCA Variance (excluding random agent)...")
        
        results = []
        for game, config in GAME_CONFIGS.items():
            game_df = self.magic_df_no_random[self.magic_df_no_random['game'] == game]
            if game_df.empty:
                continue
            
            pivot = game_df.pivot_table(index='model', columns='metric', values='mean').dropna()
            available = [m for m in config['magic_metrics'] if m in pivot.columns]
            
            if len(available) < 2:
                continue
            
            X = StandardScaler().fit_transform(pivot[available].values)
            pca = PCA().fit(X)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            
            for i, var in enumerate(pca.explained_variance_ratio_):
                results.append({
                    'game': game, 'component': f'PC{i+1}',
                    'variance_explained': round(var, 4),
                    'cumulative': round(cumulative[i], 4),
                    'n_models': len(pivot), 'n_metrics': len(available)
                })
        
        if results:
            df_out = pd.DataFrame(results)
            df_out.to_csv(self.output_dir / "T6_pca_variance.csv", index=False)
            self._save_table_as_png(df_out, "T6_pca_variance.png", "T6: PCA Variance Explained", bold_best=False)
    
    def cost_benefit_table(self, token_df: pd.DataFrame):
        """Table 8: Think vs Inst comparison with mean±std and p-values.
        
        Note: Uses perf_df_no_random to exclude random agent from RQ4 analysis.
        Output format: mean ± std for all metrics, t-test p-values for group comparisons.
        """
        logger.info("Generating T8: Think vs Inst Comparison (excluding random agent)...")
        
        # Get profit data (excluding random agent)
        profit_df = self.perf_df_no_random[self.perf_df_no_random['metric'] == 'average_profit'].copy()
        profit_df['is_thinking'] = profit_df['model'].apply(self.loader.get_thinking_status)
        profit_df['mode'] = profit_df['is_thinking'].map({True: 'Think', False: 'Inst'})
        profit_df['display_name'] = profit_df['model'].apply(self._display_name)
        
        # Merge with token data if available
        if not token_df.empty:
            profit_df = profit_df.merge(
                token_df[['model', 'game', 'condition', 'avg_reasoning_chars']],
                on=['model', 'game', 'condition'],
                how='left'
            )
            profit_df['avg_reasoning_chars'] = profit_df['avg_reasoning_chars'].fillna(0)
        else:
            profit_df['avg_reasoning_chars'] = 0
        
        # Table 8a: Detail per model with mean±std format
        detail_rows = []
        for _, row in profit_df.iterrows():
            profit_str = f"{row['mean']:.2f} ± {row.get('std', 0):.2f}"
            detail_rows.append({
                'game': row['game'],
                'condition': row['condition'],
                'model': row['display_name'],
                'mode': row['mode'],
                'avg_profit': profit_str,
                'avg_reasoning_chars': int(row.get('avg_reasoning_chars', 0)),
            })
        
        detail_df = pd.DataFrame(detail_rows)
        detail_df = detail_df.sort_values(['game', 'condition', 'mode', 'model'])
        detail_df.to_csv(self.output_dir / "T8_thinking_detail.csv", index=False)
        
        # Table 8b: Summary with mean±std and p-values
        summary_rows = []
        for game in profit_df['game'].unique():
            for condition in profit_df[profit_df['game'] == game]['condition'].unique():
                subset = profit_df[(profit_df['game'] == game) & (profit_df['condition'] == condition)]
                
                think_data = subset[subset['mode'] == 'Think']
                inst_data = subset[subset['mode'] == 'Inst']
                
                think_profits = think_data['mean'].values
                inst_profits = inst_data['mean'].values
                think_chars = think_data['avg_reasoning_chars'].values
                
                # Calculate mean ± std
                think_mean = np.mean(think_profits) if len(think_profits) > 0 else np.nan
                think_std = np.std(think_profits) if len(think_profits) > 1 else 0
                inst_mean = np.mean(inst_profits) if len(inst_profits) > 0 else np.nan
                inst_std = np.std(inst_profits) if len(inst_profits) > 1 else 0
                chars_mean = np.mean(think_chars) if len(think_chars) > 0 else 0
                chars_std = np.std(think_chars) if len(think_chars) > 1 else 0
                
                # Calculate p-value (Welch's t-test)
                p_value = np.nan
                if len(think_profits) >= 2 and len(inst_profits) >= 2:
                    from scipy.stats import ttest_ind
                    try:
                        _, p_value = ttest_ind(think_profits, inst_profits, equal_var=False)
                    except:
                        p_value = np.nan
                
                profit_diff = think_mean - inst_mean if pd.notna(think_mean) and pd.notna(inst_mean) else np.nan
                
                # Format as mean±std strings
                think_str = f"{think_mean:.2f} ± {think_std:.2f}" if pd.notna(think_mean) else "N/A"
                inst_str = f"{inst_mean:.2f} ± {inst_std:.2f}" if pd.notna(inst_mean) else "N/A"
                chars_str = f"{chars_mean:.0f} ± {chars_std:.0f}" if chars_mean > 0 else "0"
                
                summary_rows.append({
                    'game': game,
                    'condition': condition,
                    'n_think': len(think_profits),
                    'n_inst': len(inst_profits),
                    'profit_Think': think_str,
                    'profit_Inst': inst_str,
                    'profit_diff': round(profit_diff, 2) if pd.notna(profit_diff) else np.nan,
                    'p_value': round(p_value, 4) if pd.notna(p_value) else np.nan,
                    'sig': self._sig_stars(p_value) if pd.notna(p_value) else '',
                    'reasoning_chars': chars_str,
                })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(self.output_dir / "T8_thinking_summary.csv", index=False)
        
        self._save_table_as_png(summary_df, "T8_thinking_summary.png",
                               "T8: Think vs Inst (mean±std, p-values)", bold_best=False)
        
        logger.info(f"  T8: {len(detail_df)} detail rows, {len(summary_df)} summary rows")


# =============================================================================
# FIGURE GENERATOR
# =============================================================================

class FigureGenerator:
    """Generate publication figures.
    
    RQ2 similarity figures INCLUDE random agent as baseline for comparison.
    RQ3 PCA and RQ4 thinking figures EXCLUDE random agent.
    """
    
    def __init__(self, perf_df_with_random: pd.DataFrame, magic_df_with_random: pd.DataFrame,
                 perf_df_no_random: pd.DataFrame, magic_df_no_random: pd.DataFrame,
                 features_df: pd.DataFrame, output_dir: Path, loader: DataLoader):
        # Data WITH random agent (for RQ2 similarity figures)
        self.perf_df = perf_df_with_random
        self.magic_df = magic_df_with_random
        
        # Data WITHOUT random agent (for RQ3 PCA, RQ4 thinking figures)
        self.perf_df_no_random = perf_df_no_random
        self.magic_df_no_random = magic_df_no_random
        
        self.features_df = features_df
        self.output_dir = output_dir
        self.loader = loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _display_name(self, model: str) -> str:
        """Get short display name for model."""
        return self.loader.get_display_name(model)
    
    def generate_all(self, token_df: pd.DataFrame = None):
        if not PLOT_AVAILABLE or not SKLEARN_AVAILABLE:
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        
        for game in GAME_CONFIGS.keys():
            self.similarity_matrix_per_game(game)
        
        self.similarity_3v5_scores()
        self.pca_scree_plots()
        
        # Figure 5: Token usage vs outcome
        if token_df is not None and not token_df.empty:
            self.cost_benefit_scatter(token_df)
    
    def similarity_matrix_per_game(self, game: str):
        """Generate behavioral similarity matrix for a game (RQ2).
        
        Note: Uses magic_df which INCLUDES random agent as baseline for comparison.
        Random agent should cluster separately from strategic models.
        """
        logger.info(f"Generating similarity matrix for {game} (includes random agent)...")
        
        game_df = self.magic_df[self.magic_df['game'] == game].copy()
        if game_df.empty:
            return
        
        pivot = game_df.pivot_table(index='model', columns='metric', values='mean').fillna(0)
        if len(pivot) < 2:
            return
        
        sim = cosine_similarity(pivot.values)
        names = [self._display_name(m) for m in pivot.index]
        sim_df = pd.DataFrame(sim, index=names, columns=names)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(sim_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0.5, ax=ax,
                   vmin=0, vmax=1, square=True)
        ax.set_title(f'Model Behavioral Similarity: {game.replace("_", " ").title()}')
        plt.tight_layout()
        plt.savefig(self.output_dir / f"F_similarity_{game}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def similarity_3v5_scores(self):
        """Single similarity score per game: 3P vs 5P (RQ2).
        
        Note: Uses magic_df which INCLUDES random agent for comparison.
        """
        logger.info("Generating 3P vs 5P similarity scores (includes random agent)...")
        
        results = []
        
        for game in GAME_CONFIGS.keys():
            game_df = self.magic_df[self.magic_df['game'] == game].copy()
            if game_df.empty:
                continue
            
            cond_3 = game_df[game_df['condition'].str.contains('baseline|few|3', case=False, na=False)]
            cond_5 = game_df[game_df['condition'].str.contains('more|5', case=False, na=False)]
            
            if cond_3.empty or cond_5.empty:
                continue
            
            pivot_3 = cond_3.pivot_table(index='model', columns='metric', values='mean').fillna(0)
            pivot_5 = cond_5.pivot_table(index='model', columns='metric', values='mean').fillna(0)
            
            common_models = list(set(pivot_3.index) & set(pivot_5.index))
            common_metrics = list(set(pivot_3.columns) & set(pivot_5.columns))
            
            if len(common_models) < 2 or len(common_metrics) < 1:
                continue
            
            vec_3 = pivot_3.loc[common_models, common_metrics].values.flatten()
            vec_5 = pivot_5.loc[common_models, common_metrics].values.flatten()
            
            cos_sim = cosine_similarity([vec_3], [vec_5])[0, 0]
            corr, p_value = pearsonr(vec_3, vec_5)
            
            results.append({
                'game': game,
                'cosine_similarity': round(cos_sim, 4),
                'pearson_r': round(corr, 4),
                'p_value': p_value,
                'sig': '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '',
                'n_models': len(common_models),
                'n_metrics': len(common_metrics)
            })
        
        if not results:
            return
        
        df_results = pd.DataFrame(results)
        df_results.to_csv(self.output_dir / "T_similarity_3v5.csv", index=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(results))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df_results['cosine_similarity'], width, label='Cosine Similarity', color='steelblue')
        bars2 = ax.bar(x + width/2, df_results['pearson_r'], width, label='Pearson r', color='coral')
        
        for bar, row in zip(bars1, df_results.itertuples()):
            ax.annotate(f'{bar.get_height():.3f}{row.sig}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        ax.set_xlabel('Game')
        ax.set_ylabel('Similarity Score')
        ax.set_title('3P vs 5P Behavioral Similarity (with p-values)')
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace('_', ' ').title() for g in df_results['game']])
        ax.legend()
        ax.set_ylim(0, 1.15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "F_similarity_3v5.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def pca_scree_plots(self):
        """PCA scree plots for RQ3 dimensionality validation.
        
        Note: Uses magic_df_no_random to exclude random agent.
        """
        logger.info("Generating PCA scree plots (excluding random agent)...")
        
        games = [g for g, c in GAME_CONFIGS.items() if len(c['magic_metrics']) >= 2]
        if not games:
            return
        
        fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4))
        if len(games) == 1:
            axes = [axes]
        
        for ax, game in zip(axes, games):
            game_df = self.magic_df_no_random[self.magic_df_no_random['game'] == game]
            pivot = game_df.pivot_table(index='model', columns='metric', values='mean').dropna()
            available = [m for m in GAME_CONFIGS[game]['magic_metrics'] if m in pivot.columns]
            
            if len(available) < 2:
                continue
            
            X = StandardScaler().fit_transform(pivot[available].values)
            pca = PCA().fit(X)
            var = pca.explained_variance_ratio_
            
            x_pos = range(1, len(var) + 1)
            ax.bar(x_pos, var, alpha=0.7, color='steelblue', label='Individual')
            ax.plot(x_pos, np.cumsum(var), 'ro-', label='Cumulative')
            ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='80%')
            
            ax.set_title(game.replace('_', ' ').title())
            ax.set_xlabel('Component')
            ax.set_ylabel('Variance Explained')
            ax.set_ylim(0, 1.05)
            ax.set_xticks(x_pos)
            ax.legend(loc='best', fontsize=8)
        
        plt.suptitle('PCA Variance Explained', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / "F_pca_scree.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def cost_benefit_scatter(self, token_df: pd.DataFrame):
        """Figure 5: Think vs Inst performance comparison with p-values.
        
        Note: Uses perf_df_no_random to exclude random agent from RQ4 analysis.
        """
        logger.info("Generating F5: Think vs Inst Comparison (excluding random agent)...")
        
        # Get profit data (excluding random agent)
        profit_df = self.perf_df_no_random[self.perf_df_no_random['metric'] == 'average_profit'].copy()
        profit_df['is_thinking'] = profit_df['model'].apply(self.loader.get_thinking_status)
        profit_df['mode'] = profit_df['is_thinking'].map({True: 'Think', False: 'Inst'})
        
        # Merge with token data
        if not token_df.empty:
            profit_df = profit_df.merge(
                token_df[['model', 'game', 'condition', 'avg_reasoning_chars']],
                on=['model', 'game', 'condition'],
                how='left'
            )
            profit_df['avg_reasoning_chars'] = profit_df['avg_reasoning_chars'].fillna(0)
        else:
            profit_df['avg_reasoning_chars'] = 0
        
        # Calculate stats per game/condition
        stats = []
        for game in profit_df['game'].unique():
            for condition in profit_df[profit_df['game'] == game]['condition'].unique():
                subset = profit_df[(profit_df['game'] == game) & (profit_df['condition'] == condition)]
                
                think_profits = subset[subset['mode'] == 'Think']['mean'].values
                inst_profits = subset[subset['mode'] == 'Inst']['mean'].values
                think_chars = subset[subset['mode'] == 'Think']['avg_reasoning_chars'].values
                
                p_value = np.nan
                if len(think_profits) >= 2 and len(inst_profits) >= 2:
                    from scipy.stats import ttest_ind
                    _, p_value = ttest_ind(think_profits, inst_profits, equal_var=False)
                
                stats.append({
                    'game': game,
                    'condition': condition,
                    'think_mean': np.mean(think_profits) if len(think_profits) > 0 else 0,
                    'think_std': np.std(think_profits) if len(think_profits) > 1 else 0,
                    'inst_mean': np.mean(inst_profits) if len(inst_profits) > 0 else 0,
                    'inst_std': np.std(inst_profits) if len(inst_profits) > 1 else 0,
                    'p_value': p_value,
                    'avg_chars': np.mean(think_chars) if len(think_chars) > 0 else 0,
                })
        
        stats_df = pd.DataFrame(stats)
        
        # Create grouped bar chart
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Left: Performance comparison
        ax1 = axes[0]
        games = sorted(stats_df['game'].unique())
        x = np.arange(len(games))
        width = 0.35
        
        for i, condition in enumerate(['baseline', 'more_players']):
            cond_data = stats_df[stats_df['condition'] == condition]
            
            think_means = [cond_data[cond_data['game'] == g]['think_mean'].values[0] 
                          if len(cond_data[cond_data['game'] == g]) > 0 else 0 for g in games]
            think_stds = [cond_data[cond_data['game'] == g]['think_std'].values[0] 
                         if len(cond_data[cond_data['game'] == g]) > 0 else 0 for g in games]
            inst_means = [cond_data[cond_data['game'] == g]['inst_mean'].values[0] 
                         if len(cond_data[cond_data['game'] == g]) > 0 else 0 for g in games]
            inst_stds = [cond_data[cond_data['game'] == g]['inst_std'].values[0] 
                        if len(cond_data[cond_data['game'] == g]) > 0 else 0 for g in games]
            p_values = [cond_data[cond_data['game'] == g]['p_value'].values[0] 
                       if len(cond_data[cond_data['game'] == g]) > 0 else np.nan for g in games]
            
            offset = (i - 0.5) * width * 2.5
            
            bars_think = ax1.bar(x + offset - width/2, think_means, width, 
                                label=f'Think ({condition})', yerr=think_stds, capsize=3,
                                color='coral' if i == 0 else 'salmon', alpha=0.8)
            bars_inst = ax1.bar(x + offset + width/2, inst_means, width,
                               label=f'Inst ({condition})', yerr=inst_stds, capsize=3,
                               color='steelblue' if i == 0 else 'lightblue', alpha=0.8)
            
            # Add significance stars
            for j, (think_m, inst_m, p) in enumerate(zip(think_means, inst_means, p_values)):
                if pd.notna(p):
                    max_val = max(think_m, inst_m)
                    sig = ''
                    if p < 0.001:
                        sig = '***'
                    elif p < 0.01:
                        sig = '**'
                    elif p < 0.05:
                        sig = '*'
                    if sig:
                        ax1.text(x[j] + offset, max_val + 50, sig, ha='center', fontsize=10, fontweight='bold')
        
        ax1.set_xlabel('Game')
        ax1.set_ylabel('Average Profit')
        ax1.set_title('Think vs Inst Performance (* p<.05, ** p<.01, *** p<.001)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([g.replace('_', ' ').title() for g in games], rotation=15)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(0, color='black', linewidth=0.5)
        
        # Right: Reasoning chars for thinking models
        ax2 = axes[1]
        
        for i, condition in enumerate(['baseline', 'more_players']):
            cond_data = stats_df[stats_df['condition'] == condition]
            chars = [cond_data[cond_data['game'] == g]['avg_chars'].values[0] 
                    if len(cond_data[cond_data['game'] == g]) > 0 else 0 for g in games]
            
            offset = (i - 0.5) * width
            ax2.bar(x + offset, chars, width, label=condition,
                   color='coral' if i == 0 else 'salmon', alpha=0.8)
        
        ax2.set_xlabel('Game')
        ax2.set_ylabel('Avg Reasoning Characters')
        ax2.set_title('Reasoning Token Usage (Thinking Models)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([g.replace('_', ' ').title() for g in games], rotation=15)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Figure 5: Thinking vs Instruct Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / "F5_think_vs_inst.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save stats to CSV with mean±std format
        stats_df['think_profit'] = stats_df.apply(
            lambda r: f"{r['think_mean']:.2f} ± {r['think_std']:.2f}", axis=1)
        stats_df['inst_profit'] = stats_df.apply(
            lambda r: f"{r['inst_mean']:.2f} ± {r['inst_std']:.2f}", axis=1)
        stats_df['sig'] = stats_df['p_value'].apply(
            lambda p: '***' if pd.notna(p) and p < 0.001 else '**' if pd.notna(p) and p < 0.01 else '*' if pd.notna(p) and p < 0.05 else '')
        stats_df['p_value'] = stats_df['p_value'].round(4)
        stats_df.to_csv(self.output_dir / "F5_stats.csv", index=False)


# =============================================================================
# MAIN
# =============================================================================

def main():
    logger.info("=" * 80)
    logger.info("🚀 STARTING ANALYSIS PIPELINE 🚀")
    logger.info("=" * 80)
    
    experiments_dir = get_experiments_dir()
    analysis_dir = get_analysis_dir()
    analysis_dir.mkdir(exist_ok=True, parents=True)
    
    if not experiments_dir.exists() or not any(experiments_dir.iterdir()):
        logger.critical(f"No experiments found in {experiments_dir}")
        sys.exit(1)
    
    try:
        logger.info("[Step 1/5] Analyzing metrics...")
        MetricsAnalyzer().analyze_all_games()
        
        logger.info("[Step 2/5] Creating summary CSVs...")
        SummaryCreator().create_all_summaries()
        
        logger.info("[Step 3/5] Loading data...")
        config_path = Path("config/config.json")
        loader = DataLoader(
            analysis_dir, 
            config_path if config_path.exists() else None,
            experiments_dir
        )
        
        # Load data WITH random agent for RQ1 (performance) and RQ2 (behavioral profiles)
        perf_df_with_random, magic_df_with_random = loader.load(include_random=True)
        
        # Load data WITHOUT random agent for RQ3 (regressions) and RQ4 (thinking analysis)
        perf_df_no_random, magic_df_no_random = loader.load(include_random=False)
        
        # Load token data for cost-benefit analysis (excludes random by design)
        token_df = loader.load_token_data()
        
        # Extract features only for non-random models (for regressions)
        all_models_no_random = list(set(perf_df_no_random['model'].unique()) | set(magic_df_no_random['model'].unique()))
        features_df = loader.extract_model_features(all_models_no_random)
        
        logger.info("[Step 4/5] Generating tables...")
        pub_dir = analysis_dir / "publication"
        
        # RQ1 & RQ2 tables use data WITH random agent
        # RQ3 & RQ4 tables use data WITHOUT random agent
        tables = TableGenerator(
            perf_df_with_random, magic_df_with_random,  # RQ1/RQ2: include random
            perf_df_no_random, magic_df_no_random,      # RQ3/RQ4: exclude random
            features_df, pub_dir, loader
        )
        tables.generate_all(token_df)
        
        logger.info("[Step 5/5] Generating figures...")
        figures = FigureGenerator(
            perf_df_with_random, magic_df_with_random,  # RQ1/RQ2: include random
            perf_df_no_random, magic_df_no_random,      # RQ3/RQ4: exclude random  
            features_df, pub_dir, loader
        )
        figures.generate_all(token_df)
        
        logger.info("=" * 80)
        logger.info("🎉 ANALYSIS COMPLETE 🎉")
        logger.info(f"Outputs: {pub_dir}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()