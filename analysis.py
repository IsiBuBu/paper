#!/usr/bin/env python3
"""
MAgIC Analysis Pipeline - Publication Version

Usage:
    python analysis.py
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
    
    def load(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        perf_path = self.analysis_dir / "performance_metrics.csv"
        magic_path = self.analysis_dir / "magic_behavioral_metrics.csv"
        
        if not perf_path.exists() or not magic_path.exists():
            raise FileNotFoundError(f"Required CSVs not found in {self.analysis_dir}")
        
        perf_df = pd.read_csv(perf_path)
        magic_df = pd.read_csv(magic_path)
        
        perf_df = self._filter_models(perf_df)
        magic_df = self._filter_models(magic_df)
        
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                self.model_configs = config.get('model_configs', {})
                for model_name, cfg in self.model_configs.items():
                    self.display_names[model_name] = cfg.get('display_name', model_name)
        
        logger.info(f"Loaded {len(perf_df)} performance rows, {len(magic_df)} MAgIC rows")
        return perf_df, magic_df
    
    def load_token_data(self) -> pd.DataFrame:
        """Extract reasoning_char_count and tokens_used from experiment files."""
        if not self.experiments_dir or not self.experiments_dir.exists():
            logger.warning("Experiments directory not available for token extraction")
            return pd.DataFrame()
        
        records = []
        
        for exp_file in self.experiments_dir.glob("**/*.json"):
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                
                # Extract metadata
                model = data.get('model', data.get('challenger_model', ''))
                game = data.get('game', data.get('game_name', ''))
                condition = data.get('condition', data.get('condition_name', ''))
                
                if not model or not game:
                    continue
                
                # Skip excluded models
                if 'random' in model.lower() or 'gemma' in model.lower():
                    continue
                
                # Aggregate token stats across simulations
                simulations = data.get('simulations', data.get('results', []))
                if not isinstance(simulations, list):
                    simulations = [simulations]
                
                total_reasoning_chars = 0
                total_tokens = 0
                count = 0
                
                for sim in simulations:
                    if not isinstance(sim, dict):
                        continue
                    
                    # Check for round-level data
                    rounds = sim.get('rounds', sim.get('round_results', []))
                    if isinstance(rounds, list):
                        for rnd in rounds:
                            if isinstance(rnd, dict):
                                # Look for challenger response data
                                for key in ['challenger_response', 'response', 'agent_response']:
                                    resp = rnd.get(key, {})
                                    if isinstance(resp, dict):
                                        total_reasoning_chars += resp.get('reasoning_char_count', 0)
                                        total_tokens += resp.get('tokens_used', 0)
                                        count += 1
                    
                    # Also check top-level metadata
                    if 'reasoning_char_count' in sim:
                        total_reasoning_chars += sim.get('reasoning_char_count', 0)
                        count += 1
                    if 'metadata' in sim and isinstance(sim['metadata'], dict):
                        meta = sim['metadata']
                        total_reasoning_chars += meta.get('total_reasoning_chars', 0)
                        total_tokens += meta.get('total_tokens', 0)
                
                if count > 0:
                    records.append({
                        'model': model,
                        'game': game,
                        'condition': condition,
                        'reasoning_char_count': total_reasoning_chars / count,
                        'tokens_used': total_tokens / count if total_tokens > 0 else 0,
                        'n_observations': count
                    })
                    
            except Exception as e:
                logger.debug(f"Could not parse {exp_file}: {e}")
                continue
        
        if records:
            df = pd.DataFrame(records)
            # Aggregate by model/game/condition
            df = df.groupby(['model', 'game', 'condition']).agg({
                'reasoning_char_count': 'mean',
                'tokens_used': 'mean',
                'n_observations': 'sum'
            }).reset_index()
            logger.info(f"Loaded token data: {len(df)} rows")
            return df
        
        logger.warning("No token data extracted from experiments")
        return pd.DataFrame()
    
    def get_display_name(self, model: str) -> str:
        """Get display name from config.json."""
        if model in self.display_names:
            return self.display_names[model]
        return str(model).split('/')[-1]
    
    def _filter_models(self, df: pd.DataFrame) -> pd.DataFrame:
        def is_excluded(model: str) -> bool:
            m = str(model).lower()
            return 'random' in m or 'gemma' in m
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
    def __init__(self, perf_df: pd.DataFrame, magic_df: pd.DataFrame,
                 features_df: pd.DataFrame, output_dir: Path, loader: DataLoader):
        self.perf_df = perf_df
        self.magic_df = magic_df
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
        logger.info("Generating Win Rate table...")
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
        logger.info("Generating Average Profit table...")
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
        """Table: Game-specific metrics WITH 3P vs 5P."""
        logger.info("Generating Game-Specific metrics table (3P vs 5P)...")
        
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
        logger.info(f"Generating MAgIC table for {game}...")
        
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
        """MLR: Model Features → Performance Metrics."""
        if not STATSMODELS_AVAILABLE:
            return
        
        logger.info("Generating MLR: Features → Performance...")
        
        perf_pivot = self.perf_df.pivot_table(
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
        """MLR: MAgIC → Performance (T5a, T5b)."""
        if not STATSMODELS_AVAILABLE:
            return
        
        logger.info("Generating T5: MAgIC → Performance...")
        
        magic_pivot = self.magic_df.pivot_table(
            index=['game', 'model', 'condition'], columns='metric', values='mean'
        ).reset_index()
        
        perf_pivot = self.perf_df.pivot_table(
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
        if not SKLEARN_AVAILABLE:
            return
        
        logger.info("Generating T6: PCA Variance...")
        
        results = []
        for game, config in GAME_CONFIGS.items():
            game_df = self.magic_df[self.magic_df['game'] == game]
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
        """Table 8: Cost-benefit analysis - Thinking On vs Off per game/condition."""
        logger.info("Generating T8: Cost-Benefit (Tokens vs Performance)...")
        
        if token_df.empty:
            logger.warning("No token data available for cost-benefit analysis")
            return
        
        # Merge with performance data
        profit_df = self.perf_df[self.perf_df['metric'] == 'average_profit'].copy()
        
        # Add thinking flag from features_df
        thinking_map = dict(zip(self.features_df['model'], self.features_df['thinking']))
        
        merged = profit_df.merge(token_df, on=['model', 'game', 'condition'], how='inner')
        merged['thinking'] = merged['model'].map(thinking_map).fillna(0).astype(int)
        merged['thinking_mode'] = merged['thinking'].map({1: 'Think', 0: 'Inst'})
        
        if merged.empty:
            logger.warning("No merged data for cost-benefit")
            return
        
        results = []
        for game in merged['game'].unique():
            for condition in merged[merged['game'] == game]['condition'].unique():
                subset = merged[(merged['game'] == game) & (merged['condition'] == condition)]
                
                for mode in ['Think', 'Inst']:
                    mode_data = subset[subset['thinking_mode'] == mode]
                    if mode_data.empty:
                        continue
                    
                    results.append({
                        'game': game,
                        'condition': condition,
                        'mode': mode,
                        'n_models': len(mode_data),
                        'avg_profit': round(mode_data['mean'].mean(), 2),
                        'std_profit': round(mode_data['mean'].std(), 2) if len(mode_data) > 1 else 0,
                        'avg_reasoning_chars': round(mode_data['reasoning_char_count'].mean(), 0),
                        'avg_tokens': round(mode_data['tokens_used'].mean(), 0) if 'tokens_used' in mode_data else 0,
                    })
        
        if not results:
            return
        
        df_out = pd.DataFrame(results)
        
        # Calculate gain: Think profit - Inst profit per game/condition
        pivot = df_out.pivot_table(
            index=['game', 'condition'], 
            columns='mode', 
            values=['avg_profit', 'avg_reasoning_chars']
        ).reset_index()
        pivot.columns = ['_'.join(col).strip('_') for col in pivot.columns.values]
        
        if 'avg_profit_Think' in pivot.columns and 'avg_profit_Inst' in pivot.columns:
            pivot['profit_gain'] = pivot['avg_profit_Think'] - pivot['avg_profit_Inst']
            pivot['profit_gain_pct'] = ((pivot['avg_profit_Think'] / pivot['avg_profit_Inst'].replace(0, np.nan)) - 1) * 100
            pivot['token_cost'] = pivot.get('avg_reasoning_chars_Think', 0)
            pivot['efficiency'] = pivot['profit_gain'] / pivot['token_cost'].replace(0, np.nan)
            
            # Round values
            for col in ['profit_gain', 'profit_gain_pct', 'token_cost', 'efficiency']:
                if col in pivot.columns:
                    pivot[col] = pivot[col].round(2)
        
        df_out.to_csv(self.output_dir / "T8_cost_benefit_detail.csv", index=False)
        pivot.to_csv(self.output_dir / "T8_cost_benefit_summary.csv", index=False)
        
        self._save_table_as_png(pivot.round(2), "T8_cost_benefit_summary.png", 
                               "T8: Cost-Benefit Analysis (Thinking vs Instruct)", bold_best=False)


# =============================================================================
# FIGURE GENERATOR
# =============================================================================

class FigureGenerator:
    def __init__(self, perf_df: pd.DataFrame, magic_df: pd.DataFrame,
                 features_df: pd.DataFrame, output_dir: Path, loader: DataLoader):
        self.perf_df = perf_df
        self.magic_df = magic_df
        self.features_df = features_df
        self.output_dir = output_dir
        self.loader = loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def _display_name(self, model: str) -> str:
        return self.loader.get_display_name(model)
    
    def similarity_matrix_per_game(self, game: str):
        logger.info(f"Generating similarity matrix for {game}...")
        
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
        """Single similarity score per game: 3P vs 5P."""
        logger.info("Generating 3P vs 5P similarity scores...")
        
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
        logger.info("Generating PCA scree plots...")
        
        games = [g for g, c in GAME_CONFIGS.items() if len(c['magic_metrics']) >= 2]
        if not games:
            return
        
        fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4))
        if len(games) == 1:
            axes = [axes]
        
        for ax, game in zip(axes, games):
            game_df = self.magic_df[self.magic_df['game'] == game]
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
        """Figure 5: Scatter plot of token usage vs average profit."""
        logger.info("Generating F5: Token Usage vs Outcome scatter...")
        
        if token_df.empty:
            logger.warning("No token data for scatter plot")
            return
        
        # Merge with performance
        profit_df = self.perf_df[self.perf_df['metric'] == 'average_profit'].copy()
        
        # Add thinking flag
        thinking_map = dict(zip(self.features_df['model'], self.features_df['thinking']))
        
        merged = profit_df.merge(token_df, on=['model', 'game', 'condition'], how='inner')
        merged['thinking'] = merged['model'].map(thinking_map).fillna(0).astype(int)
        merged['thinking_mode'] = merged['thinking'].map({1: 'Think', 0: 'Inst'})
        merged['display_name'] = merged['model'].apply(self._display_name)
        
        if merged.empty or 'reasoning_char_count' not in merged.columns:
            logger.warning("Insufficient data for scatter plot")
            return
        
        # Filter to only models with some reasoning chars (thinking models)
        # Include all models but distinguish by color
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        games = list(merged['game'].unique())[:4]
        colors = {'Think': 'coral', 'Inst': 'steelblue'}
        
        for idx, game in enumerate(games):
            if idx >= 4:
                break
            ax = axes[idx]
            game_data = merged[merged['game'] == game]
            
            for mode, color in colors.items():
                mode_data = game_data[game_data['thinking_mode'] == mode]
                if mode_data.empty:
                    continue
                
                ax.scatter(
                    mode_data['reasoning_char_count'], 
                    mode_data['mean'],
                    c=color, 
                    label=mode,
                    alpha=0.7,
                    s=100,
                    edgecolors='black',
                    linewidths=0.5
                )
                
                # Add model labels for thinking models with significant reasoning
                for _, row in mode_data.iterrows():
                    if row['reasoning_char_count'] > 100:
                        ax.annotate(
                            row['display_name'],
                            (row['reasoning_char_count'], row['mean']),
                            fontsize=7,
                            alpha=0.8,
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
            
            # Fit regression line if enough data points
            think_data = game_data[game_data['thinking_mode'] == 'Think']
            if len(think_data) >= 3 and think_data['reasoning_char_count'].std() > 0:
                x = think_data['reasoning_char_count'].values
                y = think_data['mean'].values
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.5, linewidth=2)
                
                # Calculate correlation
                corr = np.corrcoef(x, y)[0, 1]
                ax.text(0.95, 0.05, f'r={corr:.2f}', transform=ax.transAxes, 
                       fontsize=10, ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel('Reasoning Characters')
            ax.set_ylabel('Average Profit')
            ax.set_title(f'{game.replace("_", " ").title()}')
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(games), 4):
            axes[idx].set_visible(False)
        
        plt.suptitle('Figure 5: Token Usage vs Performance Outcome', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / "F5_token_vs_outcome.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also create a combined scatter
        self._combined_token_scatter(merged)
    
    def _combined_token_scatter(self, merged: pd.DataFrame):
        """Single combined scatter plot across all games."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        game_markers = {'salop': 'o', 'spulber': 's', 'green_porter': '^', 'athey_bagwell': 'D'}
        colors = {'Think': 'coral', 'Inst': 'steelblue'}
        
        for game in merged['game'].unique():
            game_data = merged[merged['game'] == game]
            marker = game_markers.get(game, 'o')
            
            for mode, color in colors.items():
                mode_data = game_data[game_data['thinking_mode'] == mode]
                if mode_data.empty:
                    continue
                
                ax.scatter(
                    mode_data['reasoning_char_count'],
                    mode_data['mean'],
                    c=color,
                    marker=marker,
                    label=f'{game.replace("_", " ").title()} ({mode})',
                    alpha=0.7,
                    s=80,
                    edgecolors='black',
                    linewidths=0.5
                )
        
        ax.set_xlabel('Reasoning Characters (Token Proxy)', fontsize=12)
        ax.set_ylabel('Average Profit', fontsize=12)
        ax.set_title('Token Usage vs Performance (All Games)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "F5_token_vs_outcome_combined.png", dpi=300, bbox_inches='tight')
        plt.close()


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
        perf_df, magic_df = loader.load()
        
        # Load token data for cost-benefit analysis
        token_df = loader.load_token_data()
        
        all_models = list(set(perf_df['model'].unique()) | set(magic_df['model'].unique()))
        features_df = loader.extract_model_features(all_models)
        
        logger.info("[Step 4/5] Generating tables...")
        pub_dir = analysis_dir / "publication"
        tables = TableGenerator(perf_df, magic_df, features_df, pub_dir, loader)
        tables.generate_all(token_df)
        
        logger.info("[Step 5/5] Generating figures...")
        figures = FigureGenerator(perf_df, magic_df, features_df, pub_dir, loader)
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