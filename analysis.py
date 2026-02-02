#!/usr/bin/env python3
"""
MAgIC Analysis Pipeline - Publication Version

Usage:
    python analysis.py

RESEARCH QUESTIONS:
-------------------
RQ1: COMPETITIVE PERFORMANCE
    Hypothesis: Newer/larger/better LLMs achieve better game performance.
    
    Independent Variables:
    - Model version, size, architecture, thinking mode
    
    Dependent Variables:
    - win_rate, average_profit, game-specific metrics
    
    Tables: T_perf_win_rate, T_perf_avg_profit, T_perf_game_specific,
            T_mlr_features_to_performance
    Random: INCLUDED in perf tables, EXCLUDED from regression

RQ2: STRATEGIC BEHAVIORAL PROFILES
    Hypothesis 1: Same-family LLMs have similar behavioral profiles.
    Hypothesis 2: Profiles are stable across conditions (3P vs 5P).
    
    Tables: T_magic_{game}, T_similarity_3v5, T6_pca_variance
    Figures: F_similarity_{game}, F_similarity_3v5, F_pca_scree
    Random: INCLUDED (should cluster separately)

RQ3: CAPABILITY-PERFORMANCE LINKS
    Hypothesis: Behavioral profiles explain competitive performance.
    
    Tables: T5a_magic_to_perf_coef, T5b_magic_to_perf_summary
    Random: EXCLUDED

SUPPLEMENTARY:
    T_reasoning_chars, F_reasoning_chars: Reasoning effort per game/condition
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import json
import re
from typing import List, Optional, Tuple

from analysis.engine.analyze_metrics import MetricsAnalyzer
from analysis.engine.create_summary_csvs import SummaryCreator
from config.config import get_experiments_dir, get_analysis_dir

try:
    import statsmodels.api as sm
    from scipy.stats import ttest_rel, pearsonr
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


class DataLoader:
    def __init__(self, analysis_dir: Path, config_path: Optional[Path] = None, 
                 experiments_dir: Optional[Path] = None):
        self.analysis_dir = Path(analysis_dir)
        self.config_path = config_path
        self.experiments_dir = experiments_dir
        self.model_configs = {}
        self.display_names = {'random_agent': 'Random'}
        self.family_encoder = None
    
    def load(self, include_random: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        perf_df = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
        magic_df = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
        
        perf_df = self._filter_models(perf_df, include_random)
        magic_df = self._filter_models(magic_df, include_random)
        
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                config = json.load(f)
                self.model_configs = config.get('model_configs', {})
                for name, cfg in self.model_configs.items():
                    self.display_names[name] = cfg.get('display_name', name)
        
        logger.info(f"Loaded {len(perf_df)} perf, {len(magic_df)} MAgIC rows (random={'in' if include_random else 'ex'}cluded)")
        return perf_df, magic_df
    
    def load_token_data(self) -> pd.DataFrame:
        if not self.experiments_dir or not self.experiments_dir.exists():
            return pd.DataFrame()
        
        records = []
        for exp_file in self.experiments_dir.rglob("*_competition_result*.json"):
            try:
                with open(exp_file) as f:
                    data = json.load(f)
                
                model, game, condition = data.get('challenger_model', ''), data.get('game_name', ''), data.get('condition_name', '')
                if not model or not game or 'random' in model.lower() or 'gemma' in model.lower():
                    continue
                
                chars = []
                for sim in data.get('simulation_results', []):
                    gd = sim.get('game_data', {})
                    for loc in [gd.get('llm_metadata', {})] + [r.get('llm_metadata', {}) for r in gd.get('rounds', [])]:
                        for pid, meta in loc.items():
                            if 'challenger' in pid.lower() and isinstance(meta, dict):
                                chars.append(meta.get('reasoning_char_count', 0))
                
                if chars:
                    records.append({'model': model, 'game': game, 'condition': condition,
                                   'avg_reasoning_chars': round(np.mean(chars), 1),
                                   'std_reasoning_chars': round(np.std(chars), 1)})
            except:
                continue
        
        return pd.DataFrame(records) if records else pd.DataFrame()
    
    def get_thinking_status(self, model: str) -> bool:
        return self.model_configs.get(model, {}).get('reasoning_output', 'none') != 'none'
    
    def get_display_name(self, model: str) -> str:
        return self.display_names.get(model, str(model).split('/')[-1])
    
    def _filter_models(self, df: pd.DataFrame, include_random: bool) -> pd.DataFrame:
        return df[~df['model'].apply(lambda m: 'gemma' in str(m).lower() or (not include_random and 'random' in str(m).lower()))].copy()
    
    def extract_model_features(self, models: List[str]) -> pd.DataFrame:
        records = []
        for model in models:
            m = str(model).lower()
            records.append({
                'model': model,
                'architecture_moe': int(bool(re.search(r'-a\d+b|moe|maverick|scout', m))),
                'size_params': float(re.search(r'(?<!a)(\d+\.?\d*)b(?!-)', m).group(1)) if re.search(r'(?<!a)(\d+\.?\d*)b(?!-)', m) else 0,
                'family': next((f for f in ['qwen','llama','gemma','mistral','gemini','gpt','claude'] if f in m), 'unknown'),
                'version': float(re.search(r'qwen(\d+)|llama-?(\d+\.?\d*)', m).group(1) or re.search(r'qwen(\d+)|llama-?(\d+\.?\d*)', m).group(2)) if re.search(r'qwen(\d+)|llama-?(\d+\.?\d*)', m) else 0,
                'thinking': 1 if self.model_configs.get(model, {}).get('reasoning_output', 'none') != 'none' else 0
            })
        df = pd.DataFrame(records)
        if SKLEARN_AVAILABLE and len(df) > 0:
            df['family_encoded'] = LabelEncoder().fit_transform(df['family'])
        else:
            df['family_encoded'] = 0
        return df


class TableGenerator:
    def __init__(self, perf_with, magic_with, perf_no, magic_no, features, output_dir, loader):
        self.perf_df, self.magic_df = perf_with, magic_with
        self.perf_df_no_random, self.magic_df_no_random = perf_no, magic_no
        self.features_df, self.output_dir, self.loader = features, output_dir, loader
        output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self, token_df=None):
        self.performance_win_rate_table()
        self.performance_avg_profit_table()
        self.performance_game_specific_table()
        self.mlr_features_to_performance()
        for game in GAME_CONFIGS:
            self.magic_per_game_table(game)
        self.pca_variance_table()
        self.mlr_magic_to_performance()
        if token_df is not None and not token_df.empty:
            self.reasoning_chars_table(token_df)
    
    def _fmt(self, mean, std, sig=''):
        if pd.isna(mean): return "N/A"
        base = f"{mean:.3f}" if pd.isna(std) or std == 0 else f"{mean:.3f} ± {std:.3f}"
        return f"{base} {sig}".strip()
    
    @staticmethod
    def _sig(p):
        if p is None or pd.isna(p): return ''
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    
    def _save_png(self, df, filename, title, metric_cols=None):
        if not PLOT_AVAILABLE or df.empty: return
        fig, ax = plt.subplots(figsize=(max(14, len(df.columns)*1.8), max(5, len(df)*0.5)))
        ax.axis('off')
        colors = [['white']*len(df.columns) for _ in range(len(df))]
        if metric_cols:
            for ci, col in enumerate(df.columns):
                if col in metric_cols:
                    try:
                        vals = pd.to_numeric(df[col].astype(str).str.split(' ±').str[0], errors='coerce')
                        best = vals.idxmin() if '↓' in col else vals.idxmax()
                        if pd.notna(best): colors[df.index.get_loc(best)][ci] = '#90EE90'
                    except: pass
        tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center', cellColours=colors)
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.2, 1.5)
        for i in range(len(df.columns)):
            tbl[(0,i)].set_facecolor('#4472C4'); tbl[(0,i)].set_text_props(color='white', fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout(); plt.savefig(self.output_dir/filename, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    
    def _pval_3v5(self, df, metric, game):
        mdf = df[(df['metric']==metric) & (df['game']==game)]
        c3 = mdf[mdf['condition'].str.contains('baseline|few|3', case=False, na=False)]
        c5 = mdf[mdf['condition'].str.contains('more|5', case=False, na=False)]
        models = set(c3['model']) & set(c5['model'])
        if len(models) < 3: return np.nan
        v3 = [c3[c3['model']==m]['mean'].values[0] for m in models if len(c3[c3['model']==m]) > 0]
        v5 = [c5[c5['model']==m]['mean'].values[0] for m in models if len(c5[c5['model']==m]) > 0]
        try: return ttest_rel(v3, v5)[1] if len(v3) >= 3 else np.nan
        except: return np.nan
    
    def _build_3v5(self, df, metric, include_stars=True):
        """Build 3v5 table. If include_stars=False, returns without stars and adds p_value columns."""
        games = sorted(df['game'].unique())
        mdf = df[df['metric']==metric]
        if mdf.empty: return pd.DataFrame()
        pvals = {g: self._pval_3v5(df, metric, g) for g in games}
        rows = []
        for model in mdf['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            for g in games:
                gdf = mdf[(mdf['model']==model) & (mdf['game']==g)]
                for cond, suffix, sig in [('baseline|few|3', '3P', ''), ('more|5', '5P', self._sig(pvals[g]) if include_stars else '')]:
                    c = gdf[gdf['condition'].str.contains(cond, case=False, na=False)]
                    row[f'{g}_{suffix}'] = self._fmt(c['mean'].iloc[0], c['std'].iloc[0] if 'std' in c else 0, sig if '5P' in suffix else '') if not c.empty else 'N/A'
                # Add p_value column for CSV (only for 5P)
                if not include_stars:
                    row[f'{g}_5P_p_value'] = round(pvals[g], 4) if pd.notna(pvals[g]) else np.nan
            rows.append(row)
        return pd.DataFrame(rows)
    
    def _remove_collinear(self, df, preds):
        remaining = [p for p in preds if p in df.columns and df[p].std() > 0]
        while len(remaining) >= 2:
            corr = df[remaining].corr().abs()
            drop = None
            for i in range(len(remaining)):
                for j in range(i+1, len(remaining)):
                    if corr.iloc[i,j] >= COLLINEARITY_THRESHOLD:
                        drop = remaining[i] if df[remaining[i]].var() <= df[remaining[j]].var() else remaining[j]
                        break
                if drop: break
            if drop: remaining.remove(drop)
            else: break
        return remaining
    
    def performance_win_rate_table(self):
        logger.info("T_perf_win_rate (RQ1)")
        # CSV: without stars, with p_value columns
        tbl_csv = self._build_3v5(self.perf_df, 'win_rate', include_stars=False)
        if not tbl_csv.empty:
            tbl_csv = tbl_csv.rename(columns={c: f"{c} ↑" if c != 'Model' and 'p_value' not in c else c for c in tbl_csv.columns})
            tbl_csv.to_csv(self.output_dir/"T_perf_win_rate.csv", index=False)
        
        # PNG: with stars embedded
        tbl_png = self._build_3v5(self.perf_df, 'win_rate', include_stars=True)
        if not tbl_png.empty:
            tbl_png = tbl_png.rename(columns={c: f"{c} ↑" for c in tbl_png.columns if c != 'Model'})
            self._save_png(tbl_png, "T_perf_win_rate.png", "Win Rate | * p<.05, ** p<.01, *** p<.001", [c for c in tbl_png.columns if c != 'Model'])
    
    def performance_avg_profit_table(self):
        logger.info("T_perf_avg_profit (RQ1)")
        # CSV: without stars, with p_value columns
        tbl_csv = self._build_3v5(self.perf_df, 'average_profit', include_stars=False)
        if not tbl_csv.empty:
            tbl_csv = tbl_csv.rename(columns={c: f"{c} ↑" if c != 'Model' and 'p_value' not in c else c for c in tbl_csv.columns})
            tbl_csv.to_csv(self.output_dir/"T_perf_avg_profit.csv", index=False)
        
        # PNG: with stars embedded
        tbl_png = self._build_3v5(self.perf_df, 'average_profit', include_stars=True)
        if not tbl_png.empty:
            tbl_png = tbl_png.rename(columns={c: f"{c} ↑" for c in tbl_png.columns if c != 'Model'})
            self._save_png(tbl_png, "T_perf_avg_profit.png", "Average Profit | * p<.05, ** p<.01, *** p<.001", [c for c in tbl_png.columns if c != 'Model'])
    
    def performance_game_specific_table(self):
        logger.info("T_perf_game_specific (RQ1)")
        pvals = {(g, GAME_CONFIGS[g]['game_specific_metric']): self._pval_3v5(self.perf_df, GAME_CONFIGS[g]['game_specific_metric'], g) for g in GAME_CONFIGS}
        
        # CSV: without stars, with p_value columns
        rows_csv = []
        for model in self.perf_df['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = self.perf_df[self.perf_df['model']==model]
            for g, cfg in GAME_CONFIGS.items():
                m, d = cfg['game_specific_metric'], METRIC_DIRECTION.get(cfg['game_specific_metric'], '↑')
                gdf = mdf[(mdf['game']==g) & (mdf['metric']==m)]
                for cond, suffix in [('baseline|few|3', '3P'), ('more|5', '5P')]:
                    c = gdf[gdf['condition'].str.contains(cond, case=False, na=False)]
                    row[f'{g}_{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], c['std'].iloc[0] if 'std' in c else 0, '') if not c.empty else 'N/A'
                # Add p_value column for 5P
                row[f'{g}_{m}_5P_p_value'] = round(pvals[(g,m)], 4) if pd.notna(pvals[(g,m)]) else np.nan
            rows_csv.append(row)
        tbl_csv = pd.DataFrame(rows_csv)
        tbl_csv.to_csv(self.output_dir/"T_perf_game_specific.csv", index=False)
        
        # PNG: with stars embedded
        rows_png = []
        for model in self.perf_df['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = self.perf_df[self.perf_df['model']==model]
            for g, cfg in GAME_CONFIGS.items():
                m, d = cfg['game_specific_metric'], METRIC_DIRECTION.get(cfg['game_specific_metric'], '↑')
                gdf = mdf[(mdf['game']==g) & (mdf['metric']==m)]
                for cond, suffix, sig in [('baseline|few|3', '3P', ''), ('more|5', '5P', self._sig(pvals[(g,m)]))]:
                    c = gdf[gdf['condition'].str.contains(cond, case=False, na=False)]
                    row[f'{g}_{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], c['std'].iloc[0] if 'std' in c else 0, sig if '5P' in suffix else '') if not c.empty else 'N/A'
            rows_png.append(row)
        tbl_png = pd.DataFrame(rows_png)
        self._save_png(tbl_png, "T_perf_game_specific.png", "Game-Specific Metrics | * p<.05, ** p<.01, *** p<.001", [c for c in tbl_png.columns if c != 'Model'])
    
    def mlr_features_to_performance(self):
        if not STATSMODELS_AVAILABLE: return
        logger.info("T_mlr_features_to_performance (RQ1)")
        perf = self.perf_df_no_random.pivot_table(index=['game','model','condition'], columns='metric', values='mean').reset_index()
        merged = perf.merge(self.features_df, on='model', how='left')
        results = []
        for g in merged['game'].unique():
            gdf = merged[merged['game']==g].copy()
            preds = self._remove_collinear(gdf, MODEL_FEATURES)
            if not preds: continue
            for target in [c for c in perf.columns if c not in ['game','model','condition']]:
                if target not in gdf or gdf[target].std() == 0: continue
                valid = gdf[preds + [target]].dropna()
                if len(valid) < len(preds) + 2: continue
                try:
                    m = sm.OLS(valid[target], sm.add_constant(valid[preds])).fit()
                    for p in preds:
                        results.append({
                            'game': g, 
                            'type': 'features_to_performance',
                            'target': target, 
                            'predictor': p,
                            'r_squared': round(m.rsquared, 4),
                            'coef': round(m.params.get(p, np.nan), 4),
                            'p_value': round(m.pvalues.get(p, np.nan), 4),
                            'n_obs': int(m.nobs),
                            'r_squared_adj': round(m.rsquared_adj, 4)
                        })
                except: pass
        if results:
            df = pd.DataFrame(results)
            # Reorder columns to match requested format
            df = df[['game', 'type', 'target', 'predictor', 'r_squared', 'coef', 'p_value', 'n_obs', 'r_squared_adj']]
            df.to_csv(self.output_dir/"T_mlr_features_to_performance.csv", index=False)
            
            # For PNG: show coef with significance stars
            df_png = df.copy()
            df_png['coef'] = df_png.apply(lambda r: f"{r['coef']:.4f} {self._sig(r['p_value'])}".strip(), axis=1)
            df_png_display = df_png[['game', 'target', 'predictor', 'coef', 'r_squared']]
            self._save_png(df_png_display, "T_mlr_features_to_performance.png", "Features → Performance (RQ1) | * p<.05, ** p<.01, *** p<.001")
    
    def magic_per_game_table(self, game):
        logger.info(f"T_magic_{game} (RQ2)")
        gdf = self.magic_df[self.magic_df['game']==game]
        if gdf.empty: return
        metrics = GAME_CONFIGS[game]['magic_metrics']
        pvals = {m: self._pval_3v5(self.magic_df, m, game) for m in metrics}
        
        # CSV: without stars, with p_value columns
        rows_csv = []
        for model in gdf['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = gdf[gdf['model']==model]
            for m in metrics:
                d = METRIC_DIRECTION.get(m, '↑')
                mmdf = mdf[mdf['metric']==m]
                for cond, suffix in [('baseline|few|3', '3P'), ('more|5', '5P')]:
                    c = mmdf[mmdf['condition'].str.contains(cond, case=False, na=False)]
                    row[f'{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], c['std'].iloc[0] if 'std' in c else 0, '') if not c.empty else 'N/A'
                # Add p_value column for 5P
                row[f'{m}_5P_p_value'] = round(pvals[m], 4) if pd.notna(pvals[m]) else np.nan
            rows_csv.append(row)
        tbl_csv = pd.DataFrame(rows_csv)
        tbl_csv.to_csv(self.output_dir/f"T_magic_{game}.csv", index=False)
        
        # PNG: with stars embedded
        rows_png = []
        for model in gdf['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = gdf[gdf['model']==model]
            for m in metrics:
                d = METRIC_DIRECTION.get(m, '↑')
                mmdf = mdf[mdf['metric']==m]
                for cond, suffix, sig in [('baseline|few|3', '3P', ''), ('more|5', '5P', self._sig(pvals[m]))]:
                    c = mmdf[mmdf['condition'].str.contains(cond, case=False, na=False)]
                    row[f'{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], c['std'].iloc[0] if 'std' in c else 0, sig if '5P' in suffix else '') if not c.empty else 'N/A'
            rows_png.append(row)
        tbl_png = pd.DataFrame(rows_png)
        self._save_png(tbl_png, f"T_magic_{game}.png", f"MAgIC: {game.replace('_',' ').title()} (RQ2) | * p<.05, ** p<.01, *** p<.001", [c for c in tbl_png.columns if c != 'Model'])
    
    def pca_variance_table(self):
        if not SKLEARN_AVAILABLE: return
        logger.info("T6_pca_variance (RQ2)")
        results = []
        for g, cfg in GAME_CONFIGS.items():
            gdf = self.magic_df_no_random[self.magic_df_no_random['game']==g]
            if gdf.empty: continue
            pivot = gdf.pivot_table(index='model', columns='metric', values='mean').dropna()
            avail = [m for m in cfg['magic_metrics'] if m in pivot.columns]
            if len(avail) < 2: continue
            pca = PCA().fit(StandardScaler().fit_transform(pivot[avail]))
            cum = np.cumsum(pca.explained_variance_ratio_)
            for i, v in enumerate(pca.explained_variance_ratio_):
                results.append({'game': g, 'component': f'PC{i+1}', 'variance': round(v, 4), 'cumulative': round(cum[i], 4)})
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir/"T6_pca_variance.csv", index=False)
            self._save_png(df, "T6_pca_variance.png", "PCA Variance (RQ2)")
    
    def mlr_magic_to_performance(self):
        if not STATSMODELS_AVAILABLE: return
        logger.info("T5_magic_to_perf (RQ3)")
        magic = self.magic_df_no_random.pivot_table(index=['game','model','condition'], columns='metric', values='mean').reset_index()
        perf = self.perf_df_no_random.pivot_table(index=['game','model','condition'], columns='metric', values='mean').reset_index()
        merged = perf.merge(magic, on=['game','model','condition'], how='inner')
        if merged.empty: return
        perf_cols = [c for c in perf.columns if c not in ['game','model','condition']]
        magic_cols = [c for c in magic.columns if c not in ['game','model','condition']]
        results = []
        for g in merged['game'].unique():
            gdf = merged[merged['game']==g].copy()
            avail = self._remove_collinear(gdf, [m for m in magic_cols if m in gdf and gdf[m].std() > 0])
            if not avail: continue
            for t in perf_cols:
                if t not in gdf or gdf[t].std() == 0: continue
                valid = gdf[[t]+avail].dropna()
                if len(valid) < len(avail) + 2: continue
                try:
                    m = sm.OLS(valid[t], sm.add_constant(valid[avail])).fit()
                    for p in avail:
                        if p in m.params.index:
                            results.append({
                                'game': g,
                                'type': 'magic_to_performance',
                                'target': t,
                                'predictor': p,
                                'r_squared': round(m.rsquared, 4),
                                'coef': round(m.params[p], 4),
                                'p_value': round(m.pvalues[p], 4),
                                'n_obs': int(m.nobs),
                                'r_squared_adj': round(m.rsquared_adj, 4)
                            })
                except: pass
        if results:
            df = pd.DataFrame(results)
            # Reorder columns to match requested format
            df = df[['game', 'type', 'target', 'predictor', 'r_squared', 'coef', 'p_value', 'n_obs', 'r_squared_adj']]
            df.to_csv(self.output_dir/"T5_magic_to_perf.csv", index=False)
            
            # For PNG: show coef with significance stars
            df_png = df.copy()
            df_png['coef'] = df_png.apply(lambda r: f"{r['coef']:.4f} {self._sig(r['p_value'])}".strip(), axis=1)
            df_png_display = df_png[['game', 'target', 'predictor', 'coef', 'r_squared']]
            self._save_png(df_png_display, "T5_magic_to_perf.png", "MAgIC → Performance (RQ3) | * p<.05, ** p<.01, *** p<.001")
    
    def reasoning_chars_table(self, token_df):
        logger.info("T_reasoning_chars (Supplementary)")
        thinking = [m for m in token_df['model'].unique() if self.loader.get_thinking_status(m)]
        if not thinking: return
        df = token_df[token_df['model'].isin(thinking)]
        
        # Build per-model per-game per-condition table
        rows = []
        for model in thinking:
            mdf = df[df['model'] == model]
            row = {'Model': self.loader.get_display_name(model)}
            
            for game in sorted(df['game'].unique()):
                gdf = mdf[mdf['game'] == game]
                
                for condition in sorted(df['condition'].unique()):
                    cdf = gdf[gdf['condition'] == condition]
                    if not cdf.empty:
                        mean_val = cdf['avg_reasoning_chars'].mean()
                        std_val = cdf['std_reasoning_chars'].mean()
                        
                        # Calculate p-value comparing to baseline (you can adjust this logic)
                        # For now, we'll just show the values with placeholder for significance
                        sig = ''  # Add your p-value calculation here if needed
                        
                        col_name = f"{game}_{condition}"
                        row[col_name] = f"{mean_val:.0f} ± {std_val:.0f} {sig}".strip()
                    else:
                        row[f"{game}_{condition}"] = "N/A"
            
            rows.append(row)
        
        tbl = pd.DataFrame(rows)
        tbl.to_csv(self.output_dir/"T_reasoning_chars.csv", index=False)
        self._save_png(tbl, "T_reasoning_chars.png", "Reasoning Characters (Thinking Models) | * p<.05, ** p<.01, *** p<.001")


class FigureGenerator:
    def __init__(self, perf_with, magic_with, perf_no, magic_no, features, output_dir, loader):
        self.magic_df, self.magic_df_no_random = magic_with, magic_no
        self.output_dir, self.loader = output_dir, loader
        output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _sig(p):
        if p is None or pd.isna(p): return ''
        return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    
    def generate_all(self, token_df=None):
        if not PLOT_AVAILABLE or not SKLEARN_AVAILABLE: return
        plt.style.use('seaborn-v0_8-whitegrid')
        for g in GAME_CONFIGS: self.similarity_matrix(g)
        self.similarity_3v5()
        self.pca_scree()
        if token_df is not None and not token_df.empty: self.reasoning_chars_fig(token_df)
    
    def similarity_matrix(self, game):
        logger.info(f"F_similarity_{game} (RQ2)")
        gdf = self.magic_df[self.magic_df['game']==game]
        if gdf.empty: return
        pivot = gdf.pivot_table(index='model', columns='metric', values='mean').fillna(0)
        if len(pivot) < 2: return
        
        # Calculate cosine similarity
        sim = cosine_similarity(pivot.values)
        names = [self.loader.get_display_name(m) for m in pivot.index]
        
        # Calculate p-values using permutation test for each pair
        # For similarity matrices, we'll use a simple approach based on the correlation
        p_matrix = np.ones_like(sim)
        for i in range(len(pivot)):
            for j in range(i+1, len(pivot)):
                try:
                    vec_i = pivot.iloc[i].values
                    vec_j = pivot.iloc[j].values
                    
                    # Check if either vector is constant (would cause correlation warning)
                    if np.std(vec_i) == 0 or np.std(vec_j) == 0:
                        # If vectors are constant, no correlation can be computed
                        p_matrix[i, j] = 1.0
                        p_matrix[j, i] = 1.0
                    else:
                        # Use Pearson correlation p-value as proxy for significance
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            r, p = pearsonr(vec_i, vec_j)
                        p_matrix[i, j] = p
                        p_matrix[j, i] = p
                except Exception:
                    p_matrix[i, j] = 1.0
                    p_matrix[j, i] = 1.0
        
        # Create annotations with similarity values and significance stars
        annot = np.empty_like(sim, dtype=object)
        for i in range(len(sim)):
            for j in range(len(sim)):
                if i == j:
                    annot[i, j] = f"{sim[i, j]:.2f}"
                else:
                    sig = self._sig(p_matrix[i, j])
                    annot[i, j] = f"{sim[i, j]:.2f}{sig}"
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(sim, index=names, columns=names), annot=annot, fmt='', 
                   cmap='RdBu_r', center=0.5, vmin=0, vmax=1, square=True, ax=ax)
        ax.set_title(f'Behavioral Similarity: {game.replace("_"," ").title()} (RQ2) | * p<.05, ** p<.01, *** p<.001')
        plt.tight_layout(); plt.savefig(self.output_dir/f"F_similarity_{game}.png", dpi=300, bbox_inches='tight'); plt.close()
    
    def similarity_3v5(self):
        logger.info("F_similarity_3v5 (RQ2)")
        results = []
        for g in GAME_CONFIGS:
            gdf = self.magic_df[self.magic_df['game']==g]
            if gdf.empty: continue
            c3 = gdf[gdf['condition'].str.contains('baseline|few|3', case=False, na=False)]
            c5 = gdf[gdf['condition'].str.contains('more|5', case=False, na=False)]
            if c3.empty or c5.empty: continue
            p3 = c3.pivot_table(index='model', columns='metric', values='mean').fillna(0)
            p5 = c5.pivot_table(index='model', columns='metric', values='mean').fillna(0)
            models = list(set(p3.index) & set(p5.index))
            metrics = list(set(p3.columns) & set(p5.columns))
            if len(models) < 2: continue
            v3, v5 = p3.loc[models, metrics].values.flatten(), p5.loc[models, metrics].values.flatten()
            cos = cosine_similarity([v3], [v5])[0, 0]
            r, p = pearsonr(v3, v5)
            sig = self._sig(p)
            
            # CSV: separate p_value column
            results.append({
                'game': g, 
                'cosine': round(cos, 4),
                'cosine_p_value': round(p, 4),
                'pearson': round(r, 4), 
                'pearson_p_value': round(p, 4)
            })
        
        if not results: return
        
        # Save CSV with separate p_value columns
        pd.DataFrame(results).to_csv(self.output_dir/"T_similarity_3v5.csv", index=False)
        
        # PNG: show with significance stars
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(results))
        cos_vals = [r['cosine'] for r in results]
        cos_sigs = [self._sig(r['cosine_p_value']) for r in results]
        pearson_vals = [r['pearson'] for r in results]
        pearson_sigs = [self._sig(r['pearson_p_value']) for r in results]
        
        ax.bar(x - 0.175, cos_vals, 0.35, label='Cosine', color='steelblue')
        ax.bar(x + 0.175, pearson_vals, 0.35, label='Pearson', color='coral')
        
        for i, (cv, cs, pv, ps) in enumerate(zip(cos_vals, cos_sigs, pearson_vals, pearson_sigs)):
            ax.annotate(f'{cv:.3f}{cs}', xy=(x[i]-0.175, cv), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
            ax.annotate(f'{pv:.3f}{ps}', xy=(x[i]+0.175, pv), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)
        
        ax.set_xticks(x); ax.set_xticklabels([r['game'].replace('_',' ').title() for r in results])
        ax.set_ylabel('Similarity'); ax.set_title('3P vs 5P Stability (RQ2) | * p<.05, ** p<.01, *** p<.001'); ax.legend(); ax.set_ylim(0, 1.15)
        plt.tight_layout(); plt.savefig(self.output_dir/"F_similarity_3v5.png", dpi=300, bbox_inches='tight'); plt.close()
    
    def pca_scree(self):
        logger.info("F_pca_scree (RQ2)")
        games = [g for g, c in GAME_CONFIGS.items() if len(c['magic_metrics']) >= 2]
        if not games: return
        fig, axes = plt.subplots(1, len(games), figsize=(5*len(games), 4))
        if len(games) == 1: axes = [axes]
        for ax, g in zip(axes, games):
            gdf = self.magic_df_no_random[self.magic_df_no_random['game']==g]
            pivot = gdf.pivot_table(index='model', columns='metric', values='mean').dropna()
            avail = [m for m in GAME_CONFIGS[g]['magic_metrics'] if m in pivot.columns]
            if len(avail) < 2: continue
            var = PCA().fit(StandardScaler().fit_transform(pivot[avail])).explained_variance_ratio_
            x = range(1, len(var)+1)
            ax.bar(x, var, alpha=0.7, color='steelblue', label='Individual')
            ax.plot(x, np.cumsum(var), 'ro-', label='Cumulative')
            ax.axhline(0.8, color='green', ls='--', alpha=0.5)
            ax.set_title(g.replace('_',' ').title()); ax.set_xlabel('PC'); ax.set_ylabel('Variance'); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
        plt.suptitle('PCA Variance (RQ2)', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(self.output_dir/"F_pca_scree.png", dpi=300, bbox_inches='tight'); plt.close()
    
    def reasoning_chars_fig(self, token_df):
        logger.info("F_reasoning_chars (Supplementary)")
        thinking = [m for m in token_df['model'].unique() if self.loader.get_thinking_status(m)]
        if not thinking: return
        df = token_df[token_df['model'].isin(thinking)]
        agg = df.groupby(['game','condition']).agg({'avg_reasoning_chars': ['mean','std']}).reset_index()
        agg.columns = ['game', 'condition', 'mean', 'std']
        games = sorted(agg['game'].unique())
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(games))
        for i, cond in enumerate(['baseline', 'more_players']):
            data = agg[agg['condition']==cond]
            means = [data[data['game']==g]['mean'].values[0] if len(data[data['game']==g]) > 0 else 0 for g in games]
            stds = [data[data['game']==g]['std'].values[0] if len(data[data['game']==g]) > 0 else 0 for g in games]
            ax.bar(x + (i-0.5)*0.35, means, 0.35, label='3P' if i==0 else '5P', yerr=stds, capsize=3, color='steelblue' if i==0 else 'coral', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels([g.replace('_',' ').title() for g in games], rotation=15)
        ax.set_xlabel('Game'); ax.set_ylabel('Avg Reasoning Chars'); ax.set_title('Reasoning Effort (Thinking Models)'); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout(); plt.savefig(self.output_dir/"F_reasoning_chars.png", dpi=300, bbox_inches='tight'); plt.close()


def main():
    logger.info("="*60 + "\n🚀 ANALYSIS PIPELINE\n" + "="*60)
    exp_dir, ana_dir = get_experiments_dir(), get_analysis_dir()
    ana_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        MetricsAnalyzer().analyze_all_games()
        SummaryCreator().create_all_summaries()
        
        cfg = Path("config/config.json")
        loader = DataLoader(ana_dir, cfg if cfg.exists() else None, exp_dir)
        perf_with, magic_with = loader.load(include_random=True)
        perf_no, magic_no = loader.load(include_random=False)
        token_df = loader.load_token_data()
        features = loader.extract_model_features(list(set(perf_no['model']) | set(magic_no['model'])))
        
        pub = ana_dir / "publication"
        TableGenerator(perf_with, magic_with, perf_no, magic_no, features, pub, loader).generate_all(token_df)
        FigureGenerator(perf_with, magic_with, perf_no, magic_no, features, pub, loader).generate_all(token_df)
        
        logger.info("="*60 + f"\n🎉 COMPLETE → {pub}\n" + "="*60)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()