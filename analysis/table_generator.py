#!/usr/bin/env python3
"""
Table Generator - Publication Tables for MAgIC Analysis

Generates CSV and PNG tables for research questions:
- RQ1: Competitive Performance (T_perf_*, T_mlr_features_to_performance)
- RQ2: Strategic Behavioral Profiles (T_magic_*, T6_pca_variance, T_similarity_3v5)
- RQ3: Capability-Performance Links (T5_magic_to_perf, T7_combined_to_perf)
- Supplementary: T_reasoning_chars

Dependencies:
- statsmodels: For regression analysis
- matplotlib: For PNG generation
- scipy: For statistical tests
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger("TableGenerator")

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from scipy.stats import ttest_rel
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    logger.warning("statsmodels/scipy not available - regression tables disabled")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - PCA tables disabled")

try:
    import matplotlib.pyplot as plt
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    logger.warning("matplotlib not available - PNG generation disabled")

from .utils import sig_stars, format_value, remove_collinear_predictors, GAME_CONFIGS, METRIC_DIRECTION, MODEL_FEATURES

COLLINEARITY_THRESHOLD = 0.95


class TableGenerator:
    """
    Generates publication-ready tables (CSV + PNG) for MAgIC analysis.
    
    Tables are generated in two formats:
    - CSV: Raw values with separate p_value columns for reproducibility
    - PNG: Formatted with significance stars embedded for visual presentation
    
    Parameters:
        perf_with: Performance data including random agent
        magic_with: MAgIC data including random agent
        perf_no: Performance data excluding random agent
        magic_no: MAgIC data excluding random agent
        features: Model architectural features
        output_dir: Directory for output files
        loader: DataLoader instance for display names and thinking status
    """
    
    def __init__(self, perf_with, magic_with, perf_no, magic_no, features, output_dir, loader):
        self.perf_df = perf_with
        self.magic_df = magic_with
        self.perf_df_no_random = perf_no
        self.magic_df_no_random = magic_no
        self.features_df = features
        self.output_dir = Path(output_dir)
        self.loader = loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_all(self, token_df=None):
        """Generate all publication tables."""
        logger.info("Generating all tables...")
        
        # RQ1: Competitive Performance
        self.performance_win_rate_table()
        self.performance_avg_profit_table()
        self.performance_game_specific_table()
        self.mlr_features_to_performance()
        
        # RQ2: Strategic Behavioral Profiles
        for game in GAME_CONFIGS:
            self.magic_per_game_table(game)
        self.pca_variance_table()
        
        # RQ3: Capability-Performance Links
        self.mlr_magic_to_performance()
        self.mlr_combined_to_performance()
        
        # Supplementary
        if token_df is not None and not token_df.empty:
            self.reasoning_chars_table(token_df)
        
        logger.info(f"All tables generated in {self.output_dir}")
    
    def _fmt(self, mean, std, sig=''):
        """Format value with optional std and significance stars."""
        return format_value(mean, std, sig)
    
    @staticmethod
    def _sig(p):
        """Convert p-value to significance stars."""
        return sig_stars(p)
    
    def _save_png(self, df, filename, title, metric_cols=None):
        """Save table as PNG with optional highlighting of best values."""
        if not PLOT_AVAILABLE or df.empty:
            return
        
        fig, ax = plt.subplots(figsize=(max(14, len(df.columns)*1.8), max(5, len(df)*0.5)))
        ax.axis('off')
        
        # Prepare colors for highlighting best values
        colors = [['white']*len(df.columns) for _ in range(len(df))]
        
        if metric_cols:
            for ci, col in enumerate(df.columns):
                if col in metric_cols:
                    try:
                        # Extract numeric values (before ± if present)
                        vals = pd.to_numeric(df[col].astype(str).str.split(' ±').str[0], errors='coerce')
                        # Find best value (min for ↓, max for ↑)
                        best = vals.idxmin() if '↓' in col else vals.idxmax()
                        if pd.notna(best):
                            colors[df.index.get_loc(best)][ci] = '#90EE90'  # Light green
                    except:
                        pass
        
        # Create table
        tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', 
                      loc='center', cellColours=colors)
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1.2, 1.5)
        
        # Style header row
        for i in range(len(df.columns)):
            tbl[(0, i)].set_facecolor('#4472C4')
            tbl[(0, i)].set_text_props(color='white', fontweight='bold')
        
        ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
    
    def _pval_3v5(self, df, metric, game):
        """Calculate p-value for paired t-test between 3P and 5P conditions."""
        mdf = df[(df['metric'] == metric) & (df['game'] == game)]
        c3 = mdf[mdf['condition'].str.contains('baseline|few|3', case=False, na=False)]
        c5 = mdf[mdf['condition'].str.contains('more|5', case=False, na=False)]
        
        # Get models present in both conditions
        models = set(c3['model']) & set(c5['model'])
        if len(models) < 3:
            return np.nan
        
        # Extract paired values
        v3 = [c3[c3['model'] == m]['mean'].values[0] for m in models if len(c3[c3['model'] == m]) > 0]
        v5 = [c5[c5['model'] == m]['mean'].values[0] for m in models if len(c5[c5['model'] == m]) > 0]
        
        try:
            return ttest_rel(v3, v5)[1] if len(v3) >= 3 else np.nan
        except:
            return np.nan
    
    def _build_3v5(self, df, metric, include_stars=True):
        """
        Build 3P vs 5P comparison table.
        
        Parameters:
            df: Data to analyze
            metric: Metric to display
            include_stars: If False, returns without stars and adds p_value columns
        
        Returns:
            DataFrame with columns: Model, {game}_3P, {game}_5P [, {game}_5P_p_value]
        """
        games = sorted(df['game'].unique())
        mdf = df[df['metric'] == metric]
        
        if mdf.empty:
            return pd.DataFrame()
        
        # Calculate p-values for each game
        pvals = {g: self._pval_3v5(df, metric, g) for g in games}
        
        rows = []
        for model in mdf['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            
            for g in games:
                gdf = mdf[(mdf['model'] == model) & (mdf['game'] == g)]
                
                # 3P condition
                c3 = gdf[gdf['condition'].str.contains('baseline|few|3', case=False, na=False)]
                if not c3.empty:
                    row[f'{g}_3P'] = self._fmt(c3['mean'].iloc[0], c3['std'].iloc[0] if 'std' in c3 else 0)
                else:
                    row[f'{g}_3P'] = 'N/A'
                
                # 5P condition
                c5 = gdf[gdf['condition'].str.contains('more|5', case=False, na=False)]
                if not c5.empty:
                    sig = self._sig(pvals[g]) if include_stars else ''
                    row[f'{g}_5P'] = self._fmt(c5['mean'].iloc[0], c5['std'].iloc[0] if 'std' in c5 else 0, sig)
                else:
                    row[f'{g}_5P'] = 'N/A'
                
                # Add p_value column for CSV
                if not include_stars:
                    row[f'{g}_5P_p_value'] = round(pvals[g], 4) if pd.notna(pvals[g]) else np.nan
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _remove_collinear(self, df, preds):
        """Remove collinear predictors using correlation threshold."""
        return remove_collinear_predictors(df, preds, threshold=COLLINEARITY_THRESHOLD)
    
    # ========== RQ1: COMPETITIVE PERFORMANCE ==========
    
    def performance_win_rate_table(self):
        """Generate win rate table (T_perf_win_rate)."""
        logger.info("T_perf_win_rate (RQ1)")
        
        # CSV: without stars, with p_value columns
        tbl_csv = self._build_3v5(self.perf_df, 'win_rate', include_stars=False)
        if not tbl_csv.empty:
            tbl_csv = tbl_csv.rename(columns={c: f"{c} ↑" if c != 'Model' and 'p_value' not in c else c 
                                             for c in tbl_csv.columns})
            tbl_csv.to_csv(self.output_dir / "T_perf_win_rate.csv", index=False)
        
        # PNG: with stars embedded
        tbl_png = self._build_3v5(self.perf_df, 'win_rate', include_stars=True)
        if not tbl_png.empty:
            tbl_png = tbl_png.rename(columns={c: f"{c} ↑" for c in tbl_png.columns if c != 'Model'})
            self._save_png(tbl_png, "T_perf_win_rate.png", 
                          "Win Rate | * p<.05, ** p<.01, *** p<.001",
                          [c for c in tbl_png.columns if c != 'Model'])
    
    def performance_avg_profit_table(self):
        """Generate average profit table (T_perf_avg_profit)."""
        logger.info("T_perf_avg_profit (RQ1)")
        
        # CSV: without stars, with p_value columns
        tbl_csv = self._build_3v5(self.perf_df, 'average_profit', include_stars=False)
        if not tbl_csv.empty:
            tbl_csv = tbl_csv.rename(columns={c: f"{c} ↑" if c != 'Model' and 'p_value' not in c else c 
                                             for c in tbl_csv.columns})
            tbl_csv.to_csv(self.output_dir / "T_perf_avg_profit.csv", index=False)
        
        # PNG: with stars embedded
        tbl_png = self._build_3v5(self.perf_df, 'average_profit', include_stars=True)
        if not tbl_png.empty:
            tbl_png = tbl_png.rename(columns={c: f"{c} ↑" for c in tbl_png.columns if c != 'Model'})
            self._save_png(tbl_png, "T_perf_avg_profit.png",
                          "Average Profit | * p<.05, ** p<.01, *** p<.001",
                          [c for c in tbl_png.columns if c != 'Model'])
    
    def performance_game_specific_table(self):
        """Generate game-specific performance metrics table (T_perf_game_specific)."""
        logger.info("T_perf_game_specific (RQ1)")
        
        # Calculate p-values for each game-metric pair
        pvals = {(g, GAME_CONFIGS[g]['game_specific_metric']): 
                 self._pval_3v5(self.perf_df, GAME_CONFIGS[g]['game_specific_metric'], g) 
                 for g in GAME_CONFIGS}
        
        # CSV: without stars, with p_value columns
        rows_csv = []
        for model in self.perf_df['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = self.perf_df[self.perf_df['model'] == model]
            
            for g, cfg in GAME_CONFIGS.items():
                m = cfg['game_specific_metric']
                d = METRIC_DIRECTION.get(m, '↑')
                gdf = mdf[(mdf['game'] == g) & (mdf['metric'] == m)]
                
                for cond, suffix in [('baseline|few|3', '3P'), ('more|5', '5P')]:
                    c = gdf[gdf['condition'].str.contains(cond, case=False, na=False)]
                    if not c.empty:
                        row[f'{g}_{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], 
                                                                  c['std'].iloc[0] if 'std' in c else 0)
                    else:
                        row[f'{g}_{m}_{suffix} {d}'] = 'N/A'
                
                # Add p_value column for 5P
                row[f'{g}_{m}_5P_p_value'] = round(pvals[(g, m)], 4) if pd.notna(pvals[(g, m)]) else np.nan
            
            rows_csv.append(row)
        
        tbl_csv = pd.DataFrame(rows_csv)
        tbl_csv.to_csv(self.output_dir / "T_perf_game_specific.csv", index=False)
        
        # PNG: with stars embedded
        rows_png = []
        for model in self.perf_df['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = self.perf_df[self.perf_df['model'] == model]
            
            for g, cfg in GAME_CONFIGS.items():
                m = cfg['game_specific_metric']
                d = METRIC_DIRECTION.get(m, '↑')
                gdf = mdf[(mdf['game'] == g) & (mdf['metric'] == m)]
                
                for cond, suffix, sig in [('baseline|few|3', '3P', ''), 
                                          ('more|5', '5P', self._sig(pvals[(g, m)]))]:
                    c = gdf[gdf['condition'].str.contains(cond, case=False, na=False)]
                    if not c.empty:
                        row[f'{g}_{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], 
                                                                  c['std'].iloc[0] if 'std' in c else 0,
                                                                  sig if '5P' in suffix else '')
                    else:
                        row[f'{g}_{m}_{suffix} {d}'] = 'N/A'
            
            rows_png.append(row)
        
        tbl_png = pd.DataFrame(rows_png)
        self._save_png(tbl_png, "T_perf_game_specific.png",
                      "Game-Specific Metrics | * p<.05, ** p<.01, *** p<.001",
                      [c for c in tbl_png.columns if c != 'Model'])
    
    def mlr_features_to_performance(self):
        """Generate features → performance regression table (T_mlr_features_to_performance, RQ1)."""
        if not STATSMODELS_AVAILABLE:
            logger.warning("Skipping mlr_features_to_performance (statsmodels not available)")
            return
        
        logger.info("T_mlr_features_to_performance (RQ1)")
        
        # Prepare data
        perf = self.perf_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], 
            columns='metric', 
            values='mean'
        ).reset_index()
        
        merged = perf.merge(self.features_df, on='model', how='left')
        
        results = []
        
        for g in merged['game'].unique():
            gdf = merged[merged['game'] == g].copy()
            
            # Remove collinear predictors
            preds = self._remove_collinear(gdf, MODEL_FEATURES)
            if not preds:
                continue
            
            # Run regression for each performance metric
            for target in [c for c in perf.columns if c not in ['game', 'model', 'condition']]:
                if target not in gdf or gdf[target].std() == 0:
                    continue
                
                valid = gdf[preds + [target]].dropna()
                
                if len(valid) < len(preds) + 2:
                    continue
                
                try:
                    # Calculate VIF for multicollinearity detection
                    vif_dict = {}
                    try:
                        X_valid = valid[preds]
                        for i, col in enumerate(X_valid.columns):
                            vif_value = variance_inflation_factor(X_valid.values, i)
                            vif_dict[col] = round(vif_value, 2)
                    except Exception as vif_error:
                        logger.debug(f"VIF calculation failed for {g}/{target}: {vif_error}")
                        vif_dict = {col: np.nan for col in preds}
                    
                    # Fit OLS model
                    model = sm.OLS(valid[target], sm.add_constant(valid[preds])).fit()
                    
                    for p in preds:
                        results.append({
                            'game': g,
                            'type': 'features_to_performance',
                            'target': target,
                            'predictor': p,
                            'r_squared': round(model.rsquared, 4),
                            'coef': round(model.params.get(p, np.nan), 4),
                            'p_value': round(model.pvalues.get(p, np.nan), 4),
                            'vif': vif_dict.get(p, np.nan),
                            'n_obs': int(model.nobs),
                            'r_squared_adj': round(model.rsquared_adj, 4)
                        })
                except Exception as e:
                    logger.debug(f"Failed regression {g}/{target}: {e}")
                    pass
        
        if results:
            df = pd.DataFrame(results)
            df = df[['game', 'type', 'target', 'predictor', 'r_squared', 'coef', 'p_value', 
                    'vif', 'n_obs', 'r_squared_adj']]
            df.to_csv(self.output_dir / "T_mlr_features_to_performance.csv", index=False)
            
            # PNG: show coef with significance stars
            df_png = df.copy()
            df_png['coef'] = df_png.apply(
                lambda r: f"{r['coef']:.4f} {self._sig(r['p_value'])}".strip(), axis=1
            )
            df_png_display = df_png[['game', 'target', 'predictor', 'coef', 'r_squared']]
            self._save_png(df_png_display, "T_mlr_features_to_performance.png",
                          "Features → Performance (RQ1) | * p<.05, ** p<.01, *** p<.001")
    
    # ========== RQ2: STRATEGIC BEHAVIORAL PROFILES ==========
    
    def magic_per_game_table(self, game):
        """Generate MAgIC behavioral metrics table for a specific game (T_magic_{game}, RQ2)."""
        logger.info(f"T_magic_{game} (RQ2)")
        
        gdf = self.magic_df[self.magic_df['game'] == game]
        if gdf.empty:
            return
        
        metrics = GAME_CONFIGS[game]['magic_metrics']
        pvals = {m: self._pval_3v5(self.magic_df, m, game) for m in metrics}
        
        # CSV: without stars, with p_value columns
        rows_csv = []
        for model in gdf['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = gdf[gdf['model'] == model]
            
            for m in metrics:
                d = METRIC_DIRECTION.get(m, '↑')
                mmdf = mdf[mdf['metric'] == m]
                
                for cond, suffix in [('baseline|few|3', '3P'), ('more|5', '5P')]:
                    c = mmdf[mmdf['condition'].str.contains(cond, case=False, na=False)]
                    if not c.empty:
                        row[f'{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], 
                                                              c['std'].iloc[0] if 'std' in c else 0)
                    else:
                        row[f'{m}_{suffix} {d}'] = 'N/A'
                
                # Add p_value column for 5P
                row[f'{m}_5P_p_value'] = round(pvals[m], 4) if pd.notna(pvals[m]) else np.nan
            
            rows_csv.append(row)
        
        tbl_csv = pd.DataFrame(rows_csv)
        tbl_csv.to_csv(self.output_dir / f"T_magic_{game}.csv", index=False)
        
        # PNG: with stars embedded
        rows_png = []
        for model in gdf['model'].unique():
            row = {'Model': self.loader.get_display_name(model)}
            mdf = gdf[gdf['model'] == model]
            
            for m in metrics:
                d = METRIC_DIRECTION.get(m, '↑')
                mmdf = mdf[mdf['metric'] == m]
                
                for cond, suffix, sig in [('baseline|few|3', '3P', ''), 
                                          ('more|5', '5P', self._sig(pvals[m]))]:
                    c = mmdf[mmdf['condition'].str.contains(cond, case=False, na=False)]
                    if not c.empty:
                        row[f'{m}_{suffix} {d}'] = self._fmt(c['mean'].iloc[0], 
                                                              c['std'].iloc[0] if 'std' in c else 0,
                                                              sig if '5P' in suffix else '')
                    else:
                        row[f'{m}_{suffix} {d}'] = 'N/A'
            
            rows_png.append(row)
        
        tbl_png = pd.DataFrame(rows_png)
        self._save_png(tbl_png, f"T_magic_{game}.png",
                      f"MAgIC: {game.replace('_', ' ').title()} (RQ2) | * p<.05, ** p<.01, *** p<.001",
                      [c for c in tbl_png.columns if c != 'Model'])
    
    def pca_variance_table(self):
        """Generate PCA variance explained table (T6_pca_variance, RQ2)."""
        if not SKLEARN_AVAILABLE:
            logger.warning("Skipping pca_variance_table (sklearn not available)")
            return
        
        logger.info("T6_pca_variance (RQ2)")
        
        results = []
        
        for g, cfg in GAME_CONFIGS.items():
            gdf = self.magic_df_no_random[self.magic_df_no_random['game'] == g]
            if gdf.empty:
                continue
            
            pivot = gdf.pivot_table(index='model', columns='metric', values='mean').dropna()
            avail = [m for m in cfg['magic_metrics'] if m in pivot.columns]
            
            if len(avail) < 2:
                continue
            
            # Fit PCA
            pca = PCA().fit(StandardScaler().fit_transform(pivot[avail]))
            cum = np.cumsum(pca.explained_variance_ratio_)
            
            for i, v in enumerate(pca.explained_variance_ratio_):
                results.append({
                    'game': g,
                    'component': f'PC{i+1}',
                    'variance': round(v, 4),
                    'cumulative': round(cum[i], 4)
                })
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(self.output_dir / "T6_pca_variance.csv", index=False)
            self._save_png(df, "T6_pca_variance.png", "PCA Variance (RQ2)")
    
    # ========== RQ3: CAPABILITY-PERFORMANCE LINKS ==========
    
    def mlr_magic_to_performance(self):
        """Generate MAgIC → performance regression table (T5_magic_to_perf, RQ3)."""
        if not STATSMODELS_AVAILABLE:
            logger.warning("Skipping mlr_magic_to_performance (statsmodels not available)")
            return
        
        logger.info("T5_magic_to_perf (RQ3)")
        
        # Prepare data
        magic = self.magic_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], 
            columns='metric', 
            values='mean'
        ).reset_index()
        
        perf = self.perf_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], 
            columns='metric', 
            values='mean'
        ).reset_index()
        
        merged = perf.merge(magic, on=['game', 'model', 'condition'], how='inner')
        
        if merged.empty:
            return
        
        perf_cols = [c for c in perf.columns if c not in ['game', 'model', 'condition']]
        magic_cols = [c for c in magic.columns if c not in ['game', 'model', 'condition']]
        
        results = []
        
        for g in merged['game'].unique():
            gdf = merged[merged['game'] == g].copy()
            
            # Remove collinear MAgIC predictors
            avail = self._remove_collinear(gdf, [m for m in magic_cols if m in gdf and gdf[m].std() > 0])
            
            if not avail:
                continue
            
            # Run regression for each performance metric
            for t in perf_cols:
                if t not in gdf or gdf[t].std() == 0:
                    continue
                
                valid = gdf[[t] + avail].dropna()
                
                if len(valid) < len(avail) + 2:
                    continue
                
                try:
                    model = sm.OLS(valid[t], sm.add_constant(valid[avail])).fit()
                    
                    for p in avail:
                        if p in model.params.index:
                            results.append({
                                'game': g,
                                'type': 'magic_to_performance',
                                'target': t,
                                'predictor': p,
                                'r_squared': round(model.rsquared, 4),
                                'coef': round(model.params[p], 4),
                                'p_value': round(model.pvalues[p], 4),
                                'n_obs': int(model.nobs),
                                'r_squared_adj': round(model.rsquared_adj, 4)
                            })
                except Exception as e:
                    logger.debug(f"Failed regression {g}/{t}: {e}")
                    pass
        
        if results:
            df = pd.DataFrame(results)
            df = df[['game', 'type', 'target', 'predictor', 'r_squared', 'coef', 'p_value', 
                    'n_obs', 'r_squared_adj']]
            df.to_csv(self.output_dir / "T5_magic_to_perf.csv", index=False)
            
            # PNG: show coef with significance stars
            df_png = df.copy()
            df_png['coef'] = df_png.apply(
                lambda r: f"{r['coef']:.4f} {self._sig(r['p_value'])}".strip(), axis=1
            )
            df_png_display = df_png[['game', 'target', 'predictor', 'coef', 'r_squared']]
            self._save_png(df_png_display, "T5_magic_to_perf.png",
                          "MAgIC → Performance (RQ3) | * p<.05, ** p<.01, *** p<.001")
    
    def mlr_combined_to_performance(self):
        """
        Generate combined regression table: MAgIC + Features → Performance (T7_combined_to_perf, RQ3).
        
        Tests whether combining behavioral capabilities (MAgIC) and architectural features
        improves prediction of economic performance within each game.
        """
        if not STATSMODELS_AVAILABLE:
            logger.warning("Skipping mlr_combined_to_performance (statsmodels not available)")
            return
        
        logger.info("T7_combined_to_perf (Combined MAgIC + Features → Performance)")
        
        # Prepare data
        magic = self.magic_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], 
            columns='metric', 
            values='mean'
        ).reset_index()
        
        perf = self.perf_df_no_random.pivot_table(
            index=['game', 'model', 'condition'], 
            columns='metric', 
            values='mean'
        ).reset_index()
        
        # Merge all three dataframes
        merged = perf.merge(magic, on=['game', 'model', 'condition'], how='inner')
        merged = merged.merge(self.features_df, on='model', how='left')
        
        if merged.empty:
            logger.warning("No data for combined regression")
            return
        
        perf_cols = [c for c in perf.columns if c not in ['game', 'model', 'condition']]
        magic_cols = [c for c in magic.columns if c not in ['game', 'model', 'condition']]
        
        results = []
        
        for g in merged['game'].unique():
            gdf = merged[merged['game'] == g].copy()
            
            # Get available predictors
            magic_avail = self._remove_collinear(gdf, [m for m in magic_cols if m in gdf and gdf[m].std() > 0])
            features_avail = self._remove_collinear(gdf, [f for f in MODEL_FEATURES if f in gdf and gdf[f].std() > 0])
            
            # Combine all predictors
            all_predictors = magic_avail + features_avail
            
            # Remove collinearity between MAgIC and Features
            all_predictors = self._remove_collinear(gdf, all_predictors)
            
            if not all_predictors:
                logger.warning(f"No valid predictors for {g} (combined)")
                continue
            
            logger.info(f"  {g}: {len(magic_avail)} MAgIC + {len(features_avail)} features = {len(all_predictors)} total")
            
            # Run regression for each performance metric
            for target in perf_cols:
                if target not in gdf or gdf[target].std() == 0:
                    continue
                
                valid = gdf[[target] + all_predictors].dropna()
                
                if len(valid) < len(all_predictors) + 2:
                    logger.debug(f"  Skipping {g}/{target}: insufficient observations")
                    continue
                
                try:
                    # Calculate VIF
                    vif_dict = {}
                    try:
                        X_valid = valid[all_predictors]
                        for i, col in enumerate(X_valid.columns):
                            vif_value = variance_inflation_factor(X_valid.values, i)
                            vif_dict[col] = round(vif_value, 2)
                    except Exception as vif_error:
                        logger.debug(f"VIF calculation failed for {g}/{target}: {vif_error}")
                        vif_dict = {col: np.nan for col in all_predictors}
                    
                    # Fit model
                    model = sm.OLS(valid[target], sm.add_constant(valid[all_predictors])).fit()
                    
                    # Store results
                    for pred in all_predictors:
                        if pred in model.params.index:
                            pred_type = 'magic' if pred in magic_cols else 'feature'
                            
                            results.append({
                                'game': g,
                                'type': 'combined_to_performance',
                                'target': target,
                                'predictor': pred,
                                'predictor_type': pred_type,
                                'r_squared': round(model.rsquared, 4),
                                'coef': round(model.params[pred], 4),
                                'p_value': round(model.pvalues[pred], 4),
                                'vif': vif_dict.get(pred, np.nan),
                                'n_obs': int(model.nobs),
                                'r_squared_adj': round(model.rsquared_adj, 4),
                                'n_predictors': len(all_predictors)
                            })
                
                except Exception as e:
                    logger.debug(f"  Failed regression {g}/{target}: {e}")
                    pass
        
        if results:
            df = pd.DataFrame(results)
            df = df[['game', 'type', 'target', 'predictor', 'predictor_type', 'r_squared', 
                    'coef', 'p_value', 'vif', 'n_obs', 'n_predictors', 'r_squared_adj']]
            df.to_csv(self.output_dir / "T7_combined_to_perf.csv", index=False)
            
            logger.info(f"  Saved {len(df)} predictor results across {df['game'].nunique()} games")
            
            # Summary
            summary = df.groupby('game').agg({
                'r_squared': 'first',
                'n_obs': 'first',
                'n_predictors': 'first'
            }).reset_index()
            logger.info(f"\n  Average R² by game:\n{summary.to_string(index=False)}")
            
            # PNG: show coef with significance stars
            df_png = df.copy()
            df_png['coef'] = df_png.apply(
                lambda r: f"{r['coef']:.4f} {self._sig(r['p_value'])}".strip(), axis=1
            )
            df_png['predictor_display'] = df_png['predictor'] + ' (' + df_png['predictor_type'] + ')'
            df_png_display = df_png[['game', 'target', 'predictor_display', 'coef', 'r_squared']]
            df_png_display = df_png_display.rename(columns={'predictor_display': 'predictor'})
            self._save_png(df_png_display, "T7_combined_to_perf.png",
                          "Combined (MAgIC + Features) → Performance | * p<.05, ** p<.01, *** p<.001")
        else:
            logger.warning("No results for combined regression")
    
    # ========== SUPPLEMENTARY ==========
    
    def reasoning_chars_table(self, token_df):
        """Generate reasoning characters table for thinking models (T_reasoning_chars)."""
        logger.info("T_reasoning_chars (Supplementary)")
        
        thinking = [m for m in token_df['model'].unique() if self.loader.get_thinking_status(m)]
        if not thinking:
            return
        
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
                        sig = ''  # Add p-value calculation if needed
                        col_name = f"{game}_{condition}"
                        row[col_name] = f"{mean_val:.0f} ± {std_val:.0f} {sig}".strip()
                    else:
                        row[f"{game}_{condition}"] = "N/A"
            
            rows.append(row)
        
        tbl = pd.DataFrame(rows)
        tbl.to_csv(self.output_dir / "T_reasoning_chars.csv", index=False)
        self._save_png(tbl, "T_reasoning_chars.png",
                      "Reasoning Characters (Thinking Models) | * p<.05, ** p<.01, *** p<.001")
