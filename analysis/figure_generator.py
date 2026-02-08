#!/usr/bin/env python3
"""
Figure Generator - Publication Figures for MAgIC Analysis

Generates PNG figures for research questions:
- RQ2: Strategic Behavioral Profiles
  - F_similarity_{game}: Cosine similarity heatmaps per game
  - F_similarity_3v5: 3P vs 5P stability comparison
  - F_pca_scree: PCA variance explained plots
- Supplementary: F_reasoning_chars

Dependencies:
- matplotlib: For plotting
- seaborn: For heatmaps
- sklearn: For cosine similarity and PCA
- scipy: For statistical tests
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
import warnings

logger = logging.getLogger("FigureGenerator")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False
    logger.warning("matplotlib/seaborn not available - figure generation disabled")

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available - similarity and PCA figures disabled")

try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available - p-value calculations disabled")

from .utils import sig_stars, GAME_CONFIGS


class FigureGenerator:
    """
    Generates publication-ready figures (PNG) for MAgIC analysis.
    
    All figures include significance stars where applicable:
    - * p<.05
    - ** p<.01
    - *** p<.001
    
    Parameters:
        perf_with: Performance data including random agent (not used for figures)
        magic_with: MAgIC data including random agent
        perf_no: Performance data excluding random agent (not used for figures)
        magic_no: MAgIC data excluding random agent
        features: Model architectural features (not used for figures)
        output_dir: Directory for output files
        loader: DataLoader instance for display names and thinking status
    """
    
    def __init__(self, perf_with, magic_with, perf_no, magic_no, features, output_dir, loader):
        self.magic_df = magic_with
        self.magic_df_no_random = magic_no
        self.output_dir = Path(output_dir)
        self.loader = loader
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def _sig(p):
        """Convert p-value to significance stars."""
        return sig_stars(p)
    
    def generate_all(self, token_df=None):
        """Generate all publication figures."""
        if not PLOT_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.warning("Skipping figure generation (dependencies not available)")
            return
        
        logger.info("Generating all figures...")
        
        # Set plot style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # RQ2: Strategic Behavioral Profiles
        for g in GAME_CONFIGS:
            self.similarity_matrix(g)
        self.similarity_3v5()
        self.pca_scree()
        
        # Supplementary
        if token_df is not None and not token_df.empty:
            self.reasoning_chars_fig(token_df)
        
        logger.info(f"All figures generated in {self.output_dir}")
    
    # ========== RQ2: STRATEGIC BEHAVIORAL PROFILES ==========
    
    def similarity_matrix(self, game):
        """
        Generate behavioral similarity heatmap for a specific game (F_similarity_{game}, RQ2).
        
        Uses cosine similarity between behavioral profiles (MAgIC metrics).
        Significance is based on Pearson correlation p-values.
        """
        logger.info(f"F_similarity_{game} (RQ2)")
        
        gdf = self.magic_df[self.magic_df['game'] == game]
        if gdf.empty:
            return
        
        # Pivot to model × metric matrix
        pivot = gdf.pivot_table(index='model', columns='metric', values='mean').fillna(0)
        
        if len(pivot) < 2:
            return
        
        # Calculate cosine similarity
        sim = cosine_similarity(pivot.values)
        names = [self.loader.get_display_name(m) for m in pivot.index]
        
        # Calculate p-values using Pearson correlation as proxy
        p_matrix = np.ones_like(sim)
        
        if SCIPY_AVAILABLE:
            for i in range(len(pivot)):
                for j in range(i+1, len(pivot)):
                    try:
                        vec_i = pivot.iloc[i].values
                        vec_j = pivot.iloc[j].values
                        
                        # Check if either vector is constant
                        if np.std(vec_i) == 0 or np.std(vec_j) == 0:
                            p_matrix[i, j] = 1.0
                            p_matrix[j, i] = 1.0
                        else:
                            # Use Pearson correlation p-value
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
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(sim, index=names, columns=names), 
                   annot=annot, 
                   fmt='',
                   cmap='RdBu_r', 
                   center=0.5, 
                   vmin=0, 
                   vmax=1, 
                   square=True, 
                   ax=ax)
        
        ax.set_title(f'Behavioral Similarity: {game.replace("_", " ").title()} (RQ2) | * p<.05, ** p<.01, *** p<.001')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"F_similarity_{game}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def similarity_3v5(self):
        """
        Generate 3P vs 5P stability comparison figure (F_similarity_3v5, RQ2).
        
        Shows cosine similarity and Pearson correlation between behavioral profiles
        in 3-player and 5-player conditions for each game.
        """
        logger.info("F_similarity_3v5 (RQ2)")
        
        results = []
        
        for g in GAME_CONFIGS:
            gdf = self.magic_df[self.magic_df['game'] == g]
            if gdf.empty:
                continue
            
            # Separate conditions
            c3 = gdf[gdf['condition'].str.contains('baseline|few|3', case=False, na=False)]
            c5 = gdf[gdf['condition'].str.contains('more|5', case=False, na=False)]
            
            if c3.empty or c5.empty:
                continue
            
            # Pivot to model × metric matrices
            p3 = c3.pivot_table(index='model', columns='metric', values='mean').fillna(0)
            p5 = c5.pivot_table(index='model', columns='metric', values='mean').fillna(0)
            
            # Get common models and metrics
            models = list(set(p3.index) & set(p5.index))
            metrics = list(set(p3.columns) & set(p5.columns))
            
            if len(models) < 2:
                continue
            
            # Flatten vectors
            v3 = p3.loc[models, metrics].values.flatten()
            v5 = p5.loc[models, metrics].values.flatten()
            
            # Calculate cosine similarity
            cos = cosine_similarity([v3], [v5])[0, 0]
            
            # Calculate Pearson correlation and p-value
            if SCIPY_AVAILABLE:
                r, p = pearsonr(v3, v5)
            else:
                r, p = 0, 1
            
            results.append({
                'game': g,
                'cosine': round(cos, 4),
                'cosine_p_value': round(p, 4),
                'pearson': round(r, 4),
                'pearson_p_value': round(p, 4)
            })
        
        if not results:
            return
        
        # Save CSV table
        pd.DataFrame(results).to_csv(self.output_dir / "T_similarity_3v5.csv", index=False)
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(results))
        cos_vals = [r['cosine'] for r in results]
        cos_sigs = [self._sig(r['cosine_p_value']) for r in results]
        pearson_vals = [r['pearson'] for r in results]
        pearson_sigs = [self._sig(r['pearson_p_value']) for r in results]
        
        # Plot bars
        ax.bar(x - 0.175, cos_vals, 0.35, label='Cosine', color='steelblue')
        ax.bar(x + 0.175, pearson_vals, 0.35, label='Pearson', color='coral')
        
        # Add value annotations with significance stars
        for i, (cv, cs, pv, ps) in enumerate(zip(cos_vals, cos_sigs, pearson_vals, pearson_sigs)):
            ax.annotate(f'{cv:.3f}{cs}', 
                       xy=(x[i] - 0.175, cv), 
                       xytext=(0, 3), 
                       textcoords="offset points", 
                       ha='center', 
                       fontsize=9)
            ax.annotate(f'{pv:.3f}{ps}', 
                       xy=(x[i] + 0.175, pv), 
                       xytext=(0, 3), 
                       textcoords="offset points", 
                       ha='center', 
                       fontsize=9)
        
        # Format plot
        ax.set_xticks(x)
        ax.set_xticklabels([r['game'].replace('_', ' ').title() for r in results])
        ax.set_ylabel('Similarity')
        ax.set_title('3P vs 5P Stability (RQ2) | * p<.05, ** p<.01, *** p<.001')
        ax.legend()
        ax.set_ylim(0, 1.15)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "F_similarity_3v5.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def pca_scree(self):
        """
        Generate PCA scree plots for all games (F_pca_scree, RQ2).
        
        Shows individual and cumulative variance explained by principal components
        for behavioral metrics in each game.
        """
        logger.info("F_pca_scree (RQ2)")
        
        # Get games with enough metrics for PCA
        games = [g for g, c in GAME_CONFIGS.items() if len(c['magic_metrics']) >= 2]
        
        if not games:
            return
        
        # Create subplots
        fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4))
        
        # Handle single game case
        if len(games) == 1:
            axes = [axes]
        
        for ax, g in zip(axes, games):
            gdf = self.magic_df_no_random[self.magic_df_no_random['game'] == g]
            
            # Pivot to model × metric matrix
            pivot = gdf.pivot_table(index='model', columns='metric', values='mean').dropna()
            
            # Get available metrics for this game
            avail = [m for m in GAME_CONFIGS[g]['magic_metrics'] if m in pivot.columns]
            
            if len(avail) < 2:
                continue
            
            # Fit PCA
            var = PCA().fit(StandardScaler().fit_transform(pivot[avail])).explained_variance_ratio_
            
            # Plot individual and cumulative variance
            x = range(1, len(var) + 1)
            ax.bar(x, var, alpha=0.7, color='steelblue', label='Individual')
            ax.plot(x, np.cumsum(var), 'ro-', label='Cumulative')
            ax.axhline(0.8, color='green', ls='--', alpha=0.5)
            
            # Format subplot
            ax.set_title(g.replace('_', ' ').title())
            ax.set_xlabel('PC')
            ax.set_ylabel('Variance')
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
        
        # Format overall figure
        plt.suptitle('PCA Variance (RQ2)', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(self.output_dir / "F_pca_scree.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # ========== SUPPLEMENTARY ==========
    
    def reasoning_chars_fig(self, token_df):
        """
        Generate reasoning characters figure for thinking models (F_reasoning_chars).
        
        Shows average reasoning effort (character count) per game and condition
        for models with extended thinking capabilities.
        """
        logger.info("F_reasoning_chars (Supplementary)")
        
        # Filter to thinking models only
        thinking = [m for m in token_df['model'].unique() if self.loader.get_thinking_status(m)]
        
        if not thinking:
            return
        
        df = token_df[token_df['model'].isin(thinking)]
        
        # Aggregate by game and condition
        agg = df.groupby(['game', 'condition']).agg({
            'avg_reasoning_chars': ['mean', 'std']
        }).reset_index()
        agg.columns = ['game', 'condition', 'mean', 'std']
        
        games = sorted(agg['game'].unique())
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(games))
        
        for i, cond in enumerate(['baseline', 'more_players']):
            data = agg[agg['condition'] == cond]
            
            means = [data[data['game'] == g]['mean'].values[0] if len(data[data['game'] == g]) > 0 else 0 
                    for g in games]
            stds = [data[data['game'] == g]['std'].values[0] if len(data[data['game'] == g]) > 0 else 0 
                   for g in games]
            
            ax.bar(x + (i - 0.5) * 0.35, 
                  means, 
                  0.35, 
                  label='3P' if i == 0 else '5P',
                  yerr=stds, 
                  capsize=3, 
                  color='steelblue' if i == 0 else 'coral', 
                  alpha=0.8)
        
        # Format plot
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace('_', ' ').title() for g in games], rotation=15)
        ax.set_xlabel('Game')
        ax.set_ylabel('Avg Reasoning Chars')
        ax.set_title('Reasoning Effort (Thinking Models)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "F_reasoning_chars.png", dpi=300, bbox_inches='tight')
        plt.close()
