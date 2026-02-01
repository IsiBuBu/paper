# pca_analysis.py

"""
Principal Component Analysis for MAgIC Framework

Run from project root:
    python pca_analysis.py
    python pca_analysis.py --no-plot

Three PCA analyses (per game):
1. MAgIC Metrics PCA - Validate if behavioral dimensions are distinct constructs
2. Model Features PCA - Find latent model types/clusters  
3. Performance Metrics PCA - Create composite success scores

Outputs (saved to output/analysis/):
- pca_variance.csv: Explained variance per component
- pca_loadings.csv: Feature loadings on each PC
- pca_scores.csv: Model scores on principal components
- plots/: scree plots, loading heatmaps, biplots
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import json
import re
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Extraction
# =============================================================================

def load_model_configs(config_path: Optional[Path] = None) -> Dict[str, Dict]:
    """Load model_configs from config.json."""
    if config_path is None:
        config_path = Path("config/config.json")
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return json.load(f).get('model_configs', {})
    except Exception as e:
        logger.warning(f"Could not load config: {e}")
        return {}


def extract_model_features(model_name: str, model_configs: Dict) -> Dict[str, Any]:
    """Extract architecture features from model name."""
    model_lower = model_name.lower()
    
    # Architecture (MoE detection)
    is_moe = bool(re.search(r'-a\d+b', model_lower)) or 'moe' in model_lower
    if 'maverick' in model_lower or 'scout' in model_lower:
        is_moe = True
    
    # Size (in billions)
    size = 0.0
    match = re.search(r'(?<!a)(\d+\.?\d*)b(?!-)', model_lower)
    if match:
        size = float(match.group(1))
    
    # Family
    family = 'unknown'
    for fam in ['qwen', 'llama', 'gemma', 'mistral', 'claude']:
        if fam in model_lower:
            family = fam
            break
    
    # Version
    version = 0.0
    for pattern in [r'qwen(\d+)', r'llama-?(\d+\.?\d*)', r'gemma-?(\d+)']:
        match = re.search(pattern, model_lower)
        if match:
            version = float(match.group(1))
            break
    
    # Thinking (from config)
    thinking = 0
    if model_name in model_configs:
        ro = model_configs[model_name].get('reasoning_output', 'none')
        if ro in ['reasoning_tokens', 'output_tokens']:
            thinking = 1
    
    return {
        'model': model_name,
        'architecture_moe': 1 if is_moe else 0,
        'size_params': size,
        'family': family,
        'version': version,
        'thinking': thinking
    }


# =============================================================================
# PCA Analyzer
# =============================================================================

class PCAAnalyzer:
    """PCA analysis for MAgIC metrics, model features, and performance metrics."""
    
    def __init__(self, analysis_dir: Path, config_path: Optional[Path] = None):
        self.analysis_dir = Path(analysis_dir)
        self.config_path = config_path
        self.magic_df = None
        self.perf_df = None
        self.model_configs = {}
        self.results = {}
    
    def load_data(self):
        """Load metrics CSVs."""
        magic_path = self.analysis_dir / "magic_behavioral_metrics.csv"
        perf_path = self.analysis_dir / "performance_metrics.csv"
        
        if magic_path.exists():
            self.magic_df = pd.read_csv(magic_path)
            # Filter out random_agent and defender model
            self.magic_df = self._filter_models(self.magic_df)
            logger.info(f"Loaded MAgIC metrics: {len(self.magic_df)} rows")
        
        if perf_path.exists():
            self.perf_df = pd.read_csv(perf_path)
            # Filter out random_agent and defender model
            self.perf_df = self._filter_models(self.perf_df)
            logger.info(f"Loaded performance metrics: {len(self.perf_df)} rows")
        
        self.model_configs = load_model_configs(self.config_path)
    
    def _is_excluded_model(self, model_name: str) -> bool:
        """Check if model should be excluded (random_agent or defender)."""
        model_lower = model_name.lower()
        # Exclude random agent
        if 'random' in model_lower:
            return True
        # Exclude defender model (gemma is typically the defender)
        if 'gemma' in model_lower:
            return True
        return False
    
    def _filter_models(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter out random_agent and defender model from dataframe."""
        if df is None or df.empty:
            return df
        mask = ~df['model'].apply(self._is_excluded_model)
        excluded = df[~mask]['model'].unique()
        if len(excluded) > 0:
            logger.info(f"Excluding models: {list(excluded)}")
        return df[mask].copy()
    
    def _run_pca(self, data: np.ndarray, features: List[str], labels: List[str]) -> Optional[Dict]:
        """Run PCA and return results."""
        if data.shape[0] < 3 or data.shape[1] < 2:
            return None
        
        # Handle NaN
        if np.any(np.isnan(data)):
            col_means = np.nanmean(data, axis=0)
            for i in range(data.shape[1]):
                data[np.isnan(data[:, i]), i] = col_means[i]
        
        # Standardize
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # PCA
        n_comp = min(data.shape)
        pca = PCA(n_components=n_comp)
        scores = pca.fit_transform(data_scaled)
        
        # Loadings (correlation between features and PCs)
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        loadings_df = pd.DataFrame(
            loadings, 
            index=features,
            columns=[f'PC{i+1}' for i in range(n_comp)]
        )
        
        return {
            'explained_variance': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'loadings': loadings_df,
            'scores': scores,
            'labels': labels,
            'features': features,
            'n_components': n_comp
        }
    
    def pca_magic_metrics(self, game: Optional[str] = None, condition: Optional[str] = None) -> Optional[Dict]:
        """
        PCA on MAgIC behavioral metrics.
        
        Purpose: Check if 7 dimensions are distinct or have collinearity.
        """
        if self.magic_df is None:
            return None
        
        df = self.magic_df.copy()
        if game:
            df = df[df['game'] == game]
        if condition:
            df = df[df['condition'] == condition]
        
        if df.empty:
            return None
        
        # Pivot: models × metrics
        pivot = df.pivot_table(index='model', columns='metric', values='mean')
        pivot = pivot.dropna(axis=1, how='all').dropna(axis=0, how='any')
        
        if pivot.shape[0] < 3 or pivot.shape[1] < 2:
            return None
        
        result = self._run_pca(pivot.values, pivot.columns.tolist(), pivot.index.tolist())
        if result:
            result['type'] = 'magic_metrics'
            result['game'] = game
            result['condition'] = condition
        return result
    
    def pca_model_features(self) -> Optional[Dict]:
        """
        PCA on model architecture features.
        
        Purpose: Find model clusters/types.
        """
        models = set()
        if self.magic_df is not None:
            models.update(self.magic_df['model'].unique())
        if self.perf_df is not None:
            models.update(self.perf_df['model'].unique())
        
        # Filter out excluded models
        models = {m for m in models if not self._is_excluded_model(m)}
        
        if len(models) < 3:
            return None
        
        # Extract features
        features = [extract_model_features(m, self.model_configs) for m in models]
        df = pd.DataFrame(features).set_index('model')
        
        # Encode family
        le = LabelEncoder()
        df['family_encoded'] = le.fit_transform(df['family'])
        
        # Numeric columns
        cols = ['architecture_moe', 'size_params', 'version', 'thinking', 'family_encoded']
        df_num = df[cols].dropna()
        
        # Remove zero-variance columns
        cols_valid = [c for c in cols if df_num[c].std() > 0]
        if len(cols_valid) < 2:
            return None
        
        result = self._run_pca(df_num[cols_valid].values, cols_valid, df_num.index.tolist())
        if result:
            result['type'] = 'model_features'
            result['family_mapping'] = dict(zip(le.classes_, le.transform(le.classes_)))
        return result
    
    def pca_performance_metrics(self, game: str, condition: Optional[str] = None) -> Optional[Dict]:
        """
        PCA on performance metrics.
        
        Purpose: Create composite success score.
        """
        if self.perf_df is None:
            return None
        
        df = self.perf_df[self.perf_df['game'] == game].copy()
        if condition:
            df = df[df['condition'] == condition]
        
        if df.empty:
            return None
        
        pivot = df.pivot_table(index='model', columns='metric', values='mean')
        pivot = pivot.dropna(axis=1, how='all').dropna(axis=0, how='any')
        
        if pivot.shape[0] < 3 or pivot.shape[1] < 2:
            return None
        
        result = self._run_pca(pivot.values, pivot.columns.tolist(), pivot.index.tolist())
        if result:
            result['type'] = 'performance'
            result['game'] = game
            result['condition'] = condition
        return result
    
    def run_all(self) -> Dict[str, Dict]:
        """Run all PCA analyses."""
        self.load_data()
        results = {}
        
        # 1. MAgIC metrics - PER GAME ONLY (different games have different metrics)
        logger.info("\n" + "="*60)
        logger.info("PCA 1: MAgIC Behavioral Metrics (Per Game)")
        logger.info("="*60)
        
        if self.magic_df is not None:
            for game in self.magic_df['game'].unique():
                # Get metrics available for this game
                game_metrics = self.magic_df[self.magic_df['game'] == game]['metric'].unique()
                logger.info(f"\n{game}: metrics = {list(game_metrics)}")
                
                r = self.pca_magic_metrics(game=game)
                if r:
                    results[f'magic_{game}'] = r
                    self._print_summary(r, f"MAgIC Metrics ({game})")
        
        # 2. Model features
        logger.info("\n" + "="*60)
        logger.info("PCA 2: Model Features")
        logger.info("="*60)
        
        r = self.pca_model_features()
        if r:
            results['model_features'] = r
            self._print_summary(r, "Model Features")
        
        # 3. Performance metrics per game
        logger.info("\n" + "="*60)
        logger.info("PCA 3: Performance Metrics")
        logger.info("="*60)
        
        if self.perf_df is not None:
            for game in self.perf_df['game'].unique():
                r = self.pca_performance_metrics(game)
                if r:
                    results[f'perf_{game}'] = r
                    self._print_summary(r, f"Performance ({game})")
        
        self.results = results
        return results
    
    def _print_summary(self, result: Dict, title: str):
        """Print PCA summary."""
        print(f"\n--- {title} ---")
        print(f"Samples: {len(result['labels'])}, Features: {len(result['features'])}")
        
        # Variance explained
        print("\nVariance Explained:")
        for i, (v, c) in enumerate(zip(result['explained_variance'], result['cumulative_variance'])):
            print(f"  PC{i+1}: {v:.1%} (cumulative: {c:.1%})")
            if c > 0.95:
                break
        
        # Top loadings
        loadings = result['loadings']
        print("\nTop Loadings:")
        for pc in loadings.columns[:2]:
            sorted_abs = loadings[pc].abs().sort_values(ascending=False)
            print(f"  {pc}:")
            for feat in sorted_abs.head(3).index:
                print(f"    {feat}: {loadings.loc[feat, pc]:+.3f}")
        
        # Interpretation
        self._interpret(result)
    
    def _interpret(self, result: Dict):
        """Provide interpretation."""
        var = result['explained_variance']
        cum = result['cumulative_variance']
        n_feat = len(result['features'])
        
        print("\nInterpretation:")
        
        # Variance concentration
        if var[0] > 0.6:
            print(f"  ⚠️  PC1 dominates ({var[0]:.1%}) - possible redundancy")
        elif var[0] < 0.35:
            print(f"  ✓ Variance well distributed - dimensions appear distinct")
        
        # Dimensionality
        n_80 = np.argmax(cum >= 0.8) + 1
        if n_80 < n_feat * 0.5:
            print(f"  ⚠️  Only {n_80}/{n_feat} PCs needed for 80% variance")
        else:
            print(f"  ✓ {n_80}/{n_feat} PCs for 80% - good dimensionality")
        
        # Collinearity check (for MAgIC metrics)
        if result.get('type') == 'magic_metrics':
            loadings = result['loadings']
            pc1_high = loadings['PC1'].abs() > 0.5
            if pc1_high.sum() > 2:
                high_feats = loadings.index[pc1_high].tolist()
                print(f"  ⚠️  Possible collinearity: {high_feats}")
    
    def save_results(self):
        """Save results to CSV."""
        variance_rows = []
        loadings_rows = []
        scores_rows = []
        
        for name, r in self.results.items():
            # Variance
            for i, (v, c) in enumerate(zip(r['explained_variance'], r['cumulative_variance'])):
                variance_rows.append({
                    'analysis': name,
                    'component': f'PC{i+1}',
                    'explained_variance': v,
                    'cumulative_variance': c
                })
            
            # Loadings
            for feat in r['features']:
                row = {'analysis': name, 'feature': feat}
                for pc in r['loadings'].columns:
                    row[pc] = r['loadings'].loc[feat, pc]
                loadings_rows.append(row)
            
            # Scores
            for i, label in enumerate(r['labels']):
                row = {'analysis': name, 'model': label}
                for j in range(r['scores'].shape[1]):
                    row[f'PC{j+1}'] = r['scores'][i, j]
                scores_rows.append(row)
        
        # Save
        pd.DataFrame(variance_rows).to_csv(self.analysis_dir / "pca_variance.csv", index=False)
        pd.DataFrame(loadings_rows).to_csv(self.analysis_dir / "pca_loadings.csv", index=False)
        pd.DataFrame(scores_rows).to_csv(self.analysis_dir / "pca_scores.csv", index=False)
        
        logger.info(f"Saved results to {self.analysis_dir}")
    
    def plot_results(self):
        """Generate plots."""
        if not PLOTTING_AVAILABLE:
            logger.warning("Plotting libraries not available")
            return
        
        plots_dir = self.analysis_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        for name, r in self.results.items():
            self._plot_scree(r, name, plots_dir)
            self._plot_loadings(r, name, plots_dir)
            if r['scores'].shape[1] >= 2:
                self._plot_biplot(r, name, plots_dir)
        
        logger.info(f"Saved plots to {plots_dir}")
    
    def _plot_scree(self, r: Dict, name: str, out_dir: Path):
        """Scree plot."""
        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(1, len(r['explained_variance']) + 1)
        ax.bar(x, r['explained_variance'], alpha=0.7, label='Individual')
        ax.plot(x, r['cumulative_variance'], 'ro-', label='Cumulative')
        ax.axhline(0.8, color='g', linestyle='--', alpha=0.5, label='80%')
        ax.set_xlabel('Component')
        ax.set_ylabel('Variance')
        ax.set_title(f'Scree: {name}')
        ax.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"scree_{name}.png", dpi=150)
        plt.close()
    
    def _plot_loadings(self, r: Dict, name: str, out_dir: Path):
        """Loadings heatmap."""
        n_pc = min(5, r['loadings'].shape[1])
        data = r['loadings'].iloc[:, :n_pc]
        
        fig, ax = plt.subplots(figsize=(8, max(4, len(data) * 0.4)))
        sns.heatmap(data, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
        ax.set_title(f'Loadings: {name}')
        plt.tight_layout()
        plt.savefig(out_dir / f"loadings_{name}.png", dpi=150)
        plt.close()
    
    def _plot_biplot(self, r: Dict, name: str, out_dir: Path):
        """Biplot."""
        scores = r['scores']
        loadings = r['loadings']
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7, s=80)
        
        for i, label in enumerate(r['labels']):
            short = label.split('/')[-1][:20]
            ax.annotate(short, (scores[i, 0], scores[i, 1]), fontsize=7, alpha=0.7)
        
        scale = np.max(np.abs(scores)) * 0.8
        for feat in loadings.index:
            x = loadings.loc[feat, 'PC1'] * scale
            y = loadings.loc[feat, 'PC2'] * scale
            ax.arrow(0, 0, x, y, head_width=0.08, fc='red', ec='red', alpha=0.6)
            ax.text(x*1.1, y*1.1, feat, color='red', fontsize=9)
        
        ax.axhline(0, color='k', alpha=0.2)
        ax.axvline(0, color='k', alpha=0.2)
        ax.set_xlabel(f'PC1 ({r["explained_variance"][0]:.1%})')
        ax.set_ylabel(f'PC2 ({r["explained_variance"][1]:.1%})')
        ax.set_title(f'Biplot: {name}')
        plt.tight_layout()
        plt.savefig(out_dir / f"biplot_{name}.png", dpi=150)
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PCA Analysis for MAgIC')
    parser.add_argument('--analysis-dir', type=str, default='output/analysis')
    parser.add_argument('--config-path', type=str, default='config/config.json')
    parser.add_argument('--no-plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    config_path = Path(args.config_path) if args.config_path else None
    
    # Check if analysis dir exists
    if not analysis_dir.exists():
        logger.error(f"Analysis directory not found: {analysis_dir}")
        return
    
    analyzer = PCAAnalyzer(analysis_dir, config_path)
    analyzer.run_all()
    
    if analyzer.results:
        analyzer.save_results()
        if not args.no_plot:
            analyzer.plot_results()
        logger.info("Done!")
    else:
        logger.warning("No PCA results generated")


if __name__ == "__main__":
    main()