# analysis/visualization/visualize_rq3.py

import pandas as pd
import numpy as np
import logging
import json
import os
import sys
from pathlib import Path

# --- Core Plotting Libs ---
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_LIBS_INSTALLED = True
except ImportError:
    PLOT_LIBS_INSTALLED = False

# --- Libs for Advanced Predictive Modeling ---
try:
    import shap
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score
    import pymc as pm
    import arviz as az
    import pytensor.tensor as pt
    ADVANCED_LIBS_INSTALLED = True
except ImportError:
    ADVANCED_LIBS_INSTALLED = False

from config.config import get_analysis_dir, load_config

# --- Regression Analysis Component ---

class RegressionAnalyzer:
    """
    Performs regression analysis to identify significant predictors of performance.
    """

    def __init__(self, analysis_dir: Path, plots_dir: Path, tables_dir: Path):
        self.analysis_dir = analysis_dir
        self.plots_dir = plots_dir / "regression_analysis"
        self.tables_dir = tables_dir
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.results = {}
        self.importance_results = []
        self.beta_summary_results = []

    def prepare_data(self) -> pd.DataFrame:
        """Loads, merges, and engineers features for the regression analysis."""
        self.logger.info("Preparing data for regression analysis...")
        try:
            perf_df = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
            magic_df = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
            config = load_config()
        except FileNotFoundError:
            self.logger.error("Summary CSV files not found.")
            return pd.DataFrame()

        models_to_include = list(config.get('model_configs', {}).keys())
        perf_df = perf_df[perf_df['model'].isin(models_to_include)]
        magic_df = magic_df[magic_df['model'].isin(models_to_include)]

        perf_pivot = perf_df.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean').reset_index()
        magic_pivot = magic_df.pivot_table(index=['game', 'model', 'condition'], columns='metric', values='mean').reset_index()
        merged_df = pd.merge(perf_pivot, magic_pivot, on=['game', 'model', 'condition'], how='left')
        
        # Feature Engineering
        merged_df['is_gen_2_5'] = merged_df['model'].apply(lambda x: 1 if '2_5' in x else 0)
        merged_df['is_lite'] = merged_df['model'].apply(lambda x: 1 if 'lite' in x else 0)
        merged_df['is_5_player'] = merged_df['condition'].apply(lambda x: 1 if 'more_players' in x else 0)
        
        merged_df['thinking'] = 'off'
        merged_df.loc[merged_df['model'].str.contains('low', na=False), 'thinking'] = 'low'
        merged_df.loc[merged_df['model'].str.contains('medium', na=False), 'thinking'] = 'medium'
        
        thinking_dummies = pd.get_dummies(merged_df['thinking'], prefix='thinking', drop_first=True).astype(int)
        merged_df = pd.concat([merged_df, thinking_dummies], axis=1)

        return merged_df

    def run_all_regressions(self):
        """Runs regression models for each game and each performance metric."""
        df = self.prepare_data()
        if df.empty: return

        for game in df['game'].unique():
            self.logger.info("-" * 40)
            self.logger.info(f"Running Regressions for Game: {game.upper()}")
            game_df = df[df['game'] == game].dropna(axis=1, how='all')
            
            self.analyze_performance(game, game_df.copy(), 'average_profit')
            self.analyze_performance(game, game_df.copy(), 'win_rate')
            self.analyze_game_specific_metrics(game, game_df.copy())
            
        self.save_results_to_csv()

    def analyze_performance(self, game: str, game_df: pd.DataFrame, target_metric: str):
        """Dispatcher for running the correct regression model."""
        potential_predictors = [
            'is_gen_2_5', 'is_lite', 'is_5_player', 'thinking_low', 'thinking_medium',
            'rationality', 'self_awareness', 'cooperation', 'coordination', 
            'judgment', 'deception', 'reasoning'
        ]
        predictors = [p for p in potential_predictors if p in game_df.columns]
        X = game_df[predictors].fillna(0)
        y = game_df[target_metric]
        
        if y.isnull().all() or len(y) < 4:
            self.logger.warning(f"Target '{target_metric}' has insufficient data for {game}. Skipping.")
            return
            
        # --- UPDATED DISPATCH LOGIC ---
        # Proportional metrics [0,1] -> Beta Regression (PyMC)
        # Continuous metrics -> Gradient Boosting
        proportional_metrics = [
            'win_rate', 
            'allocative_efficiency',  # New Spulber metric
            'productive_efficiency',  # New Athey Bagwell metric
            'reversion_frequency'     # Green Porter
        ]
        
        if target_metric in proportional_metrics:
             self.analyze_proportion_pymc(game, X, y, target_metric)
        else:
             # Default to GBR for continuous (e.g., average_profit, market_price)
             self.analyze_continuous_gbr(game, X, y, target_metric)

    def analyze_continuous_gbr(self, game: str, X: pd.DataFrame, y: pd.Series, target_metric: str):
        """Analyzes a continuous target using Gradient Boosting and generates SHAP plot."""
        self.logger.info(f"--- Predicting {target_metric.replace('_', ' ').title()} with Gradient Boosting ---")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        model = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
        importances = pd.DataFrame({'feature': X.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
        
        importances['game'] = game
        importances['target_metric'] = target_metric
        self.importance_results.append(importances)
        self.logger.info(f"Feature Importances:\n{importances}")

        if X_test.empty:
            self.results.setdefault(game, {})[target_metric] = {"r2": np.nan}
            return

        r2 = r2_score(y_test, model.predict(X_test))
        self.results.setdefault(game, {})[target_metric] = {"r2": r2}
        self.logger.info(f"Gradient Boosting R-squared: {r2:.4f}")

        if ADVANCED_LIBS_INSTALLED:
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)
            shap.summary_plot(shap_values, X_test, show=False)
            
            plt.title(f"SHAP Summary: {target_metric} ({game.title()})")
            plot_path = self.plots_dir / f"{game}_{target_metric}_shap_summary.png"
            plt.savefig(plot_path, bbox_inches='tight')
            plt.close()

    def analyze_proportion_pymc(self, game: str, X: pd.DataFrame, y: pd.Series, target_metric: str):
        """Analyzes a proportional target using Beta Regression with PyMC."""
        self.logger.info(f"--- Predicting {target_metric.replace('_', ' ').title()} with Beta Regression ---")
        if not ADVANCED_LIBS_INSTALLED:
            self.logger.error("PyMC not installed. Skipping Beta Regression.")
            return

        n = len(y)
        # Squeeze to (0, 1) interval to avoid 0/1 errors in Beta distribution
        y_squeezed = (y * (n - 1) + 0.5) / n

        if y_squeezed.isnull().all() or y_squeezed.nunique() <= 1:
            return

        with pm.Model() as beta_model:
            intercept = pm.Normal("intercept", mu=0, sigma=10)
            betas = pm.Normal("betas", mu=0, sigma=10, shape=X.shape[1])
            kappa = pm.HalfCauchy("kappa", beta=10)
            mu = pm.invlogit(intercept + pt.dot(X.values, betas))
            alpha = mu * kappa
            beta = (1 - mu) * kappa
            y_obs = pm.Beta("y_obs", alpha=alpha, beta=beta, observed=y_squeezed)
            
            trace = pm.sample(
                2000, tune=2000, chains=4, cores=min(4, os.cpu_count()),
                target_accept=0.99, progressbar=False, random_seed=42
            )
        
        posterior_pred = pm.sample_posterior_predictive(trace, model=beta_model, random_seed=42)
        y_pred = posterior_pred.posterior_predictive["y_obs"].mean(("chain", "draw"))
        pseudo_r2 = r2_score(y_squeezed, y_pred)
        
        summary = az.summary(trace, var_names=["intercept", "betas", "kappa"])
        summary.index = ['intercept'] + list(X.columns) + ['kappa']

        self.results.setdefault(game, {})[target_metric] = {"pseudo_r2": pseudo_r2}
        summary['game'] = game
        summary['target_metric'] = target_metric
        self.beta_summary_results.append(summary)
        self.logger.info(f"PyMC Beta Regression (Pseudo) R-squared: {pseudo_r2:.4f}")

    def analyze_game_specific_metrics(self, game: str, game_df: pd.DataFrame):
        """Runs regressions for the NEW game-specific performance metrics."""
        # --- UPDATED METRIC MAPPING ---
        game_specific_metrics = {
            'salop': 'market_price',            # Continuous -> GBR
            'spulber': 'allocative_efficiency', # Proportion -> Beta
            'green_porter': 'reversion_frequency', # Proportion -> Beta
            'athey_bagwell': 'productive_efficiency' # Proportion -> Beta
        }
        target_metric = game_specific_metrics.get(game)
        if target_metric and target_metric in game_df.columns:
            self.analyze_performance(game, game_df, target_metric)
    
    def save_results_to_csv(self):
        """Saves all collected regression results to separate CSV files."""
        # Save R-squared summary
        r2_records = []
        for game, metrics in self.results.items():
            for metric_type, value in metrics.items():
                r2_type = "Pseudo R-squared" if "pseudo_r2" in value else "R-squared"
                r2_value = value.get('r2') or value.get('pseudo_r2')
                r2_records.append({
                    "game": game, "target_metric": metric_type, "r2_type": r2_type, "r2_value": r2_value
                })
        if r2_records:
            pd.DataFrame(r2_records).to_csv(self.tables_dir / "regression_r_squared_summary.csv", index=False)

        if self.importance_results:
            pd.concat(self.importance_results).to_csv(self.tables_dir / "regression_feature_importances.csv", index=False)
        if self.beta_summary_results:
            pd.concat(self.beta_summary_results).to_csv(self.tables_dir / "regression_beta_summary.csv")


# --- Correlation Visualization Component ---

def _create_master_correlation_table(corr_df: pd.DataFrame, tables_dir: Path):
    """Saves the complete, unfiltered correlation results to a CSV file."""
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("Creating Appendix Table: Master Correlation Table...")
    
    master_table = corr_df[['game_name', 'condition_type', 'magic_metric', 'performance_metric', 'correlation_coefficient', 'p_value', 'n_samples']]
    master_table.columns = ['Game', 'Condition', 'MAgIC Metric', 'Performance Metric', 'r-value', 'p-value', 'n']

    master_table_path = tables_dir / "A.1_master_correlation_table.csv"
    master_table.to_csv(master_table_path, index=False)
    logger.info(f"Saved master correlation table to {master_table_path}")

def _plot_correlation_heatmaps(corr_df: pd.DataFrame, plots_dir: Path):
    """Plot 3.1: Creates and saves separate heatmaps for conditions."""
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("Generating Plot 3.1: Correlation Heatmaps...")

    condition_map = {
        'few_players': '3-Player Baseline',
        'more_players': '5-Player Variation'
    }

    for cond_key, cond_title in condition_map.items():
        cond_df = corr_df[corr_df['condition_type'] == cond_key].copy()
        if cond_df.empty: continue
        
        cond_df['display_column'] = cond_df['game_name'].str.title() + "\n(" + cond_df['performance_metric'].str.replace('_', ' ').str.title() + ")"
        
        heatmap_pivot = cond_df.pivot_table(index='magic_metric', columns='display_column', values='correlation_coefficient')

        if heatmap_pivot.empty: continue
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(heatmap_pivot, annot=True, cmap='coolwarm', center=0, linewidths=.5, fmt=".2f", vmin=-1, vmax=1)
        plt.title(f"MAgIC vs. Performance Correlation ({cond_title})", fontsize=16)
        plt.tight_layout()
        plot_filename = plots_dir / f"P3.1_{cond_key}_correlation_heatmap.png"
        plt.savefig(plot_filename)
        plt.close()

# --- Main Visualization Function ---

def visualize_rq3():
    """Generates all tables and plots for Research Question 3."""
    if not PLOT_LIBS_INSTALLED:
        print("Plotting libraries not installed.")
        return
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("RQ3Visualizer")
    logger.info("--- Generating visualizations for RQ3: Correlations & Predictive Models ---")
    
    analysis_dir = get_analysis_dir()
    plots_dir = analysis_dir / "plots" / "rq3"
    tables_dir = analysis_dir / "tables" / "rq3"
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Part 1: Correlation Analysis
        logger.info("[Step 1/2] Generating correlation tables and heatmaps...")
        corr_df = pd.read_csv(analysis_dir / "correlations_analysis_structural.csv")
        _create_master_correlation_table(corr_df, tables_dir)
        _plot_correlation_heatmaps(corr_df, plots_dir)

        # Part 2: Regression Analysis
        logger.info("[Step 2/2] Running regression analysis and generating SHAP plots...")
        if ADVANCED_LIBS_INSTALLED:
            logging.getLogger('pymc').setLevel(logging.WARNING)
            analyzer = RegressionAnalyzer(analysis_dir=analysis_dir, plots_dir=plots_dir, tables_dir=tables_dir)
            analyzer.run_all_regressions()
        else:
            logger.warning("Advanced libraries not found. Skipping regression.")
        
        logger.info("--- Finished RQ3 visualizations ---")

    except FileNotFoundError as e:
        logger.error(f"Failed to find a necessary file for RQ3: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during RQ3 visualization: {e}", exc_info=True)

if __name__ == '__main__':
    visualize_rq3()