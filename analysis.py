"""
Unified Paper Analysis for MAgIC Benchmark
==========================================

Combines: PCA, Regression, Radar Charts, Summary CSVs

RQ1: Model Features → Performance
RQ2: Capability Profiles (PCA, Similarity)  
RQ3: Capabilities → Performance (Hierarchical Regression)

Uses config.json for all model definitions.
"""

import sys
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# =============================================================================
# Configuration Loader (matches config/config.py pattern)
# =============================================================================

def load_config(config_path: Path = None) -> Dict[str, Any]:
    """Load config.json from project root."""
    if config_path is None:
        # Try common locations
        for path in [Path("config/config.json"), Path("config.json"), 
                     Path(__file__).parent / "config" / "config.json"]:
            if path.exists():
                config_path = path
                break
    
    if config_path is None or not config_path.exists():
        raise FileNotFoundError("config.json not found")
    
    with open(config_path, 'r') as f:
        return json.load(f)


@dataclass
class ModelInfo:
    """Model information extracted from config."""
    model_name: str
    display_name: str
    model_family: str
    reasoning_effort: str
    reasoning_output: str
    is_thinking: bool
    is_large: bool
    param_size: float


class ConfigManager:
    """Manages model configurations from config.json."""
    
    def __init__(self, config: Dict = None):
        self.config = config or load_config()
        self._parse_models()
    
    def _parse_models(self):
        """Parse model configurations."""
        self.model_configs = self.config.get('model_configs', {})
        self.challenger_models = self.config.get('models', {}).get('challenger_models', [])
        self.defender_model = self.config.get('models', {}).get('defender_model', '')
        self.random_agent = 'random_agent'
        
    def get_all_model_names(self) -> List[str]:
        """Get all model names from config."""
        return list(self.model_configs.keys())
    
    def get_challenger_models(self) -> List[str]:
        """Get challenger model names."""
        return self.challenger_models
    
    def get_defender_model(self) -> str:
        """Get defender model name."""
        return self.defender_model
    
    def get_model_info(self, model_name: str) -> ModelInfo:
        """Get parsed model information."""
        cfg = self.model_configs.get(model_name, {})
        
        display_name = cfg.get('display_name', model_name)
        family = cfg.get('model_family', 'unknown')
        reasoning_effort = cfg.get('reasoning_effort', 'standard')
        reasoning_output = cfg.get('reasoning_output', 'none')
        
        # Derived features
        is_thinking = reasoning_output != 'none' or 'Thinking-On' in model_name
        is_large = any(s in model_name for s in ['70B', '235B', '32B', '30B'])
        
        # Extract param size
        import re
        match = re.search(r'(\d+)B', model_name)
        param_size = float(match.group(1)) if match else 0.0
        
        return ModelInfo(
            model_name=model_name,
            display_name=display_name,
            model_family=family,
            reasoning_effort=reasoning_effort,
            reasoning_output=reasoning_output,
            is_thinking=is_thinking,
            is_large=is_large,
            param_size=param_size
        )
    
    def create_features_dataframe(self) -> pd.DataFrame:
        """Create DataFrame with model features for regression."""
        records = []
        for model_name in self.get_all_model_names():
            info = self.get_model_info(model_name)
            records.append({
                'model': model_name,
                'display_name': info.display_name,
                'family': info.model_family,
                'is_large': int(info.is_large),
                'is_thinking': int(info.is_thinking),
                'param_size': info.param_size,
                'is_qwen3': int('qwen3' in info.model_family.lower() or 'Qwen3' in model_name),
                'is_llama': int('llama' in info.model_family.lower() or 'Llama' in model_name),
                'is_gemma': int('gemma' in info.model_family.lower()),
                'is_random': int(model_name == self.random_agent),
                'is_defender': int(model_name == self.defender_model),
            })
        return pd.DataFrame(records)


# =============================================================================
# Optional Imports
# =============================================================================

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('seaborn-v0_8-whitegrid')
    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Main Analysis Class
# =============================================================================

class UnifiedPaperAnalysis:
    """
    Unified analysis pipeline for MAgIC paper.
    
    Generates:
    - T1: Performance Descriptives
    - T2: RQ1 Regression (Model Features → Performance)
    - T3: MAgIC Descriptives  
    - T4: PCA Loadings
    - T5: Profile Stability
    - T6: Hierarchical Regression (Incremental Validity)
    - T7: Final Coefficients
    - F1: Coefficient Plot
    - F2: PCA Biplot
    - F3: Radar Chart
    - F4: Variance Decomposition
    """
    
    MAGIC_METRICS = ['rationality', 'reasoning', 'judgment', 'self_awareness',
                     'cooperation', 'coordination', 'deception']
    
    PERF_METRICS = ['win_rate', 'average_profit']
    
    def __init__(self, analysis_dir: str = "output/analysis", config: Dict = None):
        self.analysis_dir = Path(analysis_dir)
        self.output_dir = self.analysis_dir / "paper_outputs"
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load config
        self.cfg = ConfigManager(config)
        
        # Load data
        self._load_data()
        
        # Prepare merged dataset
        self.df = self._prepare_data()
        
        logger.info(f"Initialized with {len(self.df)} observations, {self.df['model'].nunique()} models")
        logger.info(f"Challenger models: {len(self.cfg.get_challenger_models())}")
        logger.info(f"Defender model: {self.cfg.get_defender_model()}")
    
    def _load_data(self):
        """Load performance and MAgIC CSVs."""
        try:
            self.perf_df = pd.read_csv(self.analysis_dir / "performance_metrics.csv")
            self.magic_df = pd.read_csv(self.analysis_dir / "magic_behavioral_metrics.csv")
            logger.info(f"Loaded performance_metrics.csv: {len(self.perf_df)} rows")
            logger.info(f"Loaded magic_behavioral_metrics.csv: {len(self.magic_df)} rows")
        except FileNotFoundError as e:
            logger.error(f"Data files not found: {e}")
            raise
    
    def _prepare_data(self) -> pd.DataFrame:
        """Merge and prepare analysis dataset with feature engineering from regression.py."""
        import re
        
        # Get valid models from config
        valid_models = self.cfg.get_all_model_names()
        
        # Filter to valid models (exclude random_agent and defender for main analysis)
        def is_excluded(model_name):
            model_lower = model_name.lower()
            if 'random' in model_lower:
                return True
            if 'gemma' in model_lower:  # defender model
                return True
            return False
        
        perf_filtered = self.perf_df[
            (self.perf_df['model'].isin(valid_models)) & 
            (~self.perf_df['model'].apply(is_excluded))
        ]
        magic_filtered = self.magic_df[
            (self.magic_df['model'].isin(valid_models)) &
            (~self.magic_df['model'].apply(is_excluded))
        ]
        
        # Pivot to wide format
        perf_pivot = perf_filtered.pivot_table(
            index=['game', 'model', 'condition'],
            columns='metric', values='mean'
        ).reset_index()
        
        magic_pivot = magic_filtered.pivot_table(
            index=['game', 'model', 'condition'],
            columns='metric', values='mean'
        ).reset_index()
        
        # Merge
        df = pd.merge(perf_pivot, magic_pivot, on=['game', 'model', 'condition'], how='outer')
        
        # =================================================================
        # FEATURE ENGINEERING (matching regression.py pattern)
        # =================================================================
        
        def extract_features(model_name: str) -> Dict[str, Any]:
            """Extract model features matching regression.py logic."""
            model_lower = model_name.lower()
            
            # --- Architecture: MoE detection ---
            is_moe = bool(re.search(r'-a\d+b', model_lower)) or 'moe' in model_lower
            if re.search(r'\d+e-instruct', model_lower) or 'maverick' in model_lower or 'scout' in model_lower:
                is_moe = True
            
            # --- Size: Extract parameter count ---
            size_params = 0.0
            size_match = re.search(r'(?<!a)(\d+\.?\d*)b(?!-)', model_lower)
            if size_match:
                size_params = float(size_match.group(1))
            if size_params == 0:
                size_match = re.search(r'-(\d+\.?\d*)b-', model_lower)
                if size_match:
                    size_params = float(size_match.group(1))
            if size_params == 0:
                size_match = re.search(r'-(\d+\.?\d*)b$', model_lower)
                if size_match:
                    size_params = float(size_match.group(1))
            
            # --- Family detection ---
            family = 'unknown'
            if 'qwen' in model_lower:
                family = 'qwen'
            elif 'llama' in model_lower:
                family = 'llama'
            elif 'gemma' in model_lower:
                family = 'gemma'
            elif 'mistral' in model_lower:
                family = 'mistral'
            
            # --- Version extraction ---
            version = 0.0
            version_patterns = [
                r'qwen(\d+)',
                r'llama-?(\d+\.?\d*)',
                r'gemma-?(\d+)',
            ]
            for pattern in version_patterns:
                match = re.search(pattern, model_lower)
                if match:
                    version = float(match.group(1))
                    break
            
            # --- Thinking mode: FROM CONFIG ---
            thinking = 0
            if model_name in self.cfg.model_configs:
                reasoning_output = self.cfg.model_configs[model_name].get('reasoning_output', 'none')
                if reasoning_output in ['reasoning_tokens', 'output_tokens']:
                    thinking = 1
            else:
                # Fallback: name-based detection
                if 'thinking-off' in model_lower or 'instruct' in model_lower:
                    thinking = 0
                elif 'thinking' in model_lower:
                    thinking = 1
            
            return {
                'architecture_moe': 1 if is_moe else 0,
                'size_params': size_params,
                'family': family,
                'version': version,
                'thinking': thinking,
            }
        
        # Extract features for each model
        unique_models = df['model'].unique()
        feature_records = []
        for model in unique_models:
            feats = extract_features(model)
            feats['model'] = model
            feature_records.append(feats)
        
        features_df = pd.DataFrame(feature_records)
        
        # Encode family as numeric
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        features_df['family_encoded'] = le.fit_transform(features_df['family'])
        self._family_encoder = le
        self._family_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Log extracted features
        logger.info("\nExtracted model features:")
        logger.info("-" * 100)
        for _, row in features_df.iterrows():
            logger.info(
                f"  {row['model'][:50]:50s} | arch_moe={row['architecture_moe']} | "
                f"size={row['size_params']:6.1f}B | family={row['family']:8s} | "
                f"ver={row['version']:4.1f} | thinking={row['thinking']}"
            )
        logger.info("-" * 100)
        
        # Merge features into main dataframe
        df = pd.merge(df, features_df, on='model', how='left')
        
        # Condition features
        df['is_5_player'] = df['condition'].apply(lambda x: 1 if 'more_players' in str(x) else 0)
        
        # Game features
        df['is_dynamic'] = df['game'].isin(['green_porter', 'athey_bagwell']).astype(int)
        
        # Add display names from config
        df['display_name'] = df['model'].apply(
            lambda x: self.cfg.get_model_info(x).display_name if x in self.cfg.get_all_model_names() else x
        )
        
        return df
    
    # =========================================================================
    # TABLE 1: Performance Descriptives
    # =========================================================================
    
    def table_1_performance_descriptives(self) -> pd.DataFrame:
        """T1: Performance by Model."""
        logger.info("Generating T1: Performance Descriptives")
        
        available = [c for c in self.PERF_METRICS if c in self.df.columns]
        
        summary = self.df.groupby('model')[available].agg(['mean', 'std', 'count'])
        summary.columns = ['_'.join(col) for col in summary.columns]
        summary = summary.round(4).reset_index()
        
        # Add display names - derive from model name or use config
        def get_display_name(model_name):
            if model_name in self.cfg.get_all_model_names():
                return self.cfg.get_model_info(model_name).display_name
            return model_name.split('/')[-1] if '/' in model_name else model_name
        
        summary['display_name'] = summary['model'].apply(get_display_name)
        
        cols = ['model', 'display_name'] + [c for c in summary.columns if c not in ['model', 'display_name']]
        summary = summary[cols]
        
        summary.to_csv(self.output_dir / "T1_performance_descriptives.csv", index=False)
        logger.info("Saved T1")
        return summary
    
    # =========================================================================
    # TABLE 2: RQ1 Regression
    # =========================================================================
    
    def table_2_rq1_regression(self) -> pd.DataFrame:
        """T2: Model Features → Performance."""
        if not STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available")
            return pd.DataFrame()
        
        logger.info("Generating T2: RQ1 Regression")
        
        # Feature columns from regression.py
        feature_cols = ['architecture_moe', 'size_params', 'family_encoded', 'version', 'thinking']
        feature_cols = [c for c in feature_cols if c in self.df.columns]
        
        results = []
        
        for outcome in self.PERF_METRICS:
            if outcome not in self.df.columns:
                continue
            
            analysis_df = self.df.dropna(subset=[outcome] + feature_cols)
            if len(analysis_df) < 10:
                continue
            
            X = analysis_df[feature_cols].astype(float)
            y = analysis_df[outcome]
            X_const = sm.add_constant(X)
            
            model = sm.OLS(y, X_const).fit()
            
            for var in model.params.index:
                results.append({
                    'outcome': outcome,
                    'predictor': var,
                    'coefficient': round(model.params[var], 4),
                    'std_error': round(model.bse[var], 4),
                    't_value': round(model.tvalues[var], 3),
                    'p_value': round(model.pvalues[var], 4),
                    'ci_lower': round(model.conf_int().loc[var, 0], 4),
                    'ci_upper': round(model.conf_int().loc[var, 1], 4),
                    'sig': '***' if model.pvalues[var] < 0.001 else 
                           '**' if model.pvalues[var] < 0.01 else
                           '*' if model.pvalues[var] < 0.05 else ''
                })
            
            # Fit stats
            results.append({'outcome': outcome, 'predictor': 'R_squared',
                          'coefficient': round(model.rsquared, 4), 'std_error': np.nan,
                          't_value': np.nan, 'p_value': np.nan, 'ci_lower': np.nan,
                          'ci_upper': np.nan, 'sig': ''})
            results.append({'outcome': outcome, 'predictor': 'N',
                          'coefficient': int(model.nobs), 'std_error': np.nan,
                          't_value': np.nan, 'p_value': np.nan, 'ci_lower': np.nan,
                          'ci_upper': np.nan, 'sig': ''})
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "T2_rq1_regression.csv", index=False)
        logger.info("Saved T2")
        return df
    
    # =========================================================================
    # TABLE 3: MAgIC Descriptives
    # =========================================================================
    
    def table_3_magic_descriptives(self) -> pd.DataFrame:
        """T3: MAgIC Metrics by Model."""
        logger.info("Generating T3: MAgIC Descriptives")
        
        available = [c for c in self.MAGIC_METRICS if c in self.df.columns]
        
        summary = self.df.groupby('model')[available].agg(['mean', 'std']).round(4)
        summary.columns = ['_'.join(col) for col in summary.columns]
        summary = summary.reset_index()
        
        summary.to_csv(self.output_dir / "T3_magic_descriptives.csv", index=False)
        logger.info("Saved T3")
        return summary
    
    # =========================================================================
    # TABLE 4: PCA Loadings
    # =========================================================================
    
    def table_4_pca_loadings(self) -> pd.DataFrame:
        """T4: PCA on MAgIC metrics."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available")
            return pd.DataFrame()
        
        logger.info("Generating T4: PCA Loadings")
        
        available = [c for c in self.MAGIC_METRICS if c in self.df.columns]
        pca_data = self.df[available].dropna()
        
        if len(pca_data) < 5:
            logger.warning("Insufficient data for PCA")
            return pd.DataFrame()
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_data)
        
        n_comp = min(len(available), 3)
        pca = PCA(n_components=n_comp)
        pca.fit(X_scaled)
        
        # Store for plots
        self._pca = pca
        self._scaler = scaler
        self._pca_cols = available
        
        # Loadings table
        loadings = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_comp)],
            index=available
        ).round(3)
        
        # Add variance explained
        var_row = pd.DataFrame({
            f'PC{i+1}': [round(pca.explained_variance_ratio_[i], 3)]
            for i in range(n_comp)
        }, index=['Variance_Explained'])
        
        loadings = pd.concat([loadings, var_row])
        loadings = loadings.reset_index().rename(columns={'index': 'metric'})
        
        loadings.to_csv(self.output_dir / "T4_pca_loadings.csv", index=False)
        logger.info("Saved T4")
        return loadings
    
    # =========================================================================
    # TABLE 5: Profile Stability
    # =========================================================================
    
    def table_5_profile_stability(self) -> pd.DataFrame:
        """T5: Cross-condition stability."""
        logger.info("Generating T5: Profile Stability")
        
        available = [c for c in self.MAGIC_METRICS if c in self.df.columns]
        conditions = self.df['condition'].dropna().unique()
        
        stability = pd.DataFrame(index=conditions, columns=conditions, dtype=float)
        
        for c1 in conditions:
            for c2 in conditions:
                p1 = self.df[self.df['condition'] == c1].groupby('model')[available].mean()
                p2 = self.df[self.df['condition'] == c2].groupby('model')[available].mean()
                
                common = p1.index.intersection(p2.index)
                if len(common) < 3:
                    continue
                
                f1 = p1.loc[common].values.flatten()
                f2 = p2.loc[common].values.flatten()
                
                if len(f1) > 2:
                    corr, _ = stats.pearsonr(f1, f2)
                    stability.loc[c1, c2] = round(corr, 3)
        
        stability.to_csv(self.output_dir / "T5_profile_stability.csv")
        logger.info("Saved T5")
        return stability
    
    # =========================================================================
    # TABLE 6: Hierarchical Regression
    # =========================================================================
    
    def table_6_hierarchical_regression(self) -> pd.DataFrame:
        """T6: Incremental validity of MAgIC."""
        if not STATSMODELS_AVAILABLE:
            return pd.DataFrame()
        
        logger.info("Generating T6: Hierarchical Regression")
        
        # Feature columns from regression.py
        feature_cols = ['architecture_moe', 'size_params', 'family_encoded', 'version', 'thinking']
        feature_cols = [c for c in feature_cols if c in self.df.columns]
        
        magic_cols = [c for c in self.MAGIC_METRICS if c in self.df.columns]
        
        results = []
        
        for outcome in self.PERF_METRICS:
            if outcome not in self.df.columns:
                continue
            
            all_cols = feature_cols + magic_cols + [outcome]
            analysis_df = self.df.dropna(subset=all_cols)
            
            if len(analysis_df) < 15:
                continue
            
            y = analysis_df[outcome]
            
            # Model 1: Features only
            X1 = sm.add_constant(analysis_df[feature_cols].astype(float))
            m1 = sm.OLS(y, X1).fit()
            
            # Model 2: Features + MAgIC
            X2 = sm.add_constant(analysis_df[feature_cols + magic_cols].astype(float))
            m2 = sm.OLS(y, X2).fit()
            
            r2_change = m2.rsquared - m1.rsquared
            df1, df2 = len(magic_cols), m2.df_resid
            
            if df2 > 0 and (1 - m2.rsquared) > 0:
                f_change = (r2_change / df1) / ((1 - m2.rsquared) / df2)
                p_change = 1 - stats.f.cdf(f_change, df1, df2)
            else:
                f_change, p_change = np.nan, np.nan
            
            results.append({
                'outcome': outcome, 'step': 'Model 1: Features Only',
                'R_squared': round(m1.rsquared, 4), 'R2_change': np.nan,
                'F_change': np.nan, 'p_change': np.nan, 'N': int(m1.nobs)
            })
            results.append({
                'outcome': outcome, 'step': 'Model 2: Features + MAgIC',
                'R_squared': round(m2.rsquared, 4), 
                'R2_change': round(r2_change, 4),
                'F_change': round(f_change, 4) if not np.isnan(f_change) else np.nan,
                'p_change': round(p_change, 4) if not np.isnan(p_change) else np.nan,
                'N': int(m2.nobs)
            })
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "T6_hierarchical_regression.csv", index=False)
        logger.info("Saved T6")
        return df
    
    # =========================================================================
    # TABLE 7: Final Coefficients
    # =========================================================================
    
    def table_7_final_coefficients(self) -> pd.DataFrame:
        """T7: Full model coefficients."""
        if not STATSMODELS_AVAILABLE:
            return pd.DataFrame()
        
        logger.info("Generating T7: Final Coefficients")
        
        # Feature columns from regression.py
        feature_cols = ['architecture_moe', 'size_params', 'family_encoded', 'version', 'thinking']
        feature_cols = [c for c in feature_cols if c in self.df.columns]
        
        magic_cols = [c for c in self.MAGIC_METRICS if c in self.df.columns]
        all_pred = feature_cols + magic_cols
        
        results = []
        
        for outcome in self.PERF_METRICS:
            if outcome not in self.df.columns:
                continue
            
            analysis_df = self.df.dropna(subset=all_pred + [outcome])
            if len(analysis_df) < 10:
                continue
            
            y = analysis_df[outcome]
            X = sm.add_constant(analysis_df[all_pred].astype(float))
            model = sm.OLS(y, X).fit()
            
            for var in model.params.index:
                pred_type = 'Feature' if var in feature_cols else 'MAgIC'
                if var == 'const':
                    pred_type = 'Intercept'
                
                results.append({
                    'outcome': outcome, 'predictor': var, 'type': pred_type,
                    'coefficient': round(model.params[var], 4),
                    'std_error': round(model.bse[var], 4),
                    'p_value': round(model.pvalues[var], 4),
                    'sig': '***' if model.pvalues[var] < 0.001 else 
                           '**' if model.pvalues[var] < 0.01 else
                           '*' if model.pvalues[var] < 0.05 else ''
                })
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / "T7_final_coefficients.csv", index=False)
        logger.info("Saved T7")
        return df
    
    # =========================================================================
    # FIGURE 1: Coefficient Plot
    # =========================================================================
    
    def figure_1_coefficient_plot(self):
        """F1: Model feature effects."""
        if not PLOT_AVAILABLE:
            return
        
        logger.info("Generating F1: Coefficient Plot")
        
        t2_path = self.output_dir / "T2_rq1_regression.csv"
        if not t2_path.exists():
            self.table_2_rq1_regression()
        
        reg_df = pd.read_csv(t2_path)
        exclude = ['const', 'R_squared', 'N']
        plot_df = reg_df[~reg_df['predictor'].isin(exclude)]
        
        outcomes = [o for o in self.PERF_METRICS if o in plot_df['outcome'].values]
        if not outcomes:
            return
        
        fig, axes = plt.subplots(1, len(outcomes), figsize=(7*len(outcomes), 6))
        if len(outcomes) == 1:
            axes = [axes]
        
        for ax, outcome in zip(axes, outcomes):
            data = plot_df[plot_df['outcome'] == outcome].sort_values('coefficient')
            if data.empty:
                continue
            
            y_pos = np.arange(len(data))
            ax.errorbar(data['coefficient'], y_pos,
                       xerr=[data['coefficient'] - data['ci_lower'],
                             data['ci_upper'] - data['coefficient']],
                       fmt='o', capsize=4, color='steelblue', markersize=8)
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(data['predictor'].str.replace('_', ' ').str.title())
            ax.set_xlabel('Coefficient (95% CI)')
            ax.set_title(f'Effect on {outcome.replace("_", " ").title()}')
        
        plt.suptitle('RQ1: Model Features → Performance', fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "F1_coefficient_plot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved F1")
    
    # =========================================================================
    # FIGURE 2: PCA Biplot
    # =========================================================================
    
    def figure_2_pca_biplot(self):
        """F2: Models in capability space."""
        if not PLOT_AVAILABLE or not hasattr(self, '_pca'):
            self.table_4_pca_loadings()
        
        if not hasattr(self, '_pca'):
            return
        
        logger.info("Generating F2: PCA Biplot")
        
        cols = self._pca_cols
        
        # Aggregate by model - family is already in the dataframe from feature extraction
        model_agg = self.df.groupby('model')[cols + ['family']].agg({
            **{c: 'mean' for c in cols}, 'family': 'first'
        }).reset_index()
        
        X_scaled = self._scaler.transform(model_agg[cols])
        scores = self._pca.transform(X_scaled)
        model_agg['PC1'], model_agg['PC2'] = scores[:, 0], scores[:, 1]
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = {'qwen': 'blue', 'llama': 'red', 'gemma': 'green', 'unknown': 'gray'}
        
        for family in model_agg['family'].unique():
            subset = model_agg[model_agg['family'] == family]
            ax.scatter(subset['PC1'], subset['PC2'], label=family.title(),
                      c=colors.get(family, 'gray'), s=100, alpha=0.7)
            for _, row in subset.iterrows():
                short = row['model'].split('/')[-1][:12]
                ax.annotate(short, (row['PC1'], row['PC2']), fontsize=7, alpha=0.7)
        
        # Loading arrows
        loadings = self._pca.components_.T
        for i, col in enumerate(cols):
            ax.arrow(0, 0, loadings[i, 0]*2.5, loadings[i, 1]*2.5,
                    head_width=0.08, fc='black', ec='black', alpha=0.5)
            ax.text(loadings[i, 0]*3, loadings[i, 1]*3, col.replace('_', '\n'), 
                   fontsize=8, ha='center')
        
        var = self._pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
        ax.set_title('RQ2: Models in Capability Space', fontweight='bold')
        ax.axhline(0, color='gray', ls='--', alpha=0.3)
        ax.axvline(0, color='gray', ls='--', alpha=0.3)
        ax.legend(title='Family')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "F2_pca_biplot.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved F2")
    
    # =========================================================================
    # FIGURE 3: Radar Chart
    # =========================================================================
    
    def figure_3_radar_chart(self):
        """F3: MAgIC capability profiles."""
        if not PLOT_AVAILABLE:
            return
        
        logger.info("Generating F3: Radar Chart")
        
        available = [c for c in self.MAGIC_METRICS if c in self.df.columns]
        if len(available) < 3:
            return
        
        profiles = self.df.groupby('model')[available].mean()
        
        num_vars = len(available)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
        
        for (model, values), color in zip(profiles.iterrows(), colors):
            vals = values.tolist() + [values.tolist()[0]]
            short = model.split('/')[-1][:15]
            ax.plot(angles, vals, 'o-', linewidth=2, label=short, color=color)
            ax.fill(angles, vals, alpha=0.1, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace('_', '\n').title() for c in available])
        ax.set_title('RQ2: MAgIC Capability Profiles', fontweight='bold', y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=7)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "F3_radar_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved F3")
    
    # =========================================================================
    # FIGURE 4: Variance Decomposition
    # =========================================================================
    
    def figure_4_variance_decomposition(self):
        """F4: Features vs MAgIC contribution."""
        if not PLOT_AVAILABLE:
            return
        
        logger.info("Generating F4: Variance Decomposition")
        
        t6_path = self.output_dir / "T6_hierarchical_regression.csv"
        if not t6_path.exists():
            self.table_6_hierarchical_regression()
        
        hier_df = pd.read_csv(t6_path)
        
        plot_data = []
        for outcome in hier_df['outcome'].unique():
            odf = hier_df[hier_df['outcome'] == outcome]
            r2_feat = odf[odf['step'].str.contains('Features Only')]['R_squared'].values
            r2_full = odf[odf['step'].str.contains('MAgIC')]['R_squared'].values
            
            if len(r2_feat) > 0 and len(r2_full) > 0:
                plot_data.append({
                    'outcome': outcome.replace('_', ' ').title(),
                    'Features': r2_feat[0],
                    'MAgIC': r2_full[0] - r2_feat[0],
                    'Unexplained': 1 - r2_full[0]
                })
        
        if not plot_data:
            return
        
        df = pd.DataFrame(plot_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(df))
        
        ax.bar(x, df['Features'], label='Model Features', color='steelblue')
        ax.bar(x, df['MAgIC'], bottom=df['Features'], label='MAgIC (ΔR²)', color='coral')
        ax.bar(x, df['Unexplained'], bottom=df['Features']+df['MAgIC'],
               label='Unexplained', color='lightgray')
        
        ax.set_ylabel('R²')
        ax.set_title('RQ3: Variance Decomposition', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df['outcome'])
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "F4_variance_decomposition.png", dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Saved F4")
    
    # =========================================================================
    # RUN ALL
    # =========================================================================
    
    def run_all(self) -> Path:
        """Run complete analysis pipeline."""
        logger.info("=" * 60)
        logger.info("UNIFIED PAPER ANALYSIS")
        logger.info(f"Models from config: {len(self.cfg.get_all_model_names())}")
        logger.info(f"Output: {self.output_dir}")
        logger.info("=" * 60)
        
        # Tables
        self.table_1_performance_descriptives()
        self.table_2_rq1_regression()
        self.table_3_magic_descriptives()
        self.table_4_pca_loadings()
        self.table_5_profile_stability()
        self.table_6_hierarchical_regression()
        self.table_7_final_coefficients()
        
        # Figures
        self.figure_1_coefficient_plot()
        self.figure_2_pca_biplot()
        self.figure_3_radar_chart()
        self.figure_4_variance_decomposition()
        
        logger.info("=" * 60)
        logger.info(f"✅ DONE! Outputs: {self.output_dir}")
        logger.info("=" * 60)
        
        return self.output_dir


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified MAgIC Paper Analysis")
    parser.add_argument('--analysis-dir', type=str, default="output/analysis",
                       help='Path to analysis directory')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config.json')
    args = parser.parse_args()
    
    config = None
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    analyzer = UnifiedPaperAnalysis(args.analysis_dir, config)
    analyzer.run_all()