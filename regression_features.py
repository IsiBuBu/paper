# regression.py

"""
Model Feature Regression Analysis

Run from project root:
    python regression.py

Analyzes how model characteristics predict MAgIC behavioral metrics and performance metrics.

Model Features (predictors):
- architecture: dense vs moe (binary)  
- size_params: parameter count in billions
- family: model family (qwen, llama, gemma, etc.)
- version: model version number
- thinking: determined from config.json's reasoning_output field
- reasoning_chars: average reasoning character count from experiment llm_metadata

Targets:
- MAgIC metrics: cooperation, rationality, reasoning, judgment, self_awareness, coordination, deception
- Performance metrics: average_profit, market_price, win_rate, allocative_efficiency

Outputs (saved to output/analysis/):
- model_feature_regression.csv: All SLR and MLR results
- model_feature_regression_significant.csv: Only p < 0.05 results
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import json
from dataclasses import dataclass
from collections import defaultdict
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Model Feature Extraction
# =============================================================================

@dataclass
class ModelFeatures:
    """Extracted features for a model."""
    model_name: str
    architecture: str  # 'dense' or 'moe'
    size_params: float  # in billions
    family: str  # 'qwen', 'llama', 'gemma', etc.
    version: float  # version number
    thinking: int  # 1 if thinking enabled (from config), 0 otherwise
    reasoning_chars: float  # average reasoning output length from experiments


def load_model_configs(config_path: Optional[Path] = None) -> Tuple[Dict[str, Dict], str]:
    """
    Load model_configs from config.json to determine thinking mode.
    
    Returns tuple: (model_configs dict, defender_model name)
    """
    if config_path is None:
        config_path = Path("config/config.json")
    
    if not config_path.exists():
        logger.warning(f"Config not found: {config_path}")
        return {}, ""
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        model_configs = config.get('model_configs', {})
        defender_model = config.get('models', {}).get('defender_model', '')
        return model_configs, defender_model
    except Exception as e:
        logger.warning(f"Could not load config.json: {e}")
        return {}, ""


def extract_model_features(model_name: str, 
                           model_configs: Dict[str, Dict],
                           reasoning_data: Optional[Dict] = None) -> ModelFeatures:
    """
    Extract features from model name and config.
    
    Thinking mode is determined by config.json's reasoning_output field:
    - reasoning_output: "reasoning_tokens" or "output_tokens" → thinking=1
    - reasoning_output: "none" → thinking=0
    
    Examples:
    - "Qwen/Qwen3-235B-A22B-Thinking-2507" with reasoning_output="reasoning_tokens" -> thinking=1
    - "Qwen/Qwen3-235B-A22B-Instruct-2507" with reasoning_output="none" -> thinking=0
    - "Qwen/Qwen3-14B-Thinking-Off" with reasoning_output="none" -> thinking=0
    """
    model_lower = model_name.lower()
    
    # --- Architecture: MoE detection ---
    # MoE indicators: "-A22B", "-A3B", "-A16B" (active params notation)
    is_moe = bool(re.search(r'-a\d+b', model_lower)) or 'moe' in model_lower
    # Llama-4 Maverick/Scout with E notation are also MoE (128E = 128 experts)
    if re.search(r'\d+e-instruct', model_lower) or 'maverick' in model_lower or 'scout' in model_lower:
        is_moe = True
    architecture = 'moe' if is_moe else 'dense'
    
    # --- Size: Extract parameter count ---
    size_params = 0.0
    # Match patterns like "235B", "70B", "27b", "8B", "17B"
    # Avoid matching "A22B" (active params for MoE)
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
    elif 'claude' in model_lower:
        family = 'claude'
    elif 'gpt' in model_lower:
        family = 'openai'
    
    # --- Version extraction ---
    version = 0.0
    version_patterns = [
        r'qwen(\d+)',           # Qwen3 -> 3
        r'llama-?(\d+\.?\d*)',  # Llama-3.3, Llama-4 -> 3.3, 4
        r'gemma-?(\d+)',        # gemma-3 -> 3
    ]
    for pattern in version_patterns:
        match = re.search(pattern, model_lower)
        if match:
            version = float(match.group(1))
            break
    
    # --- Thinking mode: FROM CONFIG (not name parsing) ---
    thinking = 0
    if model_name in model_configs:
        reasoning_output = model_configs[model_name].get('reasoning_output', 'none')
        # thinking=1 if reasoning_output is "reasoning_tokens" or "output_tokens"
        if reasoning_output in ['reasoning_tokens', 'output_tokens']:
            thinking = 1
        else:
            thinking = 0
    else:
        # Fallback: if not in config, try name-based detection
        # But this is less reliable
        if 'thinking-off' in model_lower or 'instruct' in model_lower:
            thinking = 0
        elif 'thinking' in model_lower:
            thinking = 1
    
    # --- Reasoning character count from experiments ---
    reasoning_chars = 0.0
    if reasoning_data and model_name in reasoning_data:
        reasoning_chars = reasoning_data[model_name]
    
    return ModelFeatures(
        model_name=model_name,
        architecture=architecture,
        size_params=size_params,
        family=family,
        version=version,
        thinking=thinking,
        reasoning_chars=reasoning_chars
    )


def load_reasoning_char_data(experiments_dir: Path) -> Dict[str, float]:
    """
    Load average reasoning character counts from experiment results.
    
    Extracts from llm_metadata in experiment JSON files:
    - Static games: sim['game_data']['llm_metadata']['challenger']['reasoning_char_count']
    - Dynamic games: average across round_data['llm_metadata']['challenger']['reasoning_char_count']
    
    Returns dict: model_name -> avg_reasoning_chars
    """
    reasoning_data = defaultdict(list)
    
    for json_file in experiments_dir.rglob("*_competition_result*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            model = data.get('challenger_model', '')
            if not model:
                continue
            
            for sim in data.get('simulation_results', []):
                game_data = sim.get('game_data', {})
                
                # Dynamic games: check rounds for llm_metadata
                if 'rounds' in game_data:
                    for round_data in game_data['rounds']:
                        llm_meta = round_data.get('llm_metadata', {})
                        challenger_meta = llm_meta.get('challenger', {})
                        
                        # Get reasoning_char_count directly (no fallbacks)
                        char_count = challenger_meta.get('reasoning_char_count', 0)
                        if char_count and char_count > 0:
                            reasoning_data[model].append(char_count)
                
                # Static games: check top-level llm_metadata
                else:
                    llm_meta = game_data.get('llm_metadata', {})
                    challenger_meta = llm_meta.get('challenger', {})
                    
                    # Get reasoning_char_count directly (no fallbacks)
                    char_count = challenger_meta.get('reasoning_char_count', 0)
                    if char_count and char_count > 0:
                        reasoning_data[model].append(char_count)
                        
        except Exception as e:
            logger.debug(f"Could not process {json_file}: {e}")
    
    # Average the counts
    return {model: np.mean(counts) for model, counts in reasoning_data.items() if counts}


# =============================================================================
# Regression Analysis
# =============================================================================

class ModelFeatureRegression:
    """
    Performs SLR and MLR analysis of model features predicting metrics.
    """
    
    def __init__(self, analysis_dir: Path, experiments_dir: Optional[Path] = None, config_path: Optional[Path] = None):
        self.analysis_dir = Path(analysis_dir)
        self.experiments_dir = Path(experiments_dir) if experiments_dir else None
        self.config_path = Path(config_path) if config_path else None
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Data storage
        self.magic_df = None
        self.perf_df = None
        self.model_configs = {}
        self.reasoning_data = {}
        self.family_encoder = None
        self.family_mapping = {}
        
    def load_data(self):
        """Load metrics CSVs, model configs, and reasoning data."""
        # Load metrics
        magic_path = self.analysis_dir / "magic_behavioral_metrics.csv"
        perf_path = self.analysis_dir / "performance_metrics.csv"
        
        if magic_path.exists():
            self.magic_df = pd.read_csv(magic_path)
            # Filter out random_agent and defender model
            self.magic_df = self._filter_models(self.magic_df)
            self.logger.info(f"Loaded MAgIC metrics: {len(self.magic_df)} rows")
        else:
            self.logger.warning(f"MAgIC metrics not found at {magic_path}")
            
        if perf_path.exists():
            self.perf_df = pd.read_csv(perf_path)
            # Filter out random_agent and defender model
            self.perf_df = self._filter_models(self.perf_df)
            self.logger.info(f"Loaded performance metrics: {len(self.perf_df)} rows")
        else:
            self.logger.warning(f"Performance metrics not found at {perf_path}")
        
        # Load model configs (for thinking mode detection)
        self.model_configs = load_model_configs(self.config_path)
        self.logger.info(f"Loaded config for {len(self.model_configs)} models")
        
        # Load reasoning character data
        if self.experiments_dir and self.experiments_dir.exists():
            self.reasoning_data = load_reasoning_char_data(self.experiments_dir)
            # Filter out excluded models from reasoning data
            self.reasoning_data = {k: v for k, v in self.reasoning_data.items() 
                                   if not self._is_excluded_model(k)}
            self.logger.info(f"Loaded reasoning data for {len(self.reasoning_data)} models")
            for model, chars in sorted(self.reasoning_data.items()):
                self.logger.info(f"  {model}: avg {chars:.0f} reasoning chars")
    
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
            self.logger.info(f"Excluding models: {list(excluded)}")
        return df[mask].copy()
    
    def build_feature_matrix(self, models: List[str]) -> pd.DataFrame:
        """Build feature matrix for list of models."""
        features_list = []
        
        for model in models:
            feat = extract_model_features(model, self.model_configs, self.reasoning_data)
            features_list.append({
                'model': model,
                'architecture': feat.architecture,
                'architecture_moe': 1 if feat.architecture == 'moe' else 0,
                'size_params': feat.size_params,
                'family': feat.family,
                'version': feat.version,
                'thinking': feat.thinking,
                'reasoning_chars': feat.reasoning_chars
            })
        
        df = pd.DataFrame(features_list)
        
        # Encode family as numeric
        le = LabelEncoder()
        df['family_encoded'] = le.fit_transform(df['family'])
        self.family_encoder = le
        self.family_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        
        # Log extracted features
        self.logger.info("\nExtracted model features:")
        self.logger.info("-" * 100)
        for _, row in df.iterrows():
            self.logger.info(
                f"  {row['model'][:50]:50s} | arch={row['architecture']:5s} | "
                f"size={row['size_params']:6.1f}B | family={row['family']:8s} | "
                f"ver={row['version']:4.1f} | thinking={row['thinking']} | "
                f"reasoning_chars={row['reasoning_chars']:8.0f}"
            )
        self.logger.info("-" * 100)
        
        return df
    
    def run_slr(self, X: np.ndarray, y: np.ndarray, feature_name: str) -> Dict[str, Any]:
        """Run Simple Linear Regression."""
        mask = ~(np.isnan(X) | np.isnan(y))
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < 3 or np.std(X_clean) == 0 or np.std(y_clean) == 0:
            return {
                'feature': feature_name,
                'r_squared': np.nan,
                'coef': np.nan,
                'p_value': np.nan,
                'n_obs': len(X_clean)
            }
        
        try:
            X_const = sm.add_constant(X_clean)
            model = sm.OLS(y_clean, X_const).fit()
            
            return {
                'feature': feature_name,
                'r_squared': model.rsquared,
                'coef': model.params[1] if len(model.params) > 1 else np.nan,
                'p_value': model.pvalues[1] if len(model.pvalues) > 1 else np.nan,
                'n_obs': len(X_clean)
            }
        except Exception as e:
            self.logger.warning(f"SLR failed for {feature_name}: {e}")
            return {
                'feature': feature_name,
                'r_squared': np.nan,
                'coef': np.nan,
                'p_value': np.nan,
                'n_obs': len(X_clean)
            }
    
    def run_mlr(self, X: pd.DataFrame, y: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Run Multiple Linear Regression with VIF calculation."""
        mask = ~pd.isna(y)
        for col in feature_names:
            if col in X.columns:
                mask &= ~X[col].isna()
        
        X_clean = X.loc[mask, feature_names].copy()
        y_clean = np.array(y)[mask] if not hasattr(y, 'loc') else y[mask].values
        
        if len(X_clean) < len(feature_names) + 2:
            return {
                'r_squared': np.nan,
                'r_squared_adj': np.nan,
                'coefficients': {},
                'p_values': {},
                'vif': {},
                'n_obs': len(X_clean)
            }
        
        # Drop constant columns
        X_clean = X_clean.loc[:, X_clean.std() > 0]
        
        if X_clean.empty:
            return {
                'r_squared': np.nan,
                'r_squared_adj': np.nan,
                'coefficients': {},
                'p_values': {},
                'vif': {},
                'n_obs': len(y_clean)
            }
        
        try:
            # Calculate VIF for multicollinearity check
            vif_dict = {}
            try:
                for i, col in enumerate(X_clean.columns):
                    vif_value = variance_inflation_factor(X_clean.values, i)
                    vif_dict[col] = vif_value
            except Exception as vif_error:
                self.logger.debug(f"VIF calculation failed: {vif_error}")
                vif_dict = {col: np.nan for col in X_clean.columns}
            
            X_const = sm.add_constant(X_clean)
            model = sm.OLS(y_clean, X_const).fit()
            
            coefficients = {}
            p_values = {}
            for i, name in enumerate(X_clean.columns):
                idx = i + 1
                if idx < len(model.params):
                    coefficients[name] = model.params.iloc[idx]
                    p_values[name] = model.pvalues.iloc[idx]
            
            return {
                'r_squared': model.rsquared,
                'r_squared_adj': model.rsquared_adj,
                'coefficients': coefficients,
                'p_values': p_values,
                'vif': vif_dict,
                'n_obs': len(X_clean)
            }
        except Exception as e:
            self.logger.warning(f"MLR failed: {e}")
            return {
                'r_squared': np.nan,
                'r_squared_adj': np.nan,
                'coefficients': {},
                'p_values': {},
                'vif': {},
                'n_obs': len(X_clean)
            }
    
    def analyze_game(self, game_name: str) -> Tuple[List[Dict], List[Dict]]:
        """Run full regression analysis for a single game."""
        slr_results = []
        mlr_results = []
        
        magic_game = self.magic_df[self.magic_df['game'] == game_name] if self.magic_df is not None else pd.DataFrame()
        perf_game = self.perf_df[self.perf_df['game'] == game_name] if self.perf_df is not None else pd.DataFrame()
        
        if magic_game.empty and perf_game.empty:
            self.logger.warning(f"No data for game: {game_name}")
            return slr_results, mlr_results
        
        # Get unique models
        all_models = set()
        if not magic_game.empty:
            all_models.update(magic_game['model'].unique())
        if not perf_game.empty:
            all_models.update(perf_game['model'].unique())
        
        # Build feature matrix
        features_df = self.build_feature_matrix(list(all_models))
        
        # Features for regression (reasoning_chars removed due to multicollinearity with thinking)
        numeric_features = ['architecture_moe', 'size_params', 'family_encoded', 'version', 'thinking']
        
        # Get conditions
        conditions = set()
        if not magic_game.empty:
            conditions.update(magic_game['condition'].unique())
        if not perf_game.empty:
            conditions.update(perf_game['condition'].unique())
        
        for condition in conditions:
            # --- MAgIC Metrics ---
            if not magic_game.empty:
                magic_cond = magic_game[magic_game['condition'] == condition]
                magic_pivot = magic_cond.pivot_table(index='model', columns='metric', values='mean').reset_index()
                merged = magic_pivot.merge(features_df, on='model', how='left')
                magic_metrics = [c for c in magic_pivot.columns if c != 'model']
                
                for target in magic_metrics:
                    if target not in merged.columns:
                        continue
                    y = merged[target].values
                    
                    for feat in numeric_features:
                        if feat not in merged.columns:
                            continue
                        X = merged[feat].values
                        result = self.run_slr(X, y, feat)
                        result.update({
                            'game': game_name, 'condition': condition,
                            'target': target, 'target_type': 'magic', 'analysis': 'SLR'
                        })
                        slr_results.append(result)
                    
                    available_features = [f for f in numeric_features if f in merged.columns]
                    mlr_result = self.run_mlr(merged, merged[target], available_features)
                    
                    for feat in available_features:
                        mlr_row = {
                            'game': game_name, 'condition': condition,
                            'target': target, 'target_type': 'magic', 'analysis': 'MLR_Combined',
                            'feature': feat, 'r_squared': mlr_result['r_squared'],
                            'r_squared_adj': mlr_result['r_squared_adj'],
                            'coef': mlr_result['coefficients'].get(feat, np.nan),
                            'p_value': mlr_result['p_values'].get(feat, np.nan),
                            'vif': mlr_result.get('vif', {}).get(feat, np.nan),
                            'n_obs': mlr_result['n_obs']
                        }
                        mlr_results.append(mlr_row)
            
            # --- Performance Metrics ---
            if not perf_game.empty:
                perf_cond = perf_game[perf_game['condition'] == condition]
                perf_pivot = perf_cond.pivot_table(index='model', columns='metric', values='mean').reset_index()
                merged = perf_pivot.merge(features_df, on='model', how='left')
                perf_metrics = [c for c in perf_pivot.columns if c != 'model']
                
                for target in perf_metrics:
                    if target not in merged.columns:
                        continue
                    y = merged[target].values
                    
                    for feat in numeric_features:
                        if feat not in merged.columns:
                            continue
                        X = merged[feat].values
                        result = self.run_slr(X, y, feat)
                        result.update({
                            'game': game_name, 'condition': condition,
                            'target': target, 'target_type': 'performance', 'analysis': 'SLR'
                        })
                        slr_results.append(result)
                    
                    available_features = [f for f in numeric_features if f in merged.columns]
                    mlr_result = self.run_mlr(merged, merged[target], available_features)
                    
                    for feat in available_features:
                        mlr_row = {
                            'game': game_name, 'condition': condition,
                            'target': target, 'target_type': 'performance', 'analysis': 'MLR_Combined',
                            'feature': feat, 'r_squared': mlr_result['r_squared'],
                            'r_squared_adj': mlr_result['r_squared_adj'],
                            'coef': mlr_result['coefficients'].get(feat, np.nan),
                            'p_value': mlr_result['p_values'].get(feat, np.nan),
                            'vif': mlr_result.get('vif', {}).get(feat, np.nan),
                            'n_obs': mlr_result['n_obs']
                        }
                        mlr_results.append(mlr_row)
        
        return slr_results, mlr_results
    
    def run_all_analyses(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Run regression analysis for all games."""
        self.load_data()
        
        all_slr = []
        all_mlr = []
        
        games = set()
        if self.magic_df is not None:
            games.update(self.magic_df['game'].unique())
        if self.perf_df is not None:
            games.update(self.perf_df['game'].unique())
        
        self.logger.info(f"Analyzing {len(games)} games: {games}")
        
        for game in sorted(games):
            self.logger.info(f"\nProcessing {game}...")
            slr_results, mlr_results = self.analyze_game(game)
            all_slr.extend(slr_results)
            all_mlr.extend(mlr_results)
        
        return pd.DataFrame(all_slr), pd.DataFrame(all_mlr)
    
    def save_results(self, slr_df: pd.DataFrame, mlr_df: pd.DataFrame):
        """Save regression results to CSV files."""
        combined = pd.concat([slr_df, mlr_df], ignore_index=True)
        
        col_order = ['game', 'condition', 'target_type', 'target', 'analysis', 'feature', 
                     'r_squared', 'r_squared_adj', 'coef', 'p_value', 'vif', 'n_obs']
        col_order = [c for c in col_order if c in combined.columns]
        combined = combined[col_order]
        
        output_path = self.analysis_dir / "model_feature_regression.csv"
        combined.to_csv(output_path, index=False)
        self.logger.info(f"Saved combined results to {output_path}")
        
        significant = combined[combined['p_value'] < 0.05].copy()
        significant = significant.sort_values(['game', 'target', 'r_squared'], ascending=[True, True, False])
        
        sig_path = self.analysis_dir / "model_feature_regression_significant.csv"
        significant.to_csv(sig_path, index=False)
        self.logger.info(f"Saved significant results to {sig_path}")
        
        self._print_summary(combined)
        return combined
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary of regression results."""
        print("\n" + "="*80)
        print("MODEL FEATURE REGRESSION SUMMARY")
        print("="*80)
        
        feature_display = {
            'architecture_moe': 'Architecture (MoE=1)',
            'size_params': 'Model Size (B params)',
            'family_encoded': 'Model Family',
            'version': 'Model Version',
            'thinking': 'Thinking Mode (1=On)',
            'reasoning_chars': 'Reasoning Char Count'
        }
        
        for game in sorted(df['game'].unique()):
            game_df = df[df['game'] == game]
            print(f"\n{'='*60}")
            print(f"GAME: {game.upper()}")
            print(f"{'='*60}")
            
            for target_type in ['magic', 'performance']:
                type_df = game_df[game_df['target_type'] == target_type]
                if type_df.empty:
                    continue
                
                print(f"\n--- {target_type.upper()} METRICS ---")
                
                for target in sorted(type_df['target'].unique()):
                    target_df = type_df[type_df['target'] == target]
                    
                    slr = target_df[target_df['analysis'] == 'SLR']
                    if not slr.empty and not slr['r_squared'].isna().all():
                        best_idx = slr['r_squared'].idxmax()
                        if pd.notna(best_idx):
                            best_slr = slr.loc[best_idx]
                            feat_name = feature_display.get(best_slr['feature'], best_slr['feature'])
                            p_str = f"{best_slr['p_value']:.4f}" if pd.notna(best_slr['p_value']) else "N/A"
                            r2_str = f"{best_slr['r_squared']:.3f}" if pd.notna(best_slr['r_squared']) else "N/A"
                            print(f"\n  {target}:")
                            print(f"    Best SLR: {feat_name} (R²={r2_str}, p={p_str})")
                    
                    mlr = target_df[target_df['analysis'] == 'MLR_Combined']
                    if not mlr.empty and not mlr['r_squared'].isna().all():
                        mlr_r2 = mlr['r_squared'].iloc[0]
                        mlr_r2_adj = mlr['r_squared_adj'].iloc[0]
                        if pd.notna(mlr_r2):
                            print(f"    MLR Combined: R²={mlr_r2:.3f}, R²_adj={mlr_r2_adj:.3f}")
                        
                        sig_mlr = mlr[mlr['p_value'] < 0.05]
                        if not sig_mlr.empty:
                            print(f"    Significant predictors (p<0.05):")
                            for _, row in sig_mlr.iterrows():
                                feat_name = feature_display.get(row['feature'], row['feature'])
                                print(f"      - {feat_name}: coef={row['coef']:.3f}, p={row['p_value']:.4f}")
        
        print("\n" + "="*80)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run model feature regression analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Feature Regression Analysis')
    parser.add_argument('--analysis-dir', type=str, default='output/analysis',
                        help='Directory containing metrics CSVs')
    parser.add_argument('--experiments-dir', type=str, default='output/experiments',
                        help='Directory containing experiment results (for reasoning chars)')
    parser.add_argument('--config-path', type=str, default='config/config.json',
                        help='Path to config.json (for thinking mode detection)')
    args = parser.parse_args()
    
    analysis_dir = Path(args.analysis_dir)
    experiments_dir = Path(args.experiments_dir)
    config_path = Path(args.config_path)
    
    # Check if analysis dir exists
    if not analysis_dir.exists():
        logger.error(f"Analysis directory not found: {analysis_dir}")
        return
    
    # Optional directories - set to None if not found
    if not experiments_dir.exists():
        logger.warning(f"Experiments directory not found: {experiments_dir}, skipping reasoning_chars")
        experiments_dir = None
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using name-based feature extraction")
        config_path = None
    
    analyzer = ModelFeatureRegression(analysis_dir, experiments_dir, config_path)
    slr_df, mlr_df = analyzer.run_all_analyses()
    
    if slr_df.empty and mlr_df.empty:
        logger.warning("No results generated")
        return
    
    analyzer.save_results(slr_df, mlr_df)
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()