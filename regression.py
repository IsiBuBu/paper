import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
import os
import re

# --- Configuration ---
BEHAVIOR_FILE = 'output/analysis/magic_behavioral_metrics.csv'
PERFORMANCE_FILE = 'output/analysis/performance_metrics.csv'
OUTPUT_DIR = 'output/analysis'
COLLINEARITY_THRESHOLD = 0.95 # use a stricter pairwise corr threshold (was 0.85)

# --- Helper Functions ---

def extract_arch_features(model_name):
    m = str(model_name)
    qwen3_models = {
        'Qwen/Qwen3-14B-Thinking-Off':    {'family': 'qwen3', 'param_size': '14b', 'model_architecture': 'dense', 'thinking_enabled': False, 'family_version': None},
        'Qwen/Qwen3-32B-Thinking-Off':    {'family': 'qwen3', 'param_size': '32b', 'model_architecture': 'dense', 'thinking_enabled': False, 'family_version': None},
        'Qwen/Qwen3-14B-Thinking-On':     {'family': 'qwen3', 'param_size': '14b', 'model_architecture': 'dense', 'thinking_enabled': True,  'family_version': None},
        'Qwen/Qwen3-32B-Thinking-On':     {'family': 'qwen3', 'param_size': '32b', 'model_architecture': 'dense', 'thinking_enabled': True,  'family_version': None},
        'Qwen/Qwen3-235B-A22B-Instruct-2507': {'family': 'qwen3', 'param_size': '235b', 'model_architecture': 'moe',   'thinking_enabled': False, 'family_version': None},
        'Qwen/Qwen3-30B-A3B-Thinking-On':    {'family': 'qwen3', 'param_size': '30b',  'model_architecture': 'moe',   'thinking_enabled': True,  'family_version': None},
        'Qwen/Qwen3-30B-A3B-Thinking-Off':   {'family': 'qwen3', 'param_size': '30b',  'model_architecture': 'moe',   'thinking_enabled': False, 'family_version': None},
    }
    llama_models = {
        'meta-llama/Llama-3.3-70B-Instruct-Turbo':         {'family': 'llama', 'param_size': '70b', 'model_architecture': 'dense', 'family_version': '3.3'},
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo':     {'family': 'llama', 'param_size': '8b',  'model_architecture': 'dense', 'family_version': '3.1'},
        'meta-llama/Llama-4-Scout-17B-16E-Instruct':       {'family': 'llama', 'param_size': '109b', 'model_architecture': 'moe',   'family_version': '4'},
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': {'family': 'llama', 'param_size': '400b', 'model_architecture': 'moe', 'family_version': '4'},
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo':    {'family': 'llama', 'param_size': '70b', 'model_architecture': 'dense', 'family_version': '3.1'},
    }
    if m in qwen3_models:
        d = qwen3_models[m]
        return {
            'model': model_name,
            'family': d['family'],
            'family_version': d['family_version'],
            'model_architecture': d['model_architecture'],
            'thinking_enabled': d['thinking_enabled'],
            'param_size': d['param_size']
        }
    if m in llama_models:
        d = llama_models[m]
        return {
            'model': model_name,
            'family': d['family'],
            'family_version': d['family_version'],
            'model_architecture': d['model_architecture'],
            'thinking_enabled': None,  # Not defined for llama
            'param_size': d['param_size']
        }
    # For all other models, return None (they will be excluded from arch_feats)
    return None

def load_and_merge_data():
    if not os.path.exists(BEHAVIOR_FILE) or not os.path.exists(PERFORMANCE_FILE):
        print(f"Error: Input files not found in {os.getcwd()}")
        return None, None, None

    df_behavior = pd.read_csv(BEHAVIOR_FILE)
    df_perf = pd.read_csv(PERFORMANCE_FILE)

    # Pivot Behavior (Predictors)
    pivot_behavior = df_behavior.pivot_table(
        index=['game', 'model', 'condition'], 
        columns='metric', 
        values='mean'
    ).reset_index()
    
    # Pivot Performance (Targets)
    pivot_perf = df_perf.pivot_table(
        index=['game', 'model', 'condition'], 
        columns='metric', 
        values='mean'
    ).reset_index()

    # Merge
    merged_df = pd.merge(pivot_behavior, pivot_perf, on=['game', 'model', 'condition'], how='inner')
    
    # --- FILTERING STEP ---
    models_to_exclude = ['random_agent', 'google/gemma-3-27b-it']
    merged_df = merged_df[~merged_df['model'].isin(models_to_exclude)]
    print(f"Filtered out {models_to_exclude}. Remaining models: {len(merged_df['model'].unique())}")

    # --- Architectural Features ---
    # Use model column from performance_metrics.csv for feature mapping
    arch_feats = pd.DataFrame([extract_arch_features(m) for m in df_perf['model'].unique() if extract_arch_features(m) is not None])
    merged_df = merged_df.merge(arch_feats, on='model', how='left')

    # Fill thinking_enabled for models that don't define it (e.g., llama) -> treat as 0
    if 'thinking_enabled' in merged_df.columns:
        merged_df['thinking_enabled'] = merged_df['thinking_enabled'].fillna(False).astype(bool).astype(int)
    else:
        merged_df['thinking_enabled'] = 0

    # Binary flags requested by user
    merged_df['is_qwen3'] = (merged_df['family'] == 'qwen3').astype(int)
    merged_df['is_llama3_1'] = (merged_df['family_version'] == '3.1').astype(int)

    # Derive is_moe from explicit architecture if present, otherwise fallback to model name matching
    if 'model_architecture' in merged_df.columns:
        merged_df['is_moe'] = merged_df['model_architecture'].astype(str).str.contains('moe', case=False, na=False).astype(int)
    else:
        merged_df['is_moe'] = merged_df['model'].astype(str).str.contains('Llama-4|MoE|Moe', case=False, na=False).astype(int)

    # Convert param_size to numeric (e.g., '70b' -> 70) if not already present
    def parse_param_size(s):
        if isinstance(s, str) and s.endswith('b'):
            try:
                return float(s[:-1])
            except:
                return None
        return None
    if 'param_size' in merged_df.columns:
        merged_df['param_size_num'] = merged_df['param_size'].apply(parse_param_size)
    else:
        merged_df['param_size_num'] = np.nan

    # Log-transform + z-score param size (stable numeric predictor)
    merged_df['param_size_log'] = np.log1p(merged_df['param_size_num'].fillna(0.0))
    p_mean = merged_df['param_size_log'].mean()
    p_std = merged_df['param_size_log'].std()
    if pd.isna(p_std) or p_std == 0:
        merged_df['param_size_z'] = 0.0
    else:
        merged_df['param_size_z'] = (merged_df['param_size_log'] - p_mean) / p_std

    # Keep a small family_dummies like object for compatibility with existing code paths
    family_dummies = merged_df[[c for c in ['is_qwen3', 'is_llama3_1'] if c in merged_df.columns]].copy()

    # Preserve arch_dummies (if needed elsewhere) but we'll primarily use compact binaries above
    arch_dummies = pd.get_dummies(merged_df.get('model_architecture', pd.Series(dtype=str)), prefix='arch')
    merged_df = pd.concat([merged_df, arch_dummies], axis=1)

    # We prefer the compact binary columns for downstream modeling

    # Only keep rows where family is 'llama' or 'qwen3' (as before)
    merged_df = merged_df[merged_df['family'].isin(['llama', 'qwen3'])]

    # For family version analysis, include all Qwen3 models and only Llama models with versions 4, 3.3, or 3.1
    llama_versions = ['4', '3.3', '3.1']
    analysis_type = 'family_version'  # Define the analysis type
    if analysis_type == 'family_version':
        merged_df = merged_df[
            (merged_df['family'] == 'qwen3') |
            ((merged_df['family'] == 'llama') & (merged_df['family_version'].isin(llama_versions)))
        ]

    # --- Normalize average_profit by max positive mean per (game, condition) ---
    if 'average_profit' in merged_df.columns:
        def normalize_avg_profit(group):
            max_mean = group['average_profit'][group['average_profit'] > 0].max()
            if pd.isna(max_mean) or max_mean <= 0:
                return group['average_profit']  # No normalization if no positive values
            return group['average_profit'] / max_mean
        merged_df['average_profit'] = merged_df.groupby(['game', 'condition'], group_keys=False).apply(normalize_avg_profit)

    # Identify Targets and Predictors
    targets = [c for c in pivot_perf.columns if c not in ['game', 'model', 'condition']]
    # Only use numeric predictors and the compact architectural columns
    numeric_predictors = [c for c in pivot_behavior.columns if c not in ['game', 'model', 'condition']]
    arch_numeric = ['is_qwen3', 'is_llama3_1', 'is_moe', 'thinking_enabled', 'param_size_z']
    predictors = numeric_predictors + [c for c in arch_numeric if c in merged_df.columns]

    return merged_df, targets, predictors, family_dummies, pivot_behavior, arch_dummies


def remove_collinear_predictors(df, predictor_list, threshold=0.95):
    """
    FIX 2: Detect and remove perfectly (or near-perfectly) collinear predictors.
    
    Iteratively checks the correlation matrix. When |corr(A, B)| >= threshold,
    drops the predictor with lower variance (retains the more informative one).
    Returns the cleaned list and a log of what was dropped.
    """
    if len(predictor_list) < 2:
        return predictor_list, []
    
    dropped = []
    remaining = list(predictor_list)
    
    while True:
        if len(remaining) < 2:
            break
        corr_matrix = df[remaining].corr().abs()
        
        # Find the first pair exceeding the threshold (upper triangle only)
        found_pair = False
        for i in range(len(remaining)):
            for j in range(i + 1, len(remaining)):
                if corr_matrix.iloc[i, j] >= threshold:
                    # Drop the one with lower variance
                    var_i = df[remaining[i]].var()
                    var_j = df[remaining[j]].var()
                    if var_i <= var_j:
                        to_drop = remaining[i]
                        kept = remaining[j]
                    else:
                        to_drop = remaining[j]
                        kept = remaining[i]
                    
                    dropped.append({
                        'dropped': to_drop,
                        'kept': kept,
                        'correlation': corr_matrix.iloc[i, j]
                    })
                    remaining.remove(to_drop)
                    found_pair = True
                    break  # Restart the scan after dropping
            if found_pair:
                break
        
        if not found_pair:
            break  # No more collinear pairs
    
    return remaining, dropped


# New helper: VIF calculation and high-VIF detector
from statsmodels.stats.outliers_influence import variance_inflation_factor

def calculate_vif(df, features):
    """Return DataFrame with VIF for each feature. Expects df[features] numeric and no-NaNs."""
    X = df[features].astype(float)
    # Add small ridge to avoid singular matrix issues
    X = X.fillna(0.0) + 1e-8
    vif_data = []
    for i, col in enumerate(features):
        try:
            vif = variance_inflation_factor(X.values, i)
        except Exception:
            vif = np.nan
        vif_data.append({'variable': col, 'vif': vif})
    return pd.DataFrame(vif_data)


def get_high_vif(df, features, thresh=10.0):
    """Return list of features with VIF > thresh. Safe on small feature sets."""
    if len(features) < 2:
        return []
    try:
        sub = df[features].dropna()
        if len(sub) < 2:
            return []
        vif_df = calculate_vif(sub, features)
        high = vif_df[vif_df['vif'] > thresh]['variable'].tolist()
        return high
    except Exception:
        return []

# New helper: iterative VIF-based dropping
def iterative_vif_filter(df, features, vif_thresh=10.0, min_features=1, log_prefix=''):
    """Iteratively drop the feature with highest VIF until all VIFs <= vif_thresh or len(features) <= min_features.
    Returns (kept_features, dropped_features_list).
    """
    features = list(features)
    dropped = []
    # If too few features, nothing to do
    while True:
        if len(features) <= min_features:
            break
        try:
            vif_df = calculate_vif(df, features)
        except Exception:
            break
        # If any NaN or infinite VIFs, treat as high and drop the corresponding variable
        vif_df = vif_df.replace([np.inf, -np.inf], np.nan)
        # pick highest VIF
        if vif_df['vif'].isnull().all():
            break
        max_row = vif_df.sort_values('vif', ascending=False).iloc[0]
        if pd.isna(max_row['vif']):
            break
        if max_row['vif'] > vif_thresh:
            var_to_drop = max_row['variable']
            dropped.append({'dropped': var_to_drop, 'vif': float(max_row['vif']), 'reason': f'vif>{vif_thresh}', 'context': log_prefix})
            features.remove(var_to_drop)
        else:
            break
    return features, dropped


# New helper: generate VIF reports
def generate_vif_reports(df, arch_predictors, strat_predictors, outdir):
    """Generate per-game VIF reports for Architectural, Strategic and Both predictor sets.
    Saves per-game/group CSVs to outdir/vif_reports/ and a combined summary.
    """
    reports_dir = os.path.join(outdir, 'vif_reports')
    os.makedirs(reports_dir, exist_ok=True)
    all_reports = []

    games = df['game'].unique()
    groups = [
        ('Architectural', arch_predictors),
        ('Strategic', strat_predictors),
        ('Both', arch_predictors + strat_predictors)
    ]

    for game in games:
        game_df = df[df['game'] == game]
        for group_name, preds in groups:
            # select valid predictors for this game
            valid_preds = [p for p in preds if p in game_df.columns and game_df[p].notna().any() and game_df[p].std() > 0]
            if len(valid_preds) < 1:
                continue
            sub = game_df[valid_preds].dropna()
            if sub.empty:
                continue
            try:
                vif_df = calculate_vif(sub, valid_preds)
            except Exception:
                # fallback: build empty structure
                vif_df = pd.DataFrame([{'variable': v, 'vif': np.nan} for v in valid_preds])
            vif_df['game'] = game
            vif_df['group'] = group_name
            # save per-game-group CSV
            safe_group = re.sub(r"[^0-9a-zA-Z_-]", "_", group_name.lower())
            safe_game = re.sub(r"[^0-9a-zA-Z_-]", "_", str(game))
            out_path = os.path.join(reports_dir, f"vif_{safe_game}_{safe_group}.csv")
            vif_df.to_csv(out_path, index=False)
            all_reports.append(vif_df)

    if all_reports:
        combined = pd.concat(all_reports, ignore_index=True)
        combined_path = os.path.join(reports_dir, 'vif_all_games.csv')
        combined.to_csv(combined_path, index=False)
        print(f"VIF reports written to: {reports_dir} (combined: {combined_path})")
    else:
        print("No VIF reports generated (no valid predictors present per game/group).")


def save_high_vif_summary(reports_dir, thresh=10.0):
    """Load combined VIF report (vif_all_games.csv), filter entries with vif > thresh,
    and save a compact summary to vif_high_per_game.csv.
    """
    combined_path = os.path.join(reports_dir, 'vif_all_games.csv')
    out_path = os.path.join(reports_dir, 'vif_high_per_game.csv')
    if not os.path.exists(combined_path):
        print(f"High-VIF summary: combined VIF file not found: {combined_path}")
        return
    try:
        vdf = pd.read_csv(combined_path)
        # ensure numeric
        vdf['vif'] = pd.to_numeric(vdf['vif'], errors='coerce')
        high = vdf[vdf['vif'] > float(thresh)].copy()
        if high.empty:
            print(f"No predictors with VIF > {thresh} found.")
        else:
            # sort for readability
            high = high.sort_values(['game', 'group', 'vif'], ascending=[True, True, False])
            high.to_csv(out_path, index=False)
            print(f"High-VIF summary written: {out_path}")
    except Exception as e:
        print(f"Failed to write high-VIF summary: {e}")


# --- Analysis 1: Simple Linear Regression (One-to-One) ---
def run_slr(df, targets, predictors):
    results = []
    print("\nRunning Simple Linear Regression (SLR)...")
    
    for game in df['game'].unique():
        game_df = df[df['game'] == game]
        
        for target in targets:
            if target not in game_df.columns:
                continue
            
            for pred in predictors:
                if pred not in game_df.columns:
                    continue
                
                valid = game_df[[pred, target]].dropna()
                if len(valid) < 3:
                    continue
                if valid[pred].std() == 0:
                    continue
                # FIX 1: Skip if target has zero variance
                if valid[target].std() == 0:
                    print(f"  SLR skipped: {game} / {target} has zero variance (n={len(valid)})")
                    continue
                
                slope, intercept, r, p, err = linregress(valid[pred], valid[target])
                
                results.append({
                    'game': game,
                    'type': 'SLR',
                    'target': target,
                    'predictor': pred,
                    'r_squared': r**2,
                    'coef': slope,
                    'p_value': p,
                    'n_obs': len(valid)
                })
    return pd.DataFrame(results)


# --- Analysis 2: Multiple Linear Regression (Many-to-One) ---
def run_mlr(df, targets, predictors):
    results = []
    print("Running Multiple Linear Regression (MLR)...")
    
    for game in df['game'].unique():
        game_df = df[df['game'] == game]
        
        # Identify predictors with variance for this game
        valid_preds = [p for p in predictors if p in game_df.columns and game_df[p].std() > 0]
        
        # FIX 2: Remove collinear predictors before fitting
        valid_preds, collinear_log = remove_collinear_predictors(
            game_df, valid_preds, threshold=COLLINEARITY_THRESHOLD
        )
        if collinear_log:
            for entry in collinear_log:
                print(f"  Collinearity in {game}: dropped '{entry['dropped']}' "
                      f"(corr={entry['correlation']:.4f} with '{entry['kept']}')")

        # Safety net: if still more predictors than degrees of freedom allow
        if len(valid_preds) > len(game_df) - 2:
            print(f"  Warning: Reducing predictors for {game} due to small sample size.")
            variances = game_df[valid_preds].var()
            valid_preds = variances.nlargest(3).index.tolist()

        if not valid_preds:
            continue

        for target in targets:
            if target not in game_df.columns:
                continue
            
            valid = game_df[valid_preds + [target]].dropna()
            if len(valid) < len(valid_preds) + 2:
                continue
            
            # FIX 1: Skip if target has zero variance
            if valid[target].std() == 0:
                print(f"  MLR skipped: {game} / {target} has zero variance (n={len(valid)})")
                continue
            
            X = valid[valid_preds]
            X = sm.add_constant(X)
            Y = valid[target]
            
            try:
                model = sm.OLS(Y, X).fit()
                
                for pred in valid_preds:
                    results.append({
                        'game': game,
                        'type': 'MLR_Combined',
                        'target': target,
                        'predictor': pred,
                        'r_squared': model.rsquared,
                        'r_squared_adj': model.rsquared_adj,
                        'coef': model.params.get(pred, 0),
                        'p_value': model.pvalues.get(pred, 1.0),
                        'n_obs': int(model.nobs)
                    })
            except Exception as e:
                print(f"  MLR Failed for {game} - {target}: {str(e)}")
                
    return pd.DataFrame(results)


# --- Predictor Grouping ---
# Define which predictors are architectural vs strategic
ARCHITECTURAL_FEATURES = [
    'param_count', 'version', 'thinking', 'dense_vs_moe'
]
# Strategic features: all others from behavioral metrics

def split_predictors(predictors):
    arch = [p for p in predictors if p in ARCHITECTURAL_FEATURES]
    strat = [p for p in predictors if p not in ARCHITECTURAL_FEATURES]
    return arch, strat


# --- Per-game, per-target, per-group regression ---
def run_grouped_regressions(df, targets, predictors):
    results = []
    for game in df['game'].unique():
        game_df = df[df['game'] == game]
        arch_preds, strat_preds = split_predictors(predictors)
        groups = [
            ('Architectural', arch_preds),
            ('Strategic', strat_preds),
            ('Both', arch_preds + strat_preds)
        ]
        for target in targets:
            if target not in game_df.columns:
                continue
            for group_name, group_preds in groups:
                # Only use predictors present and non-all-NaN for this game
                valid_preds = [p for p in group_preds if p in game_df.columns and game_df[p].notna().any() and game_df[p].std() > 0]
                if not valid_preds:
                    continue
                # For avg_profit, do NOT drop collinear predictors
                if target == 'average_profit':
                    preds_to_use = valid_preds
                else:
                    preds_to_use, _ = remove_collinear_predictors(game_df, valid_preds, threshold=COLLINEARITY_THRESHOLD)
                if not preds_to_use:
                    continue
                valid = game_df[preds_to_use + [target]].dropna()
                # Revert: Use len(preds_to_use) + 2 for all targets
                if len(valid) < len(preds_to_use) + 2:
                    continue
                if valid[target].std() == 0:
                    continue
                X = valid[preds_to_use]
                X = sm.add_constant(X)
                Y = valid[target]
                try:
                    model = sm.OLS(Y, X).fit()
                    for pred in preds_to_use:
                        results.append({
                            'game': game,
                            'target': target,
                            'group': group_name,
                            'predictor': pred,
                            'coef': model.params.get(pred, 0),
                            'r_squared_adj': model.rsquared_adj,
                            'n_obs': int(model.nobs)
                        })
                except Exception as e:
                    print(f"  Grouped MLR Failed for {game} - {target} [{group_name}]: {str(e)}")
    return pd.DataFrame(results)


# --- LaTeX Table Generation ---

def _get_from_row(row, candidates):
    """Safely extract a value from a pandas row-like object (DataFrame slice or Series).
    Tries each candidate key in order and returns first non-None / non-NaN value.

    This version is more defensive: it handles DataFrame (multi-row) slices by
    searching down the column for the first non-null value, Series objects,
    dict-like objects, and plain scalars.
    """
    if row is None:
        return None
    try:
        # DataFrame with potential multiple rows
        if hasattr(row, 'columns') and hasattr(row, 'iloc'):
            if getattr(row, 'empty', False):
                return None
            for key in candidates:
                if key in row.columns:
                    col = row[key].dropna()
                    if not col.empty:
                        return col.iloc[0]
            return None
        # Series-like
        if hasattr(row, 'index'):
            for key in candidates:
                try:
                    if key in row.index:
                        val = row.get(key, None)
                        if val is not None and not (isinstance(val, float) and np.isnan(val)):
                            return val
                except Exception:
                    continue
        # dict-like
        try:
            for key in candidates:
                if isinstance(row, dict) and key in row:
                    val = row[key]
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        return val
        except Exception:
            pass
        # fallback: if it's a scalar, return it if candidates include a generic key
        return row
    except Exception:
        return None

# Convenience wrappers
def get_coef(row):
    return _get_from_row(row, ['coef', 'coefficient', 'beta'])

def get_pvalue(row):
    return _get_from_row(row, ['p_value', 'pval', 'p'])

def get_adj_r2(row):
    return _get_from_row(row, ['r_squared_adj', 'adj_r2', 'adjR2', 'adj_r_squared', 'adjR_squared', 'adj_r2', 'adjR', 'r2_adj'])

def generate_latex_regression_table(metric, latex_path, df):
    games = ['athey_bagwell', 'green_porter', 'salop', 'spulber']
    game_names = ['Athey-Bagwell', 'Green-Porter', 'Salop', 'Spulber']
    arch_predictors = [
        'is_qwen3', 'param_size_z', 'is_moe', 'is_llama3_1'
    ]
    strat_predictors = [
        'cooperation', 'deception', 'rationality', 'reasoning', 'coordination', 'judgment', 'self_awareness'
    ]
    def pval_stars(p):
        if p is None:
            return ''
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    header = f"% filepath: {latex_path}\n"
    header += "\\begin{table}[ht]\n"
    header += "\\centering\n"
    header += "\\small\n"
    header += "\\begin{tabular}{llccc}\n"
    header += "\\toprule\n"
    header += "Game & Predictor & Arch. (Coef, $R^2_{adj}$) & Strat. (Coef, $R^2_{adj}$) & Both (Coef, $R^2_{adj}$) \\ \n"
    header += "\\midrule\n"
    rows = []
    for game in games:
        rows.append("\\multicolumn{5}{l}{\\textbf{Architectural Features}} \\ \n")
        for predictor in arch_predictors:
            arch = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Architectural')]
            strat = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Strategic')]
            both = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Both')]
            def cell(row):
                if row.empty:
                    return ''
                coef = get_coef(row)
                pval = get_pvalue(row)
                stars = pval_stars(pval)
                adj_r2 = get_adj_r2(row)
                coef_fmt = f"{coef:.3f}{stars}" if coef is not None else ""
                adj_r2_fmt = f"{adj_r2:.2f}" if adj_r2 is not None and not pd.isna(adj_r2) else ""
                if coef_fmt and adj_r2_fmt:
                    return f"{coef_fmt}, {adj_r2_fmt}"
                elif coef_fmt:
                    return f"{coef_fmt}"
                elif adj_r2_fmt:
                    return f", {adj_r2_fmt}"
                else:
                    return ''
            rows.append(f"{game} & \texttt{{{predictor}}} & {cell(arch)} & {cell(strat)} & {cell(both)} \\ \n")
        rows.append("\\multicolumn{5}{l}{\\textbf{Strategic Capabilities}} \\ \n")
        for predictor in strat_predictors:
            arch = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Architectural')]
            strat = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Strategic')]
            both = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Both')]
            rows.append(f"{game} & \texttt{{{predictor}}} & {cell(arch)} & {cell(strat)} & {cell(both)} \\ \n")
        rows.append("\\midrule\n")
    footer = "\\bottomrule\n\\end{tabular}\n"
    footer += f"\\caption{{Regression summary for {metric.replace('_', ' ')} by game.}}\n"
    footer += f"\\label{{tab:regression_summary_{metric}_by_game}}\n"
    footer += "\\end{table}\n"
    with open(latex_path, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(row)
        f.write(footer)


def generate_latex_tables_from_csv(csv_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(csv_path)
    targets = df['target'].unique()
    games = df['game'].unique()
    groups = ['Architectural', 'Strategic', 'Both']
    arch_predictors = [
        'is_qwen3', 'param_size_z', 'is_moe', 'is_llama3_1'
    ]
    strat_predictors = [
        'cooperation', 'deception', 'rationality', 'reasoning', 'coordination', 'judgment', 'self_awareness'
    ]
    def pval_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    for target in targets:
        table_path = os.path.join(outdir, f"regression_summary_{target}_by_game.tex")
        header = f"% filepath: {table_path}\n"
        header += "\\begin{table}[ht]\n"
        header += "\\centering\n"
        header += "\\small\n"
        header += "\\begin{tabular}{llccc}\n"
        header += "\\toprule\n"
        header += "Game & Predictor & Arch. (Coef, $R^2_{adj}$) & Strat. (Coef, $R^2_{adj}$) & Both (Coef, $R^2_{adj}$) \\ \n"
        header += "\\midrule\n"
        rows = []
        for game in games:
            game_df = df[(df['game'] == game) & (df['target'] == target)]
            # Architectural Features
            rows.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{Architectural Features}}}} \\ \n")
            arch_rows = []
            for predictor in arch_predictors:
                arch = game_df[(game_df['predictor'] == predictor) & (game_df['group'] == 'Architectural')]
                strat = game_df[(game_df['predictor'] == predictor) & (game_df['group'] == 'Strategic')]
                both = game_df[(game_df['predictor'] == predictor) & (game_df['group'] == 'Both')]
                def cell(row):
                    if row.empty:
                        return ''
                    coef = get_coef(row)
                    pval = get_pvalue(row)
                    stars = pval_stars(pval) if pval is not None else ''
                    adj_r2 = get_adj_r2(row)
                    coef_fmt = f"{coef:.3f}" if coef is not None else ""
                    adj_r2_fmt = f"{adj_r2:.2f}" if adj_r2 is not None and not pd.isna(adj_r2) else ""
                    if coef_fmt and adj_r2_fmt:
                        return f"{coef_fmt}{stars}, {adj_r2_fmt}"
                    elif coef_fmt:
                        return f"{coef_fmt}{stars}"
                    elif adj_r2_fmt:
                        return f", {adj_r2_fmt}"
                    else:
                        return ''
                arch_rows.append(f"{game} & {predictor} & {cell(arch)} & {cell(strat)} & {cell(both)} \\ \n")
            if not any([r for r in arch_rows if r.strip() and r.count('&') > 1]):
                rows.append("% No architectural features present for this game\n")
            else:
                rows.extend(arch_rows)
            # Strategic Capabilities
            rows.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{Strategic Capabilities}}}} \\ \n")
            strat_rows = []
            for predictor in strat_predictors:
                arch = game_df[(game_df['predictor'] == predictor) & (game_df['group'] == 'Architectural')]
                strat = game_df[(game_df['predictor'] == predictor) & (game_df['group'] == 'Strategic')]
                both = game_df[(game_df['predictor'] == predictor) & (game_df['group'] == 'Both')]
                strat_rows.append(f"{game} & {predictor} & {cell(arch)} & {cell(strat)} & {cell(both)} \\ \n")
            if not any([r for r in strat_rows if r.strip() and r.count('&') > 1]):
                rows.append("% No strategic capabilities present for this game\n")
            else:
                rows.extend(strat_rows)
        footer = "\\bottomrule\n\\end{tabular}\n"
        footer += f"\\caption{{Regression summary for {target.replace('_', ' ')} by game.}}\n"
        footer += f"\\label{{tab:regression_summary_{target}_by_game}}\n"
        footer += "\\end{table}\n"
        with open(table_path, 'w') as f:
            f.write(header)
            for row in rows:
                f.write(row)
            f.write(footer)
        print(f"LaTeX table written: {table_path}")


def make_regression_table(metric, latex_path, df):
    games = ['athey_bagwell', 'green_porter', 'salop', 'spulber']
    arch_predictors = [
        'is_qwen3', 'param_size_z', 'is_moe', 'is_llama3_1'
    ]
    strat_predictors = [
        'cooperation', 'deception', 'rationality', 'reasoning', 'coordination', 'judgment', 'self_awareness'
    ]
    def pval_stars(p):
        if p is None:
            return ''
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''
    header = f"% filepath: {latex_path}\n"
    header += "\\begin{table}[ht]\n"
    header += "\\centering\n"
    header += "\\small\n"
    header += "\\begin{tabular}{llccc}\n"
    header += "\\toprule\n"
    header += "Game & Predictor & Arch. (Coef, $R^2_{adj}$) & Strat. (Coef, $R^2_{adj}$) & Both (Coef, $R^2_{adj}$) \\" + "\n"
    header += "\\midrule\n"
    rows = []
    for game in games:
        rows.append("\\multicolumn{5}{l}{\\textbf{Architectural Features}} \\" + "\n")
        for predictor in arch_predictors:
            arch = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Architectural')]
            strat = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Strategic')]
            both = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Both')]
            def cell(row):
                if row.empty:
                    return ''
                coef = get_coef(row)
                pval = get_pvalue(row)
                stars = pval_stars(pval)
                adj_r2 = get_adj_r2(row)
                coef_fmt = f"{coef:.3f}{stars}" if coef is not None else ""
                adj_r2_fmt = f"{adj_r2:.2f}" if adj_r2 is not None and not pd.isna(adj_r2) else ""
                if coef_fmt and adj_r2_fmt:
                    return f"{coef_fmt}, {adj_r2_fmt}"
                elif coef_fmt:
                    return f"{coef_fmt}"
                elif adj_r2_fmt:
                    return f", {adj_r2_fmt}"
                else:
                    return ''
            row_str = f"{game} & \\texttt{{{predictor}}} & {cell(arch)} & {cell(strat)} & {cell(both)} \\" + "\n"
            rows.append(row_str)
        rows.append("\\multicolumn{5}{l}{\\textbf{Strategic Capabilities}} \\" + "\n")
        for predictor in strat_predictors:
            arch = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Architectural')]
            strat = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Strategic')]
            both = df[(df['game'] == game) & (df['target'] == metric) & (df['predictor'] == predictor) & (df['group'] == 'Both')]
            row_str = f"{game} & \\texttt{{{predictor}}} & {cell(arch)} & {cell(strat)} & {cell(both)} \\" + "\n"
            rows.append(row_str)
        rows.append("\\midrule\n")
    footer = "\\bottomrule\n\\end{tabular}\n"
    footer += f"\\caption{{Regression summary for {metric.replace('_', ' ')} by game.}}\n"
    footer += f"\\label{{tab:regression_summary_{metric}_by_game}}\n"
    footer += "\\end{table}\n"
    with open(latex_path, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(row)
        f.write(footer)


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df, targets, predictors, family_dummies, pivot_behavior, arch_dummies = load_and_merge_data()
    if df is None:
        return

    # --- Predictor Grouping ---
    # Architectural features: use compact binary features created earlier
    arch_list = ['is_qwen3', 'is_llama3_1', 'is_moe', 'thinking_enabled', 'param_size_z']
    arch_predictors = [c for c in arch_list if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    # Strategic features: all behavioral metrics from magic_behavioral_metrics.csv (numeric)
    strat_predictors = [c for c in pivot_behavior.columns if c not in ['game', 'model', 'condition'] and pd.api.types.is_numeric_dtype(df[c])]

    # Generate per-game VIF reports so we can inspect collinearity before modeling
    try:
        generate_vif_reports(df, arch_predictors, strat_predictors, OUTPUT_DIR)
        # produce high-VIF summary (default threshold = 10)
        reports_dir = os.path.join(OUTPUT_DIR, 'vif_reports')
        save_high_vif_summary(reports_dir, thresh=10.0)
    except Exception as e:
        print(f"Failed to generate VIF reports: {e}")

    results = []
    for game in df['game'].unique():
        for target in ['average_profit', 'win_rate']:
            game_df = df[df['game'] == game]
            # Architectural only
            valid_arch = [p for p in arch_predictors if p in game_df.columns and game_df[p].std() > 0]
            arch_preds, arch_collinear_log = remove_collinear_predictors(game_df, valid_arch, threshold=COLLINEARITY_THRESHOLD)
            if arch_collinear_log:
                for entry in arch_collinear_log:
                    print(f"  Collinearity in {game} {target} [Architectural]: dropped '{entry['dropped']}' (corr={entry['correlation']:.4f} with '{entry['kept']}')")
            # Iterative VIF-based dropping (prefer VIF over one-shot removal)
            arch_preds, arch_vif_dropped = iterative_vif_filter(game_df, arch_preds, vif_thresh=10.0, min_features=1, log_prefix=f"{game} {target} [Architectural]")
            if arch_vif_dropped:
                for e in arch_vif_dropped:
                    print(f"  High VIF in {e['context']}: dropped '{e['dropped']}' (VIF={e['vif']:.2f})")
            if len(arch_preds) > 0:
                valid = game_df[arch_preds + [target]].dropna()
                valid = valid.astype(float)
                if len(valid) >= len(arch_preds) + 2 and valid[target].std() > 0:
                    X = sm.add_constant(valid[arch_preds])
                    Y = valid[target]
                    try:
                        model = sm.OLS(Y, X).fit()
                        for pred in arch_preds:
                            results.append({
                                'game': game,
                                'type': 'MLR',
                                'target': target,
                                'predictor': pred,
                                'r_squared': model.rsquared,
                                'coef': model.params.get(pred, 0),
                                'p_value': model.pvalues.get(pred, 1.0),
                                'n_obs': int(model.nobs),
                                'r_squared_adj': model.rsquared_adj,
                                'group': 'Architectural'
                            })
                    except Exception as e:
                        print(f"MLR failed for {game} {target} Architectural: {e}")
            # Strategic only
            valid_strat = [p for p in strat_predictors if p in game_df.columns and game_df[p].std() > 0]
            strat_preds, strat_collinear_log = remove_collinear_predictors(game_df, valid_strat, threshold=COLLINEARITY_THRESHOLD)
            if strat_collinear_log:
                for entry in strat_collinear_log:
                    print(f"  Collinearity in {game} {target} [Strategic]: dropped '{entry['dropped']}' (corr={entry['correlation']:.4f} with '{entry['kept']}')")
            strat_preds, strat_vif_dropped = iterative_vif_filter(game_df, strat_preds, vif_thresh=10.0, min_features=1, log_prefix=f"{game} {target} [Strategic]")
            if strat_vif_dropped:
                for e in strat_vif_dropped:
                    print(f"  High VIF in {e['context']}: dropped '{e['dropped']}' (VIF={e['vif']:.2f})")
            if strat_preds:
                valid = game_df[strat_preds + [target]].dropna()
                valid = valid.astype(float)
                if len(valid) >= len(strat_preds) + 2 and valid[target].std() > 0:
                    X = sm.add_constant(valid[strat_preds])
                    Y = valid[target]
                    try:
                        model = sm.OLS(Y, X).fit()
                        for pred in strat_preds:
                            results.append({
                                'game': game,
                                'type': 'MLR',
                                'target': target,
                                'predictor': pred,
                                'r_squared': model.rsquared,
                                'coef': model.params.get(pred, 0),
                                'p_value': model.pvalues.get(pred, 1.0),
                                'n_obs': int(model.nobs),
                                'r_squared_adj': model.rsquared_adj,
                                'group': 'Strategic'
                            })
                    except Exception as e:
                        print(f"MLR failed for {game} {target} Strategic: {e}")
            # Both
            valid_both = [p for p in arch_predictors + strat_predictors if p in game_df.columns and game_df[p].std() > 0]
            both_preds, both_collinear_log = remove_collinear_predictors(game_df, valid_both, threshold=COLLINEARITY_THRESHOLD)
            if both_collinear_log:
                for entry in both_collinear_log:
                    print(f"  Collinearity in {game} {target} [Both]: dropped '{entry['dropped']}' (corr={entry['correlation']:.4f} with '{entry['kept']}')")
            both_preds, both_vif_dropped = iterative_vif_filter(game_df, both_preds, vif_thresh=10.0, min_features=1, log_prefix=f"{game} {target} [Both]")
            if both_vif_dropped:
                for e in both_vif_dropped:
                    print(f"  High VIF in {e['context']}: dropped '{e['dropped']}' (VIF={e['vif']:.2f})")
            if both_preds:
                valid = game_df[both_preds + [target]].dropna()
                valid = valid.astype(float)
                if len(valid) >= len(both_preds) + 2 and valid[target].std() > 0:
                    X = sm.add_constant(valid[both_preds])
                    Y = valid[target]
                    try:
                        model = sm.OLS(Y, X).fit()
                        for pred in both_preds:
                            results.append({
                                'game': game,
                                'type': 'MLR',
                                'target': target,
                                'predictor': pred,
                                'r_squared': model.rsquared,
                                'coef': model.params.get(pred, 0),
                                'p_value': model.pvalues.get(pred, 1.0),
                                'n_obs': int(model.nobs),
                                'r_squared_adj': model.rsquared_adj,
                                'group': 'Both'
                            })
                    except Exception as e:
                        print(f"MLR failed for {game} {target} Both: {e}")
    # Save to CSV
    out_path = os.path.join(OUTPUT_DIR, 'regressions.csv')
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"\nSuccess. Results saved to: {out_path}")

    # Generate LaTeX tables for grouped regressions
    latex_dir = os.path.join('metrics', 'latex_tables')
    for metric in ['average_profit', 'win_rate']:
        latex_path = os.path.join(latex_dir, f"regression_summary_{metric}_by_game.tex")
        make_regression_table(metric, latex_path, pd.DataFrame(results))

    # Generate LaTeX tables from regressions.csv
    generate_latex_tables_from_csv(os.path.join(OUTPUT_DIR, 'regressions.csv'), latex_dir)

if __name__ == "__main__":
    main()