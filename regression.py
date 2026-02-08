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
COLLINEARITY_THRESHOLD = 0.90  # Drop one predictor if |corr| >= this

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
    # Drop rows with missing family (only keep Qwen3/Llama models)
    merged_df = merged_df.dropna(subset=['family'])
    # One-hot encode family
    family_dummies = pd.get_dummies(merged_df['family'], prefix='family')
    merged_df = pd.concat([merged_df, family_dummies], axis=1)

    # Convert param_size to numeric (e.g., '70b' -> 70, '235b' -> 235)
    def parse_param_size(s):
        if isinstance(s, str) and s.endswith('b'):
            try:
                return float(s[:-1])
            except:
                return None
        return None
    merged_df['param_size_num'] = merged_df['param_size'].apply(parse_param_size)

    # One-hot encode model_architecture
    arch_dummies = pd.get_dummies(merged_df['model_architecture'], prefix='arch')
    merged_df = pd.concat([merged_df, arch_dummies], axis=1)

    # One-hot encode thinking_enabled (True/False)
    if 'thinking_enabled' in merged_df.columns:
        thinking_dummies = pd.get_dummies(merged_df['thinking_enabled'], prefix='thinking_enabled')
        merged_df = pd.concat([merged_df, thinking_dummies], axis=1)
    else:
        thinking_dummies = pd.DataFrame()

    # One-hot encode family_version (for llama models)
    if 'family_version' in merged_df.columns:
        version_dummies = pd.get_dummies(merged_df['family_version'], prefix='family_version')
        merged_df = pd.concat([merged_df, version_dummies], axis=1)
    else:
        version_dummies = pd.DataFrame()

    # Only keep rows where family is 'llama' or 'qwen3'
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
    # Only use numeric predictors and one-hot encoded columns
    numeric_predictors = [c for c in pivot_behavior.columns if c not in ['game', 'model', 'condition']]
    arch_numeric = [c for c in family_dummies.columns] + list(thinking_dummies.columns) + list(version_dummies.columns)
    predictors = numeric_predictors + arch_numeric

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
def generate_latex_regression_table(metric, latex_path, df):
    games = ['athey_bagwell', 'green_porter', 'salop', 'spulber']
    game_names = ['Athey-Bagwell', 'Green-Porter', 'Salop', 'Spulber']
    arch_predictors = [
        'family_qwen3', 'param_size_num', 'arch_dense', 'arch_moe', 'thinking_enabled_False', 'thinking_enabled_True',
        'family_version_3.1', 'family_version_3.3', 'family_version_4'
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
                coef = row.iloc[0]['coef']
                pval = row.iloc[0]['p_value'] if 'p_value' in row.iloc[0] else row.iloc[0].get('pval', None)
                stars = pval_stars(pval)
                adj_r2 = row.iloc[0]['r_squared_adj'] if 'r_squared_adj' in row.iloc[0] else row.iloc[0].get('adj_r2', None)
                coef_fmt = f"{coef:.3f}{stars}" if coef is not None else ""
                adj_r2_fmt = f"{adj_r2:.2f}" if adj_r2 is not None else ""
                return f"{coef_fmt}, {adj_r2_fmt}" if coef_fmt else ''
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
        'family_qwen3', 'param_size_num', 'arch_dense', 'arch_moe', 'thinking_enabled_False', 'thinking_enabled_True',
        'family_version_3.1', 'family_version_3.3', 'family_version_4'
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
                    coef = row.iloc[0]['coef'] if 'coef' in row.iloc[0] else row.iloc[0].get('coefficient', None)
                    pval = row.iloc[0]['pval'] if 'pval' in row.iloc[0] else row.iloc[0].get('p_value', None)
                    stars = pval_stars(pval) if pval is not None else ''
                    adj_r2 = row.iloc[0]['adj_r2'] if 'adj_r2' in row.iloc[0] else row.iloc[0].get('adj_r2', None)
                    coef_fmt = f"{coef:.3f}" if coef is not None else ""
                    adj_r2_fmt = f"{adj_r2:.2f}" if adj_r2 is not None else ""
                    return f"{coef_fmt}{stars}, {adj_r2_fmt}" if coef_fmt else ''
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
        'family_qwen3', 'param_size_num', 'arch_dense', 'arch_moe', 'thinking_enabled_False', 'thinking_enabled_True',
        'family_version_3.1', 'family_version_3.3', 'family_version_4'
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
                coef = row.iloc[0]['coef'] if 'coef' in row.iloc[0] else row.iloc[0].get('coef', None)
                pval = row.iloc[0]['p_value'] if 'p_value' in row.iloc[0] else row.iloc[0].get('pval', None)
                stars = pval_stars(pval)
                # Always use r_squared_adj from regressions.csv
                adj_r2 = row.iloc[0]['r_squared_adj'] if 'r_squared_adj' in row.iloc[0] else row.iloc[0].get('r_squared_adj', None)
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
    # Architectural features: one-hot encoded family columns, param_size_num, model_architecture dummies, and thinking_enabled dummies
    arch_predictors = [c for c in df.columns if (
        c.startswith('family_') or c == 'param_size_num' or c.startswith('arch_') or c.startswith('thinking_enabled_') or c.startswith('family_version_'))
        and pd.api.types.is_numeric_dtype(df[c])]
    # Strategic features: all behavioral metrics from magic_behavioral_metrics.csv (numeric)
    strat_predictors = [c for c in pivot_behavior.columns if c not in ['game', 'model', 'condition'] and pd.api.types.is_numeric_dtype(df[c])]

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