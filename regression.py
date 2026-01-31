import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
import os

# --- Configuration ---
BEHAVIOR_FILE = 'output/analysis/magic_behavioral_metrics.csv'
PERFORMANCE_FILE = 'output/analysis/performance_metrics.csv'
OUTPUT_DIR = 'output/analysis'
COLLINEARITY_THRESHOLD = 0.95  # Drop one predictor if |corr| >= this

# --- Helper Functions ---

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
    
    # Identify Targets and Predictors
    targets = [c for c in pivot_perf.columns if c not in ['game', 'model', 'condition']]
    predictors = [c for c in pivot_behavior.columns if c not in ['game', 'model', 'condition']]
    
    return merged_df, targets, predictors


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


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df, targets, predictors = load_and_merge_data()
    if df is None:
        return

    # Run Analyses
    df_slr = run_slr(df, targets, predictors)
    df_mlr = run_mlr(df, targets, predictors)
    
    # Combine and Save
    final_df = pd.concat([df_slr, df_mlr], ignore_index=True)
    
    out_path = os.path.join(OUTPUT_DIR, 'influence_magic_to_performance.csv')
    final_df.to_csv(out_path, index=False)
    print(f"\nSuccess. Results saved to: {out_path}")
    
    if not final_df.empty:
        print(final_df.head())
    else:
        print("Warning: No regression results generated. Check input data sufficiency.")


if __name__ == "__main__":
    main()