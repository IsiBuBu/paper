import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import linregress
import os

# --- Configuration ---
BEHAVIOR_FILE = 'output/analysis/magic_behavioral_metrics.csv'
PERFORMANCE_FILE = 'output/analysis/performance_metrics.csv'
OUTPUT_DIR = 'output/analysis'

# --- Helper Functions ---

def load_and_merge_data():
    if not os.path.exists(BEHAVIOR_FILE) or not os.path.exists(PERFORMANCE_FILE):
        print("Error: Input files not found.")
        return None, None

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
    
    # Identify Targets (from perf file) and Predictors (from behavior file)
    targets = [c for c in pivot_perf.columns if c not in ['game', 'model', 'condition']]
    predictors = [c for c in pivot_behavior.columns if c not in ['game', 'model', 'condition']]
    
    return merged_df, targets, predictors

# --- Analysis 1: Simple Linear Regression (One-to-One) ---
def run_slr(df, targets, predictors):
    results = []
    print("\nRunning Simple Linear Regression (SLR)...")
    
    for game in df['game'].unique():
        game_df = df[df['game'] == game]
        
        for target in targets:
            if target not in game_df.columns: continue
            
            for pred in predictors:
                if pred not in game_df.columns: continue
                
                valid = game_df[[pred, target]].dropna()
                if len(valid) < 3 or valid[pred].std() == 0: continue
                
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
        
        # Identify valid predictors for this specific game (columns with variance)
        valid_preds = [p for p in predictors if p in game_df.columns and game_df[p].std() > 0]
        
        # Constraint: Need more rows than predictors + 1
        # If N is small, we select only the top 3 predictors based on variance to avoid overfitting
        if len(valid_preds) > len(game_df) - 2:
            print(f"  Warning: Reducing predictors for {game} due to small sample size.")
            valid_preds = valid_preds[:3] 

        if not valid_preds: continue

        for target in targets:
            if target not in game_df.columns: continue
            
            valid = game_df[valid_preds + [target]].dropna()
            if len(valid) < len(valid_preds) + 2: continue
            
            X = valid[valid_preds]
            X = sm.add_constant(X) # Add Intercept
            Y = valid[target]
            
            try:
                model = sm.OLS(Y, X).fit()
                
                # We save one row per predictor to show its specific contribution (coef) in the context of the model
                for pred in valid_preds:
                    results.append({
                        'game': game,
                        'type': 'MLR_Combined',
                        'target': target,
                        'predictor': pred,
                        'r_squared_adj': model.rsquared_adj, # Adjusted R2 is better for MLR
                        'coef': model.params.get(pred, 0),
                        'p_value': model.pvalues.get(pred, 1.0),
                        'n_obs': int(model.nobs)
                    })
            except Exception as e:
                print(f"  MLR Failed for {game} - {target}: {str(e)}")
                
    return pd.DataFrame(results)

def main():
    df, targets, predictors = load_and_merge_data()
    if df is None: return

    # Run Analyses
    df_slr = run_slr(df, targets, predictors)
    df_mlr = run_mlr(df, targets, predictors)
    
    # Combine and Save
    final_df = pd.concat([df_slr, df_mlr], ignore_index=True)
    
    out_path = os.path.join(OUTPUT_DIR, 'influence_magic_to_performance.csv')
    final_df.to_csv(out_path, index=False)
    print(f"\nSuccess. Results saved to: {out_path}")
    print(final_df.head())

if __name__ == "__main__":
    main()