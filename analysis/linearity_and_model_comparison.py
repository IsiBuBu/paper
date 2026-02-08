import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os

# --- Configuration ---
MERGED_DATA_PATH = 'output/analysis/performance_metrics.csv'
BEHAVIOR_DATA_PATH = 'output/analysis/magic_behavioral_metrics.csv'
PLOT_DIR = 'output/analysis/linearity_plots'
os.makedirs(PLOT_DIR, exist_ok=True)

# --- Load Data ---
df_perf = pd.read_csv(MERGED_DATA_PATH)
df_behavior = pd.read_csv(BEHAVIOR_DATA_PATH)

# Merge on game, model, condition
merged = pd.merge(
    df_behavior.pivot_table(index=['game','model','condition'], columns='metric', values='mean').reset_index(),
    df_perf.pivot_table(index=['game','model','condition'], columns='metric', values='mean').reset_index(),
    on=['game','model','condition'], how='inner'
)

print('Merged DataFrame shape:', merged.shape)
print('Merged DataFrame columns:', merged.columns.tolist())
print('Sample merged data:')
print(merged.head())

# Architectural features
arch_features = ['param_count', 'version', 'thinking', 'dense_vs_moe']
# Extract architectural features from model names
import re
def extract_arch_features(model_name):
    m = str(model_name).lower()
    param_match = re.search(r'(\d+\.?\d*)b', m)
    param_count = float(param_match.group(1)) if param_match else 0
    version_match = re.search(r'qwen(\d+)|llama-?(\d+\.?\d*)', m)
    version = float(version_match.group(1) or version_match.group(2)) if version_match else 0
    thinking = 1 if 'thinking' in m and 'off' not in m else 0
    dense_vs_moe = 1 if any(x in m for x in ['moe','maverick','scout']) else 0
    return {'param_count': param_count, 'version': version, 'thinking': thinking, 'dense_vs_moe': dense_vs_moe}
merged = merged.copy()
for f in arch_features:
    merged[f] = [extract_arch_features(m)[f] for m in merged['model']]

# Strategic features
strat_features = [c for c in df_behavior['metric'].unique() if c not in arch_features]

# Targets
targets = [c for c in df_perf['metric'].unique() if c in merged.columns]

# --- Linearity & Model Comparison ---
for game in merged['game'].unique():
    game_df = merged[merged['game'] == game]
    for target in targets:
        if target not in game_df.columns:
            print(f"Skipping target {target} for game {game}: not in columns.")
            continue
        # Only use predictors present and not all-NaN
        predictors = [p for p in arch_features + strat_features if p in game_df.columns and game_df[p].notna().any()]
        print(f"Game: {game}, Target: {target}, Predictors: {predictors}")
        valid = game_df[predictors + [target]].dropna()
        print(f"Game: {game}, Target: {target}, Valid rows: {len(valid)}")
        if len(valid) < 3:
            print(f"Skipping {game} / {target}: not enough valid rows ({len(valid)})")
            continue
        X = valid[predictors]
        y = valid[target]
        # Scatterplots
        for pred in predictors:
            plt.figure()
            plt.scatter(X[pred], y)
            plt.title(f'{game} - {target} vs {pred}')
            plt.xlabel(pred)
            plt.ylabel(target)
            plt.savefig(f'{PLOT_DIR}/{game}_{target}_{pred}_scatter.png')
            plt.close()
        # Linear regression residuals
        lr = LinearRegression().fit(X, y)
        y_pred = lr.predict(X)
        residuals = y - y_pred
        plt.figure()
        plt.scatter(y_pred, residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.title(f'{game} - {target} Linear Residuals')
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.savefig(f'{PLOT_DIR}/{game}_{target}_linear_residuals.png')
        plt.close()
        # Model comparison
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_r2 = rf.score(X, y)
        lr_r2 = lr.score(X, y)
        print(f'{game} / {target}: Linear R²={lr_r2:.3f}, RF R²={rf_r2:.3f}')
print(f'Plots saved to {PLOT_DIR}. Check printed R² scores for model comparison.')
