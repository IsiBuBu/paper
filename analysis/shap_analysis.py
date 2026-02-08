import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap
import os

# --- Configuration ---
MERGED_DATA_PATH = 'output/analysis/performance_metrics.csv'
BEHAVIOR_DATA_PATH = 'output/analysis/magic_behavioral_metrics.csv'
SHAP_OUTPUT_DIR = 'output/analysis/shap_analysis'
os.makedirs(SHAP_OUTPUT_DIR, exist_ok=True)

# --- Load Data ---
df_perf = pd.read_csv(MERGED_DATA_PATH)
df_behavior = pd.read_csv(BEHAVIOR_DATA_PATH)

# Merge on game, model, condition
merged = pd.merge(
    df_behavior.pivot_table(index=['game','model','condition'], columns='metric', values='mean').reset_index(),
    df_perf.pivot_table(index=['game','model','condition'], columns='metric', values='mean').reset_index(),
    on=['game','model','condition'], how='inner'
)

# Architectural features
arch_features = ['param_count', 'version', 'thinking', 'dense_vs_moe']
import re

def extract_arch_features(model_name):
    m = str(model_name)
    # Explicit mapping for Qwen3 and Llama families
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
    # Qwen3
    if m in qwen3_models:
        d = qwen3_models[m]
        return {
            'param_count': float(d['param_size'].replace('b','')),
            'version': None,
            'thinking': int(d['thinking_enabled']),
            'dense_vs_moe': 0 if d['model_architecture'] == 'dense' else 1,
            'model_family': d['family']
        }
    # Llama
    if m in llama_models:
        d = llama_models[m]
        return {
            'param_count': float(d['param_size'].replace('b','')),
            'version': float(d['family_version']) if d['family_version'] else None,
            'thinking': 0,
            'dense_vs_moe': 0 if d['model_architecture'] == 'dense' else 1,
            'model_family': d['family']
        }
    # Fallback for other models
    m_lower = m.lower()
    param_match = re.search(r'(\d+\.?\d*)b', m_lower)
    param_count = float(param_match.group(1)) if param_match else 0
    version_match = re.search(r'qwen(\d+)|llama-?(\d+\.?\d*)', m_lower)
    version = float(version_match.group(1) or version_match.group(2)) if version_match else 0
    thinking = 1 if 'thinking' in m_lower and 'off' not in m_lower else 0
    dense_vs_moe = 1 if any(x in m_lower for x in ['moe','maverick','scout']) else 0
    if 'llama' in m_lower:
        model_family = 'llama'
    elif 'qwen' in m_lower:
        model_family = 'qwen3'
    elif 'mistral' in m_lower:
        model_family = 'mistral'
    elif 'gemma' in m_lower:
        model_family = 'gemma'
    else:
        model_family = 'other'
    return {'param_count': param_count, 'version': version, 'thinking': thinking, 'dense_vs_moe': dense_vs_moe, 'model_family': model_family}

# Add extracted features
merged = merged.copy()
for f in arch_features + ['model_family']:
    merged[f] = [extract_arch_features(m)[f] for m in merged['model']]

# One-hot encode model_family
model_family_dummies = pd.get_dummies(merged['model_family'], prefix='family')
merged = pd.concat([merged, model_family_dummies], axis=1)
arch_features_extended = arch_features + list(model_family_dummies.columns)

# Strategic features
strat_features = [c for c in df_behavior['metric'].unique() if c not in arch_features]

# Targets
targets = [c for c in df_perf['metric'].unique() if c in merged.columns]

# --- Helper: Remove multicollinear features ---
def remove_multicollinear_features(df, features, threshold=0.95):
    if len(features) < 2:
        return features
    corr = df[features].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [f for f in features if f not in to_drop]

# --- SHAP Analysis ---
shap_summary = []
for game in merged['game'].unique():
    game_df = merged[merged['game'] == game]
    for target in targets:
        if target not in game_df.columns:
            continue
        # Remove multicollinear features separately for arch and strat
        arch_valid = [f for f in arch_features_extended if f in game_df.columns and game_df[f].notna().any()]
        strat_valid = [f for f in strat_features if f in game_df.columns and game_df[f].notna().any()]
        arch_final = remove_multicollinear_features(game_df, arch_valid)
        strat_final = remove_multicollinear_features(game_df, strat_valid)
        predictors = arch_final + strat_final
        valid = game_df[predictors + [target]].dropna()
        if len(valid) < 5:
            continue
        X = valid[predictors]
        y = valid[target]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(X)
        # Aggregate SHAP values for architectural and strategic features
        arch_idx = [i for i, p in enumerate(predictors) if p in arch_final]
        strat_idx = [i for i, p in enumerate(predictors) if p in strat_final]
        arch_importance = np.abs(shap_values.values[:, arch_idx]).mean(axis=0)
        strat_importance = np.abs(shap_values.values[:, strat_idx]).mean(axis=0)
        shap_summary.append({
            'game': game,
            'target': target,
            'arch_total_shap': arch_importance.sum(),
            'strat_total_shap': strat_importance.sum(),
            'arch_features': dict(zip([predictors[i] for i in arch_idx], arch_importance)),
            'strat_features': dict(zip([predictors[i] for i in strat_idx], strat_importance)),
            'n_obs': len(valid)
        })
        # Save SHAP plot
        shap.summary_plot(shap_values.values, X, feature_names=predictors, show=False)
        plot_path = os.path.join(SHAP_OUTPUT_DIR, f'{game}_{target}_shap_summary.png')
        import matplotlib.pyplot as plt
        plt.savefig(plot_path)
        plt.close()
        print(f'SHAP summary plot saved: {plot_path}')
# Save summary CSV
summary_df = pd.DataFrame([
    {
        'game': s['game'],
        'target': s['target'],
        'arch_total_shap': s['arch_total_shap'],
        'strat_total_shap': s['strat_total_shap'],
        'n_obs': s['n_obs']
    } for s in shap_summary
])
summary_df.to_csv(os.path.join(SHAP_OUTPUT_DIR, 'shap_feature_importance_summary.csv'), index=False)
print('SHAP feature importance summary saved.')
