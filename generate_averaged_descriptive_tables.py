#!/usr/bin/env python3
"""
Generate RQ1 and RQ2 descriptive tables with values AVERAGED across conditions.
Each model shows a single row with metrics averaged over baseline (3P) and more_players (5P).
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
OUTPUT_DIR = Path("metrics/latex_tables")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Model display names
DISPLAY_NAMES = {
    'Qwen/Qwen3-14B-Thinking-Off': 'Q3-14B (TD)',
    'Qwen/Qwen3-14B-Thinking-On': 'Q3-14B (TE)',
    'Qwen/Qwen3-235B-A22B-Instruct-2507': 'Q3-235B Inst',
    'Qwen/Qwen3-30B-A3B-Thinking-Off': 'Qwen3-30B-A3B (TD)',
    'Qwen/Qwen3-30B-A3B-Thinking-On': 'Qwen3-30B-A3B (TE)',
    'Qwen/Qwen3-32B-Thinking-Off': 'Q3-32B (TD)',
    'Qwen/Qwen3-32B-Thinking-On': 'Q3-32B (TE)',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo': 'L3.3-70B',
    'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': 'L4-Maverick',
    'meta-llama/Llama-4-Scout-17B-16E-Instruct': 'L4-Scout',
    'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo': 'L3.1-70B',
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo': 'L3.1-8B',
    'google/gemma-3-27b-it': 'Gemma-3-27B',
    'random_agent': 'Random'
}

# Game configurations
GAME_CONFIGS = {
    'athey_bagwell': {
        'display_name': 'Athey-Bagwell',
        'abbrev': 'AB',
        'performance_metrics': ['average_profit', 'productive_efficiency', 'win_rate'],
        'magic_metrics': ['cooperation', 'deception', 'rationality', 'reasoning'],
        'column_names': ['Avg. Profit', 'Prod. Eff.', 'Win Rate']
    },
    'green_porter': {
        'display_name': 'Green-Porter',
        'abbrev': 'GP',
        'performance_metrics': ['average_profit', 'reversion_frequency', 'win_rate'],
        'magic_metrics': ['cooperation', 'coordination'],
        'column_names': ['Avg. Profit', 'Reversion Freq.', 'Win Rate']
    },
    'salop': {
        'display_name': 'Salop',
        'abbrev': 'SA',
        'performance_metrics': ['average_profit', 'market_price', 'win_rate'],
        'magic_metrics': ['cooperation', 'rationality', 'reasoning'],
        'column_names': ['Avg. Profit', 'Market Price', 'Win Rate']
    },
    'spulber': {
        'display_name': 'Spulber',
        'abbrev': 'SP',
        'performance_metrics': ['average_profit', 'allocative_efficiency', 'win_rate'],
        'magic_metrics': ['rationality', 'reasoning', 'judgment', 'self_awareness'],
        'column_names': ['Avg. Profit', 'Alloc. Eff.', 'Win Rate']
    }
}


def format_value(mean, std):
    """Format a value with mean ± std"""
    if pd.isna(mean) or pd.isna(std):
        return 'N/A'
    return f"{mean:.2f} ± {std:.2f}"


def get_averaged_value(df, model, game, metric):
    """Get metric value averaged across all conditions"""
    data = df[(df['model'] == model) & (df['game'] == game) & (df['metric'] == metric)]
    
    if len(data) == 0:
        return np.nan, np.nan
    
    # Average the means across conditions
    avg_mean = data['mean'].mean()
    
    # For std, use pooled standard deviation formula
    # σ_pooled = sqrt(mean of variances)
    avg_std = np.sqrt((data['std'] ** 2).mean())
    
    return avg_mean, avg_std


def generate_rq1_table(game, perf_df):
    """Generate RQ1 performance table for a game"""
    config = GAME_CONFIGS[game]
    
    # Get unique models (exclude Gemma as per analysis.py)
    models = [m for m in perf_df['model'].unique() 
              if 'gemma' not in m.lower()]
    
    # Build table
    lines = []
    lines.append(f"% filepath: metrics/latex_tables/RQ1_{game}_descriptive.tex")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{RQ1: {config['display_name']} - Performance Metrics (Averaged Across Conditions)}}")
    lines.append(f"\\label{{tab:rq1_{game}}}")
    
    # Table header
    col_spec = "l" + "c" * len(config['column_names'])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    header = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{name}}}" for name in config['column_names']]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Data rows
    for model in models:
        display_name = DISPLAY_NAMES.get(model, model.split('/')[-1])
        row = [display_name]
        
        for metric in config['performance_metrics']:
            mean, std = get_averaged_value(perf_df, model, game, metric)
            row.append(format_value(mean, std))
        
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


def generate_rq2_table(game, magic_df):
    """Generate RQ2 behavioral table for a game"""
    config = GAME_CONFIGS[game]
    
    # Get unique models (exclude Gemma)
    models = [m for m in magic_df['model'].unique() 
              if 'gemma' not in m.lower()]
    
    # Build table
    lines = []
    lines.append(f"% filepath: metrics/latex_tables/RQ2_{game}_Descriptive.tex")
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{RQ2: {config['display_name']} - MAgIC Behavioral Metrics (Averaged Across Conditions)}}")
    lines.append(f"\\label{{tab:rq2_{game}}}")
    
    # Table header
    col_spec = "l" + "c" * len(config['magic_metrics'])
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Capitalize metric names
    metric_names = [m.replace('_', ' ').title() for m in config['magic_metrics']]
    header = "\\textbf{Model} & " + " & ".join([f"\\textbf{{{name}}}" for name in metric_names]) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Data rows
    for model in models:
        display_name = DISPLAY_NAMES.get(model, model.split('/')[-1])
        row = [display_name]
        
        for metric in config['magic_metrics']:
            mean, std = get_averaged_value(magic_df, model, game, metric)
            row.append(format_value(mean, std))
        
        lines.append(" & ".join(row) + " \\\\")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)


# Architecture features mapping (same as regression.py)
def get_arch_features(model_name):
    qwen3_models = {
        'Qwen/Qwen3-14B-Thinking-Off':    {'family': 'qwen3', 'param_size': '14b', 'model_architecture': 'dense', 'thinking_enabled': False, 'family_version': None},
        'Qwen/Qwen3-32B-Thinking-Off':    {'family': 'qwen3', 'param_size': '32b', 'model_architecture': 'dense', 'thinking_enabled': False, 'family_version': None},
        'Qwen/Qwen3-14B-Thinking-On':     {'family': 'qwen3', 'param_size': '14b', 'model_architecture': 'dense', 'thinking_enabled': True,  'family_version': None},
        'Qwen/Qwen3-32B-Thinking-On':     {'family': 'qwen3', 'param_size': '32b', 'model_architecture': 'dense', 'thinking_enabled': True,  'family_version': None},
        'Qwen/Qwen3-235B-A22B-Instruct-2507': {'family': 'qwen3', 'param_size': '22b', 'model_architecture': 'moe',   'thinking_enabled': False, 'family_version': None},
        'Qwen/Qwen3-30B-A3B-Thinking-On':    {'family': 'qwen3', 'param_size': '3b',  'model_architecture': 'moe',   'thinking_enabled': True,  'family_version': None},
        'Qwen/Qwen3-30B-A3B-Thinking-Off':   {'family': 'qwen3', 'param_size': '3b',  'model_architecture': 'moe',   'thinking_enabled': False, 'family_version': None},
    }
    llama_models = {
        'meta-llama/Llama-3.3-70B-Instruct-Turbo':         {'family': 'llama', 'param_size': '70b', 'model_architecture': 'dense', 'family_version': '3.3'},
        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo':     {'family': 'llama', 'param_size': '8b',  'model_architecture': 'dense', 'family_version': '3.1'},
        'meta-llama/Llama-4-Scout-17B-16E-Instruct':       {'family': 'llama', 'param_size': '17b', 'model_architecture': 'moe',   'family_version': '4'},
        'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8': {'family': 'llama', 'param_size': '17b', 'model_architecture': 'moe', 'family_version': '4'},
        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo':    {'family': 'llama', 'param_size': '70b', 'model_architecture': 'dense', 'family_version': '3.1'},
    }
    if model_name in qwen3_models:
        
        with open(output_file, 'w') as f:
            f.write(table_content)
        
        print(f"  ✓ {output_file.name}", flush=True)
    
    print(flush=True)
    
    # Generate RQ2 tables
    print("Generating RQ2 Behavioral Tables...", flush=True)
    for game in GAME_CONFIGS.keys():
        output_file = OUTPUT_DIR / f"RQ2_{game}_Descriptive.tex"
        table_content = generate_rq2_table(game, magic_df)
        
        with open(output_file, 'w') as f:
            f.write(table_content)
        
        print(f"  ✓ {output_file.name}", flush=True)
    
    print(flush=True)
    print("=" * 80, flush=True)
    print("✅ COMPLETE: All tables now show values averaged across conditions", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    print("Changes made:", flush=True)
    print("  - Each model has ONE row (not separate 3P/5P sections)", flush=True)
    print("  - Values are averaged over baseline (3P) and more_players (5P)", flush=True)
    print("  - Standard deviations use pooled formula: sqrt(mean of variances)", flush=True)
    print(flush=True)


if __name__ == "__main__":
    main()
