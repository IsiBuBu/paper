# visualize_magic.py

"""
MAgIC Metrics Visualization

Run from project root:
    python visualize_magic.py

Generates radar charts and quadrant plots for MAgIC behavioral metrics.
Excludes random_agent and defender model (gemma).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi

# --- Configuration ---
INPUT_FILE = 'output/analysis/magic_behavioral_metrics.csv'
OUTPUT_DIR = 'output/analysis/plots/visualizations'

# MAgIC Metrics per Game (v5 - No Collinearity)
# These are the only metrics computed for each game
GAME_METRICS = {
    'salop': ['rationality', 'reasoning', 'cooperation'],
    'spulber': ['rationality', 'judgment', 'reasoning', 'self_awareness'],
    'green_porter': ['cooperation', 'coordination'],
    'athey_bagwell': ['rationality', 'reasoning', 'deception', 'cooperation']
}

# Define Metric Pairs for Quadrant Plots (using available metrics per game)
QUADRANT_CONFIG = {
    'salop': {'x': 'reasoning', 'y': 'cooperation'},
    'spulber': {'x': 'judgment', 'y': 'self_awareness'},
    'green_porter': {'x': 'cooperation', 'y': 'coordination'},
    'athey_bagwell': {'x': 'deception', 'y': 'cooperation'}
}


# --- Helper Functions ---

def is_excluded_model(model_name: str) -> bool:
    """Check if model should be excluded (random_agent or defender)."""
    model_lower = model_name.lower()
    if 'random' in model_lower:
        return True
    if 'gemma' in model_lower:
        return True
    return False


def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    df = pd.read_csv(filepath)
    
    # Filter out excluded models
    original_count = len(df)
    df = df[~df['model'].apply(is_excluded_model)]
    excluded_count = original_count - len(df)
    if excluded_count > 0:
        print(f"Excluded {excluded_count} rows (random_agent/defender)")
    
    return df


def normalize_data(df):
    """Min-Max normalization (0-1) so all metrics fit on the same Radar Chart"""
    return (df - df.min()) / (df.max() - df.min()).replace(0, 1)


def make_radar_chart(df, title, filename, game_name=None):
    """Generates a Radar Chart (Spider Plot) for all models"""
    if df.empty:
        return

    # Filter to only game-specific metrics if game_name provided
    if game_name and game_name in GAME_METRICS:
        valid_metrics = [m for m in GAME_METRICS[game_name] if m in df.columns]
        df = df[valid_metrics]
    
    if df.empty or len(df.columns) < 2:
        print(f"Skipping Radar {title}: Not enough metrics")
        return

    # 1. Prepare Data
    categories = list(df.columns)
    N = len(categories)
    
    # 2. Setup Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Define distinct colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    # 4. Plot Each Model
    for i, (model_name, row) in enumerate(df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]  # Close the loop
        
        # Shorten model name for legend
        short_name = model_name.split('/')[-1][:30]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=short_name, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.05)
    
    # 5. Aesthetics
    plt.xticks(angles[:-1], [c.replace('_', ' ').title() for c in categories], color='black', size=10)
    
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=7)
    plt.ylim(0, 1.05)
    
    plt.title(title, size=16, y=1.1, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), title="Models", fontsize=8)
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Radar: {filename}")


def make_quadrant_plot(df, config, title, filename):
    """Generates a Scatter Plot for 2 specific metrics"""
    x_metric = config['x']
    y_metric = config['y']
    
    if x_metric not in df.columns or y_metric not in df.columns:
        print(f"Skipping Quadrant {title}: Metrics '{x_metric}' or '{y_metric}' not found")
        return

    plt.figure(figsize=(11, 9))
    
    # Scatter Plot
    sns.scatterplot(
        data=df.reset_index(),
        x=x_metric,
        y=y_metric,
        hue='model',
        style='model',
        s=300,
        palette='tab10',
        alpha=0.9
    )
    
    # Add Center Lines (mean-based crosshairs)
    mid_x = df[x_metric].mean()
    mid_y = df[y_metric].mean()
    
    plt.axvline(x=mid_x, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=mid_y, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and Titles
    plt.title(title, fontsize=18, pad=20, weight='bold')
    plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=14, weight='bold')
    plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=14, weight='bold')
    
    # Legend with shortened names
    handles, labels = plt.gca().get_legend_handles_labels()
    short_labels = [l.split('/')[-1][:25] for l in labels]
    plt.legend(handles, short_labels, bbox_to_anchor=(1.02, 1), loc='upper left', 
               borderaxespad=0., title="Models", fontsize=9)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Quadrant: {filename}")


def make_bar_chart(df, title, filename, game_name=None):
    """Generates a grouped bar chart comparing models across metrics"""
    if df.empty:
        return
    
    # Filter to game-specific metrics
    if game_name and game_name in GAME_METRICS:
        valid_metrics = [m for m in GAME_METRICS[game_name] if m in df.columns]
        df = df[valid_metrics]
    
    if df.empty or len(df.columns) < 1:
        return
    
    # Reshape for seaborn
    df_melted = df.reset_index().melt(id_vars='model', var_name='Metric', value_name='Value')
    df_melted['Model'] = df_melted['model'].apply(lambda x: x.split('/')[-1][:20])
    
    plt.figure(figsize=(14, 8))
    
    ax = sns.barplot(
        data=df_melted,
        x='Metric',
        y='Value',
        hue='Model',
        palette='tab10'
    )
    
    plt.title(title, fontsize=16, weight='bold')
    plt.xlabel('MAgIC Metric', fontsize=12)
    plt.ylabel('Normalized Value (0-1)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Bar Chart: {filename}")


# --- Main Logic ---

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    raw_df = load_data(INPUT_FILE)
    if raw_df is None:
        return

    games = raw_df['game'].unique()
    print(f"\nFound games: {list(games)}")
    
    for game in games:
        print(f"\n{'='*50}")
        print(f"Processing {game}...")
        print(f"Available metrics: {GAME_METRICS.get(game, 'unknown')}")
        print(f"{'='*50}")
        
        game_subset = raw_df[raw_df['game'] == game]
        
        # --- 1. Overall Game Plot (Averaged across conditions) ---
        pivot_overall = game_subset.pivot_table(index='model', columns='metric', values='mean')
        
        # Filter to game-specific metrics only
        if game in GAME_METRICS:
            valid_cols = [c for c in GAME_METRICS[game] if c in pivot_overall.columns]
            pivot_overall = pivot_overall[valid_cols]
        
        # Normalize for Radar
        norm_overall = normalize_data(pivot_overall).fillna(0)
        
        # Radar Chart
        make_radar_chart(
            norm_overall, 
            f"MAgIC Profile: {game.replace('_', ' ').title()} (Overall)", 
            os.path.join(OUTPUT_DIR, f"radar_{game}_overall.png"),
            game_name=game
        )
        
        # Bar Chart
        make_bar_chart(
            norm_overall,
            f"MAgIC Metrics: {game.replace('_', ' ').title()} (Overall)",
            os.path.join(OUTPUT_DIR, f"bar_{game}_overall.png"),
            game_name=game
        )
        
        # Quadrant Plot
        if game in QUADRANT_CONFIG:
            make_quadrant_plot(
                norm_overall, 
                QUADRANT_CONFIG[game], 
                f"MAgIC Quadrant: {game.replace('_', ' ').title()} (Overall)",
                os.path.join(OUTPUT_DIR, f"quadrant_{game}_overall.png")
            )

        # --- 2. Per Condition Plots ---
        conditions = game_subset['condition'].unique()
        for cond in conditions:
            cond_subset = game_subset[game_subset['condition'] == cond]
            
            pivot_cond = cond_subset.pivot_table(index='model', columns='metric', values='mean')
            
            # Filter to game-specific metrics
            if game in GAME_METRICS:
                valid_cols = [c for c in GAME_METRICS[game] if c in pivot_cond.columns]
                pivot_cond = pivot_cond[valid_cols]
            
            norm_cond = normalize_data(pivot_cond).fillna(0)
            
            # Radar
            make_radar_chart(
                norm_cond, 
                f"MAgIC Profile: {game.replace('_', ' ').title()} ({cond})", 
                os.path.join(OUTPUT_DIR, f"radar_{game}_{cond}.png"),
                game_name=game
            )
            
            # Quadrant
            if game in QUADRANT_CONFIG:
                make_quadrant_plot(
                    norm_cond, 
                    QUADRANT_CONFIG[game], 
                    f"MAgIC Quadrant: {game.replace('_', ' ').title()} ({cond})",
                    os.path.join(OUTPUT_DIR, f"quadrant_{game}_{cond}.png")
                )
    
    print(f"\n{'='*50}")
    print(f"Visualization complete! Files saved to: {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()