import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from math import pi

# --- Configuration ---
INPUT_FILE = 'output/analysis/magic_behavioral_metrics.csv'
OUTPUT_DIR = 'output/analysis/plots/visualizations'

# Metrics to Plot: (Empty list = Include Everything)
# We are now using all metrics including profit, win_rate, etc.
EXCLUDE_METRICS = []

# Define Metric Pairs for Quadrant Plots
# We only define X and Y axes now, no specific "names" for the quadrants.
QUADRANT_CONFIG = {
    'athey_bagwell': {'x': 'rationality', 'y': 'coordination'},
    'green_porter':  {'x': 'deception',   'y': 'cooperation'},
    'salop':         {'x': 'reasoning',   'y': 'judgment'},
    'spulber':       {'x': 'self_awareness', 'y': 'rationality'}
}

# --- Helper Functions ---

def load_data(filepath):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    return pd.read_csv(filepath)

def normalize_data(df):
    """Min-Max normalization (0-1) so all metrics fit on the same Radar Chart"""
    return (df - df.min()) / (df.max() - df.min()).replace(0, 1)

def make_radar_chart(df, title, filename):
    """Generates a Radar Chart (Spider Plot) for all models"""
    if df.empty: return

    # 1. Prepare Data
    categories = list(df.columns)
    N = len(categories)
    
    # 2. Setup Angles
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1] # Close the loop
    
    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Define distinct colors for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    # 4. Plot Each Model
    for i, (model_name, row) in enumerate(df.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1] # Close the loop
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.05) # Very light fill to avoid clutter
    
    # 5. Aesthetics
    # Rotate labels to be readable
    plt.xticks(angles[:-1], categories, color='black', size=9)
    
    # Y-labels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0"], color="grey", size=7)
    plt.ylim(0, 1.05)
    
    # Title & Legend
    plt.title(title, size=16, y=1.1, weight='bold')
    # Move legend outside
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), title="Models")
    
    # Save
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Radar: {filename}")

def make_quadrant_plot(df, config, title, filename):
    """Generates a Scatter Plot for 2 specific metrics"""
    x_metric = config['x']
    y_metric = config['y']
    
    # Check if metrics exist in data (since we might have filtered different things)
    if x_metric not in df.columns or y_metric not in df.columns:
        # Fallback: check if we can find them in the raw columns before pivot
        # (The df passed here is already pivoted, so columns are metrics)
        print(f"Skipping Quadrant {title}: Metrics '{x_metric}' or '{y_metric}' not found in data columns: {list(df.columns)}")
        return

    plt.figure(figsize=(11, 9))
    
    # Scatter Plot
    # Using 'hue' and 'style' to differentiate models clearly
    sns.scatterplot(
        data=df, 
        x=x_metric, 
        y=y_metric, 
        hue=df.index, 
        style=df.index, 
        s=300, # Large markers
        palette='tab10',
        alpha=0.9
    )
    
    # Add Center Lines to create visual quadrants
    # Assuming normalized data (0-1), center is 0.5. 
    # If using raw data, we might use mean/median, but let's stick to 0.5 for normalized view or mean for raw.
    # Let's use the mean of the data for the crosshairs to show relative performance.
    mid_x = df[x_metric].mean()
    mid_y = df[y_metric].mean()
    
    plt.axvline(x=mid_x, color='gray', linestyle='--', alpha=0.5)
    plt.axhline(y=mid_y, color='gray', linestyle='--', alpha=0.5)
    
    # Labels and Titles
    plt.title(title, fontsize=18, pad=20, weight='bold')
    plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=14, weight='bold')
    plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=14, weight='bold')
    
    # Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., title="Models", fontsize=10)
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Quadrant: {filename}")

# --- Main Logic ---

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    raw_df = load_data(INPUT_FILE)
    if raw_df is None: return

    # No filtering - use all metrics
    df = raw_df.copy()
    
    games = df['game'].unique()
    
    for game in games:
        print(f"\nProcessing {game}...")
        game_subset = df[df['game'] == game]
        
        # --- 1. Overall Game Plot (Averaged across conditions) ---
        pivot_overall = game_subset.pivot_table(index='model', columns='metric', values='mean')
        
        # Normalize for Radar (0-1 range is required for radar charts to look right)
        norm_overall = normalize_data(pivot_overall).fillna(0)
        
        # Radar Chart
        make_radar_chart(norm_overall, f"Metric Profile: {game.title()} (Overall)", 
                         os.path.join(OUTPUT_DIR, f"radar_{game}_overall.png"))
        
        # Quadrant Plot (Use raw values or normalized? Normalized is better for comparison if units differ)
        # We will use the normalized data so the scale is 0-1 for all
        if game in QUADRANT_CONFIG:
            make_quadrant_plot(norm_overall, QUADRANT_CONFIG[game], 
                               f"Metric Comparison: {game.title()} (Overall)",
                               os.path.join(OUTPUT_DIR, f"quadrant_{game}_overall.png"))

        # --- 2. Per Condition Plots ---
        conditions = game_subset['condition'].unique()
        for cond in conditions:
            cond_subset = game_subset[game_subset['condition'] == cond]
            
            # Pivot
            pivot_cond = cond_subset.pivot_table(index='model', columns='metric', values='mean')
            norm_cond = normalize_data(pivot_cond).fillna(0)
            
            # Radar
            make_radar_chart(norm_cond, f"Metric Profile: {game.title()} ({cond})", 
                             os.path.join(OUTPUT_DIR, f"radar_{game}_{cond}.png"))
            
            # Quadrant
            if game in QUADRANT_CONFIG:
                make_quadrant_plot(norm_cond, QUADRANT_CONFIG[game], 
                                   f"Metric Comparison: {game.title()} ({cond})",
                                   os.path.join(OUTPUT_DIR, f"quadrant_{game}_{cond}.png"))

if __name__ == "__main__":
    main()