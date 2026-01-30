import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Configuration ---
INPUT_FILE = 'output/analysis/magic_behavioral_metrics.csv'
OUTPUT_DIR = 'output/analysis/plots'

# --- Helper Functions ---

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found.")
        return None

def calculate_matrices(pivot_df):
    """
    Calculates Cosine Similarity and Significance (Pearson p-value) matrices.
    """
    models = pivot_df.index.tolist()
    n_models = len(models)
    
    sim_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    pval_matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    
    # Fill NaN with 0 for similarity calculation
    X = pivot_df.fillna(0).values
    
    if X.shape[1] == 0:
        return None, None
        
    # 1. Cosine Similarity
    cos_sim = cosine_similarity(X)
    sim_matrix[:] = cos_sim
    
    # 2. Significance (Pearson Correlation P-value)
    for i in range(n_models):
        for j in range(n_models):
            if i == j:
                pval_matrix.iloc[i, j] = 0.0
                continue
            
            vec_i = X[i]
            vec_j = X[j]
            
            # Check for constant vectors to avoid warnings
            if np.std(vec_i) == 0 or np.std(vec_j) == 0:
                pval_matrix.iloc[i, j] = np.nan
            else:
                try:
                    stat, p = pearsonr(vec_i, vec_j)
                    pval_matrix.iloc[i, j] = p
                except:
                    pval_matrix.iloc[i, j] = np.nan
                
    return sim_matrix, pval_matrix

def format_title(name):
    # Clean up filename for plot title
    name = name.replace('similarity_matrix_', '').replace('significance_matrix_', '')
    name = name.replace('.png', '')
    return name.replace('_', ' ').title()

def create_heatmap(df, output_path, matrix_type):
    try:
        plt.figure(figsize=(12, 10))
        sns.set(style="white", font_scale=0.9)
        
        if matrix_type == 'similarity':
            cmap = "RdBu_r" # Red-Blue (Blue=High sim)
            vmin, vmax = 0, 1
            fmt = ".2f"
            label = "Cosine Similarity"
            center = 0.5
        else: # significance
            cmap = "Greens_r" # Reversed Greens (Dark=Low p-value/High Sig)
            vmin, vmax = 0, 0.1 # Focus visual range on significant values (p < 0.1)
            fmt = ".3f"
            label = "P-Value"
            center = None

        heatmap = sns.heatmap(df, 
                              annot=True, 
                              fmt=fmt, 
                              cmap=cmap, 
                              vmin=vmin, 
                              vmax=vmax, 
                              center=center,
                              square=True, 
                              linewidths=.5, 
                              cbar_kws={"shrink": .5, "label": label})
        
        title = format_title(os.path.basename(output_path))
        plt.title(title, fontsize=16, pad=20, weight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Generated: {output_path}")
        
    except Exception as e:
        print(f"Error plotting {output_path}: {e}")

def process_and_plot(pivot_df, name_suffix):
    """
    Calculates matrices and generates both Similarity and Significance plots.
    """
    sim, pval = calculate_matrices(pivot_df)
    
    if sim is not None:
        # Plot Similarity
        sim_path = os.path.join(OUTPUT_DIR, f"similarity_matrix_{name_suffix}.png")
        create_heatmap(sim, sim_path, 'similarity')
        
        # Plot Significance
        pval_path = os.path.join(OUTPUT_DIR, f"significance_matrix_{name_suffix}.png")
        create_heatmap(pval, pval_path, 'significance')

# --- Main Execution ---

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = load_data(INPUT_FILE)
    if df is None or df.empty:
        return

    # Note: Exclusion filter removed. All metrics in the CSV will be used.

    # 1. Overall (All Games, All Conditions)
    print("\n--- Processing Overall ---")
    df['feat_overall'] = df['game'] + "_" + df['condition'] + "_" + df['metric']
    pivot_overall = df.pivot_table(index='model', columns='feat_overall', values='mean')
    process_and_plot(pivot_overall, "overall")

    # 2. Per Condition (Aggregated over All Games)
    print("\n--- Processing Per Condition ---")
    conditions = df['condition'].unique()
    for cond in conditions:
        sub_df = df[df['condition'] == cond].copy()
        if sub_df.empty: continue
        sub_df['feat_cond'] = sub_df['game'] + "_" + sub_df['metric']
        pivot_cond = sub_df.pivot_table(index='model', columns='feat_cond', values='mean')
        process_and_plot(pivot_cond, f"condition_allgames_{cond}")

    # 3. Per Game (Aggregated over All Conditions)
    print("\n--- Processing Per Game ---")
    games = df['game'].unique()
    for game in games:
        sub_df = df[df['game'] == game].copy()
        if sub_df.empty: continue
        sub_df['feat_game'] = sub_df['condition'] + "_" + sub_df['metric']
        pivot_game = sub_df.pivot_table(index='model', columns='feat_game', values='mean')
        process_and_plot(pivot_game, f"game_allconditions_{game}")

    # 4. Granular (Per Game, Per Condition)
    print("\n--- Processing Granular (Game x Condition) ---")
    for game in games:
        for cond in conditions:
            sub_df = df[(df['game'] == game) & (df['condition'] == cond)].copy()
            if sub_df.empty: continue
            
            # Pivot directly on metric since game/cond are fixed
            pivot_granular = sub_df.pivot_table(index='model', columns='metric', values='mean')
            process_and_plot(pivot_granular, f"game_{game}_condition_{cond}")

if __name__ == "__main__":
    main()