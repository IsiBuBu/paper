import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import os

CSV_PATH = "output/analysis/magic_behavioral_metrics.csv"
GAMES = ["athey_bagwell", "green_porter", "salop", "spulber"]
EXCLUDE_MODELS = ["random_agent", "google/gemma-3-27b-it", "meta-llama/Gemma-7B-Instruct"]
OUTDIR = "metrics/latex_tables/"
MODEL_ORDER = [
    'Q3-235B Inst', 'Q3-32B (TE)', 'Q3-32B (TD)', 'Q3-14B (TE)', 'Q3-14B (TD)',
    'Qwen3-30B-A3B (TE)', 'Qwen3-30B-A3B (TD)', 'L4-Maverick', 'L4-Scout',
    'L3.1-70B', 'L3.3-70B', 'L3.1-8B'
]
MODEL_MAP = {
    'Q3-235B Inst': 'Qwen/Qwen3-235B-A22B-Instruct-2507',
    'Q3-32B (TE)': 'Qwen/Qwen3-32B-Thinking-On',
    'Q3-32B (TD)': 'Qwen/Qwen3-32B-Thinking-Off',
    'Q3-14B (TE)': 'Qwen/Qwen3-14B-Thinking-On',
    'Q3-14B (TD)': 'Qwen/Qwen3-14B-Thinking-Off',
    'Qwen3-30B-A3B (TE)': 'Qwen/Qwen3-30B-A3B-Thinking-On',
    'Qwen3-30B-A3B (TD)': 'Qwen/Qwen3-30B-A3B-Thinking-Off',
    'L4-Maverick': 'meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8',
    'L4-Scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'L3.1-70B': 'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
    'L3.3-70B': 'meta-llama/Llama-3.3-70B-Instruct-Turbo',
    'L3.1-8B': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
}

def get_metrics_for_game(df, game):
    return df[df['game'] == game]['metric'].unique()

def get_pooled_means(df, game):
    metrics = get_metrics_for_game(df, game)
    data = []
    labels = []
    for model_disp in MODEL_ORDER:
        model = MODEL_MAP[model_disp]
        if model in EXCLUDE_MODELS:
            continue
        vec = [df[(df['game'] == game) & (df['model'] == model) & (df['metric'] == m)]['mean'].mean() for m in metrics]
        data.append(vec)
        labels.append(model_disp)
    return np.array(data), labels, metrics

def plot_dendrogram(data, labels, title, filename):
    Z = linkage(data, metric='euclidean', method='ward')
    plt.figure(figsize=(12, 5))
    dendrogram(Z, labels=labels, leaf_rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    df = pd.read_csv(CSV_PATH)
    for game in GAMES:
        data, labels, metrics = get_pooled_means(df, game)
        title = f"Dendrogram for {game.replace('_', ' ').title()} (Pooled Means)"
        filename = os.path.join(OUTDIR, f"dendrogram_{game}.png")
        plot_dendrogram(data, labels, title, filename)
        print(f"Saved {filename}")
    # Dendrogram over all games
    all_data = []
    all_labels = []
    for model_disp in MODEL_ORDER:
        model = MODEL_MAP[model_disp]
        if model in EXCLUDE_MODELS:
            continue
        vec = []
        for game in GAMES:
            metrics = get_metrics_for_game(df, game)
            vec.extend([df[(df['game'] == game) & (df['model'] == model) & (df['metric'] == m)]['mean'].mean() for m in metrics])
        all_data.append(vec)
        all_labels.append(model_disp)
    title = "Dendrogram for All Games (Pooled Means)"
    filename = os.path.join(OUTDIR, "dendrogram_all_games.png")
    plot_dendrogram(np.array(all_data), all_labels, title, filename)
    print(f"Saved {filename}")

if __name__ == '__main__':
    main()
