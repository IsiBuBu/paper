import pandas as pd
import numpy as np

CSV_PATH = "/Users/ismetbuyar/git-repos/paper/output/analysis/magic_behavioral_metrics.csv"
GAMES = ["athey_bagwell", "green_porter", "salop", "spulber"]
EXCLUDE_MODELS = ["random_agent", "google/gemma-3-27b-it"]

# Helper: get metrics for each game
def get_metrics_for_game(df, game):
    return df[df['game'] == game]['metric'].unique()

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

def main():
    df = pd.read_csv(CSV_PATH)
    # Exclude models by exact name
    df = df[~df['model'].isin(EXCLUDE_MODELS)]
    game_distances = []
    for game in GAMES:
        metrics = get_metrics_for_game(df, game)
        rows_3 = df[(df['game'] == game) & (df['condition'] == 'baseline')]
        rows_5 = df[(df['game'] == game) & (df['condition'] == 'more_players')]
        vec_3 = [rows_3[rows_3['metric'] == m]['mean'].mean() for m in metrics]
        vec_5 = [rows_5[rows_5['metric'] == m]['mean'].mean() for m in metrics]
        dist = euclidean_distance(vec_3, vec_5)
        game_distances.append(dist)
    overall_distance = np.mean(game_distances)
    # Output LaTeX table to file
    latex = ""
    latex += "\\begin{table}[h]\n"
    latex += "\\centering\n"
    latex += "\\caption{Euclidean Distance Similarity between 3-player and 5-player models (excluding Gemma and random_agent)}\n"
    latex += "\\begin{tabular}{lcccccc}\n"
    latex += "\\toprule\n"
    latex += "Game & " + " & ".join(GAMES) + " & Overall \\" + "\n"
    latex += "\\midrule\n"
    latex += "Distance & " + " & ".join([f"{d:.3f}" for d in game_distances]) + f" & {overall_distance:.3f} \\" + "\n"
    latex += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open("metrics/latex_tables/euclidean_similarity_NEW.tex", "w") as f:
        f.write(latex)
    print("Saved metrics/latex_tables/euclidean_similarity_NEW.tex")

if __name__ == '__main__':
    main()
