import pandas as pd
import io

csv_path = "output/analysis/magic_behavioral_metrics.csv"
rq2_tables = [
    ("metrics/latex_tables/RQ2_athey_bagwell_descriptive_NEW.tex", "Athey-Bagwell"),
    ("metrics/latex_tables/RQ2_green_porter_descriptive_NEW.tex", "Green-Porter"),
    ("metrics/latex_tables/RQ2_salop_descriptive_NEW.tex", "Salop"),
    ("metrics/latex_tables/RQ2_Spulber_Descriptive_NEW.tex", "Spulber"),
]
model_order = [
    'Q3-235B Inst', 'Q3-32B (TE)', 'Q3-32B (TD)', 'Q3-14B (TE)', 'Q3-14B (TD)',
    'Qwen3-30B-A3B (TE)', 'Qwen3-30B-A3B (TD)', 'L4-Maverick', 'L4-Scout',
    'L3.1-70B', 'L3.3-70B', 'L3.1-8B', 'Random'
]
model_map = {
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
    'Random': 'random_agent',
}
def pooled_std(std1, std2):
    std1 = 0 if pd.isna(std1) else std1
    std2 = 0 if pd.isna(std2) else std2
    return ((std1 ** 2 + std2 ** 2) / 2) ** 0.5

def get_pooled_values(df, model_csv, game_key, metric):
    d = df[(df['game'] == game_key) & (df['model'] == model_csv) & (df['metric'] == metric)]
    if len(d) < 2:
        return None, None
    mean1, mean2 = d.iloc[0]['mean'], d.iloc[1]['mean']
    std1, std2 = d.iloc[0]['std'], d.iloc[1]['std']
    pooled_mean = (mean1 + mean2) / 2
    pooled_std_val = pooled_std(std1, std2)
    return pooled_mean, pooled_std_val

def get_metrics_for_game(df, game_key):
    return [m for m in df[df['game'] == game_key]['metric'].unique() if m not in ('model','game')]

def make_table(game, latex_path, df):
    game_key = game.lower().replace('-', '_')
    metrics = get_metrics_for_game(df, game_key)
    def escape_latex(s):
        return s.replace('_', '\\_')
    header = f"% filepath: {latex_path}\n"
    header += "\\begin{table}[h]\n"
    header += f"\\centering\n"
    header += f"\\caption{{RQ2: {game} - Behavioral Metrics (Pooled Mean/Std)}}\n"
    header += f"\\label{{tab:rq2_{game_key}_behavioral}}\n"
    header += f"\\begin{{tabular}}{{l {'c'*len(metrics)}}}\n"
    header += "\\toprule\n"
    header += "\\textbf{Model} & " + " & ".join([f"\\textbf{{{escape_latex(m)}}}" for m in metrics]) + " \\\\" + "\n"
    header += "\\midrule\n"
    pooled_results = []
    for model_latex in model_order:
        model_csv = model_map.get(model_latex, model_latex)
        row = [model_latex]
        for metric in metrics:
            mean, std = get_pooled_values(df, model_csv, game_key, metric)
            if mean is not None and std is not None:
                row.append((mean, std))
            else:
                row.append("")
        pooled_results.append(row)
    rows = []
    for row in pooled_results:
        model_latex = row[0]
        vals = []
        for cell in row[1:]:
            if isinstance(cell, tuple):
                mean, std = cell
                val = f"${mean:.3f} \\pm {std:.3f}$"
            else:
                val = ""
            vals.append(val)
        # Ensure row ends with double backslash and nothing else
        rows.append(f"{model_latex} & " + " & ".join(vals) + " \\\\" + "\n")
    footer = "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(latex_path, 'w') as f:
        f.write(header)
        for row in rows:
            f.write(row)
        f.write(footer)

def main():
    with open(csv_path, 'r') as f:
        lines = [line for line in f if not line.lstrip().startswith('//')]
    df = pd.read_csv(io.StringIO(''.join(lines)), skip_blank_lines=True)
    df['model'] = df['model'].astype(str).str.strip()
    df['game'] = df['game'].astype(str).str.strip().str.lower()
    for latex_path, game in rq2_tables:
        make_table(game, latex_path, df)
        print(f"Created {latex_path} for {game}")

if __name__ == "__main__":
    main()
