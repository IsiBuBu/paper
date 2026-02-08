import pandas as pd
import io

csv_path = "output/analysis/performance_metrics.csv"
table_configs = [
    ("metrics/latex_tables/RQ1_summary_avg_profit_NEW.tex", "average_profit"),
    ("metrics/latex_tables/RQ1_summary_win_rate_NEW.tex", "win_rate"),
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

def get_pooled_values(df, model_csv, metric):
    d = df[(df['model'] == model_csv) & (df['metric'] == metric)]
    if len(d) < 2:
        return None, None
    mean1, mean2 = d.iloc[0]['mean'], d.iloc[1]['mean']
    std1, std2 = d.iloc[0]['std'], d.iloc[1]['std']
    pooled_mean = (mean1 + mean2) / 2
    pooled_std_val = pooled_std(std1, std2)
    return pooled_mean, pooled_std_val

def make_table(metric, latex_path, df):
    games = ['athey_bagwell', 'green_porter', 'salop', 'spulber']
    game_names = ['Athey-Bagwell', 'Green-Porter', 'Salop', 'Spulber']
    header = f"% filepath: {latex_path}\n"
    header += "\\begin{table}[h]\n"
    header += f"\\centering\n"
    header += f"\\caption{{RQ1: {metric.replace('_', ' ').title()} (Pooled Mean/Std) by Game}}\n"
    header += f"\\label{{tab:rq1_{metric}_summary}}\n"
    header += f"\\begin{{tabular}}{{l {'c'*len(games)}}}\n"
    header += "\\toprule\n"
    header += "\\textbf{Model} & " + " & ".join([f"\\textbf{{{g}}}" for g in game_names]) + " \\\\" + "\n"
    header += "\\midrule\n"
    pooled_results = []
    for model_latex in model_order:
        model_csv = model_map.get(model_latex, model_latex)
        row = [model_latex]
        for game in games:
            d = df[(df['model'] == model_csv) & (df['game'] == game) & (df['metric'] == metric)]
            if len(d) < 2:
                row.append("")
                continue
            mean1, mean2 = d.iloc[0]['mean'], d.iloc[1]['mean']
            std1, std2 = d.iloc[0]['std'], d.iloc[1]['std']
            pooled_mean = (mean1 + mean2) / 2
            pooled_std_val = pooled_std(std1, std2)
            row.append((pooled_mean, pooled_std_val))
        pooled_results.append(row)
    if metric == 'average_profit':
        for j in range(1, len(games)+1):
            max_mean = max([row[j][0] for row in pooled_results if isinstance(row[j], tuple) and row[j][0] > 0], default=1)
            for row in pooled_results:
                if isinstance(row[j], tuple) and max_mean > 0:
                    mean, std = row[j]
                    row[j] = (mean / max_mean, std / max_mean)
    rows = []
    for row in pooled_results:
        model_latex = row[0]
        vals = []
        for cell in row[1:]:
            if isinstance(cell, tuple):
                mean, std = cell
                if metric == 'win_rate':
                    val = f"${mean:.3f} \\pm {std:.3f}$"
                else:
                    val = f"${mean:.2f} \\pm {std:.2f}$"
            else:
                val = ""
            vals.append(val)
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
    for latex_path, metric in table_configs:
        make_table(metric, latex_path, df)
        print(f"Created {latex_path} for metric {metric}")

if __name__ == "__main__":
    main()
