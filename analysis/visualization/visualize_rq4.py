# analysis/visualization/visualize_rq4.py

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from scipy import stats
from scipy.stats import t

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_LIBS_INSTALLED = True
except ImportError:
    PLOT_LIBS_INSTALLED = False

from config.config import get_analysis_dir, get_experiments_dir
from metrics import GameResult, MAgICMetricsCalculator

# --- Data Preparation Helpers ---

def _get_thinking_data_per_sim(results_dir: Path) -> pd.DataFrame:
    """Extracts the average thinking tokens per simulation for thinking-enabled models."""
    records = []
    thinking_models = [
        "gemini_2_5_flash_thinking_low", "gemini_2_5_flash_thinking_medium",
        "gemini_2_5_flash_lite_thinking_low", "gemini_2_5_flash_lite_thinking_medium"
    ]
    for file_path in results_dir.glob("*/*/*_competition_result*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if data.get('challenger_model') not in thinking_models:
            continue

        for sim in data.get('simulation_results', []):
            if 'rounds' in sim['game_data']:
                thinking_tokens_per_round = [
                    r.get('llm_metadata', {}).get('challenger', {}).get('thinking_tokens', 0)
                    for r in sim['game_data']['rounds'] if r.get('llm_metadata')
                ]
                avg_tokens = np.mean([t for t in thinking_tokens_per_round if t is not None]) if thinking_tokens_per_round else 0
            else:
                avg_tokens = sim.get('game_data', {}).get('llm_metadata', {}).get('challenger', {}).get('thinking_tokens', 0)
            
            records.append({
                'game': sim['game_name'], 'model': sim['challenger_model'], 'condition': sim['condition_name'],
                'simulation_id': sim['simulation_id'], 'avg_thinking_tokens': avg_tokens
            })
    return pd.DataFrame(records)

def _get_thinking_data_per_round(results_dir: Path) -> pd.DataFrame:
    """Extracts per-round thinking tokens for dynamic games."""
    records = []
    thinking_models = [
        "gemini_2_5_flash_thinking_low", "gemini_2_5_flash_thinking_medium",
        "gemini_2_5_flash_lite_thinking_low", "gemini_2_5_flash_lite_thinking_medium"
    ]
    for file_path in results_dir.glob("*/*/*_competition_result*.json"):
        with open(file_path, 'r') as f: data = json.load(f)

        if data.get('challenger_model') not in thinking_models or data.get('game_name') not in ['green_porter', 'athey_bagwell']:
            continue

        for sim in data.get('simulation_results', []):
            for i, round_data in enumerate(sim['game_data'].get('rounds', [])):
                if round_data.get('llm_metadata'):
                    records.append({
                        'game': sim['game_name'], 'model': sim['challenger_model'], 'condition': sim['condition_name'],
                        'simulation_id': sim['simulation_id'], 'round': i + 1,
                        'thinking_tokens': round_data.get('llm_metadata', {}).get('challenger', {}).get('thinking_tokens', 0)
                    })
    return pd.DataFrame(records)

def _get_magic_data_per_sim(results_dir: Path) -> pd.DataFrame:
    """Calculates MAgIC metrics for each individual simulation from raw data."""
    magic_calc = MAgICMetricsCalculator()
    records = []
    for file_path in results_dir.glob("*/*/*_competition_result*.json"):
        with open(file_path, 'r') as f:
            data = json.load(f)
        for sim in data.get('simulation_results', []):
            game_result = GameResult(**sim)
            # We also need performance metrics for green_porter's reversion_frequency
            if game_result.game_name == 'green_porter':
                reversion_freq = game_result.game_data.get('reversion_frequency', 0)
                records.append({
                    'game': sim['game_name'], 'model': sim['challenger_model'], 'condition': sim['condition_name'],
                    'simulation_id': sim['simulation_id'], 'metric': 'reversion_frequency', 'value': reversion_freq
                })
            
            magic_metrics = magic_calc.calculate_all_magic_metrics([game_result], 'challenger')
            for name, metric_obj in magic_metrics.items():
                records.append({
                    'game': sim['game_name'], 'model': sim['challenger_model'], 'condition': sim['condition_name'],
                    'simulation_id': sim['simulation_id'], 'metric': name, 'value': metric_obj.value
                })
    return pd.DataFrame(records)


# --- Plot and Table Generation Functions ---

def _plot_avg_token_usage(thinking_sim_df: pd.DataFrame, plots_dir: Path):
    """Plot 1: Bar chart of average total thinking token count per decision."""
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("Generating Plot 4.1: Average Thinking Token Usage...")
    
    if thinking_sim_df.empty:
        logger.warning("No thinking data available to generate token usage plot.")
        return

    plot_data = thinking_sim_df.copy()
    model_order = [
        "gemini_2_5_flash_lite_thinking_low", "gemini_2_5_flash_lite_thinking_medium",
        "gemini_2_5_flash_thinking_low", "gemini_2_5_flash_thinking_medium"
    ]
    
    plt.figure(figsize=(16, 9))
    sns.barplot(data=plot_data, x='game', y='avg_thinking_tokens', hue='model', 
                hue_order=model_order, errorbar=None, palette='viridis',
                order=['salop', 'spulber', 'green_porter', 'athey_bagwell'])
    
    plt.title('Average Thinking Tokens Consumed Per Strategic Decision', fontsize=18)
    plt.xlabel('Game Environment', fontsize=14)
    plt.ylabel('Average Thinking Tokens', fontsize=14)
    plt.xticks(rotation=0)
    plt.legend(title='Challenger Model')
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plots_dir / "P4.1_avg_token_usage.png")
    plt.close()

def _plot_token_usage_over_time(thinking_round_df: pd.DataFrame, plots_dir: Path):
    """Plot 2: Line plots of average token count per round for dynamic games."""
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("Generating Plot 4.2: Thinking Token Usage Over Time...")

    if thinking_round_df.empty:
        logger.warning("No per-round thinking data available to generate time-series plots.")
        return
        
    plot_data = thinking_round_df.copy()
    plot_data['Market Size'] = np.where(plot_data['condition'].str.contains('more_players'), '5-Player', '3-Player')

    for game_name in ['green_porter', 'athey_bagwell']:
        game_specific_data = plot_data[plot_data['game'] == game_name]
        if game_specific_data.empty: continue

        g = sns.relplot(data=game_specific_data, x='round', y='thinking_tokens', hue='model', 
                        style='Market Size', kind='line', height=7, aspect=1.5, 
                        palette='plasma', facet_kws={'legend_out': True})
        g.fig.suptitle(f"{game_name.replace('_', ' ').title()}: Thinking Dynamics Over Time", y=1.03, fontsize=16)
        g.set_axis_labels("Game Round", "Average Thinking Tokens")
        plt.savefig(plots_dir / f"P4.2_{game_name}_token_usage_over_time.png")
        plt.close()

def _plot_thinking_impact_scatter(thinking_sim_df: pd.DataFrame, magic_sim_df: pd.DataFrame, plots_dir: Path):
    """Plot 3: Scatter plot of thinking tokens vs. key behavioral metrics."""
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("Generating Plot 4.3: Scatter Plots of Thinking vs. Behavior...")

    merged_data = pd.merge(thinking_sim_df, magic_sim_df, on=['game', 'model', 'condition', 'simulation_id'])
    
    scatter_map = {'green_porter': 'reversion_frequency', 'athey_bagwell': 'deception'}

    for game, metric in scatter_map.items():
        plot_data = merged_data[(merged_data['game'] == game) & (merged_data['metric'] == metric)].copy()
        if plot_data.empty: continue
        
        plot_data.loc[:, 'Market Size'] = np.where(plot_data['condition'].str.contains('more_players'), '5-Player', '3-Player')

        plt.figure(figsize=(12, 8))
        sns.scatterplot(data=plot_data, x='avg_thinking_tokens', y='value', hue='Market Size', style='model', s=150)
        
        plt.title(f"{game.replace('_', ' ').title()}: Thinking vs. {metric.replace('_', ' ').title()}", fontsize=16)
        plt.xlabel("Average Thinking Tokens per Simulation", fontsize=12)
        plt.ylabel(f"Final {metric.replace('_', ' ').title().replace('Reversion Frequency', 'Reversion Rate')}", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(title='Market Size & Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(plots_dir / f"P4.3_{game}_thinking_vs_{metric}.png")
        plt.close()

def _create_cost_benefit_table(thinking_sim_df: pd.DataFrame, perf_df: pd.DataFrame, magic_df: pd.DataFrame, tables_dir: Path):
    """Table 1: Creates the final cost-benefit summary table."""
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("Creating Table 4.1: Cost-Benefit Analysis of Thinking Budget...")

    # Filter for Gemini 2.5 models only
    perf_df = perf_df[perf_df['model'].str.contains('gemini_2_5')]
    magic_df = magic_df[magic_df['model'].str.contains('gemini_2_5')]
    thinking_sim_df = thinking_sim_df[thinking_sim_df['model'].str.contains('gemini_2_5')]
    
    avg_tokens = thinking_sim_df.groupby(['game', 'model', 'condition'])['avg_thinking_tokens'].mean().reset_index()
    
    # Create zero token rows for non-thinking 2.5 models
    non_thinking_models = [m for m in perf_df['model'].unique() if 'thinking_off' in m]
    all_conditions = perf_df[['game', 'condition']].drop_duplicates()
    zero_token_rows = []
    for model in non_thinking_models:
        for _, row in all_conditions.iterrows():
            zero_token_rows.append({'game': row['game'], 'model': model, 'condition': row['condition'], 'avg_thinking_tokens': 0})
    thinking_data = pd.concat([avg_tokens, pd.DataFrame(zero_token_rows)], ignore_index=True)

    key_metrics_map = {'salop': 'self_awareness', 'spulber': 'self_awareness', 'green_porter': 'cooperation', 'athey_bagwell': 'reasoning'}
    
    perf_subset = perf_df[perf_df['metric'] == 'average_profit'][['game', 'model', 'condition', 'mean']].rename(columns={'mean': 'Average Profit/NPV'})
    magic_subset = magic_df[magic_df['metric'].isin(key_metrics_map.values())][['game', 'model', 'condition', 'metric', 'mean']]

    filtered_magic = magic_subset.copy()
    filtered_magic['is_key_metric'] = filtered_magic.apply(lambda row: key_metrics_map.get(row['game']) == row['metric'], axis=1)
    filtered_magic = filtered_magic[filtered_magic['is_key_metric']].drop(columns=['metric', 'is_key_metric']).rename(columns={'mean': 'Key Capability Score'})

    final_df = pd.merge(thinking_data, perf_subset, on=['game', 'model', 'condition'])
    final_df = pd.merge(final_df, filtered_magic, on=['game', 'model', 'condition'])
    
    final_df['Market Size'] = np.where(final_df['condition'].str.contains('more_players'), '5-Player', '3-Player')
    
    # Keep the full model name
    final_df = final_df.sort_values(['game', 'Market Size', 'model'])
    
    # Reorder columns for clarity
    final_df = final_df[['game', 'Market Size', 'model', 'avg_thinking_tokens', 'Key Capability Score', 'Average Profit/NPV']]

    # Save to a single CSV file
    table_path = tables_dir / "T4.1_cost_benefit_analysis.csv"
    final_df.to_csv(table_path, index=False, float_format='%.2f')
    logger.info(f"Saved consolidated cost-benefit table to {table_path}")

# --- Main Visualization Function ---

def visualize_rq4():
    """Generates all tables and plots for Research Question 4."""
    if not PLOT_LIBS_INSTALLED:
        print("Plotting libraries not installed. Skipping visualization.")
        return
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)-20s - %(levelname)-8s - %(message)s')
    logger = logging.getLogger("RQ4Visualizer")
    logger.info("--- Generating visualizations for RQ4: Thinking & Strategic Capability ---")
    
    analysis_dir = get_analysis_dir()
    results_dir = get_experiments_dir()

    plots_dir = analysis_dir / "plots" / "rq4"
    tables_dir = analysis_dir / "tables" / "rq4"
    plots_dir.mkdir(exist_ok=True, parents=True)
    tables_dir.mkdir(exist_ok=True, parents=True)

    try:
        # Load aggregated data
        perf_df = pd.read_csv(analysis_dir / "performance_metrics.csv")
        magic_df_agg = pd.read_csv(analysis_dir / "magic_behavioral_metrics.csv")

        # Process raw data for thinking and per-simulation metrics
        thinking_sim_df = _get_thinking_data_per_sim(results_dir)
        thinking_round_df = _get_thinking_data_per_round(results_dir)
        magic_sim_df = _get_magic_data_per_sim(results_dir) # <-- Correct per-sim data

        # Generate all outputs based on your plan
        _plot_avg_token_usage(thinking_sim_df, plots_dir)
        _plot_token_usage_over_time(thinking_round_df, plots_dir)
        _plot_thinking_impact_scatter(thinking_sim_df, magic_sim_df, plots_dir)
        _create_cost_benefit_table(thinking_sim_df, perf_df, magic_df_agg, tables_dir)
        
        logger.info("--- Finished RQ4 visualizations ---")
        
    except FileNotFoundError as e:
        logger.error(f"Failed to find necessary file for RQ4 visualizations: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during RQ4 visualization: {e}", exc_info=True)


if __name__ == '__main__':
    visualize_rq4()