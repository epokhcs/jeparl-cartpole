"""Compare results across multiple experimental runs.

This script loads results from multiple experiments and generates
comparison plots and statistical tests.

Usage:
    python scripts/compare_results.py --results-dir results/experiments/20240315_120000
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tensorboard_scalars(log_dir, tags):
    """Load scalar data from TensorBoard logs.

    Args:
        log_dir: Path to TensorBoard logs
        tags: List of scalar tags to load

    Returns:
        Dictionary mapping tags to DataFrames with 'step' and 'value' columns
    """
    ea = event_accumulator.EventAccumulator(str(log_dir))
    ea.Reload()

    data = {}
    available_tags = ea.Tags().get('scalars', [])

    for tag in tags:
        if tag in available_tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = pd.DataFrame({'step': steps, 'value': values})

    return data


def load_all_experiments(results_dir, configs=(1, 2, 3, 4)):
    """Load all experiment results.

    Args:
        results_dir: Base results directory
        configs: Configuration IDs to load

    Returns:
        Dictionary mapping config_id to list of run data
    """
    results_dir = Path(results_dir)

    config_data = {config_id: [] for config_id in configs}

    # Find all experiment directories
    for config_id in configs:
        pattern = f"config{config_id}_*"
        config_dirs = list(results_dir.glob(pattern))

        for exp_dir in config_dirs:
            # Find tensorboard logs
            tb_dir = exp_dir / 'logs' / 'tensorboard'

            if tb_dir.exists():
                try:
                    # Load key metrics
                    data = load_tensorboard_scalars(
                        tb_dir,
                        tags=[
                            'train/episode_reward',
                            'train/episode_length',
                            'train/total_loss',
                            'train/jepa_loss',
                            'train/actor_loss',
                            'train/critic_loss',
                            'train/avg_variance'
                        ]
                    )

                    if data:
                        config_data[config_id].append({
                            'path': exp_dir,
                            'data': data
                        })
                except Exception as e:
                    print(f"Warning: Failed to load {exp_dir}: {e}")

    return config_data


def plot_learning_curves(config_data, save_path=None):
    """Plot learning curves for all configurations.

    Args:
        config_data: Dictionary mapping config_id to list of runs
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'orange'}
    labels = {
        1: 'Config 1: Baseline (No JEPA)',
        2: 'Config 2: JEPA + RL Gradients (Best)',
        3: 'Config 3: JEPA Only (Collapses)',
        4: 'Config 4: JEPA + Regularization'
    }

    for config_id in sorted(config_data.keys()):
        runs = config_data[config_id]

        if not runs:
            continue

        # Aggregate episode rewards
        all_curves = []

        for run in runs:
            if 'train/episode_reward' in run['data']:
                df = run['data']['train/episode_reward']
                all_curves.append(df)

        if not all_curves:
            continue

        # Align curves by step
        # Find common step range
        min_steps = min(df['step'].max() for df in all_curves)
        step_grid = np.arange(0, min_steps, 1000)

        # Interpolate each curve to common grid
        interpolated = []
        for df in all_curves:
            interp_values = np.interp(step_grid, df['step'], df['value'])
            interpolated.append(interp_values)

        # Compute mean and std
        mean_values = np.mean(interpolated, axis=0)
        std_values = np.std(interpolated, axis=0)

        # Plot
        ax.plot(step_grid, mean_values, color=colors[config_id],
                label=labels[config_id], linewidth=2)
        ax.fill_between(step_grid,
                        mean_values - std_values,
                        mean_values + std_values,
                        color=colors[config_id], alpha=0.2)

    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Episode Return', fontsize=14)
    ax.set_title('Learning Curves: All Configurations', fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_variance_over_time(config_data, save_path=None):
    """Plot embedding variance over time.

    Args:
        config_data: Dictionary mapping config_id to list of runs
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    colors = {2: 'green', 3: 'red', 4: 'orange'}
    labels = {
        2: 'Config 2: JEPA + RL Gradients',
        3: 'Config 3: JEPA Only (Collapses)',
        4: 'Config 4: JEPA + Regularization'
    }

    for config_id in [2, 3, 4]:  # Skip Config 1 (no JEPA)
        runs = config_data.get(config_id, [])

        for run in runs:
            if 'train/avg_variance' in run['data']:
                df = run['data']['train/avg_variance']
                ax.plot(df['step'], df['value'],
                       color=colors[config_id], alpha=0.3, linewidth=1)

    # Add collapse threshold
    ax.axhline(y=0.01, color='red', linestyle='--',
              label='Collapse Threshold', linewidth=2)

    # Legend with config labels
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=colors[2], lw=2, label=labels[2]),
        Line2D([0], [0], color=colors[3], lw=2, label=labels[3]),
        Line2D([0], [0], color=colors[4], lw=2, label=labels[4]),
        Line2D([0], [0], color='red', lw=2, linestyle='--', label='Collapse Threshold')
    ]
    ax.legend(handles=legend_elements, fontsize=11)

    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel('Average Embedding Variance', fontsize=14)
    ax.set_title('Embedding Variance Over Time', fontsize=16)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def compute_final_performance(config_data, last_n_episodes=10):
    """Compute final performance statistics.

    Args:
        config_data: Dictionary mapping config_id to list of runs
        last_n_episodes: Number of final episodes to average

    Returns:
        DataFrame with statistics per configuration
    """
    results = []

    for config_id in sorted(config_data.keys()):
        runs = config_data[config_id]

        final_returns = []

        for run in runs:
            if 'train/episode_reward' in run['data']:
                df = run['data']['train/episode_reward']

                # Get last N episodes
                final_return = df['value'].tail(last_n_episodes).mean()
                final_returns.append(final_return)

        if final_returns:
            results.append({
                'Config': f'Config {config_id}',
                'Mean': np.mean(final_returns),
                'Std': np.std(final_returns),
                'Min': np.min(final_returns),
                'Max': np.max(final_returns),
                'N': len(final_returns)
            })

    return pd.DataFrame(results)


def plot_performance_comparison(stats_df, save_path=None):
    """Plot final performance comparison.

    Args:
        stats_df: DataFrame with performance statistics
        save_path: Optional save path
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['blue', 'green', 'red', 'orange']
    x = np.arange(len(stats_df))

    bars = ax.bar(x, stats_df['Mean'], yerr=stats_df['Std'],
                   color=colors[:len(stats_df)], alpha=0.7,
                   capsize=5, error_kw={'linewidth': 2})

    ax.set_xticks(x)
    ax.set_xticklabels(stats_df['Config'], fontsize=12)
    ax.set_ylabel('Mean Episode Return', fontsize=14)
    ax.set_title('Final Performance Comparison\n(Last 10 Episodes Average)', fontsize=16)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=11)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def statistical_comparison(config_data):
    """Perform statistical tests between configurations.

    Args:
        config_data: Dictionary mapping config_id to list of runs

    Returns:
        DataFrame with pairwise comparison results
    """
    # Extract final returns for each config
    final_returns = {}

    for config_id in sorted(config_data.keys()):
        runs = config_data[config_id]
        returns = []

        for run in runs:
            if 'train/episode_reward' in run['data']:
                df = run['data']['train/episode_reward']
                final_return = df['value'].tail(10).mean()
                returns.append(final_return)

        final_returns[config_id] = returns

    # Pairwise t-tests
    results = []
    config_ids = sorted(final_returns.keys())

    for i, config_i in enumerate(config_ids):
        for config_j in config_ids[i+1:]:
            if final_returns[config_i] and final_returns[config_j]:
                t_stat, p_value = stats.ttest_ind(
                    final_returns[config_i],
                    final_returns[config_j]
                )

                results.append({
                    'Comparison': f'Config {config_i} vs Config {config_j}',
                    't-statistic': t_stat,
                    'p-value': p_value,
                    'Significant (p<0.05)': 'Yes' if p_value < 0.05 else 'No'
                })

    return pd.DataFrame(results)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Compare experimental results')
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/comparisons',
        help='Output directory for comparison plots'
    )
    parser.add_argument(
        '--configs',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help='Configuration IDs to compare'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EXPERIMENT COMPARISON")
    print("="*60)
    print(f"Results directory: {args.results_dir}")
    print(f"Configurations: {args.configs}")
    print(f"Output: {output_dir}")

    # Load all experiments
    print("\nLoading experiments...")
    config_data = load_all_experiments(args.results_dir, configs=args.configs)

    for config_id, runs in config_data.items():
        print(f"  Config {config_id}: {len(runs)} runs")

    total_runs = sum(len(runs) for runs in config_data.values())
    if total_runs == 0:
        print("\nNo experiment data found!")
        return

    print(f"\nTotal: {total_runs} runs")

    # Generate comparisons
    print("\nGenerating comparison plots...")

    print("  1. Learning curves...")
    plot_learning_curves(
        config_data,
        save_path=output_dir / 'learning_curves_comparison.png'
    )

    print("  2. Variance over time...")
    plot_variance_over_time(
        config_data,
        save_path=output_dir / 'variance_comparison.png'
    )

    print("  3. Final performance...")
    stats_df = compute_final_performance(config_data)
    print("\n" + stats_df.to_string(index=False))

    plot_performance_comparison(
        stats_df,
        save_path=output_dir / 'performance_comparison.png'
    )

    # Statistical tests
    print("\n  4. Statistical comparison...")
    stat_results = statistical_comparison(config_data)

    if not stat_results.empty:
        print("\n" + stat_results.to_string(index=False))

        # Save to CSV
        stat_results.to_csv(output_dir / 'statistical_tests.csv', index=False)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print("="*60)
    print(f"Plots and statistics saved to: {output_dir}")


if __name__ == "__main__":
    main()
