"""Visualize evaluation results for Solar Merchant RL.

Generates performance comparison charts from pre-computed evaluation CSVs
and training curves from TensorBoard logs.
Outputs publication-quality PNG figures for README embedding.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    import matplotlib.figure

# Paths
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results'
DEFAULT_METRICS = RESULTS_PATH / 'metrics'
DEFAULT_FIGURES = RESULTS_PATH / 'figures'
DEFAULT_TENSORBOARD = Path(__file__).parent.parent.parent / 'outputs' / 'tensorboard'


def load_comparison_data(metrics_dir: Path) -> pd.DataFrame:
    """Load evaluation results for visualization.

    Tries multi_seed_evaluation.csv first (has std for error bars),
    falls back to agent_vs_baselines.csv, then individual files.

    Args:
        metrics_dir: Path to results/metrics/ directory.

    Returns:
        DataFrame with columns: policy, net_profit (or net_profit_mean),
        and optionally net_profit_std for error bars.

    Raises:
        FileNotFoundError: If no evaluation CSV files found.
    """
    multi_seed = metrics_dir / 'multi_seed_evaluation.csv'
    if multi_seed.exists():
        return pd.read_csv(multi_seed)

    combined = metrics_dir / 'agent_vs_baselines.csv'
    if combined.exists():
        return pd.read_csv(combined)

    # Fallback: merge individual files
    agent_f = metrics_dir / 'agent_evaluation.csv'
    baseline_f = metrics_dir / 'baseline_comparison.csv'
    if agent_f.exists() and baseline_f.exists():
        return pd.concat([pd.read_csv(agent_f), pd.read_csv(baseline_f)], ignore_index=True)

    raise FileNotFoundError(
        f"No evaluation CSV files found in {metrics_dir}.\n"
        "Run evaluation scripts first:\n"
        "  python src/evaluation/evaluate_baselines.py\n"
        "  python src/evaluation/evaluate_agent.py\n"
        "  python src/evaluation/evaluate_multi_seed.py"
    )


def load_tensorboard_data(log_dir: Path, all_runs: bool = False) -> pd.DataFrame:
    """Load training metrics from TensorBoard event files.

    Reads SAC run directories and extracts episode reward scalars for
    training curve visualization.

    Args:
        log_dir: Path to TensorBoard log directory (e.g., outputs/tensorboard).
        all_runs: If True, load all runs and include 'run' column for labels.
            If False (default), load only the most recent run.

    Returns:
        DataFrame with columns: step, value (episode reward).
        If all_runs=True, also includes 'run' column with run names.

    Raises:
        FileNotFoundError: If no TensorBoard logs found with helpful message.
    """
    from tensorboard.backend.event_processing import event_accumulator

    # Find run directories
    run_dirs = sorted(log_dir.glob('SAC_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(
            f"No TensorBoard logs found in {log_dir}.\n"
            "Run training first:\n"
            "  python src/training/train.py"
        )

    # Select which runs to load
    dirs_to_load = run_dirs if all_runs else [run_dirs[0]]

    all_data = []
    for run_dir in dirs_to_load:
        event_files = list(run_dir.glob('events.out.tfevents.*'))
        if not event_files:
            if not all_runs:
                raise FileNotFoundError(
                    f"No event files found in {run_dir}.\n"
                    "Run training first:\n"
                    "  python src/training/train.py"
                )
            continue  # Skip empty runs when loading all

        # Load event accumulator
        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()

        # Extract rollout/ep_rew_mean scalar
        try:
            scalars = ea.Scalars('rollout/ep_rew_mean')
        except KeyError:
            if not all_runs:
                raise FileNotFoundError(
                    f"No 'rollout/ep_rew_mean' scalar found in {run_dir}.\n"
                    "This may indicate an incomplete training run."
                )
            continue  # Skip runs without the scalar when loading all

        run_name = run_dir.name
        for s in scalars:
            row = {'step': s.step, 'value': s.value}
            if all_runs:
                row['run'] = run_name
            all_data.append(row)

    if not all_data:
        raise FileNotFoundError(
            f"No valid TensorBoard data found in {log_dir}.\n"
            "Run training first:\n"
            "  python src/training/train.py"
        )

    return pd.DataFrame(all_data)


def smooth_data(values: pd.Series, weight: float = 0.9) -> pd.Series:
    """Apply exponential moving average smoothing.

    Args:
        values: Raw values to smooth.
        weight: Smoothing weight (0-1, higher = smoother). Default 0.9.
            Weight of 0 returns original values, 1 returns constant.

    Returns:
        Smoothed values as pandas Series.

    Raises:
        ValueError: If weight is not in [0, 1] range.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"Smoothing weight must be in [0, 1], got {weight}")

    if len(values) == 0:
        return values.copy()

    if weight == 0.0:
        return values.copy()

    smoothed = []
    last = None
    for v in values:
        # Handle NaN: skip NaN values, keep last valid smoothed value
        if pd.isna(v):
            smoothed.append(v)  # Preserve NaN in output
            continue
        if last is None:
            last = v
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return pd.Series(smoothed)


def plot_training_curves(
    df: pd.DataFrame,
    output_path: Path | None = None,
    backend: str = 'Agg',
    smoothing: float = 0.9,
) -> matplotlib.figure.Figure:
    """Generate training reward curve chart.

    Creates a publication-quality line chart showing episode reward
    over training steps. Optionally applies smoothing with raw data
    as light background.

    Args:
        df: DataFrame with 'step' and 'value' columns.
        output_path: Path to save PNG. If None, not saved.
        backend: Matplotlib backend. Default 'Agg' for headless.
        smoothing: EMA smoothing weight (0 = no smoothing, 1 = full). Default 0.9.

    Returns:
        Matplotlib Figure object.

    Raises:
        ValueError: If DataFrame is empty or missing required columns.
    """
    if df.empty:
        raise ValueError(
            "Cannot plot training curves: DataFrame is empty.\n"
            "This may indicate incomplete training logs."
        )
    if 'step' not in df.columns or 'value' not in df.columns:
        raise ValueError(
            f"DataFrame must have 'step' and 'value' columns, got: {list(df.columns)}"
        )

    import matplotlib
    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Check if multiple runs (has 'run' column)
    has_multiple_runs = 'run' in df.columns and df['run'].nunique() > 1

    if has_multiple_runs:
        # Plot each run with different colors
        colors = plt.cm.tab10.colors
        runs = df['run'].unique()
        for i, run_name in enumerate(runs):
            run_data = df[df['run'] == run_name]
            color = colors[i % len(colors)]

            # Plot raw data as light background
            ax.plot(run_data['step'], run_data['value'], alpha=0.2, color=color, linewidth=0.5)

            # Plot smoothed line
            if smoothing > 0:
                smoothed = smooth_data(run_data['value'].reset_index(drop=True), smoothing)
                ax.plot(run_data['step'], smoothed.values, color=color, linewidth=2, label=run_name)
            else:
                ax.plot(run_data['step'], run_data['value'], color=color, linewidth=2, label=run_name)
    else:
        # Single run plotting (original behavior)
        # Plot raw data as light background
        ax.plot(df['step'], df['value'], alpha=0.3, color='#2196F3', linewidth=0.5)

        # Plot smoothed line (or raw if no smoothing)
        if smoothing > 0:
            smoothed = smooth_data(df['value'], smoothing)
            ax.plot(df['step'], smoothed, color='#2196F3', linewidth=2, label='Episode Reward')
        else:
            ax.plot(df['step'], df['value'], color='#2196F3', linewidth=2, label='Episode Reward')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward (EUR)', fontsize=12)
    title = 'SAC Training Progress — Episode Reward over Time'
    if has_multiple_runs:
        title = 'SAC Training Progress — All Runs Comparison'
    ax.set_title(title, fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def plot_net_profit_comparison(
    df: pd.DataFrame,
    output_path: Path | None = None,
    backend: str = 'Agg',
) -> matplotlib.figure.Figure:
    """Generate bar chart comparing net profit across policies.

    Args:
        df: DataFrame with policy names and net_profit values.
            Supports both plain columns (net_profit) and
            multi-seed columns (net_profit_mean, net_profit_std).
        output_path: Path to save PNG file. If None, figure is not saved.
        backend: Matplotlib backend to use. Default 'Agg' for headless.

    Returns:
        The matplotlib Figure object (for testing or interactive display).
    """
    import matplotlib
    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    # Detect columns
    if 'net_profit_mean' in df.columns:
        values = df['net_profit_mean']
        errors = df['net_profit_std'] if 'net_profit_std' in df.columns else None
    else:
        values = df['net_profit']
        errors = None

    policies = df['policy']

    # Color: highlight RL agent differently
    colors = []
    labels = []
    for p in policies:
        if 'rl' in p.lower() or 'sac' in p.lower() or 'agent' in p.lower():
            colors.append('#2196F3')  # Blue for RL agent
            labels.append('RL Agent')
        else:
            colors.append('#9E9E9E')  # Grey for baselines
            labels.append('Baseline')

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(policies)),
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
    )

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=15, ha='right', fontsize=11)
    ax.set_ylabel('Net Profit (EUR / episode)', fontsize=12)
    ax.set_title('RL Agent vs Baseline Policies — Net Profit Comparison', fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(axis='y', alpha=0.3)

    # Add legend for color coding
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2196F3', edgecolor='black', label='RL Agent'),
        Patch(facecolor='#9E9E9E', edgecolor='black', label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # Value labels on bars
    for bar, val in zip(bars, values):
        y_pos = bar.get_height()
        # For positive bars: label above; for negative bars: label below
        if y_pos >= 0:
            offset = 5
            va = 'bottom'
        else:
            offset = -5
            va = 'top'
        ax.annotate(
            f'EUR {val:,.0f}',
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, offset),
            textcoords='offset points',
            ha='center', va=va,
            fontsize=9, fontweight='bold',
        )

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig


def main() -> None:
    """Entry point for performance visualization CLI."""
    parser = argparse.ArgumentParser(description='Generate performance charts')
    parser.add_argument('--metrics-dir', type=str, default=str(DEFAULT_METRICS))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_FIGURES))
    parser.add_argument('--tensorboard-dir', type=str, default=str(DEFAULT_TENSORBOARD),
                        help='Path to TensorBoard logs directory')
    parser.add_argument('--show', action='store_true', help='Display chart interactively')
    parser.add_argument('--training-curves', action='store_true',
                        help='Generate training curves chart from TensorBoard logs')
    parser.add_argument('--all-runs', action='store_true',
                        help='Plot all training runs with labels (default: most recent only)')
    parser.add_argument('--performance', action='store_true',
                        help='Generate performance comparison chart')
    parser.add_argument('--all', action='store_true', help='Generate all charts')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)
    tensorboard_dir = Path(args.tensorboard_dir)

    # Default: generate performance comparison if no specific flag
    if not args.training_curves and not args.performance and not args.all:
        args.performance = True

    # Choose backend based on --show flag
    backend = 'TkAgg' if args.show else 'Agg'

    import matplotlib.pyplot as plt
    figures = []

    # Generate performance comparison chart
    if args.performance or args.all:
        try:
            df = load_comparison_data(metrics_dir)
            output_path = output_dir / 'performance_comparison.png'
            fig = plot_net_profit_comparison(df, output_path, backend=backend)
            figures.append(fig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            if not args.all:
                sys.exit(1)

    # Generate training curves chart
    if args.training_curves or args.all:
        try:
            df = load_tensorboard_data(tensorboard_dir, all_runs=args.all_runs)
            output_path = output_dir / 'training_curves.png'
            fig = plot_training_curves(df, output_path, backend=backend)
            figures.append(fig)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            if not args.all:
                sys.exit(1)

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)


if __name__ == '__main__':
    main()
