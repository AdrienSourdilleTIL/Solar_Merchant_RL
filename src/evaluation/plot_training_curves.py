"""
Plot Training Curves from TensorBoard Logs
==========================================

Visualizes training convergence for battery and commitment agents.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

# Paths
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs'
BATTERY_TB = OUTPUT_PATH / 'battery_agent' / 'tensorboard'
COMMITMENT_TB = OUTPUT_PATH / 'commitment_agent' / 'tensorboard'


def load_tensorboard_scalars(
    log_dir: Path,
    scalar_name: str = 'rollout/ep_rew_mean',
    run_name: str = None,
    all_runs: bool = False
) -> pd.DataFrame:
    """
    Load scalar data from TensorBoard event files.

    Args:
        log_dir: Path to TensorBoard log directory
        scalar_name: Name of scalar to extract
        run_name: Specific run to load (e.g., 'SAC_1')
        all_runs: If True, load all runs with 'run' column

    Returns:
        DataFrame with 'step' and 'value' columns (and 'run' if all_runs)
    """
    from tensorboard.backend.event_processing import event_accumulator

    # Find run directories
    run_dirs = sorted(log_dir.glob('SAC_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(f"No SAC_* directories found in {log_dir}")

    if run_name:
        run_dirs = [d for d in run_dirs if d.name == run_name]
        if not run_dirs:
            raise FileNotFoundError(f"Run '{run_name}' not found in {log_dir}")

    if not all_runs:
        run_dirs = [run_dirs[0]]

    all_data = []
    for run_dir in run_dirs:
        print(f"  Loading from: {run_dir.name}")

        # Load event accumulator
        ea = event_accumulator.EventAccumulator(str(run_dir))
        ea.Reload()

        # Get available scalars
        available = ea.Tags().get('scalars', [])
        if scalar_name not in available:
            print(f"    Available scalars: {available}")
            continue

        # Extract data
        scalars = ea.Scalars(scalar_name)
        for s in scalars:
            row = {'step': s.step, 'value': s.value}
            if all_runs:
                row['run'] = run_dir.name
            all_data.append(row)

    if not all_data:
        raise KeyError(f"Scalar '{scalar_name}' not found in any run")

    return pd.DataFrame(all_data)


def smooth_ema(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Apply exponential moving average smoothing."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)


def plot_training_curves(
    agents_data: Dict[str, pd.DataFrame],
    output_path: Path = None,
    smoothing: float = 0.9,
):
    """
    Plot training curves for multiple agents.

    Args:
        agents_data: Dict mapping agent names to DataFrames
        output_path: Path to save figure
        smoothing: EMA smoothing weight
    """
    n_agents = len(agents_data)
    fig, axes = plt.subplots(1, n_agents, figsize=(6 * n_agents, 5))

    if n_agents == 1:
        axes = [axes]

    colors = {
        'Battery Agent': '#2196F3',
        'Commitment Agent': '#4CAF50',
    }

    for ax, (agent_name, df) in zip(axes, agents_data.items()):
        if df.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(agent_name)
            continue

        color = colors.get(agent_name, '#333333')

        # Plot raw data (faded)
        ax.plot(df['step'], df['value'], alpha=0.2, color=color, linewidth=0.5)

        # Plot smoothed data
        smoothed = smooth_ema(df['value'].values, smoothing)
        ax.plot(df['step'], smoothed, color=color, linewidth=2, label='Smoothed')

        ax.set_xlabel('Training Steps', fontsize=11)
        ax.set_ylabel('Episode Reward (EUR)', fontsize=11)
        ax.set_title(f'{agent_name} Training Convergence', fontsize=12)
        ax.grid(alpha=0.3)

        # Add final value annotation
        final_val = smoothed[-1] if len(smoothed) > 0 else 0
        ax.axhline(y=final_val, color=color, linestyle='--', alpha=0.5, linewidth=1)
        ax.annotate(f'Final: {final_val:.0f}',
                   xy=(df['step'].iloc[-1], final_val),
                   xytext=(-60, 10), textcoords='offset points',
                   fontsize=9, color=color)

    fig.suptitle('SAC Training Convergence', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_combined_training_curves(
    agents_data: Dict[str, pd.DataFrame],
    output_path: Path = None,
    smoothing: float = 0.9,
):
    """
    Plot all training curves on a single normalized plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {
        'Battery Agent': '#2196F3',
        'Commitment Agent': '#4CAF50',
    }

    for agent_name, df in agents_data.items():
        if df.empty:
            continue

        color = colors.get(agent_name, '#333333')

        # Normalize steps to percentage of training
        max_steps = df['step'].max()
        normalized_steps = df['step'] / max_steps * 100

        # Plot raw (faded)
        ax.plot(normalized_steps, df['value'], alpha=0.15, color=color, linewidth=0.5)

        # Plot smoothed
        smoothed = smooth_ema(df['value'].values, smoothing)
        ax.plot(normalized_steps, smoothed, color=color, linewidth=2.5, label=agent_name)

    ax.set_xlabel('Training Progress (%)', fontsize=12)
    ax.set_ylabel('Episode Reward (EUR)', fontsize=12)
    ax.set_title('Training Convergence: Battery vs Commitment Agent', fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(alpha=0.3)

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def plot_battery_runs_comparison(
    battery_df: pd.DataFrame,
    output_path: Path = None,
    smoothing: float = 0.9,
    min_steps: int = 1000,  # Skip noisy initial points
):
    """Plot both battery agent training runs for comparison."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'SAC_1': '#FF9800', 'SAC_2': '#2196F3'}
    labels = {
        'SAC_1': 'Battery Agent (random commits)',
        'SAC_2': 'Battery Agent (trained commits)'
    }

    for run_name in battery_df['run'].unique():
        run_data = battery_df[battery_df['run'] == run_name].sort_values('step')
        # Skip noisy initial points (first few episodes)
        run_data = run_data[run_data['step'] >= min_steps]

        if run_data.empty:
            continue

        color = colors.get(run_name, '#333333')
        label = labels.get(run_name, run_name)

        # Plot raw (faded)
        ax.plot(run_data['step'], run_data['value'], alpha=0.15, color=color, linewidth=0.5)

        # Plot smoothed
        smoothed = smooth_ema(run_data['value'].values, smoothing)
        ax.plot(run_data['step'], smoothed, color=color, linewidth=2.5, label=label)

        # Add final value annotation
        final_val = smoothed[-1]
        ax.axhline(y=final_val, color=color, linestyle='--', alpha=0.3, linewidth=1)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward (EUR)', fontsize=12)
    ax.set_title('Battery Agent Training Convergence\n(Training data: 2015-2021)', fontsize=13)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(alpha=0.3)

    # Add note about data
    ax.text(0.02, 0.02, 'Note: Rewards measured on training rollouts, not test data',
            transform=ax.transAxes, fontsize=8, color='gray', style='italic')

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.close(fig)
    return fig


def main():
    """Load TensorBoard data and generate plots."""
    print("=" * 70)
    print("TRAINING CURVES VISUALIZATION")
    print("=" * 70)

    agents_data = {}

    # Load Battery Agent data - check all runs
    print("\nLoading Battery Agent logs (all runs)...")
    battery_df_all = pd.DataFrame()
    try:
        battery_df_all = load_tensorboard_scalars(BATTERY_TB, all_runs=True)
        print(f"  Loaded {len(battery_df_all)} total data points")

        # Show stats per run
        if 'run' in battery_df_all.columns:
            for run_name in sorted(battery_df_all['run'].unique()):
                run_data = battery_df_all[battery_df_all['run'] == run_name]
                print(f"    {run_name}: steps {run_data['step'].min()}-{run_data['step'].max()}, "
                      f"final reward: {run_data['value'].iloc[-1]:.2f}")

        # Use most recent run (SAC_2) for main plots
        if 'run' in battery_df_all.columns and 'SAC_2' in battery_df_all['run'].values:
            agents_data['Battery Agent'] = battery_df_all[battery_df_all['run'] == 'SAC_2'].copy()
        else:
            agents_data['Battery Agent'] = battery_df_all
    except (FileNotFoundError, KeyError) as e:
        print(f"  Error: {e}")
        agents_data['Battery Agent'] = pd.DataFrame()

    # Load Commitment Agent data
    print("\nLoading Commitment Agent logs...")
    try:
        commitment_df = load_tensorboard_scalars(COMMITMENT_TB)
        agents_data['Commitment Agent'] = commitment_df
        print(f"  Loaded {len(commitment_df)} data points")
        print(f"  Steps: {commitment_df['step'].min()} to {commitment_df['step'].max()}")
        print(f"  Final reward: {commitment_df['value'].iloc[-1]:.2f}")
    except (FileNotFoundError, KeyError) as e:
        print(f"  Error: {e}")
        agents_data['Commitment Agent'] = pd.DataFrame()

    if all(df.empty for df in agents_data.values()):
        print("\nNo training data found!")
        return

    # Generate battery runs comparison (if multiple runs exist)
    if not battery_df_all.empty and 'run' in battery_df_all.columns and battery_df_all['run'].nunique() > 1:
        print("\nGenerating battery runs comparison...")
        output_file = OUTPUT_PATH / 'battery_training_runs.png'
        plot_battery_runs_comparison(battery_df_all, output_file)

    # Generate separate plots
    print("Generating separate training curves...")
    output_file = OUTPUT_PATH / 'training_curves_separate.png'
    plot_training_curves(agents_data, output_file)

    # Generate combined plot
    print("Generating combined training curves...")
    output_file = OUTPUT_PATH / 'training_curves_combined.png'
    plot_combined_training_curves(agents_data, output_file)

    print("\nDone!")


if __name__ == '__main__':
    main()
