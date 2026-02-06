"""Visualize evaluation results for Solar Merchant RL.

Generates performance comparison charts from pre-computed evaluation CSVs.
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
    parser.add_argument('--show', action='store_true', help='Display chart interactively')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)

    try:
        df = load_comparison_data(metrics_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    output_path = output_dir / 'performance_comparison.png'

    # Choose backend based on --show flag
    backend = 'TkAgg' if args.show else 'Agg'
    fig = plot_net_profit_comparison(df, output_path, backend=backend)

    import matplotlib.pyplot as plt
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
