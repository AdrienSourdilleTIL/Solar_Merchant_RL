"""
Plot Cumulative Rewards for All Policies
=========================================

Compares trained hierarchical agents against baseline policies
by plotting cumulative profit over time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from environment.hierarchical_orchestrator import HierarchicalOrchestrator
from environment.solar_plant import PlantConfig
from baselines.hierarchical_baselines import (
    conservative_commitment_policy,
    aggressive_commitment_policy,
    price_aware_commitment_policy,
    greedy_battery_policy,
    conservative_battery_policy,
    do_nothing_battery_policy,
)

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs'

# Plant config
PLANT_CONFIG = PlantConfig()


def create_policy_orchestrator(
    data: pd.DataFrame,
    commitment_policy,
    battery_policy,
    policy_name: str
) -> Tuple[HierarchicalOrchestrator, str]:
    """Create orchestrator with specific policies."""
    return HierarchicalOrchestrator(
        data=data,
        plant_config=PLANT_CONFIG,
        commitment_policy=commitment_policy,
        battery_policy=battery_policy,
    ), policy_name


def run_continuous_evaluation(
    orchestrator: HierarchicalOrchestrator,
    start_idx: int,
    num_days: int,
    initial_soc: float = 0.5
) -> List[float]:
    """
    Run evaluation and return daily profits.

    Returns list of daily profits for cumulative plotting.
    """
    try:
        result = orchestrator.run_episode(
            start_idx=start_idx,
            num_days=num_days,
            initial_soc=initial_soc
        )
        return result.daily_profits
    except (ValueError, IndexError) as e:
        print(f"Error during evaluation: {e}")
        return []


def plot_cumulative_rewards(
    policies_data: Dict[str, List[float]],
    output_path: Path = None,
    show: bool = True
):
    """
    Plot cumulative rewards for all policies.

    Args:
        policies_data: Dict mapping policy names to daily profits
        output_path: Path to save figure
        show: Whether to display figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    # Color scheme
    colors = {
        'Hierarchical RL': '#2196F3',  # Blue
        'Conservative + Greedy': '#4CAF50',  # Green
        'Aggressive + Greedy': '#F44336',  # Red
        'Price-Aware + Greedy': '#FF9800',  # Orange
        'Conservative + Conservative': '#9C27B0',  # Purple
        'Conservative + Do-Nothing': '#607D8B',  # Gray
    }

    for policy_name, daily_profits in policies_data.items():
        if not daily_profits:
            continue

        cumulative = np.cumsum(daily_profits)
        days = np.arange(1, len(cumulative) + 1)

        color = colors.get(policy_name, '#000000')
        linewidth = 3 if 'RL' in policy_name else 2
        linestyle = '-' if 'RL' in policy_name else '--'

        ax.plot(days, cumulative, label=policy_name,
                color=color, linewidth=linewidth, linestyle=linestyle)

    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Cumulative Profit (EUR)', fontsize=12)
    ax.set_title('Cumulative Profit Comparison: RL Agents vs Baselines', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='-')

    fig.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def main():
    """Run comparison and generate plot."""
    test_path = DATA_PATH / 'test.csv'

    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        print("Run prepare_dataset.py first.")
        return

    print("=" * 70)
    print("CUMULATIVE REWARDS COMPARISON")
    print("=" * 70)

    # Load data
    data = pd.read_csv(test_path, parse_dates=['datetime'])
    print(f"Loaded {len(data)} hours of test data")

    # Evaluation parameters
    num_days = 60  # Simulate 60 days for clear trends
    seed = 42
    np.random.seed(seed)

    # Find a good starting point (need commitment hour)
    start_idx = 0
    while start_idx < len(data) - (num_days + 2) * 24:
        if data.iloc[start_idx]['hour'] == PLANT_CONFIG.commitment_hour:
            break
        start_idx += 1

    initial_soc = 0.5

    print(f"\nSimulating {num_days} days starting from index {start_idx}")
    print(f"Initial battery SOC: {initial_soc * 100:.0f}%\n")

    policies_data = {}

    # 1. Hierarchical RL (trained agents)
    print("Evaluating: Hierarchical RL...")
    rl_orchestrator = HierarchicalOrchestrator.from_trained_agents(
        data_path=str(test_path),
        plant_config=PLANT_CONFIG,
    )
    rl_profits = run_continuous_evaluation(
        rl_orchestrator, start_idx, num_days, initial_soc
    )
    policies_data['Hierarchical RL'] = rl_profits
    print(f"  Total profit: {sum(rl_profits):.2f} EUR")

    # 2. Conservative commitment + Greedy battery
    print("Evaluating: Conservative + Greedy...")
    orchestrator, name = create_policy_orchestrator(
        data, conservative_commitment_policy, greedy_battery_policy,
        'Conservative + Greedy'
    )
    profits = run_continuous_evaluation(orchestrator, start_idx, num_days, initial_soc)
    policies_data[name] = profits
    print(f"  Total profit: {sum(profits):.2f} EUR")

    # 3. Aggressive commitment + Greedy battery
    print("Evaluating: Aggressive + Greedy...")
    orchestrator, name = create_policy_orchestrator(
        data, aggressive_commitment_policy, greedy_battery_policy,
        'Aggressive + Greedy'
    )
    profits = run_continuous_evaluation(orchestrator, start_idx, num_days, initial_soc)
    policies_data[name] = profits
    print(f"  Total profit: {sum(profits):.2f} EUR")

    # 4. Price-aware commitment + Greedy battery
    print("Evaluating: Price-Aware + Greedy...")
    orchestrator, name = create_policy_orchestrator(
        data, price_aware_commitment_policy, greedy_battery_policy,
        'Price-Aware + Greedy'
    )
    profits = run_continuous_evaluation(orchestrator, start_idx, num_days, initial_soc)
    policies_data[name] = profits
    print(f"  Total profit: {sum(profits):.2f} EUR")

    # 5. Conservative commitment + Conservative battery
    print("Evaluating: Conservative + Conservative...")
    orchestrator, name = create_policy_orchestrator(
        data, conservative_commitment_policy, conservative_battery_policy,
        'Conservative + Conservative'
    )
    profits = run_continuous_evaluation(orchestrator, start_idx, num_days, initial_soc)
    policies_data[name] = profits
    print(f"  Total profit: {sum(profits):.2f} EUR")

    # 6. Conservative commitment + Do-Nothing battery
    print("Evaluating: Conservative + Do-Nothing...")
    orchestrator, name = create_policy_orchestrator(
        data, conservative_commitment_policy, do_nothing_battery_policy,
        'Conservative + Do-Nothing'
    )
    profits = run_continuous_evaluation(orchestrator, start_idx, num_days, initial_soc)
    policies_data[name] = profits
    print(f"  Total profit: {sum(profits):.2f} EUR")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Policy':<35} {'Total Profit':>15} {'Daily Avg':>12}")
    print("-" * 70)
    for name, profits in policies_data.items():
        total = sum(profits)
        avg = np.mean(profits) if profits else 0
        print(f"{name:<35} {total:>15.2f} {avg:>12.2f}")

    # Generate plot
    output_file = OUTPUT_PATH / 'cumulative_rewards.png'
    print(f"\nGenerating plot: {output_file}")
    plot_cumulative_rewards(policies_data, output_file, show=False)

    print("\nDone!")


if __name__ == '__main__':
    main()
