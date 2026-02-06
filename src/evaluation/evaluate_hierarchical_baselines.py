"""
Evaluation Script for Hierarchical Baselines
=============================================

Compares all combinations of commitment and battery policies,
including trained agents when available.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate
from itertools import product

from environment.hierarchical_orchestrator import HierarchicalOrchestrator
from environment.solar_plant import PlantConfig
from baselines.hierarchical_baselines import (
    COMMITMENT_POLICIES,
    BATTERY_POLICIES,
    get_commitment_policy,
    get_battery_policy,
)

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models'

# Plant config
PLANT_CONFIG = PlantConfig()


def evaluate_policy_combination(
    data: pd.DataFrame,
    commitment_policy_name: str,
    battery_policy_name: str,
    num_episodes: int = 20,
    days_per_episode: int = 7,
    seed: int = 42,
) -> dict:
    """Evaluate a specific combination of policies."""
    commitment_policy = get_commitment_policy(commitment_policy_name)
    battery_policy = get_battery_policy(battery_policy_name)

    orch = HierarchicalOrchestrator(
        data=data,
        plant_config=PLANT_CONFIG,
        commitment_policy=commitment_policy,
        battery_policy=battery_policy,
    )

    metrics = orch.evaluate(
        num_episodes=num_episodes,
        days_per_episode=days_per_episode,
        seed=seed,
    )

    return {
        'commitment_policy': commitment_policy_name,
        'battery_policy': battery_policy_name,
        **metrics,
    }


def evaluate_trained_agents(
    data_path: str,
    num_episodes: int = 20,
    days_per_episode: int = 7,
    seed: int = 42,
) -> dict:
    """Evaluate trained agents if available."""
    orch = HierarchicalOrchestrator.from_trained_agents(
        data_path=data_path,
        plant_config=PLANT_CONFIG,
    )

    metrics = orch.evaluate(
        num_episodes=num_episodes,
        days_per_episode=days_per_episode,
        seed=seed,
    )

    return {
        'commitment_policy': metrics['commitment_type'],
        'battery_policy': metrics['battery_type'],
        **metrics,
    }


def main():
    """Run comprehensive baseline evaluation."""
    test_path = DATA_PATH / 'test.csv'

    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        print("Run prepare_dataset.py first.")
        return

    print("=" * 80)
    print("HIERARCHICAL BASELINES EVALUATION")
    print("=" * 80)

    data = pd.read_csv(test_path, parse_dates=['datetime'])

    num_episodes = 20
    days_per_episode = 7

    results = []

    # Evaluate all baseline combinations
    commitment_names = ['conservative', 'aggressive', 'price_aware']
    battery_names = ['greedy', 'conservative', 'do_nothing', 'smoothing']

    total_combinations = len(commitment_names) * len(battery_names)
    print(f"\nEvaluating {total_combinations} baseline combinations...")

    for i, (commit_name, battery_name) in enumerate(product(commitment_names, battery_names)):
        print(f"  [{i+1}/{total_combinations}] {commit_name} + {battery_name}...")

        metrics = evaluate_policy_combination(
            data=data,
            commitment_policy_name=commit_name,
            battery_policy_name=battery_name,
            num_episodes=num_episodes,
            days_per_episode=days_per_episode,
        )

        results.append({
            'Commitment': commit_name,
            'Battery': battery_name,
            'Daily Profit': metrics['mean_daily_profit'],
            'Episode Profit': metrics['mean_episode_profit'],
            'Imbalance Cost': metrics['mean_imbalance_cost'],
            'Episodes': metrics['num_episodes'],
        })

    # Evaluate trained agents
    print("\nEvaluating trained agents...")
    trained_metrics = evaluate_trained_agents(
        data_path=str(test_path),
        num_episodes=num_episodes,
        days_per_episode=days_per_episode,
    )

    results.append({
        'Commitment': f"RL ({trained_metrics['commitment_policy']})",
        'Battery': f"RL ({trained_metrics['battery_policy']})",
        'Daily Profit': trained_metrics['mean_daily_profit'],
        'Episode Profit': trained_metrics['mean_episode_profit'],
        'Imbalance Cost': trained_metrics['mean_imbalance_cost'],
        'Episodes': trained_metrics['num_episodes'],
    })

    # Sort by daily profit
    results_sorted = sorted(results, key=lambda x: x['Daily Profit'], reverse=True)

    # Format for display
    for r in results_sorted:
        r['Daily Profit'] = f"{r['Daily Profit']:,.2f}"
        r['Episode Profit'] = f"{r['Episode Profit']:,.2f}"
        r['Imbalance Cost'] = f"{r['Imbalance Cost']:,.2f}"

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS (sorted by Daily Profit)")
    print("=" * 80)
    print()
    print(tabulate(results_sorted, headers='keys', tablefmt='grid'))

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    profits = [float(r['Daily Profit'].replace(',', '')) for r in results_sorted]
    print(f"\nBest daily profit: {max(profits):,.2f} EUR")
    print(f"Worst daily profit: {min(profits):,.2f} EUR")
    print(f"Range: {max(profits) - min(profits):,.2f} EUR")

    # Find best baseline (excluding trained)
    baseline_results = [r for r in results_sorted if 'RL' not in r['Commitment']]
    if baseline_results:
        best_baseline = baseline_results[0]
        print(f"\nBest baseline: {best_baseline['Commitment']} + {best_baseline['Battery']}")
        print(f"  Daily profit: {best_baseline['Daily Profit']} EUR")

    # Compare RL to best baseline
    rl_results = [r for r in results_sorted if 'RL' in r['Commitment']]
    if rl_results and baseline_results:
        rl_profit = float(rl_results[0]['Daily Profit'].replace(',', ''))
        baseline_profit = float(baseline_results[0]['Daily Profit'].replace(',', ''))
        improvement = ((rl_profit - baseline_profit) / abs(baseline_profit)) * 100 if baseline_profit != 0 else 0
        print(f"\nRL vs best baseline: {improvement:+.1f}%")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
