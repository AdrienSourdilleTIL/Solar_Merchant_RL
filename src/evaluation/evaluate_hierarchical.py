"""
Evaluation Script for Hierarchical Agents
==========================================

Evaluates the hierarchical agent system (commitment + battery)
and compares against baselines.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from tabulate import tabulate

from environment.hierarchical_orchestrator import HierarchicalOrchestrator
from environment.solar_plant import PlantConfig

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models'

# Plant config
PLANT_CONFIG = PlantConfig()


def create_heuristic_orchestrator(data_path: str) -> HierarchicalOrchestrator:
    """Create orchestrator with heuristic policies."""
    data = pd.read_csv(data_path, parse_dates=['datetime'])
    return HierarchicalOrchestrator(
        data=data,
        plant_config=PLANT_CONFIG,
    )


def create_trained_orchestrator(data_path: str) -> HierarchicalOrchestrator:
    """Create orchestrator with trained agents."""
    return HierarchicalOrchestrator.from_trained_agents(
        data_path=data_path,
        plant_config=PLANT_CONFIG,
    )


def main():
    """Run hierarchical agent evaluation."""
    test_path = DATA_PATH / 'test.csv'

    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        print("Run prepare_dataset.py first.")
        return

    print("=" * 70)
    print("HIERARCHICAL AGENT EVALUATION")
    print("=" * 70)

    num_episodes = 20
    days_per_episode = 7

    results = []

    # Evaluate heuristic baseline
    print("\nEvaluating heuristic baseline...")
    heuristic_orch = create_heuristic_orchestrator(str(test_path))
    heuristic_metrics = heuristic_orch.evaluate(
        num_episodes=num_episodes,
        days_per_episode=days_per_episode,
        seed=42
    )
    results.append({
        'Policy': 'Heuristic (baseline)',
        'Commitment': heuristic_metrics['commitment_type'],
        'Battery': heuristic_metrics['battery_type'],
        'Mean Daily Profit': f"{heuristic_metrics['mean_daily_profit']:.2f}",
        'Mean Episode Profit': f"{heuristic_metrics['mean_episode_profit']:.2f}",
        'Mean Imbalance Cost': f"{heuristic_metrics['mean_imbalance_cost']:.2f}",
        'Episodes': heuristic_metrics['num_episodes'],
    })

    # Evaluate trained agents (if available)
    print("\nEvaluating trained agents...")
    trained_orch = create_trained_orchestrator(str(test_path))
    trained_metrics = trained_orch.evaluate(
        num_episodes=num_episodes,
        days_per_episode=days_per_episode,
        seed=42
    )
    results.append({
        'Policy': 'Hierarchical RL',
        'Commitment': trained_metrics['commitment_type'],
        'Battery': trained_metrics['battery_type'],
        'Mean Daily Profit': f"{trained_metrics['mean_daily_profit']:.2f}",
        'Mean Episode Profit': f"{trained_metrics['mean_episode_profit']:.2f}",
        'Mean Imbalance Cost': f"{trained_metrics['mean_imbalance_cost']:.2f}",
        'Episodes': trained_metrics['num_episodes'],
    })

    # Print results table
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(tabulate(results, headers='keys', tablefmt='grid'))

    # Comparison
    if len(results) >= 2:
        heur_profit = float(results[0]['Mean Daily Profit'])
        trained_profit = float(results[1]['Mean Daily Profit'])
        improvement = ((trained_profit - heur_profit) / abs(heur_profit)) * 100 if heur_profit != 0 else 0

        print(f"\nImprovement over heuristic: {improvement:+.1f}%")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
