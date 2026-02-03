"""Multi-seed statistical evaluation for Solar Merchant RL.

Runs evaluation across multiple random seeds for the RL agent and all
baseline policies, then reports mean +/- std for each metric.

This tests whether results are robust to different episode orderings,
not just a single lucky draw.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from src.evaluation.evaluate import evaluate_policy, save_results

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results' / 'metrics'
DEFAULT_MODEL = Path(__file__).parent.parent.parent / 'models' / 'solar_merchant_final.zip'

DEFAULT_SEEDS = "42,123,456,789,1024"


def run_multi_seed(
    policy: Callable[[np.ndarray], np.ndarray],
    env_factory: Callable,
    seeds: list[int],
    n_episodes: int = 10,
) -> list[dict[str, float]]:
    """Run evaluate_policy() with multiple base seeds.

    Args:
        policy: Policy function (obs -> action).
        env_factory: Callable that creates a fresh env instance.
        seeds: List of base seeds to evaluate with.
        n_episodes: Episodes per seed.

    Returns:
        List of result dicts, one per seed.
    """
    results = []
    for seed in seeds:
        env = env_factory()
        result = evaluate_policy(policy, env, n_episodes=n_episodes, seed=seed)
        results.append(result)
        env.close()
    return results


def aggregate_results(seed_results: list[dict[str, float]]) -> dict[str, float]:
    """Compute mean and std across seed results.

    Args:
        seed_results: List of result dicts from run_multi_seed.

    Returns:
        Dict with {metric}_mean and {metric}_std keys for each numeric metric.
        Also includes n_seeds count.

    Raises:
        ValueError: If seed_results is empty.
    """
    if not seed_results:
        raise ValueError("seed_results must not be empty")
    numeric_keys = [
        k for k in seed_results[0]
        if isinstance(seed_results[0][k], (int, float))
    ]
    n = len(seed_results)
    agg: dict[str, float] = {}
    for key in numeric_keys:
        values = [r[key] for r in seed_results]
        agg[f"{key}_mean"] = float(np.mean(values))
        # Use sample std (ddof=1) for n >= 2, 0.0 for single seed
        agg[f"{key}_std"] = float(np.std(values, ddof=1)) if n >= 2 else 0.0
    agg["n_seeds"] = n
    return agg


def _order_aggregate_keys(agg: dict[str, float]) -> dict[str, float]:
    """Reorder aggregate result dict for readable CSV output.

    Places policy first, then metric pairs (mean, std) grouped together,
    then n_seeds last.

    Args:
        agg: Aggregate result dict with _mean/_std suffixed keys.

    Returns:
        OrderedDict-style dict with keys in logical order.
    """
    ordered: dict[str, float] = {}
    if "policy" in agg:
        ordered["policy"] = agg["policy"]

    # Group metric pairs: net_profit_mean, net_profit_std, revenue_mean, ...
    metric_order = [
        "net_profit", "revenue", "imbalance_cost", "degradation_cost",
        "total_reward", "delivered", "committed", "pv_produced",
        "delivery_ratio", "battery_cycles", "hours", "n_episodes",
    ]
    for metric in metric_order:
        mean_key = f"{metric}_mean"
        std_key = f"{metric}_std"
        if mean_key in agg:
            ordered[mean_key] = agg[mean_key]
        if std_key in agg:
            ordered[std_key] = agg[std_key]

    # Add n_seeds and any remaining keys
    if "n_seeds" in agg:
        ordered["n_seeds"] = agg["n_seeds"]
    for k, v in agg.items():
        if k not in ordered:
            ordered[k] = v

    return ordered


def print_multi_seed_table(all_policy_results: list[dict[str, float]]) -> None:
    """Print formatted table with mean +/- std for each policy.

    Args:
        all_policy_results: List of aggregate result dicts, each containing
            a 'policy' key and {metric}_mean / {metric}_std keys.
    """
    n_seeds = int(all_policy_results[0].get("n_seeds", 0))
    n_episodes = int(all_policy_results[0].get("n_episodes_mean", 0))

    print("=" * 80)
    print(f"MULTI-SEED STATISTICAL EVALUATION ({n_seeds} seeds x {n_episodes} episodes each)")
    print("=" * 80)
    print()
    print(f"{'Policy':<30}  {'Net Profit (mean +/- std)':>28}  {'Imb Cost (mean +/- std)':>26}")
    print("-" * 88)

    best_idx = 0
    best_profit = -float('inf')

    for i, r in enumerate(all_policy_results):
        name = r.get("policy", f"Policy {i}")
        net_mean = r.get("net_profit_mean", 0.0)
        net_std = r.get("net_profit_std", 0.0)
        imb_mean = r.get("imbalance_cost_mean", 0.0)
        imb_std = r.get("imbalance_cost_std", 0.0)

        print(
            f"{name:<30}  EUR {net_mean:>7,.0f} +/- {net_std:>5,.0f}"
            f"          EUR {imb_mean:>5,.0f} +/- {imb_std:>5,.0f}"
        )

        if net_mean > best_profit:
            best_profit = net_mean
            best_idx = i

    print("-" * 88)
    print()
    best_name = all_policy_results[best_idx].get("policy", f"Policy {best_idx}")
    print(f"Best policy: {best_name}")
    print(f"Mean net profit: EUR {best_profit:,.0f} per episode")


def main() -> None:
    """Run multi-seed evaluation for RL agent and all baselines."""
    import pandas as pd
    from src.environment import SolarMerchantEnv
    from src.training.train import load_model
    from src.baselines import conservative_policy, aggressive_policy, price_aware_policy
    from src.evaluation.evaluate_agent import make_agent_policy

    parser = argparse.ArgumentParser(
        description='Multi-seed statistical evaluation of RL agent and baselines'
    )
    parser.add_argument(
        '--model', type=str, default=str(DEFAULT_MODEL),
        help='Path to trained model .zip file'
    )
    parser.add_argument(
        '--seeds', type=str, default=DEFAULT_SEEDS,
        help='Comma-separated list of base seeds (default: 42,123,456,789,1024)'
    )
    parser.add_argument(
        '--episodes', type=int, default=10,
        help='Number of episodes per seed (default: 10)'
    )
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',')]

    # Check test data exists
    test_data_path = DATA_PATH / 'test.csv'
    if not test_data_path.exists():
        print(f"Test data not found at {test_data_path}")
        print("Run prepare_dataset.py first.")
        return

    test_df = pd.read_csv(test_data_path, parse_dates=['datetime'])

    def env_factory():
        return SolarMerchantEnv(test_df.copy())

    # Load model
    model_path = Path(args.model)
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Define all policies
    policies = [
        ("RL Agent (SAC)", make_agent_policy(model)),
        ("Conservative (80%)", conservative_policy),
        ("Aggressive (100%)", aggressive_policy),
        ("Price-Aware", price_aware_policy),
    ]

    # Run multi-seed evaluation for each policy
    all_aggregates = []
    all_per_seed = []
    for name, policy in policies:
        print(f"Evaluating {name} with {len(seeds)} seeds...")
        seed_results = run_multi_seed(
            policy, env_factory, seeds, n_episodes=args.episodes
        )
        # Preserve per-seed results for inspection (AC #2)
        for seed, result in zip(seeds, seed_results):
            per_seed_row = dict(result)
            per_seed_row["policy"] = name
            per_seed_row["seed"] = seed
            all_per_seed.append(per_seed_row)

        agg = aggregate_results(seed_results)
        agg["policy"] = name
        all_aggregates.append(agg)

    # Print and save results
    print_multi_seed_table(all_aggregates)

    # Save aggregate summary with ordered columns
    ordered_aggregates = [_order_aggregate_keys(a) for a in all_aggregates]
    save_results(ordered_aggregates, RESULTS_PATH / 'multi_seed_evaluation.csv')
    print(f"\nAggregate results saved to {RESULTS_PATH / 'multi_seed_evaluation.csv'}")

    # Save per-seed details for inspection
    save_results(all_per_seed, RESULTS_PATH / 'multi_seed_per_seed.csv')
    print(f"Per-seed details saved to {RESULTS_PATH / 'multi_seed_per_seed.csv'}")


if __name__ == '__main__':
    main()
