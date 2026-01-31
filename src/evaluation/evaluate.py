"""Policy evaluation and comparison for Solar Merchant RL.

Provides functions to evaluate any trading policy over multiple episodes,
print formatted comparison tables, and save results to CSV.
"""

from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
import pandas as pd

# Plant constants for derived metrics
BATTERY_CAPACITY_MWH = 10.0
BATTERY_DEGRADATION_COST = 0.01  # EUR per MWh throughput


def evaluate_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    env: gym.Env,
    n_episodes: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a policy over multiple episodes and return averaged metrics.

    Runs the policy for n_episodes complete episodes, collecting financial
    and operational metrics from the environment's info dict at each step.

    Args:
        policy: Function mapping observation to action array.
        env: Gymnasium environment instance.
        n_episodes: Number of episodes to evaluate.
        seed: Base random seed (incremented per episode).

    Returns:
        Dict with averaged metrics:
            - revenue: Mean total revenue per episode (EUR)
            - imbalance_cost: Mean total imbalance cost per episode (EUR)
            - net_profit: Mean net profit per episode (EUR, revenue - imbalance_cost)
            - degradation_cost: Mean battery degradation cost per episode (EUR)
            - total_reward: Mean total env reward per episode (revenue - imbalance - degradation)
            - delivered: Mean total delivered energy per episode (MWh)
            - committed: Mean total committed energy per episode (MWh)
            - pv_produced: Mean total PV produced per episode (MWh)
            - delivery_ratio: Mean delivered/committed ratio
            - battery_cycles: Mean battery cycles per episode
            - hours: Mean episode length in hours
            - n_episodes: Number of episodes evaluated (not averaged)

    Raises:
        ValueError: If n_episodes < 1.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")

    episode_metrics: list[dict[str, float]] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        episode_revenue = 0.0
        episode_imbalance = 0.0
        episode_delivered = 0.0
        episode_committed = 0.0
        episode_pv = 0.0
        episode_battery_throughput = 0.0
        episode_reward = 0.0
        steps = 0

        done = False
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_revenue += info['revenue']
            episode_imbalance += info['imbalance_cost']
            episode_delivered += info['delivered']
            episode_committed += info['committed']
            episode_pv += info['pv_actual']
            episode_battery_throughput += info['battery_throughput']
            episode_reward += reward
            steps += 1

        delivery_ratio = episode_delivered / max(episode_committed, 1e-8)
        battery_cycles = episode_battery_throughput / (2 * BATTERY_CAPACITY_MWH)
        degradation_cost = episode_battery_throughput * BATTERY_DEGRADATION_COST

        episode_metrics.append({
            "revenue": episode_revenue,
            "imbalance_cost": episode_imbalance,
            "net_profit": episode_revenue - episode_imbalance,
            "degradation_cost": degradation_cost,
            "total_reward": episode_reward,
            "delivered": episode_delivered,
            "committed": episode_committed,
            "pv_produced": episode_pv,
            "delivery_ratio": delivery_ratio,
            "battery_cycles": battery_cycles,
            "hours": float(steps),
        })

    # Average across episodes
    avg: dict[str, float] = {}
    keys = episode_metrics[0].keys()
    for key in keys:
        avg[key] = sum(m[key] for m in episode_metrics) / n_episodes
    avg["n_episodes"] = float(n_episodes)

    return avg


def print_comparison(results: list[dict[str, float]]) -> None:
    """Print a formatted comparison table of policy evaluation results.

    Args:
        results: List of result dicts, each containing a 'policy' key with
            the policy name and metric values from evaluate_policy.
    """
    print("=" * 80)
    print("BASELINE POLICY COMPARISON")
    print("=" * 80)
    print()
    print(f"{'Policy':<30}  {'Net Profit':>10}  {'Imb Cost':>10}  {'Deliv %':>7}  {'Episodes':>8}")
    print("-" * 80)

    best_idx = 0
    best_profit = -float('inf')

    for i, r in enumerate(results):
        name = r.get("policy", f"Policy {i}")
        net_profit = r["net_profit"]
        imb_cost = r["imbalance_cost"]
        delivery = r["delivery_ratio"] * 100
        episodes = int(r.get("n_episodes", 0))

        print(f"{name:<30}  EUR {net_profit:>7,.0f}  EUR {imb_cost:>5,.0f}  {delivery:>6.1f}%  {episodes:>8}")

        if net_profit > best_profit:
            best_profit = net_profit
            best_idx = i

    print("-" * 80)
    print()
    best_name = results[best_idx].get("policy", f"Policy {best_idx}")
    print(f"Best policy: {best_name}")
    print(f"Mean net profit: EUR {best_profit:,.0f} per episode")


def save_results(results: list[dict[str, float]], output_path: Path) -> None:
    """Save evaluation results to a CSV file.

    Args:
        results: List of result dicts from evaluate_policy (with 'policy' key).
        output_path: Path to the output CSV file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure readable column order: policy name first, then key metrics
    preferred_order = [
        "policy", "net_profit", "revenue", "imbalance_cost", "degradation_cost",
        "total_reward", "delivered", "committed", "pv_produced", "delivery_ratio",
        "battery_cycles", "hours", "n_episodes",
    ]
    all_keys = list(results[0].keys())
    columns = [c for c in preferred_order if c in all_keys]
    columns += [c for c in all_keys if c not in columns]
    df = pd.DataFrame(results, columns=columns)
    df.to_csv(output_path, index=False)
