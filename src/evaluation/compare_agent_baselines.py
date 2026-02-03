"""Compare trained RL agent against baseline policies.

Loads pre-computed evaluation results for the agent and baselines,
calculates percentage improvement, and determines if RL beats rule-based strategies.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.evaluation.evaluate import print_comparison, save_results

# Paths
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results' / 'metrics'
DEFAULT_AGENT_RESULTS = RESULTS_PATH / 'agent_evaluation.csv'
DEFAULT_BASELINE_RESULTS = RESULTS_PATH / 'baseline_comparison.csv'


def load_results(csv_path: Path) -> list[dict[str, float]]:
    """Load evaluation results from CSV.

    Args:
        csv_path: Path to CSV file with policy results.

    Returns:
        List of result dicts, each with 'policy' key and metric values.

    Raises:
        FileNotFoundError: If CSV file does not exist.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df.to_dict(orient='records')


def calculate_improvement(agent_value: float, baseline_value: float) -> float:
    """Calculate percentage improvement of agent over baseline.

    Args:
        agent_value: Agent metric value.
        baseline_value: Baseline metric value.

    Returns:
        Percentage improvement: (agent - baseline) / abs(baseline) * 100.
    """
    if abs(baseline_value) < 1e-8:
        return float('inf') if agent_value > 0 else 0.0
    return (agent_value - baseline_value) / abs(baseline_value) * 100


def identify_best_baseline(baselines: list[dict[str, float]]) -> dict[str, float]:
    """Find the baseline with the highest net_profit.

    Args:
        baselines: List of baseline result dicts.

    Returns:
        The best-performing baseline dict.
    """
    return max(baselines, key=lambda b: b['net_profit'])


def determine_verdict(agent_profit: float, baselines: list[dict[str, float]]) -> str:
    """Determine if agent beats all baselines.

    Args:
        agent_profit: Agent net_profit.
        baselines: List of baseline result dicts with net_profit.

    Returns:
        "PASS" if agent beats all baselines,
        "FAIL_SOME" if agent beats some but not all,
        "FAIL_NONE" if agent beats none.
    """
    beaten = sum(1 for b in baselines if agent_profit > b['net_profit'])
    total = len(baselines)
    if beaten == total:
        return "PASS"
    if beaten == 0:
        return "FAIL_NONE"
    return "FAIL_SOME"


def main() -> None:
    """Load results, compare, and print verdict."""
    parser = argparse.ArgumentParser(description='Compare RL agent vs baselines')
    parser.add_argument('--agent-results', type=str,
                        default=str(DEFAULT_AGENT_RESULTS))
    parser.add_argument('--baseline-results', type=str,
                        default=str(DEFAULT_BASELINE_RESULTS))
    args = parser.parse_args()

    agent_path = Path(args.agent_results)
    baseline_path = Path(args.baseline_results)

    # Load results with user-friendly error messages
    try:
        agent_results = load_results(agent_path)
    except FileNotFoundError:
        print(f"Agent results not found at {agent_path}")
        print("Run evaluate_agent.py first to generate agent evaluation results.")
        sys.exit(1)

    try:
        baseline_results = load_results(baseline_path)
    except FileNotFoundError:
        print(f"Baseline results not found at {baseline_path}")
        print("Run evaluate_baselines.py first to generate baseline results.")
        sys.exit(1)

    # Combine into unified results list
    all_results = agent_results + baseline_results

    # Print formatted comparison table
    print_comparison(all_results)

    # Extract agent and baselines
    agent = agent_results[0]
    agent_profit = agent['net_profit']

    # Calculate improvement over each baseline
    print()
    print("IMPROVEMENT OVER BASELINES:")
    for baseline in baseline_results:
        improvement = calculate_improvement(agent_profit, baseline['net_profit'])
        name = baseline.get('policy', 'Unknown')
        print(f"  vs {name:<25} {improvement:>+.1f}%")

    # Identify best baseline and determine verdict
    best_baseline = identify_best_baseline(baseline_results)
    verdict = determine_verdict(agent_profit, baseline_results)
    beaten = sum(1 for b in baseline_results if agent_profit > b['net_profit'])
    total = len(baseline_results)

    print()
    if verdict == "PASS":
        print(f"VERDICT: PASS - Agent beats all {total} baselines.")
    elif verdict == "FAIL_NONE":
        print(f"VERDICT: FAIL - Agent beats none of {total} baselines.")
    else:
        print(f"VERDICT: FAIL - Agent beats {beaten} of {total} baselines.")
    best_name = best_baseline.get('policy', 'Unknown')
    print(f"Best baseline: {best_name} at EUR {best_baseline['net_profit']:,.0f}")

    # Build results with improvement columns (without mutating originals)
    save_list = []
    for r in all_results:
        entry = dict(r)
        if r is agent:
            entry['improvement_pct'] = 0.0
        else:
            entry['improvement_pct'] = calculate_improvement(
                agent_profit, r['net_profit']
            )
        save_list.append(entry)
    save_results(save_list, RESULTS_PATH / 'agent_vs_baselines.csv')
    print(f"\nResults saved to {RESULTS_PATH / 'agent_vs_baselines.csv'}")


if __name__ == '__main__':
    main()
