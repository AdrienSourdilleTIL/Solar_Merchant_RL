"""Evaluate all baseline policies and produce comparison report.

Runs conservative, aggressive, and price-aware baselines on the test dataset,
prints a formatted comparison table, and saves results to CSV.
"""

import sys
from pathlib import Path

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.baselines import aggressive_policy, conservative_policy, price_aware_policy
from src.environment import SolarMerchantEnv
from src.evaluation.evaluate import evaluate_policy, print_comparison, save_results

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results' / 'metrics'


def main():
    """Evaluate all three baseline policies on the test dataset."""
    test_df = pd.read_csv(DATA_PATH / 'test.csv', parse_dates=['datetime'])

    policies = [
        ("Conservative (80%)", conservative_policy),
        ("Aggressive (100%)", aggressive_policy),
        ("Price-Aware", price_aware_policy),
    ]

    results = []
    for name, policy in policies:
        env = SolarMerchantEnv(test_df.copy())
        result = evaluate_policy(policy, env, n_episodes=10)
        result["policy"] = name
        results.append(result)
        env.close()

    print_comparison(results)
    save_results(results, RESULTS_PATH / 'baseline_comparison.csv')
    print(f"\nResults saved to {RESULTS_PATH / 'baseline_comparison.csv'}")


if __name__ == '__main__':
    main()
