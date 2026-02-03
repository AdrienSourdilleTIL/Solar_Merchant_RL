"""Evaluate trained RL agent on test dataset.

Loads a trained SAC model, evaluates on 2022-2023 test data,
prints metrics, and saves results to CSV.
"""

import argparse
import sys
from pathlib import Path
from typing import Callable

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np

from src.evaluation.evaluate import evaluate_policy, print_comparison, save_results

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results' / 'metrics'
DEFAULT_MODEL = Path(__file__).parent.parent.parent / 'models' / 'solar_merchant_final.zip'


def make_agent_policy(model) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an SB3 model into the policy interface for evaluate_policy().

    Args:
        model: Trained SB3 model with predict() method.

    Returns:
        Policy function: obs -> action.
    """
    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def main() -> None:
    """Evaluate trained RL agent on the test dataset."""
    import pandas as pd
    from src.environment import SolarMerchantEnv
    from src.training.train import load_model

    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument('--model', type=str, default=str(DEFAULT_MODEL),
                        help='Path to trained model .zip file')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    model_path = Path(args.model)

    # Check test data exists
    test_data_path = DATA_PATH / 'test.csv'
    if not test_data_path.exists():
        print(f"Test data not found at {test_data_path}")
        print("Run prepare_dataset.py first.")
        return

    # Load trained model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Create test environment (constructor defaults, consistent with evaluate_baselines.py)
    print(f"Creating test environment from {test_data_path}...")
    test_df = pd.read_csv(test_data_path, parse_dates=['datetime'])
    env = SolarMerchantEnv(test_df.copy())

    # Wrap model into policy interface
    policy = make_agent_policy(model)

    # Evaluate
    print(f"Evaluating agent over {args.episodes} episodes (seed={args.seed})...")
    result = evaluate_policy(policy, env, n_episodes=args.episodes, seed=args.seed)
    result["policy"] = "RL Agent (SAC)"
    env.close()

    # Print results
    print_comparison([result])

    # Save results
    save_results([result], RESULTS_PATH / 'agent_evaluation.csv')
    print(f"\nResults saved to {RESULTS_PATH / 'agent_evaluation.csv'}")


if __name__ == '__main__':
    main()
