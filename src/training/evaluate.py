"""
Evaluation Script for Solar Merchant RL Agent

Evaluates a trained agent on the test set and outputs detailed metrics.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from stable_baselines3 import SAC

from environment.solar_merchant_env import SolarMerchantEnv

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs'

# Plant configuration (must match training)
PLANT_CONFIG = {
    'plant_capacity_mw': 20.0,
    'battery_capacity_mwh': 10.0,
    'battery_power_mw': 5.0,
    'battery_efficiency': 0.92,
    'battery_degradation_cost': 0.01,
}


def evaluate_agent(model, env, num_steps: int = None):
    """
    Run agent through environment and collect detailed metrics.

    Returns:
        DataFrame with hourly data and summary dict
    """
    obs, _ = env.reset(seed=42)

    if num_steps is None:
        num_steps = len(env.data) - env.current_idx - 1

    records = []
    total_reward = 0

    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        records.append({
            'step': step,
            'hour': info['hour'],
            'pv_actual': info['pv_actual'],
            'committed': info['committed'],
            'delivered': info['delivered'],
            'imbalance': info['imbalance'],
            'price': info['price'],
            'revenue': info['revenue'],
            'imbalance_cost': info['imbalance_cost'],
            'battery_soc': info['battery_soc'],
            'battery_throughput': info['battery_throughput'],
            'reward': reward,
        })

        if terminated or truncated:
            break

    df = pd.DataFrame(records)

    # Calculate summary metrics
    summary = {
        'total_steps': len(df),
        'total_days': len(df) / 24,
        'total_reward': total_reward,
        'total_revenue': df['revenue'].sum(),
        'total_imbalance_cost': df['imbalance_cost'].sum(),
        'net_profit': df['revenue'].sum() - df['imbalance_cost'].sum(),
        'total_pv_produced': df['pv_actual'].sum(),
        'total_delivered': df['delivered'].sum(),
        'total_committed': df['committed'].sum(),
        'mean_absolute_imbalance': df['imbalance'].abs().mean(),
        'imbalance_hours_short': (df['imbalance'] < -0.1).sum(),
        'imbalance_hours_long': (df['imbalance'] > 0.1).sum(),
        'battery_cycles': df['battery_throughput'].sum() / (2 * PLANT_CONFIG['battery_capacity_mwh']),
        'mean_battery_soc': df['battery_soc'].mean(),
    }

    # Annualize
    hours_per_year = 8760
    scale_factor = hours_per_year / len(df)
    summary['annual_revenue'] = summary['total_revenue'] * scale_factor
    summary['annual_imbalance_cost'] = summary['total_imbalance_cost'] * scale_factor
    summary['annual_net_profit'] = summary['net_profit'] * scale_factor
    summary['annual_battery_cycles'] = summary['battery_cycles'] * scale_factor

    return df, summary


def print_summary(summary: dict, label: str = "Agent"):
    """Print formatted summary."""
    print(f"\n{'='*60}")
    print(f"{label} PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Evaluation period: {summary['total_days']:.0f} days ({summary['total_steps']:,} hours)")
    print(f"\nFinancial Performance:")
    print(f"  Total Revenue:        EUR {summary['total_revenue']:>12,.0f}")
    print(f"  Total Imbalance Cost: EUR {summary['total_imbalance_cost']:>12,.0f}")
    print(f"  Net Profit:           EUR {summary['net_profit']:>12,.0f}")
    print(f"\nAnnualized (EUR/year):")
    print(f"  Revenue:              EUR {summary['annual_revenue']:>12,.0f}")
    print(f"  Imbalance Cost:       EUR {summary['annual_imbalance_cost']:>12,.0f}")
    print(f"  Net Profit:           EUR {summary['annual_net_profit']:>12,.0f}")
    print(f"\nOperational Metrics:")
    print(f"  PV Produced:          {summary['total_pv_produced']:>12,.0f} MWh")
    print(f"  Energy Delivered:     {summary['total_delivered']:>12,.0f} MWh")
    print(f"  Energy Committed:     {summary['total_committed']:>12,.0f} MWh")
    print(f"  Mean Abs Imbalance:   {summary['mean_absolute_imbalance']:>12.2f} MWh")
    print(f"  Hours Short:          {summary['imbalance_hours_short']:>12,}")
    print(f"  Hours Long:           {summary['imbalance_hours_long']:>12,}")
    print(f"\nBattery Usage:")
    print(f"  Total Cycles:         {summary['battery_cycles']:>12.1f}")
    print(f"  Annual Cycles:        {summary['annual_battery_cycles']:>12.1f}")
    print(f"  Mean SOC:             {summary['mean_battery_soc']:>12.1f} MWh "
          f"({100*summary['mean_battery_soc']/PLANT_CONFIG['battery_capacity_mwh']:.0f}%)")


def main():
    # Find best model
    model_candidates = [
        MODEL_PATH / 'best' / 'best_model.zip',
        MODEL_PATH / 'solar_merchant_final.zip',
    ]

    model_path = None
    for candidate in model_candidates:
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        print("No trained model found. Run train.py first.")
        print(f"Looked in: {MODEL_PATH}")
        return

    print(f"Loading model from {model_path}")
    model = SAC.load(str(model_path))

    # Load test data
    test_path = DATA_PATH / 'test.csv'
    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        print("Run prepare_dataset.py first.")
        return

    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, parse_dates=['datetime'])
    env = SolarMerchantEnv(test_df, **PLANT_CONFIG)

    # Evaluate
    print("\nRunning evaluation...")
    results_df, summary = evaluate_agent(model, env)

    # Print summary
    print_summary(summary, "RL Agent")

    # Save detailed results
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_PATH / 'evaluation_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to {output_file}")

    # Save summary
    summary_file = OUTPUT_PATH / 'evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    print(f"Summary saved to {summary_file}")


if __name__ == '__main__':
    main()
