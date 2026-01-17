"""
Rule-Based Baseline for Solar Merchant

Implements simple heuristic policies to compare against the RL agent.
This establishes the minimum bar that RL must beat to be useful.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from environment.solar_merchant_env import SolarMerchantEnv

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs'

# Plant configuration (must match training)
PLANT_CONFIG = {
    'plant_capacity_mw': 20.0,
    'battery_capacity_mwh': 10.0,
    'battery_power_mw': 5.0,
    'battery_efficiency': 0.92,
    'battery_degradation_cost': 0.01,
}


class ConservativePolicy:
    """
    Conservative bidding policy.

    Strategy:
    - Commit only 80% of forecast (conservative to avoid short imbalance)
    - Use battery to fill gaps when under-producing
    - Charge battery when over-producing
    """

    def __init__(self, commitment_fraction: float = 0.80):
        self.commitment_fraction = commitment_fraction
        self.name = f"Conservative ({int(commitment_fraction*100)}% commitment)"

    def predict(self, obs: np.ndarray, env: SolarMerchantEnv) -> np.ndarray:
        """Generate action based on current state."""
        action = np.zeros(25, dtype=np.float32)

        # If commitment hour, set commitment schedule
        if env._is_commitment_hour():
            # Get tomorrow's forecast from observation
            # Obs structure: [hour, soc, committed(24), cumul_imb, forecast(24), prices(24), ...]
            forecast_start = 1 + 1 + 24 + 1  # After hour, soc, committed, cumul_imb
            forecast = obs[forecast_start:forecast_start + 24]

            # Commit conservative fraction of forecast
            # Denormalize forecast (it's normalized by plant capacity)
            forecast_mwh = forecast * PLANT_CONFIG['plant_capacity_mw']
            commitment_mwh = forecast_mwh * self.commitment_fraction

            # Convert back to fraction for action (relative to max possible)
            max_possible = forecast_mwh + PLANT_CONFIG['battery_power_mw']
            max_possible = np.maximum(max_possible, 0.01)  # Avoid division by zero
            action[:24] = commitment_mwh / max_possible

        # Battery action: try to meet commitment
        # Get current hour info
        hour = int(obs[0] * 24)
        soc = obs[1] * PLANT_CONFIG['battery_capacity_mwh']
        committed = obs[2 + hour] * PLANT_CONFIG['plant_capacity_mw']

        # Estimate current PV from observation
        pv_idx = 1 + 1 + 24 + 1 + 24 + 24  # Position of current PV in obs
        current_pv = obs[pv_idx] * PLANT_CONFIG['plant_capacity_mw']

        gap = committed - current_pv

        if gap > 0:
            # Under-producing: discharge battery
            discharge_needed = min(gap, PLANT_CONFIG['battery_power_mw'], soc)
            action[24] = 0.5 - (discharge_needed / PLANT_CONFIG['battery_power_mw']) * 0.5
        elif gap < 0:
            # Over-producing: charge battery
            charge_available = min(-gap, PLANT_CONFIG['battery_power_mw'],
                                   PLANT_CONFIG['battery_capacity_mwh'] - soc)
            action[24] = 0.5 + (charge_available / PLANT_CONFIG['battery_power_mw']) * 0.5
        else:
            action[24] = 0.5  # Idle

        return action


class AggressivePolicy:
    """
    Aggressive bidding policy.

    Strategy:
    - Commit 100% of forecast plus battery capacity
    - Maximize revenue by committing to deliver as much as possible
    - Risk: higher imbalance penalties when forecast is wrong
    """

    def __init__(self):
        self.name = "Aggressive (100% + battery)"

    def predict(self, obs: np.ndarray, env: SolarMerchantEnv) -> np.ndarray:
        """Generate action based on current state."""
        action = np.zeros(25, dtype=np.float32)

        if env._is_commitment_hour():
            # Commit maximum: full forecast + battery discharge capacity
            forecast_start = 1 + 1 + 24 + 1
            forecast = obs[forecast_start:forecast_start + 24]
            forecast_mwh = forecast * PLANT_CONFIG['plant_capacity_mw']

            # Commit forecast + assume battery helps during peak hours
            commitment_mwh = forecast_mwh + 0.5 * PLANT_CONFIG['battery_power_mw']

            max_possible = forecast_mwh + PLANT_CONFIG['battery_power_mw']
            max_possible = np.maximum(max_possible, 0.01)
            action[:24] = np.clip(commitment_mwh / max_possible, 0, 1)

        # Battery: always try to maximize delivery
        hour = int(obs[0] * 24)
        soc = obs[1] * PLANT_CONFIG['battery_capacity_mwh']
        committed = obs[2 + hour] * PLANT_CONFIG['plant_capacity_mw']
        pv_idx = 1 + 1 + 24 + 1 + 24 + 24
        current_pv = obs[pv_idx] * PLANT_CONFIG['plant_capacity_mw']

        gap = committed - current_pv

        if gap > 0:
            # Discharge aggressively
            discharge = min(gap * 1.2, PLANT_CONFIG['battery_power_mw'], soc)
            action[24] = 0.5 - (discharge / PLANT_CONFIG['battery_power_mw']) * 0.5
        else:
            # Charge when excess
            charge = min(-gap * 0.8, PLANT_CONFIG['battery_power_mw'],
                        PLANT_CONFIG['battery_capacity_mwh'] - soc)
            action[24] = 0.5 + (charge / PLANT_CONFIG['battery_power_mw']) * 0.5

        return action


class PriceAwarePolicy:
    """
    Price-aware bidding policy.

    Strategy:
    - Commit more during high-price hours
    - Commit less during low-price hours (save battery for better times)
    - Use price forecasts to optimize battery charge/discharge timing
    """

    def __init__(self, base_fraction: float = 0.85):
        self.base_fraction = base_fraction
        self.name = "Price-Aware"

    def predict(self, obs: np.ndarray, env: SolarMerchantEnv) -> np.ndarray:
        """Generate action based on current state."""
        action = np.zeros(25, dtype=np.float32)

        if env._is_commitment_hour():
            forecast_start = 1 + 1 + 24 + 1
            price_start = forecast_start + 24

            forecast = obs[forecast_start:forecast_start + 24]
            prices = obs[price_start:price_start + 24]

            forecast_mwh = forecast * PLANT_CONFIG['plant_capacity_mw']

            # Adjust commitment based on price
            # Higher prices -> commit more (accept more risk for more revenue)
            # Lower prices -> commit less (reduce risk when stakes are lower)
            mean_price = np.mean(prices)
            price_factor = prices / (mean_price + 1e-8)
            price_factor = np.clip(price_factor, 0.7, 1.3)

            commitment_fraction = self.base_fraction * price_factor
            commitment_mwh = forecast_mwh * commitment_fraction

            max_possible = forecast_mwh + PLANT_CONFIG['battery_power_mw']
            max_possible = np.maximum(max_possible, 0.01)
            action[:24] = np.clip(commitment_mwh / max_possible, 0, 1)

        # Battery: discharge during high prices, charge during low
        hour = int(obs[0] * 24)
        soc = obs[1] * PLANT_CONFIG['battery_capacity_mwh']

        price_start = 1 + 1 + 24 + 1 + 24
        current_price = obs[price_start + hour] if hour < 24 else obs[price_start]
        mean_price = np.mean(obs[price_start:price_start + 24])

        pv_idx = 1 + 1 + 24 + 1 + 24 + 24
        current_pv = obs[pv_idx] * PLANT_CONFIG['plant_capacity_mw']
        committed = obs[2 + hour] * PLANT_CONFIG['plant_capacity_mw']

        gap = committed - current_pv

        if current_price > mean_price * 1.2:
            # High price: discharge to maximize sales
            if gap > 0:
                discharge = min(gap, PLANT_CONFIG['battery_power_mw'], soc)
            else:
                discharge = min(PLANT_CONFIG['battery_power_mw'] * 0.5, soc)
            action[24] = 0.5 - (discharge / PLANT_CONFIG['battery_power_mw']) * 0.5
        elif current_price < mean_price * 0.8:
            # Low price: charge if we have excess
            if gap < 0:
                charge = min(-gap, PLANT_CONFIG['battery_power_mw'],
                            PLANT_CONFIG['battery_capacity_mwh'] - soc)
                action[24] = 0.5 + (charge / PLANT_CONFIG['battery_power_mw']) * 0.5
            else:
                action[24] = 0.5  # Idle
        else:
            # Normal price: just meet commitment
            if gap > 0:
                discharge = min(gap, PLANT_CONFIG['battery_power_mw'], soc)
                action[24] = 0.5 - (discharge / PLANT_CONFIG['battery_power_mw']) * 0.5
            elif gap < 0:
                charge = min(-gap * 0.5, PLANT_CONFIG['battery_power_mw'],
                            PLANT_CONFIG['battery_capacity_mwh'] - soc)
                action[24] = 0.5 + (charge / PLANT_CONFIG['battery_power_mw']) * 0.5
            else:
                action[24] = 0.5

        return action


def evaluate_policy(policy, env, num_steps: int = None):
    """Evaluate a policy and return metrics."""
    obs, _ = env.reset(seed=42)

    if num_steps is None:
        num_steps = len(env.data) - env.current_idx - 1

    total_revenue = 0
    total_imbalance_cost = 0
    total_delivered = 0
    total_committed = 0
    total_pv = 0
    battery_throughput = 0

    for step in range(num_steps):
        action = policy.predict(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)

        total_revenue += info['revenue']
        total_imbalance_cost += info['imbalance_cost']
        total_delivered += info['delivered']
        total_committed += info['committed']
        total_pv += info['pv_actual']
        battery_throughput += info['battery_throughput']

        if terminated or truncated:
            break

    hours = step + 1
    annual_factor = 8760 / hours

    return {
        'policy': policy.name,
        'hours': hours,
        'total_revenue': total_revenue,
        'total_imbalance_cost': total_imbalance_cost,
        'net_profit': total_revenue - total_imbalance_cost,
        'annual_revenue': total_revenue * annual_factor,
        'annual_imbalance_cost': total_imbalance_cost * annual_factor,
        'annual_net_profit': (total_revenue - total_imbalance_cost) * annual_factor,
        'total_pv': total_pv,
        'total_delivered': total_delivered,
        'total_committed': total_committed,
        'delivery_ratio': total_delivered / max(total_committed, 1),
        'battery_cycles': battery_throughput / (2 * PLANT_CONFIG['battery_capacity_mwh']),
    }


def print_comparison(results: list):
    """Print comparison table."""
    print("\n" + "="*80)
    print("RULE-BASED POLICY COMPARISON")
    print("="*80)

    # Header
    print(f"\n{'Policy':<30} {'Net Profit':>12} {'Annual':>12} {'Imb Cost':>10} {'Deliv %':>8}")
    print("-"*80)

    for r in results:
        print(f"{r['policy']:<30} "
              f"EUR {r['net_profit']:>9,.0f} "
              f"EUR {r['annual_net_profit']:>9,.0f} "
              f"EUR {r['total_imbalance_cost']:>7,.0f} "
              f"{100*r['delivery_ratio']:>7.1f}%")

    print("-"*80)

    # Find best
    best = max(results, key=lambda x: x['annual_net_profit'])
    print(f"\nBest policy: {best['policy']}")
    print(f"Annual net profit: EUR {best['annual_net_profit']:,.0f}")


def main():
    # Load test data
    test_path = DATA_PATH / 'test.csv'
    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        print("Run prepare_dataset.py first.")
        return

    print(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path, parse_dates=['datetime'])

    # Define policies to test
    policies = [
        ConservativePolicy(0.70),
        ConservativePolicy(0.80),
        ConservativePolicy(0.90),
        AggressivePolicy(),
        PriceAwarePolicy(),
    ]

    results = []

    for policy in policies:
        print(f"\nEvaluating: {policy.name}")
        env = SolarMerchantEnv(test_df.copy(), **PLANT_CONFIG)
        result = evaluate_policy(policy, env)
        results.append(result)
        print(f"  Net profit: EUR {result['net_profit']:,.0f} "
              f"(annual: EUR {result['annual_net_profit']:,.0f})")

    # Print comparison
    print_comparison(results)

    # Save results
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_PATH / 'baseline_comparison.csv', index=False)
    print(f"\nResults saved to {OUTPUT_PATH / 'baseline_comparison.csv'}")


if __name__ == '__main__':
    main()
