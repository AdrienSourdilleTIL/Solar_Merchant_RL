"""Baseline trading policies for the Solar Merchant environment.

Provides rule-based strategies that serve as performance benchmarks
for the RL agent. Each policy follows the interface:
    def policy_name(obs: np.ndarray) -> np.ndarray

Plant constants (for reference, not used for denormalization):
    PLANT_CAPACITY_MW = 20.0
    BATTERY_CAPACITY_MWH = 10.0
    BATTERY_POWER_MW = 5.0
"""

import numpy as np

# Plant constants
PLANT_CAPACITY_MW = 20.0
BATTERY_CAPACITY_MWH = 10.0
BATTERY_POWER_MW = 5.0

# Conservative policy parameters
COMMITMENT_FRACTION = 0.8


def _parse_observation(obs: np.ndarray) -> dict:
    """Extract named fields from the 84-dimensional observation vector.

    Args:
        obs: 84-dimensional observation from SolarMerchantEnv.

    Returns:
        Dictionary with named observation fields (normalized values).
    """
    return {
        "hour": obs[0],
        "soc": obs[1],
        "commitments": obs[2:26],
        "cumulative_imbalance": obs[26],
        "pv_forecast": obs[27:51],
        "prices": obs[51:75],
        "actual_pv": obs[75],
        "weather": obs[76:78],
        "time_features": obs[78:84],
    }


def conservative_policy(obs: np.ndarray) -> np.ndarray:
    """Conservative baseline: commit 80% of forecast, use battery to fill gaps.

    Strategy:
        - Commitment: Set all 24 hourly fractions to 0.8 (80% of forecast).
        - Battery: Discharge when under-delivering, charge when over-delivering
          or surplus PV exists, idle otherwise.

    Args:
        obs: 84-dimensional observation from SolarMerchantEnv.

    Returns:
        25-dimensional action array in [0, 1] range (float32).
    """
    parsed = _parse_observation(obs)

    action = np.full(25, COMMITMENT_FRACTION, dtype=np.float32)

    # Battery heuristic based on cumulative imbalance and SOC
    cumulative_imbalance = parsed["cumulative_imbalance"]
    soc = parsed["soc"]

    if cumulative_imbalance < 0 and soc > 0:
        # Under-delivering and have battery charge: discharge
        action[24] = 0.0
    elif cumulative_imbalance > 0:
        # Over-delivering: charge battery to store surplus
        action[24] = 1.0
    else:
        # Balanced: idle
        action[24] = 0.5

    return action
