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

# Aggressive policy parameters
AGGRESSIVE_FRACTION = 1.0

# Price-aware policy parameters
PRICE_AWARE_HIGH_FRACTION = 1.0
PRICE_AWARE_LOW_FRACTION = 0.5


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

    # Battery heuristic based on cumulative imbalance, SOC, and PV surplus
    cumulative_imbalance = parsed["cumulative_imbalance"]
    soc = parsed["soc"]

    # Detect PV surplus above commitment for the current hour
    hour_idx = int(round(parsed["hour"] * 24)) % 24
    current_commitment = parsed["commitments"][hour_idx]
    has_pv_surplus = parsed["actual_pv"] > current_commitment

    if cumulative_imbalance < 0 and soc > 0:
        # Under-delivering and have battery charge: discharge
        action[24] = 0.0
    elif cumulative_imbalance > 0 or has_pv_surplus:
        # Over-delivering or PV surplus: charge battery
        action[24] = 1.0
    else:
        # Balanced: idle
        action[24] = 0.5

    return action


def aggressive_policy(obs: np.ndarray) -> np.ndarray:
    """Aggressive baseline: commit 100% of max capacity, discharge battery aggressively.

    Strategy:
        - Commitment: Set all 24 hourly fractions to 1.0 (100% of forecast + battery).
        - Battery: Discharge by default to meet high commitments, charge only when
          PV surplus exists above commitment and SOC is not full, idle when SOC is empty.

    Args:
        obs: 84-dimensional observation from SolarMerchantEnv.

    Returns:
        25-dimensional action array in [0, 1] range (float32).
    """
    parsed = _parse_observation(obs)

    action = np.full(25, AGGRESSIVE_FRACTION, dtype=np.float32)

    # Battery heuristic: aggressive defaults to discharge
    soc = parsed["soc"]
    hour_idx = int(round(parsed["hour"] * 24)) % 24
    current_commitment = parsed["commitments"][hour_idx]
    has_pv_surplus = parsed["actual_pv"] > current_commitment

    if has_pv_surplus and soc < 1.0:
        # PV surplus above commitment and battery not full: charge
        action[24] = 1.0
    elif soc > 1e-6:
        # Default: discharge aggressively to meet high commitments
        action[24] = 0.0
    else:
        # SOC effectively empty and no surplus: idle
        action[24] = 0.5

    return action


def price_aware_policy(obs: np.ndarray) -> np.ndarray:
    """Price-aware baseline: adjust commitment and battery based on price levels.

    Strategy:
        - Commitment: Set per-hour fractions based on price relative to 24h median.
          High-price hours get 1.0, low-price hours get 0.5.
        - Battery: Discharge during high-price hours, charge during low-price hours.

    Args:
        obs: 84-dimensional observation from SolarMerchantEnv.

    Returns:
        25-dimensional action array in [0, 1] range (float32).
    """
    parsed = _parse_observation(obs)

    action = np.full(25, PRICE_AWARE_LOW_FRACTION, dtype=np.float32)

    # Compute median of 24h price window as threshold
    prices = parsed["prices"]
    price_median = np.median(prices)

    # Set per-hour commitment fractions based on price level (vectorized)
    action[0:24] = np.where(
        prices > price_median,
        PRICE_AWARE_HIGH_FRACTION,
        PRICE_AWARE_LOW_FRACTION,
    )

    # Battery heuristic: price-driven charge/discharge with PV surplus awareness
    soc = parsed["soc"]
    hour_idx = int(round(parsed["hour"] * 24)) % 24
    current_price = prices[hour_idx]
    current_commitment = parsed["commitments"][hour_idx]
    has_pv_surplus = parsed["actual_pv"] > current_commitment

    if current_price > price_median and soc > 1e-6:
        # High-price hour with charge available: discharge
        action[24] = 0.0
    elif current_price < price_median and has_pv_surplus and soc < 1.0 - 1e-6:
        # Low-price hour with PV surplus and room to charge: charge
        action[24] = 1.0
    else:
        # SOC boundary, no PV surplus, or neutral price: idle
        action[24] = 0.5

    return action
