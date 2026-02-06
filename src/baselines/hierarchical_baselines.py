"""
Baseline Policies for Hierarchical System
==========================================

Provides various baseline commitment and battery policies for comparison
with trained agents.

Commitment Policies:
- Conservative: 80% of forecast
- Aggressive: 100% of forecast + battery capacity
- Price-Aware: Commit more during high-price hours

Battery Policies:
- Greedy: Always try to meet commitment exactly
- Conservative: Keep SOC buffer for later hours
- Do-Nothing: No battery usage (baseline)
"""

import numpy as np
from typing import Callable

from src.environment.solar_plant import PlantConfig, calculate_max_commitment


# =============================================================================
# COMMITMENT POLICIES
# =============================================================================

def conservative_commitment_policy(
    forecasts: np.ndarray,
    prices: np.ndarray,
    battery_soc: float,
    config: PlantConfig = None,
    **kwargs
) -> np.ndarray:
    """
    Conservative commitment: 80% of forecast.

    Low risk of being short, but leaves money on the table.
    """
    config = config or PlantConfig()
    commitment_fractions = np.ones(24) * 0.80
    max_commits = calculate_max_commitment(forecasts, config.battery_power_mw)
    return commitment_fractions * max_commits


def aggressive_commitment_policy(
    forecasts: np.ndarray,
    prices: np.ndarray,
    battery_soc: float,
    config: PlantConfig = None,
    **kwargs
) -> np.ndarray:
    """
    Aggressive commitment: 100% of forecast + partial battery.

    High risk of being short if forecast is wrong.
    """
    config = config or PlantConfig()
    # 100% of forecast + 50% of battery power
    base_commitment = forecasts + config.battery_power_mw * 0.5
    max_commits = calculate_max_commitment(forecasts, config.battery_power_mw)
    return np.minimum(base_commitment, max_commits)


def price_aware_commitment_policy(
    forecasts: np.ndarray,
    prices: np.ndarray,
    battery_soc: float,
    config: PlantConfig = None,
    **kwargs
) -> np.ndarray:
    """
    Price-aware commitment: Commit more aggressively during high-price hours.

    Uses price percentile to adjust commitment fraction:
    - Low price hours: 70% of max
    - High price hours: 95% of max
    """
    config = config or PlantConfig()
    max_commits = calculate_max_commitment(forecasts, config.battery_power_mw)

    # Calculate price percentiles for tomorrow
    if len(prices) > 0 and prices.max() > prices.min():
        price_percentiles = (prices - prices.min()) / (prices.max() - prices.min())
    else:
        price_percentiles = np.ones(24) * 0.5

    # Scale commitment: 0.70 to 0.95 based on price
    commitment_fractions = 0.70 + 0.25 * price_percentiles

    return commitment_fractions * max_commits


def perfect_hindsight_commitment_policy(
    forecasts: np.ndarray,
    prices: np.ndarray,
    battery_soc: float,
    actuals: np.ndarray = None,
    config: PlantConfig = None,
    **kwargs
) -> np.ndarray:
    """
    Perfect hindsight: Commit exactly what will be produced (cheating!).

    This is an upper bound on performance - not achievable in practice.
    Requires actuals to be passed in.
    """
    config = config or PlantConfig()
    if actuals is not None:
        return actuals.copy()
    else:
        # Fall back to forecast if no actuals
        return forecasts.copy()


# =============================================================================
# BATTERY POLICIES
# =============================================================================

def greedy_battery_policy(
    soc: float,
    committed: float,
    pv_actual: float,
    config: PlantConfig = None,
    **kwargs
) -> float:
    """
    Greedy battery: Always try to exactly meet commitment.

    Discharge if short, charge if long (and capacity available).
    """
    config = config or PlantConfig()
    gap = committed - pv_actual

    if gap > 0:
        # Short: discharge to cover
        discharge_needed = min(gap, config.battery_power_mw)
        discharge_possible = soc * np.sqrt(config.battery_efficiency)
        actual_discharge = min(discharge_needed, discharge_possible)
        # Convert to action: 0 = full discharge
        action = 0.5 - (actual_discharge / config.battery_power_mw) * 0.5
    elif gap < 0:
        # Long: charge excess
        charge_needed = min(-gap, config.battery_power_mw)
        room = config.battery_capacity_mwh - soc
        if room > 0.1:
            action = 0.5 + (charge_needed / config.battery_power_mw) * 0.5
        else:
            action = 0.5
    else:
        action = 0.5

    return np.clip(action, 0.0, 1.0)


def conservative_battery_policy(
    soc: float,
    committed: float,
    pv_actual: float,
    hour: int = 12,
    config: PlantConfig = None,
    **kwargs
) -> float:
    """
    Conservative battery: Keep SOC buffer for later hours.

    Only discharge if really needed, prefer to save battery for later.
    """
    config = config or PlantConfig()
    gap = committed - pv_actual

    # Reserve more battery for later hours
    hours_remaining = max(1, 24 - hour)
    min_soc = 0.3 * config.battery_capacity_mwh  # Keep at least 30%

    if gap > 0:
        # Short: only discharge if we have buffer
        available_soc = max(0, soc - min_soc)
        if available_soc > 0:
            discharge_needed = min(gap, config.battery_power_mw, available_soc)
            action = 0.5 - (discharge_needed / config.battery_power_mw) * 0.5
        else:
            action = 0.5  # Don't discharge below reserve
    elif gap < 0 and soc < 0.8 * config.battery_capacity_mwh:
        # Long and room to charge: charge moderately
        charge = min(-gap * 0.5, config.battery_power_mw)
        action = 0.5 + (charge / config.battery_power_mw) * 0.5
    else:
        action = 0.5

    return np.clip(action, 0.0, 1.0)


def do_nothing_battery_policy(
    soc: float,
    committed: float,
    pv_actual: float,
    **kwargs
) -> float:
    """
    Do-nothing battery: Always idle.

    Baseline to show battery value.
    """
    return 0.5


def smoothing_battery_policy(
    soc: float,
    committed: float,
    pv_actual: float,
    hour: int = 12,
    config: PlantConfig = None,
    **kwargs
) -> float:
    """
    Smoothing battery: Charge during peak PV, discharge during low PV.

    Aims to flatten production profile regardless of commitments.
    """
    config = config or PlantConfig()

    # Peak solar hours (10-14): tend to charge
    # Low solar hours (0-6, 18-24): tend to discharge
    if 10 <= hour <= 14:
        # Peak hours: charge if PV high
        if pv_actual > 0.5 * config.plant_capacity_mw:
            action = 0.7  # Moderate charge
        else:
            action = 0.5
    elif hour < 6 or hour > 18:
        # Low hours: discharge if soc available
        if soc > 0.3 * config.battery_capacity_mwh:
            action = 0.3  # Moderate discharge
        else:
            action = 0.5
    else:
        # Shoulder hours: try to meet commitment
        gap = committed - pv_actual
        if gap > 0.5:
            action = 0.4
        elif gap < -0.5:
            action = 0.6
        else:
            action = 0.5

    return np.clip(action, 0.0, 1.0)


# =============================================================================
# POLICY FACTORY
# =============================================================================

COMMITMENT_POLICIES = {
    'conservative': conservative_commitment_policy,
    'aggressive': aggressive_commitment_policy,
    'price_aware': price_aware_commitment_policy,
    'perfect_hindsight': perfect_hindsight_commitment_policy,
}

BATTERY_POLICIES = {
    'greedy': greedy_battery_policy,
    'conservative': conservative_battery_policy,
    'do_nothing': do_nothing_battery_policy,
    'smoothing': smoothing_battery_policy,
}


def get_commitment_policy(name: str) -> Callable:
    """Get commitment policy by name."""
    if name not in COMMITMENT_POLICIES:
        raise ValueError(f"Unknown commitment policy: {name}. "
                        f"Available: {list(COMMITMENT_POLICIES.keys())}")
    return COMMITMENT_POLICIES[name]


def get_battery_policy(name: str) -> Callable:
    """Get battery policy by name."""
    if name not in BATTERY_POLICIES:
        raise ValueError(f"Unknown battery policy: {name}. "
                        f"Available: {list(BATTERY_POLICIES.keys())}")
    return BATTERY_POLICIES[name]
