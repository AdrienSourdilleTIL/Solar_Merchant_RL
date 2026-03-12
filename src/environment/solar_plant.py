"""
Solar Plant Module
==================

Shared physical simulation logic for solar plant with battery storage.
Used by both CommitmentEnv and BatteryEnv to ensure consistent physics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PlantConfig:
    """Configuration for solar plant physical parameters.

    Attributes:
        plant_capacity_mw: Solar plant nameplate capacity (MW)
        battery_capacity_mwh: Battery energy storage capacity (MWh)
        battery_power_mw: Battery max charge/discharge power (MW)
        battery_efficiency: Round-trip battery efficiency (0-1)
        battery_degradation_cost: Cost per MWh of battery throughput (EUR/MWh)
        commitment_hour: Hour when day-ahead commitments are due (0-23)
    """
    plant_capacity_mw: float = 20.0
    battery_capacity_mwh: float = 10.0
    battery_power_mw: float = 5.0
    battery_efficiency: float = 0.92
    battery_degradation_cost: float = 0.01
    commitment_hour: int = 11


class Battery:
    """Battery storage simulation with efficiency losses.

    Handles charge/discharge operations with:
    - Power limits (max charge/discharge rate)
    - Capacity limits (SOC bounds)
    - Efficiency losses (sqrt of round-trip applied each way)
    - Constraint: Can only charge from PV, not from grid

    Example:
        >>> battery = Battery(capacity_mwh=10.0, power_mw=5.0, efficiency=0.92)
        >>> battery.reset(initial_soc=5.0)
        >>> charged, throughput = battery.charge(3.0, available_pv=4.0)
        >>> discharged, throughput = battery.discharge(2.0)
    """

    def __init__(
        self,
        capacity_mwh: float,
        power_mw: float,
        efficiency: float,
        degradation_cost: float = 0.01
    ):
        self.capacity_mwh = capacity_mwh
        self.power_mw = power_mw
        self.efficiency = efficiency
        self.one_way_efficiency = np.sqrt(efficiency)
        self.degradation_cost = degradation_cost
        self.soc = 0.5 * capacity_mwh  # Default 50% SOC

    def reset(self, initial_soc: Optional[float] = None) -> None:
        """Reset battery to initial state.

        Args:
            initial_soc: Initial state of charge in MWh.
                        If None, defaults to 50% capacity.
        """
        if initial_soc is None:
            self.soc = 0.5 * self.capacity_mwh
        else:
            self.soc = np.clip(initial_soc, 0, self.capacity_mwh)

    def charge(self, target_mwh: float, available_pv: float) -> Tuple[float, float]:
        """Attempt to charge battery from available PV.

        DESIGN: Battery can only charge from PV surplus, not from grid.
        This eliminates pure price arbitrage strategies.

        Args:
            target_mwh: Desired charge amount (MWh)
            available_pv: Available PV power that could be used for charging (MWh)

        Returns:
            Tuple of (actual_energy_used, throughput_for_degradation)
            - actual_energy_used: Energy taken from PV for charging
            - throughput: Amount for degradation cost calculation
        """
        if target_mwh <= 0:
            return 0.0, 0.0

        # Constraints: power limit, available PV, remaining capacity
        max_charge = min(
            target_mwh,
            self.power_mw,  # Power limit
            available_pv,   # Can only charge from PV
            (self.capacity_mwh - self.soc) / self.one_way_efficiency  # Capacity limit
        )
        actual_charge = max(0, max_charge)

        # Apply efficiency loss
        self.soc += actual_charge * self.one_way_efficiency
        self.soc = min(self.soc, self.capacity_mwh)  # Clamp

        return actual_charge, actual_charge

    def discharge(self, target_mwh: float) -> Tuple[float, float]:
        """Attempt to discharge battery.

        Args:
            target_mwh: Desired discharge amount (MWh)

        Returns:
            Tuple of (actual_energy_delivered, throughput_for_degradation)
            - actual_energy_delivered: Energy available for delivery/sale
            - throughput: Amount for degradation cost calculation
        """
        if target_mwh <= 0:
            return 0.0, 0.0

        # Constraints: power limit, available SOC
        max_discharge = min(
            target_mwh,
            self.power_mw,  # Power limit
            self.soc * self.one_way_efficiency  # Available energy after efficiency
        )
        actual_discharge = max(0, max_discharge)

        # Reduce SOC (more energy removed than delivered due to efficiency)
        self.soc -= actual_discharge / self.one_way_efficiency
        self.soc = max(self.soc, 0)  # Clamp

        return actual_discharge, actual_discharge

    def step(self, action: float, available_pv: float) -> Tuple[float, float, float]:
        """Execute battery action for one timestep.

        Args:
            action: Battery action in [0, 1] where:
                   0 = full discharge, 0.5 = idle, 1 = full charge
            available_pv: PV power available for charging (MWh)

        Returns:
            Tuple of (energy_delta, throughput, degradation_cost)
            - energy_delta: Change in available energy for delivery
                           (positive = discharge added energy, negative = charge consumed)
            - throughput: Battery throughput for this step
            - degradation_cost: Cost incurred from degradation
        """
        # Convert action from [0,1] to [-1,1] range
        battery_action = (action - 0.5) * 2
        target_power = battery_action * self.power_mw

        if target_power > 0:
            # Charging
            energy_used, throughput = self.charge(target_power, available_pv)
            energy_delta = -energy_used  # Charging consumes available energy
        elif target_power < 0:
            # Discharging
            energy_delivered, throughput = self.discharge(-target_power)
            energy_delta = energy_delivered  # Discharging adds energy
        else:
            energy_delta = 0.0
            throughput = 0.0

        degradation_cost = throughput * self.degradation_cost

        return energy_delta, throughput, degradation_cost

    @property
    def soc_normalized(self) -> float:
        """Get normalized SOC in [0, 1] range."""
        return self.soc / self.capacity_mwh


class Settlement:
    """Market settlement calculations.

    Implements simplified European balancing market settlement:
    - Short (under-delivered): Pay higher imbalance price
    - Long (over-delivered): Receive lower imbalance price

    Example:
        >>> settlement = Settlement()
        >>> revenue, imbalance_cost = settlement.calculate(
        ...     committed=10.0, delivered=8.0,
        ...     price_da=50.0, price_short=75.0, price_long=30.0
        ... )
    """

    @staticmethod
    def calculate(
        committed: float,
        delivered: float,
        price_da: float,
        price_short: float,
        price_long: float
    ) -> Tuple[float, float]:
        """Calculate revenue and imbalance cost for an hour.

        Args:
            committed: Energy committed in day-ahead market (MWh)
            delivered: Energy actually delivered (MWh)
            price_da: Day-ahead price (EUR/MWh)
            price_short: Imbalance price for short positions (EUR/MWh)
            price_long: Imbalance price for long positions (EUR/MWh)

        Returns:
            Tuple of (revenue, imbalance_cost)
            - revenue: Total revenue earned (EUR)
            - imbalance_cost: Penalty for imbalance (EUR, always >= 0)
        """
        imbalance = delivered - committed

        if imbalance >= 0:
            # Long position: over-delivered
            # Get DA price for commitment + imbalance price for excess
            revenue = committed * price_da + imbalance * price_long
            imbalance_cost = 0.0
        else:
            # Short position: under-delivered
            # Get DA price for commitment - cost to cover shortage
            revenue = committed * price_da
            imbalance_cost = abs(imbalance) * price_short

        return revenue, imbalance_cost


class DataManager:
    """Manages data loading and normalization for environments.

    Provides consistent data access and normalization factors across
    both CommitmentEnv and BatteryEnv.
    """

    REQUIRED_COLUMNS = [
        'datetime', 'hour', 'price_eur_mwh', 'pv_actual_mwh', 'pv_forecast_mwh',
        'price_imbalance_short', 'price_imbalance_long',
        'temperature_c', 'irradiance_direct',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
    ]

    def __init__(self, data: pd.DataFrame, plant_config: PlantConfig):
        """Initialize data manager.

        Args:
            data: DataFrame with required columns
            plant_config: Plant configuration for normalization

        Raises:
            ValueError: If required columns are missing
        """
        self._validate_columns(data)
        self.data = data.reset_index(drop=True)
        self.config = plant_config
        self.norm_factors = self._compute_normalization_factors()

    def _validate_columns(self, data: pd.DataFrame) -> None:
        """Validate required columns exist."""
        missing = [col for col in self.REQUIRED_COLUMNS if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _compute_normalization_factors(self) -> dict:
        """Compute normalization factors from data statistics."""
        return {
            'price': self.data['price_eur_mwh'].abs().max() + 1e-8,
            'pv': self.config.plant_capacity_mw,
            'temperature': max(
                abs(self.data['temperature_c'].min()),
                abs(self.data['temperature_c'].max())
            ) + 1e-8,
            'irradiance': self.data['irradiance_direct'].max() + 1e-8,
        }

    def get_row(self, idx: int) -> pd.Series:
        """Get data row at index."""
        return self.data.iloc[idx]

    def get_hour(self, idx: int) -> int:
        """Get hour of day at index."""
        return int(self.data.iloc[idx]['hour'])

    def get_forecasts(self, start_idx: int, hours: int = 24) -> np.ndarray:
        """Get PV forecasts starting from index.

        Args:
            start_idx: Starting index in data
            hours: Number of hours to retrieve

        Returns:
            Array of PV forecasts, zero-padded if near data end
        """
        forecasts = []
        for i in range(hours):
            idx = start_idx + i
            if idx < len(self.data):
                forecasts.append(self.data.iloc[idx]['pv_forecast_mwh'])
            else:
                forecasts.append(0.0)
        return np.array(forecasts)

    def get_prices(self, start_idx: int, hours: int = 24) -> np.ndarray:
        """Get day-ahead prices starting from index.

        Args:
            start_idx: Starting index in data
            hours: Number of hours to retrieve

        Returns:
            Array of prices, zero-padded if near data end
        """
        prices = []
        for i in range(hours):
            idx = start_idx + i
            if idx < len(self.data):
                prices.append(self.data.iloc[idx]['price_eur_mwh'])
            else:
                prices.append(0.0)
        return np.array(prices)

    def get_actuals(self, start_idx: int, hours: int = 24) -> np.ndarray:
        """Get actual PV production starting from index.

        Args:
            start_idx: Starting index in data
            hours: Number of hours to retrieve

        Returns:
            Array of actual PV values, zero-padded if near data end
        """
        actuals = []
        for i in range(hours):
            idx = start_idx + i
            if idx < len(self.data):
                actuals.append(self.data.iloc[idx]['pv_actual_mwh'])
            else:
                actuals.append(0.0)
        return np.array(actuals)

    def get_imbalance_prices(self, idx: int) -> Tuple[float, float]:
        """Get imbalance prices at index.

        Returns:
            Tuple of (price_short, price_long)
        """
        row = self.data.iloc[idx]
        return row['price_imbalance_short'], row['price_imbalance_long']

    def normalize_price(self, price: float) -> float:
        """Normalize price value."""
        return price / self.norm_factors['price']

    def normalize_pv(self, pv: float) -> float:
        """Normalize PV value."""
        return pv / self.norm_factors['pv']

    def normalize_prices(self, prices: np.ndarray) -> np.ndarray:
        """Normalize price array."""
        return prices / self.norm_factors['price']

    def normalize_forecasts(self, forecasts: np.ndarray) -> np.ndarray:
        """Normalize forecast array."""
        return forecasts / self.norm_factors['pv']

    def __len__(self) -> int:
        return len(self.data)


def calculate_max_commitment(
    forecasts: np.ndarray,
    battery_power_mw: float
) -> np.ndarray:
    """Calculate maximum possible commitment for each hour.

    Max commitment = forecast + battery discharge potential.
    This represents the upper bound of what could be delivered.

    Args:
        forecasts: PV forecasts for each hour (MWh)
        battery_power_mw: Battery max discharge power (MW)

    Returns:
        Array of maximum commitment values
    """
    return forecasts + battery_power_mw


def heuristic_battery_policy(
    soc: float,
    committed: float,
    pv_actual: float,
    battery_power_mw: float,
    battery_capacity_mwh: float
) -> float:
    """Simple heuristic battery policy for commitment environment training.

    Strategy:
    - If short (PV < committed): discharge to cover gap
    - If long (PV > committed): charge excess if possible
    - Otherwise: idle

    Args:
        soc: Current battery state of charge (MWh)
        committed: Energy committed for this hour (MWh)
        pv_actual: Actual PV production (MWh)
        battery_power_mw: Battery power limit (MW)
        battery_capacity_mwh: Battery capacity (MWh)

    Returns:
        Battery action in [0, 1] range (0=discharge, 0.5=idle, 1=charge)
    """
    gap = committed - pv_actual  # Positive = short, negative = long

    if gap > 0:
        # Short: need to discharge
        # Scale action based on how much discharge is needed
        discharge_needed = min(gap, battery_power_mw)
        action = 0.5 - (discharge_needed / battery_power_mw) * 0.5
    elif gap < 0:
        # Long: can charge if capacity available
        excess = min(-gap, battery_power_mw)
        room = battery_capacity_mwh - soc
        if room > 0.1:  # Only charge if meaningful room
            action = 0.5 + (excess / battery_power_mw) * 0.5
        else:
            action = 0.5
    else:
        action = 0.5

    return np.clip(action, 0.0, 1.0)
