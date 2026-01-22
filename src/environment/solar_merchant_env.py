"""
Solar Merchant Environment (V1)
================================

Simulates a utility-scale solar farm with battery storage trading on
the day-ahead electricity market.

The agent makes daily decisions about:
1. How much energy to commit selling for each hour of the next day
2. Battery charge/discharge schedule to meet commitments

Key features:
- Day-ahead market with 24-hour commitment horizon
- Battery for arbitrage and commitment smoothing
- Imbalance penalties for under/over-delivery
- Realistic forecast uncertainty

Episode structure:
- Each step = 1 hour of real-time operation
- Every 24 hours (at noon), agent commits to next day's delivery
- Rewards based on: revenue - imbalance costs - battery degradation
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from pathlib import Path
from typing import Optional


class SolarMerchantEnv(gym.Env):
    """
    Gym environment for solar merchant trading.

    The agent operates a solar farm + battery and trades on the day-ahead market.

    Observation Space (84 dimensions):
        - Current hour (1): [0, 1] normalized
        - Battery SOC (1): [0, 1] normalized by capacity
        - Today's committed schedule (24): Hourly commitments normalized
        - Cumulative imbalance today (1): Normalized by capacity
        - PV forecast next 24h (24): Normalized by plant capacity
        - Prices next 24h (24): Normalized by max abs price
        - Current actual PV (1): [0, 1] normalized
        - Temperature (1): Normalized
        - Irradiance (1): Normalized
        - Time features (6): hour_sin/cos, day_sin/cos, month_sin/cos

    Action Space (25 dimensions):
        - Commitment fractions (24): [0, 1] for each hour tomorrow
          (Used only at commitment_hour=11)
        - Battery action (1): [0, 1] where 0=discharge, 0.5=idle, 1=charge

    Reward:
        Revenue from sales - imbalance costs - battery degradation

    Example:
        >>> import gymnasium
        >>> import src.environment  # Triggers registration
        >>> from src.environment import load_environment
        >>>
        >>> # Load from CSV
        >>> env = load_environment('data/processed/train.csv')
        >>> obs, info = env.reset(seed=42)
        >>> print(f"Observation shape: {obs.shape}")  # (84,)
        >>>
        >>> # Or use gymnasium.make()
        >>> import pandas as pd
        >>> data = pd.read_csv('data/processed/train.csv', parse_dates=['datetime'])
        >>> env = gymnasium.make('SolarMerchant-v0', data=data)
        >>>
        >>> # Run episode
        >>> obs, info = env.reset()
        >>> for _ in range(24):
        ...     action = env.action_space.sample()
        ...     obs, reward, terminated, truncated, info = env.step(action)
        ...     if terminated:
        ...         break
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        plant_capacity_mw: float = 20.0,
        battery_capacity_mwh: float = 10.0,
        battery_power_mw: float = 5.0,
        battery_efficiency: float = 0.92,
        battery_degradation_cost: float = 0.01,  # EUR/MWh throughput
        commitment_hour: int = 11,  # Hour when day-ahead bids are due
        render_mode: Optional[str] = None
    ) -> None:
        """
        Initialize the environment.

        Args:
            data: DataFrame with columns:
                - datetime, price_eur_mwh, pv_actual_mwh, pv_forecast_mwh
                - price_imbalance_short, price_imbalance_long
            plant_capacity_mw: Solar plant nameplate capacity
            battery_capacity_mwh: Battery energy capacity
            battery_power_mw: Battery max charge/discharge power
            battery_efficiency: Round-trip efficiency (sqrt applied each way)
            battery_degradation_cost: Cost per MWh of battery throughput
            commitment_hour: Hour when day-ahead commitments are due (default 11 = noon)
        """
        super().__init__()

        # Validate required columns
        required_columns = [
            'datetime', 'hour', 'price_eur_mwh', 'pv_actual_mwh', 'pv_forecast_mwh',
            'price_imbalance_short', 'price_imbalance_long',
            'temperature_c', 'irradiance_direct',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos'
        ]
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in data: {missing_columns}. "
                f"Required columns: {required_columns}"
            )

        self.data = data.reset_index(drop=True)
        self.plant_capacity_mw = plant_capacity_mw
        self.battery_capacity_mwh = battery_capacity_mwh
        self.battery_power_mw = battery_power_mw
        self.battery_efficiency = battery_efficiency
        self.one_way_efficiency = np.sqrt(battery_efficiency)
        self.battery_degradation_cost = battery_degradation_cost
        self.commitment_hour = commitment_hour
        self.render_mode = render_mode

        # Episode tracking
        self.current_idx = 0
        self.episode_start_idx = 0  # Track episode start for 24-hour termination
        self.battery_soc = 0.5 * battery_capacity_mwh  # Start at 50% SOC
        self.committed_schedule = np.zeros(24)  # Today's hourly commitments
        self.episode_revenue = 0.0
        self.episode_imbalance_cost = 0.0
        self.episode_degradation_cost = 0.0

        # Normalization factors (computed from data)
        self._compute_normalization_factors()

        # Action space:
        # At commitment hour: 24 commitment values (fraction of capacity, 0-1)
        # At other hours: battery action (-1 to 1, discharge to charge)
        # We use a unified space and interpret based on hour
        # Max dim = 24 (for commitment) + 1 (for battery) = 25
        # But we'll simplify: always output 25 values, ignore unused ones
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(25,),  # 24 commitment fractions + 1 battery action (rescaled)
            dtype=np.float32
        )

        # Observation space
        # - hour (1)
        # - battery SOC normalized (1)
        # - committed schedule for remaining hours today (24)
        # - cumulative imbalance so far today (1)
        # - PV forecast next 24h normalized (24)
        # - prices next 24h normalized (24)
        # - current actual PV (1)
        # - weather features (2: temperature, irradiance)
        # - time features (6: hour_sin/cos, day_sin/cos, month_sin/cos)
        obs_dim = 1 + 1 + 24 + 1 + 24 + 24 + 1 + 2 + 6
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def _compute_normalization_factors(self) -> None:
        """Compute normalization factors from data."""
        self.norm_factors = {
            'price': self.data['price_eur_mwh'].abs().max() + 1e-8,
            'pv': self.plant_capacity_mw,
            'temperature': max(abs(self.data['temperature_c'].min()),
                              abs(self.data['temperature_c'].max())) + 1e-8,
            'irradiance': self.data['irradiance_direct'].max() + 1e-8,
        }

    def _get_observation(self) -> np.ndarray:
        """Build observation vector for current state."""
        row = self.data.iloc[self.current_idx]
        hour = row['hour']

        # Get next 24 hours of forecasts/prices (pad with zeros if near end)
        forecast_window = []
        price_window = []
        for i in range(24):
            if self.current_idx + i < len(self.data):
                forecast_window.append(
                    self.data.iloc[self.current_idx + i]['pv_forecast_mwh'] / self.norm_factors['pv']
                )
                price_window.append(
                    self.data.iloc[self.current_idx + i]['price_eur_mwh'] / self.norm_factors['price']
                )
            else:
                forecast_window.append(0.0)
                price_window.append(0.0)

        # Calculate cumulative imbalance so far today
        # (difference between committed and delivered for hours already passed)
        cumulative_imbalance = 0.0
        for h in range(int(hour)):
            if hasattr(self, 'hourly_delivered'):
                committed = self.committed_schedule[h]
                delivered = self.hourly_delivered.get(h, 0.0)
                cumulative_imbalance += delivered - committed

        obs = np.concatenate([
            # Current hour (normalized)
            [hour / 24.0],
            # Battery SOC (normalized)
            [self.battery_soc / self.battery_capacity_mwh],
            # Committed schedule (normalized by capacity)
            self.committed_schedule / self.plant_capacity_mw,
            # Cumulative imbalance (normalized)
            [cumulative_imbalance / self.plant_capacity_mw],
            # PV forecast (normalized)
            np.array(forecast_window),
            # Prices (normalized)
            np.array(price_window),
            # Current actual PV (normalized)
            [row['pv_actual_mwh'] / self.norm_factors['pv']],
            # Weather
            [row['temperature_c'] / self.norm_factors['temperature'],
             row['irradiance_direct'] / self.norm_factors['irradiance']],
            # Time features (already -1 to 1)
            [row['hour_sin'], row['hour_cos'],
             row['day_sin'], row['day_cos'],
             row['month_sin'], row['month_cos']],
        ]).astype(np.float32)

        return obs

    def _is_commitment_hour(self) -> bool:
        """Check if current hour is when day-ahead bids are due."""
        return self.data.iloc[self.current_idx]['hour'] == self.commitment_hour

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment to start of a new episode."""
        super().reset(seed=seed)

        # Start at a random position (but not too close to end)
        if seed is not None:
            np.random.seed(seed)

        max_start = len(self.data) - 24 * 30  # Leave at least 30 days
        self.current_idx = np.random.randint(0, max(1, max_start))
        self.episode_start_idx = self.current_idx  # Store episode start

        # Validate episode will encounter commitment hour
        # Check if commitment hour appears in next 24 hours
        episode_hours = []
        for i in range(24):
            if self.current_idx + i < len(self.data):
                episode_hours.append(int(self.data.iloc[self.current_idx + i]['hour']))

        if self.commitment_hour not in episode_hours:
            # Adjust start to ensure we hit commitment hour
            # Find next occurrence of commitment hour
            for i in range(len(self.data) - self.current_idx - 24):
                if int(self.data.iloc[self.current_idx + i]['hour']) == self.commitment_hour:
                    self.current_idx += i
                    self.episode_start_idx = self.current_idx
                    break

        # Reset state
        self.battery_soc = 0.5 * self.battery_capacity_mwh
        self.committed_schedule = np.zeros(24)
        self.hourly_delivered = {}
        self.episode_revenue = 0.0
        self.episode_imbalance_cost = 0.0
        self.episode_degradation_cost = 0.0

        return self._get_observation(), {}

    def step(
        self,
        action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one hour of operation.

        Args:
            action: Array of shape (25,)
                - action[0:24]: commitment fractions for next day (used at commitment hour)
                - action[24]: battery action (0=full discharge, 0.5=idle, 1=full charge)

        Returns:
            observation, reward, terminated, truncated, info
        """
        row = self.data.iloc[self.current_idx]
        hour = int(row['hour'])
        price = row['price_eur_mwh']
        pv_actual = row['pv_actual_mwh']
        price_short = row['price_imbalance_short']
        price_long = row['price_imbalance_long']

        reward = 0.0
        info = {}

        # Handle day-ahead commitment at commitment hour
        if self._is_commitment_hour():
            # Interpret first 24 action values as commitment fractions
            commitment_fractions = np.clip(action[:24], 0, 1)

            # Get forecasts for tomorrow (next 24 hours starting from midnight)
            # Calculate hours until next midnight (00:00)
            current_hour = int(row['hour'])
            hours_until_midnight = (24 - current_hour) % 24
            if hours_until_midnight == 0:
                hours_until_midnight = 24  # We're at 00:00, tomorrow is 24h away

            # Tomorrow's 24 hours start at hours_until_midnight from now
            tomorrow_forecasts = []
            for i in range(hours_until_midnight, hours_until_midnight + 24):
                if self.current_idx + i < len(self.data):
                    tomorrow_forecasts.append(
                        self.data.iloc[self.current_idx + i]['pv_forecast_mwh']
                    )
                else:
                    tomorrow_forecasts.append(0.0)
            tomorrow_forecasts = np.array(tomorrow_forecasts)

            # Commitment = fraction * (forecast + battery discharge potential)
            # Agent decides how aggressively to commit based on forecast
            max_commitment = tomorrow_forecasts + self.battery_power_mw
            self.committed_schedule = commitment_fractions * max_commitment
            self.hourly_delivered = {}  # Reset for new day

            info['new_commitment'] = self.committed_schedule.copy()

        # Battery decision
        # action[24] in [0, 1]: 0 = full discharge, 0.5 = idle, 1 = full charge
        battery_action = (action[24] - 0.5) * 2  # Convert to [-1, 1]
        battery_target = battery_action * self.battery_power_mw  # MWh (hourly = MW)

        # Energy available for delivery
        available_energy = pv_actual

        # Battery charging (positive battery_target)
        if battery_target > 0:
            # DESIGN DECISION: Battery can only charge from PV surplus, not from grid
            # This simplifies the model for V1 - future versions could add grid charging
            # Impact: No pure arbitrage, battery is only for smoothing PV production
            charge_potential = min(
                battery_target,
                available_energy,  # Can only charge from available PV
                (self.battery_capacity_mwh - self.battery_soc) / self.one_way_efficiency
            )
            actual_charge = max(0, charge_potential)
            self.battery_soc += actual_charge * self.one_way_efficiency
            available_energy -= actual_charge
            battery_throughput = actual_charge

        # Battery discharging (negative battery_target)
        elif battery_target < 0:
            discharge_potential = min(
                -battery_target,
                self.battery_soc * self.one_way_efficiency,
                self.battery_power_mw
            )
            actual_discharge = max(0, discharge_potential)
            self.battery_soc -= actual_discharge / self.one_way_efficiency
            available_energy += actual_discharge
            battery_throughput = actual_discharge
        else:
            battery_throughput = 0.0

        # Clamp SOC
        self.battery_soc = np.clip(self.battery_soc, 0, self.battery_capacity_mwh)

        # Determine delivery vs commitment
        committed = self.committed_schedule[hour]
        delivered = available_energy  # We deliver whatever we have

        # Calculate revenue and imbalance
        # Revenue for delivered energy at day-ahead price
        revenue = delivered * price
        self.episode_revenue += revenue

        # Imbalance settlement
        # NOTE: Revenue already paid at day-ahead price for delivered amount
        # Imbalance cost represents additional penalties/adjustments
        imbalance = delivered - committed
        if imbalance < 0:
            # Short: under-delivered, pay penalty at short price
            # Total cost = committed * price (already in revenue) + shortage * price_short
            # We already got revenue for delivered, so we owe: shortage * price_short
            imbalance_cost = abs(imbalance) * price_short
        else:
            # Long: over-delivered, receive long price instead of DA price for excess
            # Total revenue = committed * price + excess * price_long (not price)
            # Since we already counted delivered * price in revenue,
            # we need to subtract the excess that should have been at long price
            imbalance_cost = imbalance * (price - price_long)

        self.episode_imbalance_cost += imbalance_cost

        # Battery degradation cost
        degradation = battery_throughput * self.battery_degradation_cost
        self.episode_degradation_cost += degradation

        # Track delivery
        self.hourly_delivered[hour] = delivered

        # Reward = revenue - imbalance cost - degradation
        reward = revenue - imbalance_cost - degradation

        # Store info
        info.update({
            'hour': hour,
            'pv_actual': pv_actual,
            'committed': committed,
            'delivered': delivered,
            'imbalance': imbalance,
            'price': price,
            'revenue': revenue,
            'imbalance_cost': imbalance_cost,
            'battery_soc': self.battery_soc,
            'battery_throughput': battery_throughput,
        })

        # Advance time
        self.current_idx += 1

        # Check termination: Episode ends after 24 hours OR at data boundary
        episode_hours = self.current_idx - self.episode_start_idx
        terminated = episode_hours >= 24
        truncated = self.current_idx >= len(self.data) - 1

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        """Render current state."""
        if self.render_mode == 'human':
            row = self.data.iloc[self.current_idx]
            print(f"Hour: {row['hour']:02d} | "
                  f"SOC: {self.battery_soc:.1f}/{self.battery_capacity_mwh} MWh | "
                  f"Price: {row['price_eur_mwh']:.1f} EUR/MWh | "
                  f"PV: {row['pv_actual_mwh']:.1f} MWh")


def load_environment(data_path: str, **kwargs) -> SolarMerchantEnv:
    """Helper to load environment from processed data file."""
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    return SolarMerchantEnv(df, **kwargs)


if __name__ == '__main__':
    # Test the environment
    from pathlib import Path

    data_path = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'train.csv'

    if data_path.exists():
        env = load_environment(str(data_path))
        obs, info = env.reset(seed=42)

        print(f"Observation shape: {obs.shape}")
        print(f"Action space: {env.action_space}")

        # Run a few steps
        total_reward = 0
        for i in range(48):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if i % 24 == 0:
                print(f"\nStep {i}: reward={reward:.2f}, total={total_reward:.2f}")
                print(f"  Hour: {info['hour']}, PV: {info['pv_actual']:.1f}, "
                      f"Committed: {info['committed']:.1f}, Delivered: {info['delivered']:.1f}")

            if terminated:
                break
    else:
        print(f"Data file not found: {data_path}")
        print("Run prepare_dataset.py first to generate the data.")
