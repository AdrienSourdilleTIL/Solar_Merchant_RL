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

    State:
        - Current hour (0-23)
        - Battery SOC (0-1)
        - Today's committed delivery schedule (24 values)
        - Today's remaining hours actual vs committed
        - PV forecast for next 24 hours
        - Price forecast for next 24 hours (known day-ahead)
        - Weather features (temperature, irradiance)

    Actions:
        At the commitment hour (11:00), agent provides:
        - 24 hourly commitment values (energy to deliver, in MWh)

        At other hours, agent provides:
        - Battery charge/discharge decision for this hour

    Reward:
        Revenue from sales - imbalance costs - battery degradation
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
    ):
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

    def _compute_normalization_factors(self):
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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to start of a new episode."""
        super().reset(seed=seed)

        # Start at a random position (but not too close to end)
        if seed is not None:
            np.random.seed(seed)

        max_start = len(self.data) - 24 * 30  # Leave at least 30 days
        self.current_idx = np.random.randint(0, max(1, max_start))

        # Reset state
        self.battery_soc = 0.5 * self.battery_capacity_mwh
        self.committed_schedule = np.zeros(24)
        self.hourly_delivered = {}
        self.episode_revenue = 0.0
        self.episode_imbalance_cost = 0.0
        self.episode_degradation_cost = 0.0

        return self._get_observation(), {}

    def step(self, action: np.ndarray):
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

            # Get forecasts for tomorrow (next 13-36 hours, i.e., tomorrow 00:00-23:00)
            tomorrow_forecasts = []
            for i in range(13, 37):  # Hours 13-36 from now = tomorrow 00:00-23:00
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
            # Charge from PV surplus (can't charge from grid in this simple model)
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
        imbalance = delivered - committed
        if imbalance < 0:
            # Short: under-delivered, pay penalty
            imbalance_cost = abs(imbalance) * price_short
        else:
            # Long: over-delivered, receive less (opportunity cost)
            # Actually receive long price instead of DA price
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

        # Check termination
        terminated = self.current_idx >= len(self.data) - 1
        truncated = False

        return self._get_observation(), reward, terminated, truncated, info

    def render(self):
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
