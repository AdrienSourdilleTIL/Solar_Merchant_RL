"""
Battery Environment
===================

Low-level environment for battery management agent.
Given a fixed commitment schedule, the agent learns to optimally
charge/discharge the battery to minimize imbalance costs.

Episode: 24 hours of hourly battery decisions.
Observation: Current state including SOC, commitment, actual PV, etc.
Action: Single continuous value [0,1] for charge/discharge.
Reward: Negative of imbalance cost + degradation penalty (immediate).
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple

from .solar_plant import (
    PlantConfig,
    Battery,
    Settlement,
    DataManager,
)


class BatteryEnv(gym.Env):
    """
    Gym environment for battery management.

    The agent manages a battery to meet pre-committed energy delivery
    schedules, minimizing imbalance costs.

    Observation Space (21 dimensions):
        - Current hour (1): [0, 1] normalized
        - Battery SOC (1): [0, 1] normalized by capacity
        - Current hour's commitment (1): normalized by plant capacity
        - Current hour's actual PV (1): normalized by plant capacity
        - Next 6 hours' commitments (6): normalized (rolling lookahead)
        - Next 6 hours' forecasts (6): normalized (rolling lookahead)
        - Cumulative imbalance today (1): normalized
        - Current price (1): normalized
        - Current imbalance status (1): (delivered - committed) / plant_cap
        - Time features (2): hour_sin, hour_cos

    Action Space (1 dimension):
        - Battery action: [0, 1] where 0=full discharge, 0.5=idle, 1=full charge

    Reward:
        - revenue - imbalance_cost - degradation (immediate, each step)

    Episode:
        - 24 hours of battery management
        - Commitment schedule is fixed at episode start
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        plant_config: Optional[PlantConfig] = None,
        render_mode: Optional[str] = None
    ) -> None:
        """
        Initialize the battery environment.

        Args:
            data: DataFrame with required columns (see DataManager)
            plant_config: Plant configuration. Uses defaults if None.
            render_mode: Render mode ('human' or None)
        """
        super().__init__()

        self.config = plant_config or PlantConfig()
        self.data_manager = DataManager(data, self.config)
        self.render_mode = render_mode

        # Create battery
        self.battery = Battery(
            capacity_mwh=self.config.battery_capacity_mwh,
            power_mw=self.config.battery_power_mw,
            efficiency=self.config.battery_efficiency,
            degradation_cost=self.config.battery_degradation_cost
        )

        # Episode state
        self.current_idx = 0
        self.episode_start_idx = 0
        self.episode_step = 0
        self.commitments = np.zeros(24)  # Fixed at episode start
        self.cumulative_imbalance = 0.0

        # Episode tracking
        self.episode_revenue = 0.0
        self.episode_imbalance_cost = 0.0
        self.episode_degradation = 0.0

        # Observation space: 21 dimensions
        # hour(1) + soc(1) + commit(1) + pv(1) + future_commit(6) +
        # future_forecast(6) + cum_imbalance(1) + price(1) + imbalance_status(1) + time(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(21,),
            dtype=np.float32
        )

        # Action space: single continuous value [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

    def _get_observation(self) -> np.ndarray:
        """Build observation vector for current state."""
        row = self.data_manager.get_row(self.current_idx)
        hour = int(row['hour'])

        # Current values
        current_commit = self.commitments[hour]
        pv_actual = row['pv_actual_mwh']
        price = row['price_eur_mwh']

        # Lookahead window (next 6 hours, wrapping if needed)
        future_commits = []
        future_forecasts = []
        for i in range(1, 7):
            future_hour = (hour + i) % 24
            future_commits.append(self.commitments[future_hour])

            future_idx = self.current_idx + i
            if future_idx < len(self.data_manager):
                future_forecasts.append(
                    self.data_manager.get_row(future_idx)['pv_forecast_mwh']
                )
            else:
                future_forecasts.append(0.0)

        # Current imbalance status (what would happen with no battery action)
        imbalance_status = (pv_actual - current_commit) / self.config.plant_capacity_mw

        obs = np.array([
            # Current hour (normalized)
            hour / 24.0,
            # Battery SOC (normalized)
            self.battery.soc_normalized,
            # Current commitment (normalized)
            current_commit / self.config.plant_capacity_mw,
            # Current actual PV (normalized)
            pv_actual / self.config.plant_capacity_mw,
            # Future commitments (normalized)
            *[c / self.config.plant_capacity_mw for c in future_commits],
            # Future forecasts (normalized)
            *[f / self.config.plant_capacity_mw for f in future_forecasts],
            # Cumulative imbalance (normalized)
            self.cumulative_imbalance / self.config.plant_capacity_mw,
            # Current price (normalized)
            self.data_manager.normalize_price(price),
            # Imbalance status
            imbalance_status,
            # Time features
            row['hour_sin'],
            row['hour_cos'],
        ], dtype=np.float32)

        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to start of a new episode.

        Args:
            seed: Random seed for reproducibility
            options: Configuration dict with optional keys:
                - initial_soc: Initial battery SOC as fraction [0, 1]
                - commitments: Array of 24 hourly commitments (MWh)
                - start_idx: Specific data index to start from
                - start_at_midnight: If True, find next midnight to start

        Returns:
            observation: Initial observation array (20,)
            info: Info dict with episode configuration
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Determine starting index
        if options and 'start_idx' in options:
            self.current_idx = options['start_idx']
        else:
            # Random start, leaving room for 24-hour episode
            max_start = len(self.data_manager) - 48
            self.current_idx = np.random.randint(0, max(1, max_start))

        # Optionally align to midnight
        if options and options.get('start_at_midnight', False):
            # Find next hour 0
            while self.data_manager.get_hour(self.current_idx) != 0:
                self.current_idx += 1
                if self.current_idx >= len(self.data_manager) - 24:
                    self.current_idx = 0
                    break

        self.episode_start_idx = self.current_idx
        self.episode_step = 0

        # Set commitments
        if options and 'commitments' in options:
            self.commitments = np.array(options['commitments'], dtype=np.float32)
            if len(self.commitments) != 24:
                raise ValueError("Commitments must have 24 values")
        else:
            # Generate random commitments for training diversity
            self._generate_random_commitments()

        # Reset battery
        initial_soc_fraction = 0.5
        if options and 'initial_soc' in options:
            initial_soc_fraction = np.clip(float(options['initial_soc']), 0.0, 1.0)
        self.battery.reset(initial_soc_fraction * self.config.battery_capacity_mwh)

        # Reset tracking
        self.cumulative_imbalance = 0.0
        self.episode_revenue = 0.0
        self.episode_imbalance_cost = 0.0
        self.episode_degradation = 0.0

        info = {
            'commitments': self.commitments.copy(),
            'start_idx': self.episode_start_idx,
            'initial_soc': self.battery.soc,
        }

        return self._get_observation(), info

    def _generate_random_commitments(self) -> None:
        """Generate random but realistic commitment schedule.

        Uses forecasts from the episode window with random scaling
        to create diverse training scenarios.
        """
        # Get forecasts for this 24-hour window
        forecasts = self.data_manager.get_forecasts(self.current_idx, hours=24)

        # Random commitment strategy
        strategy = np.random.choice(['conservative', 'aggressive', 'mixed'])

        if strategy == 'conservative':
            # 70-90% of forecast
            scale = np.random.uniform(0.7, 0.9)
            self.commitments = forecasts * scale
        elif strategy == 'aggressive':
            # 90-110% of forecast + some battery
            scale = np.random.uniform(0.9, 1.1)
            battery_add = np.random.uniform(0, self.config.battery_power_mw * 0.5)
            self.commitments = forecasts * scale + battery_add
        else:  # mixed
            # Per-hour random scaling
            scales = np.random.uniform(0.6, 1.2, size=24)
            self.commitments = forecasts * scales

        # Clamp to reasonable bounds
        max_possible = forecasts + self.config.battery_power_mw
        self.commitments = np.clip(self.commitments, 0, max_possible)

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one hour of battery operation.

        Args:
            action: Array of shape (1,) with battery action [0, 1]
                   0 = full discharge, 0.5 = idle, 1 = full charge

        Returns:
            observation, reward, terminated, truncated, info
        """
        row = self.data_manager.get_row(self.current_idx)
        hour = int(row['hour'])
        pv_actual = row['pv_actual_mwh']
        price_da = row['price_eur_mwh']
        price_short, price_long = self.data_manager.get_imbalance_prices(self.current_idx)

        # Get commitment for current hour
        committed = self.commitments[hour]

        # Execute battery action
        battery_action = float(action[0])
        energy_delta, throughput, degradation = self.battery.step(
            battery_action, available_pv=pv_actual
        )

        # Calculate delivery
        # If battery charged: energy_delta < 0 (took from available)
        # If battery discharged: energy_delta > 0 (added to available)
        delivered = pv_actual + energy_delta

        # Calculate settlement
        revenue, imbalance_cost = Settlement.calculate(
            committed=committed,
            delivered=delivered,
            price_da=price_da,
            price_short=price_short,
            price_long=price_long
        )

        # Update tracking
        imbalance = delivered - committed
        self.cumulative_imbalance += imbalance
        self.episode_revenue += revenue
        self.episode_imbalance_cost += imbalance_cost
        self.episode_degradation += degradation

        # Reward: revenue - imbalance_cost - degradation
        reward = revenue - imbalance_cost - degradation

        # Build info
        info = {
            'hour': hour,
            'pv_actual': pv_actual,
            'committed': committed,
            'delivered': delivered,
            'imbalance': imbalance,
            'price': price_da,
            'revenue': revenue,
            'imbalance_cost': imbalance_cost,
            'battery_soc': self.battery.soc,
            'battery_action': battery_action,
            'throughput': throughput,
            'degradation': degradation,
        }

        # Advance time
        self.current_idx += 1
        self.episode_step += 1

        # Check termination: 24 hours or data boundary
        terminated = self.episode_step >= 24
        truncated = self.current_idx >= len(self.data_manager) - 1

        if terminated or truncated:
            info['episode_revenue'] = self.episode_revenue
            info['episode_imbalance_cost'] = self.episode_imbalance_cost
            info['episode_degradation'] = self.episode_degradation
            info['episode_profit'] = (
                self.episode_revenue -
                self.episode_imbalance_cost -
                self.episode_degradation
            )

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        """Render current state."""
        if self.render_mode == 'human':
            row = self.data_manager.get_row(self.current_idx)
            hour = int(row['hour'])
            print(
                f"Hour: {hour:02d} | "
                f"SOC: {self.battery.soc:.1f}/{self.config.battery_capacity_mwh} MWh | "
                f"Commit: {self.commitments[hour]:.1f} MWh | "
                f"PV: {row['pv_actual_mwh']:.1f} MWh | "
                f"Price: {row['price_eur_mwh']:.1f} EUR/MWh"
            )


def load_battery_env(data_path: str, **kwargs) -> BatteryEnv:
    """Helper to load BatteryEnv from processed data file.

    Args:
        data_path: Path to CSV file with required columns
        **kwargs: Additional arguments passed to BatteryEnv

    Returns:
        Configured BatteryEnv instance
    """
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    return BatteryEnv(df, **kwargs)
