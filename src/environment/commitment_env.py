"""
Commitment Environment
======================

High-level environment for commitment agent (day-ahead bidding).
The agent decides how much energy to commit selling for each hour
of the next day, then observes the result after 24 hours of execution.

Episode: Single commitment decision + simulated 24h execution
Observation: Forecasts, prices, battery SOC, weather for tomorrow
Action: 24 commitment fractions [0, 1] for each hour
Reward: End-of-day P&L (revenue - imbalance - degradation)
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Optional, Tuple, Callable, Union
from pathlib import Path

from .solar_plant import (
    PlantConfig,
    Battery,
    Settlement,
    DataManager,
    calculate_max_commitment,
    heuristic_battery_policy,
)


class CommitmentEnv(gym.Env):
    """
    Gym environment for day-ahead commitment decisions.

    The agent makes daily commitment decisions for the next 24 hours.
    The environment then simulates execution with a battery policy
    and returns the total P&L as reward.

    Observation Space (56 dimensions):
        - PV forecast for tomorrow (24): normalized by plant capacity
        - Day-ahead prices for tomorrow (24): normalized
        - Current battery SOC (1): normalized
        - Weather features (2): temperature, irradiance (normalized)
        - Forecast confidence proxy (1): recent forecast error magnitude
        - Time features (4): day_sin, day_cos, month_sin, month_cos

    Action Space (24 dimensions):
        - Commitment fractions for each hour [0, 1]
        - Actual commitment = fraction * max_possible
        - max_possible = forecast + battery_power

    Reward:
        - Total 24h P&L: revenue - imbalance_cost - degradation

    Episode:
        - Agent observes state at commitment hour (11:00)
        - Agent outputs 24 commitment fractions
        - Environment simulates 24h with battery policy
        - Reward is the total P&L
        - Episode terminates
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        plant_config: Optional[PlantConfig] = None,
        battery_policy: Optional[Union[Callable, str]] = None,
        render_mode: Optional[str] = None
    ) -> None:
        """
        Initialize the commitment environment.

        Args:
            data: DataFrame with required columns (see DataManager)
            plant_config: Plant configuration. Uses defaults if None.
            battery_policy: Battery policy for simulation. Options:
                - None or 'heuristic': Use built-in heuristic
                - Callable: Function(soc, committed, pv_actual, power, capacity) -> action
                - 'trained': Load trained battery agent (requires model file)
            render_mode: Render mode ('human' or None)
        """
        super().__init__()

        self.config = plant_config or PlantConfig()
        self.data_manager = DataManager(data, self.config)
        self.render_mode = render_mode

        # Set up battery policy
        self._setup_battery_policy(battery_policy)

        # Create battery for simulation
        self.battery = Battery(
            capacity_mwh=self.config.battery_capacity_mwh,
            power_mw=self.config.battery_power_mw,
            efficiency=self.config.battery_efficiency,
            degradation_cost=self.config.battery_degradation_cost
        )

        # Episode state
        self.current_idx = 0
        self.commitment_hour_idx = 0
        self.battery_soc_at_commitment = 0.0

        # Episode tracking
        self.last_commitments = np.zeros(24)
        self.last_simulation_results = {}

        # Observation space: 56 dimensions
        # forecast(24) + prices(24) + soc(1) + weather(2) + forecast_conf(1) + time(4)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(56,),
            dtype=np.float32
        )

        # Action space: 24 commitment fractions [0, 1]
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(24,),
            dtype=np.float32
        )

    def _setup_battery_policy(self, policy) -> None:
        """Set up the battery policy for simulation."""
        if policy is None or policy == 'heuristic':
            self.battery_policy = self._heuristic_battery_action
            self._battery_policy_type = 'heuristic'
        elif policy == 'trained':
            self._load_trained_battery_agent()
            self._battery_policy_type = 'trained'
        elif callable(policy):
            self.battery_policy = policy
            self._battery_policy_type = 'custom'
        else:
            raise ValueError(f"Unknown battery policy: {policy}")

    def _load_trained_battery_agent(self) -> None:
        """Load trained battery agent for simulation."""
        model_path = Path(__file__).parent.parent.parent / 'models' / 'battery_agent' / 'best' / 'best_model.zip'

        if not model_path.exists():
            print(f"Warning: Trained battery agent not found at {model_path}")
            print("Falling back to heuristic policy.")
            self.battery_policy = self._heuristic_battery_action
            self._battery_policy_type = 'heuristic'
            return

        try:
            from stable_baselines3 import SAC
            self._trained_battery_model = SAC.load(str(model_path))
            self.battery_policy = self._trained_battery_action
            print(f"Loaded trained battery agent from {model_path}")
        except Exception as e:
            print(f"Warning: Failed to load battery agent: {e}")
            print("Falling back to heuristic policy.")
            self.battery_policy = self._heuristic_battery_action
            self._battery_policy_type = 'heuristic'

    def _heuristic_battery_action(
        self,
        soc: float,
        committed: float,
        pv_actual: float,
        hour: int,
        future_commitments: np.ndarray,
        future_forecasts: np.ndarray,
        cumulative_imbalance: float,
        price: float
    ) -> float:
        """Heuristic battery policy wrapper."""
        return heuristic_battery_policy(
            soc=soc,
            committed=committed,
            pv_actual=pv_actual,
            battery_power_mw=self.config.battery_power_mw,
            battery_capacity_mwh=self.config.battery_capacity_mwh
        )

    def _trained_battery_action(
        self,
        soc: float,
        committed: float,
        pv_actual: float,
        hour: int,
        future_commitments: np.ndarray,
        future_forecasts: np.ndarray,
        cumulative_imbalance: float,
        price: float
    ) -> float:
        """Get action from trained battery agent."""
        # Build observation for battery agent (21 dimensions)
        obs = np.array([
            hour / 24.0,
            soc / self.config.battery_capacity_mwh,
            committed / self.config.plant_capacity_mw,
            pv_actual / self.config.plant_capacity_mw,
            *[c / self.config.plant_capacity_mw for c in future_commitments[:6]],
            *[f / self.config.plant_capacity_mw for f in future_forecasts[:6]],
            cumulative_imbalance / self.config.plant_capacity_mw,
            self.data_manager.normalize_price(price),
            (pv_actual - committed) / self.config.plant_capacity_mw,
            np.sin(hour * 2 * np.pi / 24),
            np.cos(hour * 2 * np.pi / 24),
        ], dtype=np.float32)

        action, _ = self._trained_battery_model.predict(obs, deterministic=True)
        return float(action[0])

    def _find_next_commitment_hour(self, start_idx: int) -> int:
        """Find the next index where hour == commitment_hour."""
        idx = start_idx
        max_search = min(48, len(self.data_manager) - idx)  # Search at most 48 hours

        for _ in range(max_search):
            if self.data_manager.get_hour(idx) == self.config.commitment_hour:
                return idx
            idx += 1

        # If not found, wrap around or raise
        raise ValueError("Could not find commitment hour in data range")

    def _get_tomorrow_window_start(self, commitment_idx: int) -> int:
        """Get the data index for tomorrow 00:00 given commitment hour index."""
        current_hour = self.data_manager.get_hour(commitment_idx)
        hours_until_midnight = (24 - current_hour) % 24
        if hours_until_midnight == 0:
            hours_until_midnight = 24
        return commitment_idx + hours_until_midnight

    def _get_observation(self) -> np.ndarray:
        """Build observation vector for commitment decision.

        Observation contains information about tomorrow (the commitment period).
        """
        row = self.data_manager.get_row(self.commitment_hour_idx)

        # Get tomorrow's window
        tomorrow_start = self._get_tomorrow_window_start(self.commitment_hour_idx)

        # PV forecasts for tomorrow
        forecasts = self.data_manager.get_forecasts(tomorrow_start, hours=24)
        forecasts_norm = self.data_manager.normalize_forecasts(forecasts)

        # Prices for tomorrow
        prices = self.data_manager.get_prices(tomorrow_start, hours=24)
        prices_norm = self.data_manager.normalize_prices(prices)

        # Current battery SOC
        soc_norm = self.battery.soc_normalized

        # Weather (current, as proxy for tomorrow)
        temp_norm = row['temperature_c'] / self.data_manager.norm_factors['temperature']
        irr_norm = row['irradiance_direct'] / self.data_manager.norm_factors['irradiance']

        # Forecast confidence proxy: recent forecast error (simplified)
        # In a real system, would track actual forecast errors
        forecast_confidence = 0.85  # Placeholder

        # Time features
        time_features = [
            row['day_sin'], row['day_cos'],
            row['month_sin'], row['month_cos']
        ]

        obs = np.concatenate([
            forecasts_norm,       # 24
            prices_norm,          # 24
            [soc_norm],           # 1
            [temp_norm, irr_norm], # 2
            [forecast_confidence], # 1
            time_features,        # 4
        ]).astype(np.float32)

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
                - start_idx: Specific data index to start searching from

        Returns:
            observation: Initial observation array (56,)
            info: Info dict with episode configuration
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Determine starting index
        if options and 'start_idx' in options:
            search_start = options['start_idx']
        else:
            # Random start, leaving room for 48 hours (commitment + execution)
            max_start = len(self.data_manager) - 72
            search_start = np.random.randint(0, max(1, max_start))

        # Find next commitment hour
        self.commitment_hour_idx = self._find_next_commitment_hour(search_start)
        self.current_idx = self.commitment_hour_idx

        # Reset battery
        initial_soc_fraction = 0.5
        if options and 'initial_soc' in options:
            initial_soc_fraction = np.clip(float(options['initial_soc']), 0.0, 1.0)
        self.battery.reset(initial_soc_fraction * self.config.battery_capacity_mwh)
        self.battery_soc_at_commitment = self.battery.soc

        # Clear tracking
        self.last_commitments = np.zeros(24)
        self.last_simulation_results = {}

        info = {
            'commitment_hour_idx': self.commitment_hour_idx,
            'initial_soc': self.battery.soc,
            'battery_policy': self._battery_policy_type,
        }

        return self._get_observation(), info

    def _simulate_day(self, commitments: np.ndarray) -> dict:
        """Simulate 24 hours of operation with given commitments.

        Uses the battery policy to manage the battery hour by hour.

        Args:
            commitments: Array of 24 hourly commitments (MWh)

        Returns:
            Dict with simulation results
        """
        tomorrow_start = self._get_tomorrow_window_start(self.commitment_hour_idx)

        # Get actual PV and prices for tomorrow
        actuals = self.data_manager.get_actuals(tomorrow_start, hours=24)
        prices = self.data_manager.get_prices(tomorrow_start, hours=24)

        # Initialize simulation state
        self.battery.reset(self.battery_soc_at_commitment)
        cumulative_imbalance = 0.0
        total_revenue = 0.0
        total_imbalance_cost = 0.0
        total_degradation = 0.0

        hourly_results = []

        for hour in range(24):
            idx = tomorrow_start + hour
            if idx >= len(self.data_manager):
                break

            pv_actual = actuals[hour]
            price_da = prices[hour]
            price_short, price_long = self.data_manager.get_imbalance_prices(idx)
            committed = commitments[hour]

            # Get future info for battery policy
            future_commits = commitments[hour+1:min(hour+7, 24)]
            future_commits = np.pad(future_commits, (0, 6 - len(future_commits)))

            # Get future forecasts
            future_forecasts = []
            for i in range(1, 7):
                future_idx = idx + i
                if future_idx < len(self.data_manager):
                    future_forecasts.append(
                        self.data_manager.get_row(future_idx)['pv_forecast_mwh']
                    )
                else:
                    future_forecasts.append(0.0)
            future_forecasts = np.array(future_forecasts)

            # Get battery action
            battery_action = self.battery_policy(
                soc=self.battery.soc,
                committed=committed,
                pv_actual=pv_actual,
                hour=hour,
                future_commitments=future_commits,
                future_forecasts=future_forecasts,
                cumulative_imbalance=cumulative_imbalance,
                price=price_da
            )

            # Execute battery action
            energy_delta, throughput, degradation = self.battery.step(
                battery_action, available_pv=pv_actual
            )

            # Calculate delivery
            delivered = pv_actual + energy_delta

            # Calculate settlement
            revenue, imbalance_cost = Settlement.calculate(
                committed=committed,
                delivered=delivered,
                price_da=price_da,
                price_short=price_short,
                price_long=price_long
            )

            # Update totals
            imbalance = delivered - committed
            cumulative_imbalance += imbalance
            total_revenue += revenue
            total_imbalance_cost += imbalance_cost
            total_degradation += degradation

            hourly_results.append({
                'hour': hour,
                'committed': committed,
                'pv_actual': pv_actual,
                'delivered': delivered,
                'imbalance': imbalance,
                'revenue': revenue,
                'imbalance_cost': imbalance_cost,
                'battery_soc': self.battery.soc,
                'battery_action': battery_action,
            })

        return {
            'total_revenue': total_revenue,
            'total_imbalance_cost': total_imbalance_cost,
            'total_degradation': total_degradation,
            'total_profit': total_revenue - total_imbalance_cost - total_degradation,
            'final_soc': self.battery.soc,
            'cumulative_imbalance': cumulative_imbalance,
            'hourly_results': hourly_results,
        }

    def step(
        self,
        action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute commitment decision and simulate 24h.

        Args:
            action: Array of shape (24,) with commitment fractions [0, 1]

        Returns:
            observation: Final observation (same as initial for single-step episode)
            reward: Total 24h profit
            terminated: Always True (single-step episode)
            truncated: False unless data boundary reached
            info: Dict with simulation details
        """
        # Get tomorrow's forecasts for max commitment calculation
        tomorrow_start = self._get_tomorrow_window_start(self.commitment_hour_idx)
        forecasts = self.data_manager.get_forecasts(tomorrow_start, hours=24)

        # Convert action fractions to actual commitments
        action_clipped = np.clip(action, 0, 1)
        max_commitments = calculate_max_commitment(forecasts, self.config.battery_power_mw)
        commitments = action_clipped * max_commitments
        self.last_commitments = commitments.copy()

        # Simulate the day
        results = self._simulate_day(commitments)
        self.last_simulation_results = results

        # Reward is total profit
        reward = results['total_profit']

        # Episode always terminates after single step
        terminated = True
        truncated = (tomorrow_start + 24) >= len(self.data_manager)

        # Build info
        info = {
            'commitments': commitments,
            'forecasts': forecasts,
            'total_revenue': results['total_revenue'],
            'total_imbalance_cost': results['total_imbalance_cost'],
            'total_degradation': results['total_degradation'],
            'total_profit': results['total_profit'],
            'final_soc': results['final_soc'],
            'cumulative_imbalance': results['cumulative_imbalance'],
            # Include hourly breakdown for analysis
            'hourly_results': results['hourly_results'],
        }

        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        """Render current state."""
        if self.render_mode == 'human':
            row = self.data_manager.get_row(self.commitment_hour_idx)
            tomorrow_start = self._get_tomorrow_window_start(self.commitment_hour_idx)
            forecasts = self.data_manager.get_forecasts(tomorrow_start, hours=24)

            print(f"\n{'='*60}")
            print(f"Commitment Decision at hour {self.config.commitment_hour}")
            print(f"{'='*60}")
            print(f"Battery SOC: {self.battery.soc:.1f} / {self.config.battery_capacity_mwh} MWh")
            print(f"Tomorrow's total forecast: {forecasts.sum():.1f} MWh")

            if self.last_simulation_results:
                r = self.last_simulation_results
                print(f"\nSimulation Results:")
                print(f"  Revenue: {r['total_revenue']:.2f} EUR")
                print(f"  Imbalance cost: {r['total_imbalance_cost']:.2f} EUR")
                print(f"  Degradation: {r['total_degradation']:.2f} EUR")
                print(f"  Profit: {r['total_profit']:.2f} EUR")


def load_commitment_env(
    data_path: str,
    battery_policy: Optional[Union[Callable, str]] = None,
    **kwargs
) -> CommitmentEnv:
    """Helper to load CommitmentEnv from processed data file.

    Args:
        data_path: Path to CSV file with required columns
        battery_policy: Battery policy ('heuristic', 'trained', or callable)
        **kwargs: Additional arguments passed to CommitmentEnv

    Returns:
        Configured CommitmentEnv instance
    """
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    return CommitmentEnv(df, battery_policy=battery_policy, **kwargs)
