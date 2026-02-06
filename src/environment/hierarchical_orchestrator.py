"""
Hierarchical Orchestrator
=========================

Combines the trained commitment and battery agents for deployment
and evaluation. Runs the full decision loop:

1. At commitment hour (11:00): Commitment agent decides 24h schedule
2. Every hour: Battery agent manages charge/discharge

This orchestrator can be used with:
- Trained agents (loaded from model files)
- Heuristic policies (for comparison)
- Any combination of the above
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Callable, Dict, List, Tuple
from dataclasses import dataclass

from .solar_plant import (
    PlantConfig,
    Battery,
    Settlement,
    DataManager,
    calculate_max_commitment,
    heuristic_battery_policy,
)


@dataclass
class EpisodeResult:
    """Results from a single episode (multiple days)."""
    total_revenue: float
    total_imbalance_cost: float
    total_degradation: float
    total_profit: float
    num_days: int
    daily_profits: List[float]
    daily_imbalance_costs: List[float]
    commitment_decisions: List[np.ndarray]


class HierarchicalOrchestrator:
    """
    Orchestrates hierarchical agents for solar merchant trading.

    Combines a commitment agent (makes daily decisions at hour 11)
    with a battery agent (makes hourly decisions) for full operation.

    Example usage:
        >>> orchestrator = HierarchicalOrchestrator.from_trained_agents(
        ...     data_path='data/processed/test.csv'
        ... )
        >>> result = orchestrator.run_episode(start_idx=0, num_days=7)
        >>> print(f"7-day profit: {result.total_profit:.2f} EUR")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        plant_config: Optional[PlantConfig] = None,
        commitment_agent=None,
        battery_agent=None,
        commitment_policy: Optional[Callable] = None,
        battery_policy: Optional[Callable] = None,
    ):
        """
        Initialize orchestrator.

        Args:
            data: DataFrame with required columns
            plant_config: Plant configuration
            commitment_agent: Trained SB3 model for commitment decisions
            battery_agent: Trained SB3 model for battery decisions
            commitment_policy: Callable policy (alternative to agent)
            battery_policy: Callable policy (alternative to agent)

        Note: Provide either agent or policy for each role, not both.
        """
        self.config = plant_config or PlantConfig()
        self.data_manager = DataManager(data, self.config)

        # Set up commitment policy
        if commitment_agent is not None:
            self._commitment_agent = commitment_agent
            self._commitment_type = 'trained'
        elif commitment_policy is not None:
            self._commitment_policy = commitment_policy
            self._commitment_type = 'policy'
        else:
            self._commitment_policy = self._default_commitment_policy
            self._commitment_type = 'heuristic'

        # Set up battery policy
        if battery_agent is not None:
            self._battery_agent = battery_agent
            self._battery_type = 'trained'
        elif battery_policy is not None:
            self._battery_policy = battery_policy
            self._battery_type = 'policy'
        else:
            self._battery_policy = self._default_battery_policy
            self._battery_type = 'heuristic'

        # Battery for simulation
        self.battery = Battery(
            capacity_mwh=self.config.battery_capacity_mwh,
            power_mw=self.config.battery_power_mw,
            efficiency=self.config.battery_efficiency,
            degradation_cost=self.config.battery_degradation_cost
        )

    @classmethod
    def from_trained_agents(
        cls,
        data_path: str,
        commitment_model_path: Optional[str] = None,
        battery_model_path: Optional[str] = None,
        plant_config: Optional[PlantConfig] = None,
    ) -> 'HierarchicalOrchestrator':
        """
        Create orchestrator from trained model files.

        Args:
            data_path: Path to data CSV
            commitment_model_path: Path to commitment agent .zip file
            battery_model_path: Path to battery agent .zip file
            plant_config: Plant configuration

        Returns:
            Configured HierarchicalOrchestrator
        """
        from stable_baselines3 import SAC

        data = pd.read_csv(data_path, parse_dates=['datetime'])
        config = plant_config or PlantConfig()

        # Load commitment agent
        commitment_agent = None
        if commitment_model_path is None:
            # Try default path
            default_path = Path(__file__).parent.parent.parent / 'models' / 'commitment_agent' / 'best' / 'best_model.zip'
            if default_path.exists():
                commitment_agent = SAC.load(str(default_path))
                print(f"Loaded commitment agent from {default_path}")
        elif Path(commitment_model_path).exists():
            commitment_agent = SAC.load(commitment_model_path)
            print(f"Loaded commitment agent from {commitment_model_path}")

        # Load battery agent
        battery_agent = None
        if battery_model_path is None:
            # Try default path
            default_path = Path(__file__).parent.parent.parent / 'models' / 'battery_agent' / 'best' / 'best_model.zip'
            if default_path.exists():
                battery_agent = SAC.load(str(default_path))
                print(f"Loaded battery agent from {default_path}")
        elif Path(battery_model_path).exists():
            battery_agent = SAC.load(battery_model_path)
            print(f"Loaded battery agent from {battery_model_path}")

        if commitment_agent is None:
            print("Warning: No commitment agent loaded, using heuristic")
        if battery_agent is None:
            print("Warning: No battery agent loaded, using heuristic")

        return cls(
            data=data,
            plant_config=config,
            commitment_agent=commitment_agent,
            battery_agent=battery_agent,
        )

    def _default_commitment_policy(
        self,
        forecasts: np.ndarray,
        prices: np.ndarray,
        battery_soc: float,
        **kwargs
    ) -> np.ndarray:
        """Default heuristic commitment policy.

        Conservative strategy: commit 85% of forecast.
        """
        commitment_fractions = np.ones(24) * 0.85
        max_commitments = calculate_max_commitment(forecasts, self.config.battery_power_mw)
        return commitment_fractions * max_commitments

    def _default_battery_policy(
        self,
        soc: float,
        committed: float,
        pv_actual: float,
        **kwargs
    ) -> float:
        """Default heuristic battery policy."""
        return heuristic_battery_policy(
            soc=soc,
            committed=committed,
            pv_actual=pv_actual,
            battery_power_mw=self.config.battery_power_mw,
            battery_capacity_mwh=self.config.battery_capacity_mwh
        )

    def _get_commitment_observation(self, idx: int) -> np.ndarray:
        """Build observation for commitment agent."""
        row = self.data_manager.get_row(idx)
        current_hour = int(row['hour'])

        # Get tomorrow's window
        hours_until_midnight = (24 - current_hour) % 24
        if hours_until_midnight == 0:
            hours_until_midnight = 24
        tomorrow_start = idx + hours_until_midnight

        # Get forecasts and prices
        forecasts = self.data_manager.get_forecasts(tomorrow_start, 24)
        prices = self.data_manager.get_prices(tomorrow_start, 24)

        obs = np.concatenate([
            self.data_manager.normalize_forecasts(forecasts),
            self.data_manager.normalize_prices(prices),
            [self.battery.soc_normalized],
            [row['temperature_c'] / self.data_manager.norm_factors['temperature'],
             row['irradiance_direct'] / self.data_manager.norm_factors['irradiance']],
            [0.85],  # Forecast confidence placeholder
            [row['day_sin'], row['day_cos'], row['month_sin'], row['month_cos']],
        ]).astype(np.float32)

        return obs, forecasts, prices, tomorrow_start

    def _get_battery_observation(
        self,
        idx: int,
        hour: int,
        commitments: np.ndarray,
        cumulative_imbalance: float
    ) -> np.ndarray:
        """Build observation for battery agent."""
        row = self.data_manager.get_row(idx)
        pv_actual = row['pv_actual_mwh']
        price = row['price_eur_mwh']

        # Future info
        future_commits = []
        future_forecasts = []
        for i in range(1, 7):
            future_hour = (hour + i) % 24
            future_commits.append(commitments[future_hour])
            future_idx = idx + i
            if future_idx < len(self.data_manager):
                future_forecasts.append(
                    self.data_manager.get_row(future_idx)['pv_forecast_mwh']
                )
            else:
                future_forecasts.append(0.0)

        imbalance_status = (pv_actual - commitments[hour]) / self.config.plant_capacity_mw

        obs = np.array([
            hour / 24.0,
            self.battery.soc_normalized,
            commitments[hour] / self.config.plant_capacity_mw,
            pv_actual / self.config.plant_capacity_mw,
            *[c / self.config.plant_capacity_mw for c in future_commits],
            *[f / self.config.plant_capacity_mw for f in future_forecasts],
            cumulative_imbalance / self.config.plant_capacity_mw,
            self.data_manager.normalize_price(price),
            imbalance_status,
            row['hour_sin'],
            row['hour_cos'],
        ], dtype=np.float32)

        return obs

    def _get_commitment_action(
        self,
        obs: np.ndarray,
        forecasts: np.ndarray,
        prices: np.ndarray
    ) -> np.ndarray:
        """Get commitment decision from agent or policy."""
        if self._commitment_type == 'trained':
            action, _ = self._commitment_agent.predict(obs, deterministic=True)
            action = np.clip(action, 0, 1)
            max_commits = calculate_max_commitment(forecasts, self.config.battery_power_mw)
            return action * max_commits
        else:
            return self._commitment_policy(
                forecasts=forecasts,
                prices=prices,
                battery_soc=self.battery.soc
            )

    def _get_battery_action(self, obs: np.ndarray, committed: float, pv_actual: float) -> float:
        """Get battery action from agent or policy."""
        if self._battery_type == 'trained':
            action, _ = self._battery_agent.predict(obs, deterministic=True)
            return float(action[0])
        else:
            return self._battery_policy(
                soc=self.battery.soc,
                committed=committed,
                pv_actual=pv_actual
            )

    def run_episode(
        self,
        start_idx: int = 0,
        num_days: int = 7,
        initial_soc: float = 0.5
    ) -> EpisodeResult:
        """
        Run a full episode spanning multiple days.

        Args:
            start_idx: Starting index in data
            num_days: Number of days to simulate
            initial_soc: Initial battery SOC as fraction [0, 1]

        Returns:
            EpisodeResult with aggregated metrics
        """
        # Find first commitment hour
        idx = start_idx
        while self.data_manager.get_hour(idx) != self.config.commitment_hour:
            idx += 1
            if idx >= len(self.data_manager) - 48:
                raise ValueError("Cannot find commitment hour in data range")

        # Initialize
        self.battery.reset(initial_soc * self.config.battery_capacity_mwh)

        total_revenue = 0.0
        total_imbalance_cost = 0.0
        total_degradation = 0.0
        daily_profits = []
        daily_imbalance_costs = []
        commitment_decisions = []

        for day in range(num_days):
            # Check if we have enough data
            if idx + 48 >= len(self.data_manager):
                break

            # Get commitment observation and make decision
            obs, forecasts, prices, tomorrow_start = self._get_commitment_observation(idx)
            commitments = self._get_commitment_action(obs, forecasts, prices)
            commitment_decisions.append(commitments.copy())

            # Simulate the day
            day_revenue = 0.0
            day_imbalance_cost = 0.0
            day_degradation = 0.0
            cumulative_imbalance = 0.0

            for hour in range(24):
                sim_idx = tomorrow_start + hour
                if sim_idx >= len(self.data_manager):
                    break

                row = self.data_manager.get_row(sim_idx)
                pv_actual = row['pv_actual_mwh']
                price_da = row['price_eur_mwh']
                price_short, price_long = self.data_manager.get_imbalance_prices(sim_idx)
                committed = commitments[hour]

                # Get battery action
                battery_obs = self._get_battery_observation(
                    sim_idx, hour, commitments, cumulative_imbalance
                )
                battery_action = self._get_battery_action(battery_obs, committed, pv_actual)

                # Execute battery action
                energy_delta, throughput, degradation = self.battery.step(
                    battery_action, available_pv=pv_actual
                )

                # Calculate delivery and settlement
                delivered = pv_actual + energy_delta
                revenue, imbalance_cost = Settlement.calculate(
                    committed=committed,
                    delivered=delivered,
                    price_da=price_da,
                    price_short=price_short,
                    price_long=price_long
                )

                # Update tracking
                imbalance = delivered - committed
                cumulative_imbalance += imbalance
                day_revenue += revenue
                day_imbalance_cost += imbalance_cost
                day_degradation += degradation

            # Record daily results
            day_profit = day_revenue - day_imbalance_cost - day_degradation
            daily_profits.append(day_profit)
            daily_imbalance_costs.append(day_imbalance_cost)

            total_revenue += day_revenue
            total_imbalance_cost += day_imbalance_cost
            total_degradation += day_degradation

            # Move to next commitment hour (24 hours later)
            idx += 24

        return EpisodeResult(
            total_revenue=total_revenue,
            total_imbalance_cost=total_imbalance_cost,
            total_degradation=total_degradation,
            total_profit=total_revenue - total_imbalance_cost - total_degradation,
            num_days=len(daily_profits),
            daily_profits=daily_profits,
            daily_imbalance_costs=daily_imbalance_costs,
            commitment_decisions=commitment_decisions,
        )

    def evaluate(
        self,
        num_episodes: int = 10,
        days_per_episode: int = 7,
        seed: int = 42
    ) -> Dict:
        """
        Evaluate orchestrator over multiple episodes.

        Args:
            num_episodes: Number of episodes to run
            days_per_episode: Days per episode
            seed: Random seed

        Returns:
            Dict with evaluation metrics
        """
        np.random.seed(seed)

        all_profits = []
        all_imbalance_costs = []
        all_daily_profits = []

        max_start = len(self.data_manager) - (days_per_episode + 2) * 24

        for _ in range(num_episodes):
            start_idx = np.random.randint(0, max(1, max_start))
            initial_soc = np.random.uniform(0.3, 0.7)

            try:
                result = self.run_episode(
                    start_idx=start_idx,
                    num_days=days_per_episode,
                    initial_soc=initial_soc
                )
                all_profits.append(result.total_profit)
                all_imbalance_costs.append(result.total_imbalance_cost)
                all_daily_profits.extend(result.daily_profits)
            except (ValueError, IndexError):
                continue

        return {
            'mean_episode_profit': np.mean(all_profits) if all_profits else 0,
            'std_episode_profit': np.std(all_profits) if all_profits else 0,
            'mean_daily_profit': np.mean(all_daily_profits) if all_daily_profits else 0,
            'mean_imbalance_cost': np.mean(all_imbalance_costs) if all_imbalance_costs else 0,
            'num_episodes': len(all_profits),
            'commitment_type': self._commitment_type,
            'battery_type': self._battery_type,
        }


def load_orchestrator(
    data_path: str,
    commitment_model_path: Optional[str] = None,
    battery_model_path: Optional[str] = None,
) -> HierarchicalOrchestrator:
    """Helper to load orchestrator with trained agents."""
    return HierarchicalOrchestrator.from_trained_agents(
        data_path=data_path,
        commitment_model_path=commitment_model_path,
        battery_model_path=battery_model_path,
    )
