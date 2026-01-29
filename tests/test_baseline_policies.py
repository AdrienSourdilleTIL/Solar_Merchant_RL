"""
Tests for baseline trading policies.

This module validates the conservative baseline policy according to Story 3-1:
- Action shape and dtype correctness
- Action values within valid range
- Commitment fractions at 80% of forecast
- Battery heuristic responds to delivery gaps and surplus
- Full episode execution without errors
- Module imports correctly
"""

import numpy as np
import pytest

from src.baselines import conservative_policy
from src.baselines.baseline_policies import _parse_observation


class TestModuleImports:
    """Task 4: Test module imports correctly (AC: #3)."""

    def test_conservative_policy_importable_from_package(self):
        """Verify conservative_policy is exported from src.baselines."""
        from src.baselines import conservative_policy as cp
        assert callable(cp)

    def test_conservative_policy_importable_from_module(self):
        """Verify conservative_policy is in baseline_policies module."""
        from src.baselines.baseline_policies import conservative_policy as cp
        assert callable(cp)


class TestObservationParsing:
    """Task 4: Test observation parsing helper (AC: #1)."""

    def test_parse_observation_keys(self):
        """Verify _parse_observation returns all expected fields."""
        obs = np.zeros(84, dtype=np.float32)
        parsed = _parse_observation(obs)
        expected_keys = {
            "hour", "soc", "commitments", "cumulative_imbalance",
            "pv_forecast", "prices", "actual_pv", "weather", "time_features",
        }
        assert set(parsed.keys()) == expected_keys

    def test_parse_observation_shapes(self):
        """Verify parsed field shapes match observation layout."""
        obs = np.arange(84, dtype=np.float32)
        parsed = _parse_observation(obs)
        assert parsed["hour"] == 0.0
        assert parsed["soc"] == 1.0
        assert parsed["commitments"].shape == (24,)
        assert parsed["cumulative_imbalance"] == 26.0
        assert parsed["pv_forecast"].shape == (24,)
        assert parsed["prices"].shape == (24,)
        assert parsed["actual_pv"] == 75.0
        assert parsed["weather"].shape == (2,)
        assert parsed["time_features"].shape == (6,)


class TestConservativePolicyAction:
    """Task 4: Test action shape, range, and commitment values (AC: #1)."""

    def test_action_shape(self, env):
        """Verify action output is (25,) array."""
        obs, _ = env.reset(seed=42)
        action = conservative_policy(obs)
        assert action.shape == (25,)

    def test_action_dtype(self, env):
        """Verify action output is float32."""
        obs, _ = env.reset(seed=42)
        action = conservative_policy(obs)
        assert action.dtype == np.float32

    def test_action_range(self, env):
        """Verify all action values are in [0, 1]."""
        obs, _ = env.reset(seed=42)
        action = conservative_policy(obs)
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    def test_commitment_fractions_are_080(self, env):
        """Verify commitment fractions (action[0:24]) are all 0.8."""
        obs, _ = env.reset(seed=42)
        action = conservative_policy(obs)
        np.testing.assert_allclose(action[0:24], 0.8, atol=1e-7)


class TestConservativePolicyBattery:
    """Task 4: Test battery heuristic logic (AC: #1, #2)."""

    def test_battery_discharge_when_under_delivering(self):
        """Battery should discharge when cumulative imbalance < 0 and SOC > 0."""
        obs = np.zeros(84, dtype=np.float32)
        obs[1] = 0.5    # SOC > 0 (normalized, means ~5 MWh)
        obs[26] = -0.1   # Cumulative imbalance < 0 (under-delivering)
        action = conservative_policy(obs)
        assert action[24] == 0.0  # Full discharge

    def test_battery_charge_when_over_delivering(self):
        """Battery should charge when cumulative imbalance > 0."""
        obs = np.zeros(84, dtype=np.float32)
        obs[1] = 0.3     # Some SOC
        obs[26] = 0.1    # Cumulative imbalance > 0 (over-delivering)
        action = conservative_policy(obs)
        assert action[24] == 1.0  # Full charge

    def test_battery_idle_when_balanced(self):
        """Battery should idle when cumulative imbalance is zero."""
        obs = np.zeros(84, dtype=np.float32)
        obs[1] = 0.5     # Some SOC
        obs[26] = 0.0    # Balanced
        action = conservative_policy(obs)
        assert action[24] == 0.5  # Idle

    def test_battery_idle_when_under_delivering_but_empty(self):
        """Battery should idle when under-delivering but SOC is 0."""
        obs = np.zeros(84, dtype=np.float32)
        obs[1] = 0.0     # No SOC
        obs[26] = -0.1   # Under-delivering
        action = conservative_policy(obs)
        assert action[24] == 0.5  # Idle (can't discharge empty battery)

    def test_battery_charges_on_pv_surplus(self):
        """Battery should charge when PV surplus exists above commitment."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 14.0 / 24.0  # Hour 14 (normalized)
        obs[1] = 0.3           # Some SOC
        obs[26] = 0.0          # Balanced cumulative imbalance
        obs[16] = 0.2          # Commitment for hour 14 (index 2+14=16): 0.2 normalized
        obs[75] = 0.5          # Actual PV: 0.5 normalized (surplus above 0.2 commitment)
        action = conservative_policy(obs)
        assert action[24] == 1.0  # Should charge due to PV surplus


class TestConservativePolicyEpisode:
    """Task 4: Test full episode execution (AC: #2)."""

    def test_full_episode_runs_without_errors(self, env):
        """Run a full 48-hour episode with conservative policy."""
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        steps = 0

        done = False
        while not done:
            action = conservative_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        assert steps == 48  # 48-hour episode
        assert np.isfinite(total_reward)

    def test_episode_metrics_are_finite(self, env):
        """Verify revenue, imbalance_cost, and net_profit are finite."""
        obs, _ = env.reset(seed=42)
        total_revenue = 0.0
        total_imbalance_cost = 0.0

        done = False
        while not done:
            action = conservative_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_revenue += info["revenue"]
            total_imbalance_cost += info["imbalance_cost"]
            done = terminated or truncated

        assert np.isfinite(total_revenue)
        assert np.isfinite(total_imbalance_cost)
        net_profit = total_revenue - total_imbalance_cost
        assert np.isfinite(net_profit)

    def test_commitment_values_are_80_percent(self, env):
        """Verify commitments during episode are 80% of forecast."""
        obs, _ = env.reset(seed=42)

        done = False
        while not done:
            action = conservative_policy(obs)
            # Commitment fractions should always be 0.8
            np.testing.assert_allclose(action[0:24], 0.8, atol=1e-7)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    def test_battery_responds_during_episode(self, env):
        """Verify battery action varies in response to conditions during episode."""
        obs, _ = env.reset(seed=42)
        battery_actions = []

        done = False
        while not done:
            action = conservative_policy(obs)
            battery_actions.append(float(action[24]))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Battery should not idle the entire 48-hour episode
        unique_actions = set(battery_actions)
        assert len(unique_actions) > 1, (
            f"Battery was stuck at {unique_actions} for entire episode — "
            "should respond to imbalance/surplus"
        )
