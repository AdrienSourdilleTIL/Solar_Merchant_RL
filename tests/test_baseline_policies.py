"""
Tests for baseline trading policies.

This module validates baseline policies:
- Conservative policy (Story 3-1): commits 80% of forecast, battery fills gaps
- Aggressive policy (Story 3-2): commits 100% of max, battery discharges aggressively
- Price-aware policy (Story 3-3): adjusts commitment and battery based on price levels

Tests cover:
- Action shape, dtype, and range correctness
- Commitment fraction values
- Battery heuristic responds to delivery gaps, surplus, and SOC conditions
- Full 48-hour episode execution without errors
- Module imports correctly
"""

import numpy as np
import pytest

from src.baselines import aggressive_policy, conservative_policy, price_aware_policy
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


# ============================================================================
# Aggressive Policy Tests (Story 3-2)
# ============================================================================


class TestAggressivePolicyImport:
    """Test module import of aggressive_policy (AC: #3)."""

    def test_aggressive_policy_importable_from_package(self):
        """Verify aggressive_policy is exported from src.baselines."""
        from src.baselines import aggressive_policy as ap
        assert callable(ap)

    def test_aggressive_policy_importable_from_module(self):
        """Verify aggressive_policy is in baseline_policies module."""
        from src.baselines.baseline_policies import aggressive_policy as ap
        assert callable(ap)


class TestAggressivePolicyAction:
    """Test action shape, dtype, range, and commitment values (AC: #1)."""

    def test_action_shape(self, env):
        """Verify action output is (25,) array."""
        obs, _ = env.reset(seed=42)
        action = aggressive_policy(obs)
        assert action.shape == (25,)

    def test_action_dtype(self, env):
        """Verify action output is float32."""
        obs, _ = env.reset(seed=42)
        action = aggressive_policy(obs)
        assert action.dtype == np.float32

    def test_action_range(self, env):
        """Verify all action values are in [0, 1]."""
        obs, _ = env.reset(seed=42)
        action = aggressive_policy(obs)
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    def test_commitment_fractions_are_100(self, env):
        """Verify commitment fractions (action[0:24]) are all 1.0."""
        obs, _ = env.reset(seed=42)
        action = aggressive_policy(obs)
        np.testing.assert_allclose(action[0:24], 1.0, atol=1e-7)


class TestAggressivePolicyBattery:
    """Test battery heuristic logic (AC: #1, #2)."""

    def test_battery_discharge_by_default(self):
        """Battery should discharge by default (aggressive strategy)."""
        obs = np.zeros(84, dtype=np.float32)
        obs[1] = 0.5    # SOC > 0
        obs[26] = 0.0    # Balanced
        action = aggressive_policy(obs)
        assert action[24] == 0.0  # Full discharge (aggressive default)

    def test_battery_idle_when_soc_empty(self):
        """Battery should idle when SOC is 0 (can't discharge empty battery)."""
        obs = np.zeros(84, dtype=np.float32)
        obs[1] = 0.0     # SOC = 0 (empty)
        action = aggressive_policy(obs)
        assert action[24] == 0.5  # Idle

    def test_battery_charges_on_pv_surplus(self):
        """Battery should charge when PV surplus exists above commitment."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 14.0 / 24.0  # Hour 14 (normalized)
        obs[1] = 0.3           # Some SOC (not full)
        obs[16] = 0.2          # Commitment for hour 14 (index 2+14=16): 0.2 normalized
        obs[75] = 0.5          # Actual PV: 0.5 normalized (surplus above 0.2 commitment)
        action = aggressive_policy(obs)
        assert action[24] == 1.0  # Should charge due to PV surplus

    def test_battery_charges_when_soc_empty_and_pv_surplus(self):
        """Battery should charge when SOC is 0 but PV surplus exists (SOC=0 is not full)."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 14.0 / 24.0  # Hour 14 (normalized)
        obs[1] = 0.0           # SOC = 0 (empty, but NOT full)
        obs[16] = 0.2          # Commitment for hour 14 (index 2+14=16)
        obs[75] = 0.5          # Actual PV: surplus above 0.2 commitment
        action = aggressive_policy(obs)
        assert action[24] == 1.0  # Should charge (empty battery has room)

    def test_battery_discharges_when_no_surplus(self):
        """Battery should discharge when no PV surplus and SOC > 0."""
        obs = np.zeros(84, dtype=np.float32)
        obs[1] = 0.5     # SOC > 0
        obs[26] = -0.1   # Under-delivering
        action = aggressive_policy(obs)
        assert action[24] == 0.0  # Full discharge (aggressive default)

    def test_battery_discharges_when_soc_full_and_surplus(self):
        """Battery should discharge when SOC is full even with PV surplus."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 12.0 / 24.0  # Hour 12
        obs[1] = 1.0           # SOC = full (normalized = 1.0)
        obs[14] = 0.2          # Commitment for hour 12 (index 2+12=14)
        obs[75] = 0.5          # Actual PV surplus
        action = aggressive_policy(obs)
        assert action[24] == 0.0  # Discharge (can't charge full battery)


class TestAggressivePolicyEpisode:
    """Test full episode execution (AC: #2)."""

    def test_full_episode_runs_without_errors(self, env):
        """Run a full 48-hour episode with aggressive policy."""
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        steps = 0

        done = False
        while not done:
            action = aggressive_policy(obs)
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
            action = aggressive_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_revenue += info["revenue"]
            total_imbalance_cost += info["imbalance_cost"]
            done = terminated or truncated

        assert np.isfinite(total_revenue)
        assert np.isfinite(total_imbalance_cost)
        net_profit = total_revenue - total_imbalance_cost
        assert np.isfinite(net_profit)

    def test_commitment_values_are_100_percent(self, env):
        """Verify commitments during episode are 100% of forecast."""
        obs, _ = env.reset(seed=42)

        done = False
        while not done:
            action = aggressive_policy(obs)
            # Commitment fractions should always be 1.0
            np.testing.assert_allclose(action[0:24], 1.0, atol=1e-7)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    def test_battery_responds_during_episode(self, env):
        """Verify battery action varies in response to conditions during episode."""
        obs, _ = env.reset(seed=42)
        battery_actions = []

        done = False
        while not done:
            action = aggressive_policy(obs)
            battery_actions.append(float(action[24]))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Battery should not be stuck at a single value the entire episode
        unique_actions = set(battery_actions)
        assert len(unique_actions) > 1, (
            f"Battery was stuck at {unique_actions} for entire episode — "
            "should respond to SOC and surplus conditions"
        )


# ============================================================================
# Price-Aware Policy Tests (Story 3-3)
# ============================================================================


class TestPriceAwarePolicyImport:
    """Test module import of price_aware_policy (AC: #3)."""

    def test_price_aware_policy_importable_from_package(self):
        """Verify price_aware_policy is exported from src.baselines."""
        from src.baselines import price_aware_policy as pap
        assert callable(pap)

    def test_price_aware_policy_importable_from_module(self):
        """Verify price_aware_policy is in baseline_policies module."""
        from src.baselines.baseline_policies import price_aware_policy as pap
        assert callable(pap)


class TestPriceAwarePolicyAction:
    """Tests for price-aware policy action output (AC: #1)."""

    def test_action_shape(self, env):
        """Verify action output is (25,) array."""
        obs, _ = env.reset(seed=42)
        action = price_aware_policy(obs)
        assert action.shape == (25,)

    def test_action_dtype(self, env):
        """Verify action output is float32."""
        obs, _ = env.reset(seed=42)
        action = price_aware_policy(obs)
        assert action.dtype == np.float32

    def test_action_range(self, env):
        """Verify all action values are in [0, 1]."""
        obs, _ = env.reset(seed=42)
        action = price_aware_policy(obs)
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    def test_commitment_fractions_vary(self, env):
        """Commitment fractions should NOT be uniform (unlike conservative/aggressive)."""
        obs, _ = env.reset(seed=42)
        action = price_aware_policy(obs)
        unique_fractions = set(action[0:24].tolist())
        assert len(unique_fractions) > 1  # Should have both high and low fractions

    def test_high_price_gets_high_commitment(self):
        """Hours with above-median price should get fraction 1.0."""
        obs = np.zeros(84, dtype=np.float32)
        # Set up prices: hours 0-11 = 0.1 (low), hours 12-23 = 0.9 (high)
        obs[51:63] = 0.1   # Low prices
        obs[63:75] = 0.9   # High prices
        action = price_aware_policy(obs)
        # Median = 0.5 → hours 12-23 (high) should get 1.0, hours 0-11 (low) should get 0.5
        np.testing.assert_allclose(action[0:12], 0.5, atol=1e-7)  # Low price hours
        np.testing.assert_allclose(action[12:24], 1.0, atol=1e-7)  # High price hours

    def test_low_price_gets_low_commitment(self):
        """Hours with at-or-below-median price should get fraction 0.5."""
        obs = np.zeros(84, dtype=np.float32)
        # All prices equal → all at/below median → all 0.5
        obs[51:75] = 0.5
        action = price_aware_policy(obs)
        np.testing.assert_allclose(action[0:24], 0.5, atol=1e-7)
        assert action[24] == 0.5  # Battery idles on flat prices

    def test_negative_prices_commitment(self):
        """Negative prices should still produce correct relative commitment fractions."""
        obs = np.zeros(84, dtype=np.float32)
        # Mix of negative prices: hours 0-11 very negative, hours 12-23 slightly negative
        obs[51:63] = -0.5  # More negative (below median)
        obs[63:75] = -0.1  # Less negative (above median)
        action = price_aware_policy(obs)
        # Less negative hours should get high commitment, more negative get low
        np.testing.assert_allclose(action[0:12], 0.5, atol=1e-7)   # More negative → low
        np.testing.assert_allclose(action[12:24], 1.0, atol=1e-7)  # Less negative → high


class TestPriceAwarePolicyBattery:
    """Tests for price-aware policy battery heuristic (AC: #2)."""

    def test_battery_discharge_on_high_price(self):
        """Battery should discharge during high-price hours when SOC > 0."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 14.0 / 24.0  # Hour 14 (normalized)
        obs[1] = 0.5           # SOC > 0
        # Set hour 14 price high, others low → hour 14 is above median
        obs[51:75] = 0.1       # All prices low
        obs[65] = 0.9          # Price at hour 14 (index 51+14=65) is high
        action = price_aware_policy(obs)
        assert action[24] == 0.0  # Discharge during high price

    def test_battery_charge_on_low_price(self):
        """Battery should charge during low-price hours when PV surplus exists and SOC < 1.0."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 3.0 / 24.0   # Hour 3 (normalized)
        obs[1] = 0.3           # SOC < 1.0
        # Set hour 3 price low, others high → hour 3 is below median
        obs[51:75] = 0.9       # All prices high
        obs[54] = 0.1          # Price at hour 3 (index 51+3=54) is low
        obs[75] = 0.5          # Actual PV: surplus above commitment (0.0)
        action = price_aware_policy(obs)
        assert action[24] == 1.0  # Charge during low price with PV surplus

    def test_battery_idle_on_low_price_without_pv_surplus(self):
        """Battery should idle during low-price hours when no PV surplus exists."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 3.0 / 24.0   # Hour 3 (normalized)
        obs[1] = 0.3           # SOC < 1.0
        obs[51:75] = 0.9       # All prices high
        obs[54] = 0.1          # Price at hour 3 low
        # No PV surplus (actual_pv = 0, commitment = 0 → no surplus)
        action = price_aware_policy(obs)
        assert action[24] == 0.5  # Idle (no PV surplus to charge from)

    def test_battery_idle_on_flat_prices(self):
        """Battery should idle when all prices are identical (no strategic advantage)."""
        obs = np.zeros(84, dtype=np.float32)
        obs[51:75] = 0.5  # All prices equal
        obs[1] = 0.3      # SOC not empty/full
        action = price_aware_policy(obs)
        assert action[24] == 0.5  # Idle on flat prices

    def test_battery_idle_when_soc_empty_and_high_price(self):
        """Battery should idle when SOC is empty during high-price hour."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 14.0 / 24.0  # Hour 14
        obs[1] = 0.0           # SOC = 0 (empty)
        obs[51:75] = 0.1       # All prices low
        obs[65] = 0.9          # Hour 14 high
        action = price_aware_policy(obs)
        assert action[24] == 0.5  # Idle (can't discharge empty)

    def test_battery_idle_when_soc_full_and_low_price(self):
        """Battery should idle when SOC is full during low-price hour."""
        obs = np.zeros(84, dtype=np.float32)
        obs[0] = 3.0 / 24.0   # Hour 3
        obs[1] = 1.0           # SOC = 1.0 (full)
        obs[51:75] = 0.9       # All prices high
        obs[54] = 0.1          # Hour 3 low
        action = price_aware_policy(obs)
        assert action[24] == 0.5  # Idle (can't charge full)


class TestPriceAwarePolicyEpisode:
    """Test full 48h episode execution with price-aware policy (AC: #2)."""

    def test_full_episode_runs_without_errors(self, env):
        """Run a full 48-hour episode with price-aware policy."""
        obs, _ = env.reset(seed=42)
        total_reward = 0.0
        steps = 0

        done = False
        while not done:
            action = price_aware_policy(obs)
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
            action = price_aware_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_revenue += info["revenue"]
            total_imbalance_cost += info["imbalance_cost"]
            done = terminated or truncated

        assert np.isfinite(total_revenue)
        assert np.isfinite(total_imbalance_cost)
        net_profit = total_revenue - total_imbalance_cost
        assert np.isfinite(net_profit)

    def test_commitment_fractions_vary_during_episode(self, env):
        """Commitment fractions should vary across the episode (price-driven)."""
        obs, _ = env.reset(seed=42)
        all_commitment_sets = []

        done = False
        while not done:
            action = price_aware_policy(obs)
            unique_this_step = set(action[0:24].tolist())
            all_commitment_sets.append(unique_this_step)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # At least some steps should have non-uniform commitments
        steps_with_variation = sum(1 for s in all_commitment_sets if len(s) > 1)
        assert steps_with_variation > 0, (
            "Commitment fractions were uniform every step — "
            "price-aware should produce variable commitments"
        )

    def test_battery_responds_during_episode(self, env):
        """Verify battery action varies in response to price conditions during episode."""
        obs, _ = env.reset(seed=42)
        battery_actions = []

        done = False
        while not done:
            action = price_aware_policy(obs)
            battery_actions.append(float(action[24]))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        # Battery should not be stuck at a single value the entire episode
        unique_actions = set(battery_actions)
        assert len(unique_actions) > 1, (
            f"Battery was stuck at {unique_actions} for entire episode — "
            "should respond to price conditions"
        )
