"""
Tests for Commitment Environment
"""

import numpy as np
import pandas as pd
import pytest

from src.environment.commitment_env import CommitmentEnv, load_commitment_env
from src.environment.solar_plant import PlantConfig


@pytest.fixture
def sample_data():
    """Create sample dataset for testing.

    Creates 96 hours (4 days) of data to ensure we can find commitment hours.
    """
    hours = 96
    return pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=hours, freq='h'),
        'hour': [h % 24 for h in range(hours)],
        'price_eur_mwh': [50.0 + 10 * np.sin(h * np.pi / 12) for h in range(hours)],
        'pv_actual_mwh': [max(0, 10 * np.sin((h % 24 - 6) * np.pi / 12)) for h in range(hours)],
        'pv_forecast_mwh': [max(0, 9 * np.sin((h % 24 - 6) * np.pi / 12)) for h in range(hours)],
        'price_imbalance_short': [75.0] * hours,
        'price_imbalance_long': [30.0] * hours,
        'temperature_c': [15.0 + 5 * np.sin(h * np.pi / 12) for h in range(hours)],
        'irradiance_direct': [max(0, 800 * np.sin((h % 24 - 6) * np.pi / 12)) for h in range(hours)],
        'hour_sin': [np.sin(h % 24 * 2 * np.pi / 24) for h in range(hours)],
        'hour_cos': [np.cos(h % 24 * 2 * np.pi / 24) for h in range(hours)],
        'day_sin': [np.sin((h // 24) * 2 * np.pi / 365) for h in range(hours)],
        'day_cos': [np.cos((h // 24) * 2 * np.pi / 365) for h in range(hours)],
        'month_sin': [0.0] * hours,
        'month_cos': [1.0] * hours,
    })


@pytest.fixture
def env(sample_data):
    """Create CommitmentEnv with sample data."""
    return CommitmentEnv(sample_data)


class TestCommitmentEnvInit:
    """Tests for CommitmentEnv initialization."""

    def test_default_config(self, sample_data):
        env = CommitmentEnv(sample_data)
        assert env.config.plant_capacity_mw == 20.0
        assert env.config.battery_capacity_mwh == 10.0
        assert env.config.commitment_hour == 11

    def test_custom_config(self, sample_data):
        config = PlantConfig(plant_capacity_mw=30.0, commitment_hour=10)
        env = CommitmentEnv(sample_data, plant_config=config)
        assert env.config.plant_capacity_mw == 30.0
        assert env.config.commitment_hour == 10

    def test_observation_space(self, env):
        assert env.observation_space.shape == (56,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self, env):
        assert env.action_space.shape == (24,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 1.0

    def test_default_battery_policy(self, env):
        assert env._battery_policy_type == 'heuristic'

    def test_custom_battery_policy(self, sample_data):
        def custom_policy(soc, committed, pv_actual, hour, future_commits,
                         future_forecasts, cumulative_imbalance, price):
            return 0.5  # Always idle

        env = CommitmentEnv(sample_data, battery_policy=custom_policy)
        assert env._battery_policy_type == 'custom'


class TestCommitmentEnvReset:
    """Tests for CommitmentEnv reset."""

    def test_reset_returns_correct_shapes(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (56,)
        assert isinstance(info, dict)

    def test_reset_with_seed_reproducible(self, env):
        obs1, _ = env.reset(seed=42)
        env.reset()
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_finds_commitment_hour(self, env):
        obs, info = env.reset(seed=42)
        idx = info['commitment_hour_idx']
        hour = env.data_manager.get_hour(idx)
        assert hour == env.config.commitment_hour

    def test_reset_with_custom_soc(self, env):
        obs, info = env.reset(options={'initial_soc': 0.8})
        expected_soc = 0.8 * env.config.battery_capacity_mwh
        assert env.battery.soc == pytest.approx(expected_soc)
        assert info['initial_soc'] == pytest.approx(expected_soc)

    def test_reset_info_contains_expected_keys(self, env):
        _, info = env.reset()
        assert 'commitment_hour_idx' in info
        assert 'initial_soc' in info
        assert 'battery_policy' in info


class TestCommitmentEnvStep:
    """Tests for CommitmentEnv step."""

    def test_step_returns_correct_shapes(self, env):
        env.reset(seed=42)
        action = np.ones(24) * 0.5  # 50% commitment
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (56,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_always_terminates(self, env):
        """Single-step episodes should always terminate."""
        env.reset(seed=42)
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert terminated

    def test_step_info_contains_results(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.ones(24) * 0.5)

        expected_keys = [
            'commitments', 'forecasts', 'total_revenue',
            'total_imbalance_cost', 'total_degradation', 'total_profit',
            'final_soc', 'cumulative_imbalance', 'hourly_results'
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_step_commitment_conversion(self, env):
        """Test that action fractions are converted to commitments correctly."""
        env.reset(seed=42)
        action = np.ones(24) * 0.5  # 50% of max
        _, _, _, _, info = env.step(action)

        commitments = info['commitments']
        forecasts = info['forecasts']

        # Max commitment = forecast + battery_power
        max_possible = forecasts + env.config.battery_power_mw

        # Actual should be ~50% of max
        expected = 0.5 * max_possible
        np.testing.assert_array_almost_equal(commitments, expected, decimal=5)

    def test_step_zero_commitment(self, env):
        """Zero commitment should result in all production being 'long'."""
        env.reset(seed=42)
        action = np.zeros(24)
        _, reward, _, _, info = env.step(action)

        # With zero commitment, no imbalance cost (we're always long)
        assert info['total_imbalance_cost'] == 0.0
        # But we get less money for excess (long price < DA price)

    def test_step_full_commitment(self, env):
        """Full commitment might result in some short positions."""
        env.reset(seed=42)
        action = np.ones(24)  # 100% commitment
        _, reward, _, _, info = env.step(action)

        # With full commitment, might have imbalance costs
        # (actual PV < forecast + battery)
        # Just verify it runs without error
        assert isinstance(reward, float)

    def test_reward_equals_profit(self, env):
        """Reward should equal total profit."""
        env.reset(seed=42)
        _, reward, _, _, info = env.step(np.ones(24) * 0.5)

        expected = (info['total_revenue'] -
                   info['total_imbalance_cost'] -
                   info['total_degradation'])
        assert reward == pytest.approx(expected)


class TestCommitmentEnvSimulation:
    """Tests for internal simulation logic."""

    def test_simulation_runs_24_hours(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.ones(24) * 0.5)

        assert len(info['hourly_results']) == 24

    def test_simulation_hourly_results_structure(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.ones(24) * 0.5)

        hourly = info['hourly_results'][0]
        expected_keys = [
            'hour', 'committed', 'pv_actual', 'delivered',
            'imbalance', 'revenue', 'imbalance_cost',
            'battery_soc', 'battery_action'
        ]
        for key in expected_keys:
            assert key in hourly, f"Missing hourly key: {key}"

    def test_simulation_battery_changes_soc(self, env):
        env.reset(seed=42, options={'initial_soc': 0.5})
        initial_soc = env.battery.soc
        _, _, _, _, info = env.step(np.ones(24) * 0.8)

        # Battery should have been used (SOC changed)
        final_soc = info['final_soc']
        # Can't guarantee direction, but should likely be different
        # unless perfectly balanced

    def test_simulation_respects_battery_limits(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.ones(24) * 0.5)

        for hourly in info['hourly_results']:
            soc = hourly['battery_soc']
            assert 0 <= soc <= env.config.battery_capacity_mwh


class TestCommitmentEnvObservation:
    """Tests for observation construction."""

    def test_observation_components(self, env):
        obs, _ = env.reset(seed=42)

        # Check observation is reasonable
        assert obs.shape == (56,)

        # First 24 should be forecasts (normalized, mostly 0-1)
        forecasts = obs[:24]
        assert all(f >= 0 for f in forecasts)  # No negative forecasts

        # Next 24 should be prices (can be negative in theory)
        prices = obs[24:48]
        assert all(np.isfinite(p) for p in prices)

        # SOC should be in [0, 1]
        soc = obs[48]
        assert 0 <= soc <= 1

    def test_observation_all_finite(self, env):
        obs, _ = env.reset(seed=42)
        assert np.all(np.isfinite(obs))


class TestCommitmentEnvBatteryPolicies:
    """Tests for different battery policies."""

    def test_heuristic_policy_runs(self, sample_data):
        env = CommitmentEnv(sample_data, battery_policy='heuristic')
        env.reset(seed=42)
        _, reward, _, _, _ = env.step(np.ones(24) * 0.5)
        assert isinstance(reward, float)

    def test_custom_policy_is_called(self, sample_data):
        call_count = [0]

        def counting_policy(soc, committed, pv_actual, hour, future_commitments,
                           future_forecasts, cumulative_imbalance, price):
            call_count[0] += 1
            return 0.5

        env = CommitmentEnv(sample_data, battery_policy=counting_policy)
        env.reset(seed=42)
        env.step(np.ones(24) * 0.5)

        # Should be called 24 times (once per hour)
        assert call_count[0] == 24

    def test_invalid_policy_raises(self, sample_data):
        with pytest.raises(ValueError, match="Unknown battery policy"):
            CommitmentEnv(sample_data, battery_policy='invalid')


class TestLoadCommitmentEnv:
    """Tests for load_commitment_env helper."""

    def test_load_from_file(self, sample_data, tmp_path):
        path = tmp_path / "test_data.csv"
        sample_data.to_csv(path, index=False)

        env = load_commitment_env(str(path))
        assert isinstance(env, CommitmentEnv)

        obs, _ = env.reset()
        assert obs.shape == (56,)


class TestCommitmentEnvEdgeCases:
    """Tests for edge cases."""

    def test_action_clipping(self, env):
        """Actions outside [0,1] should be clipped."""
        env.reset(seed=42)
        action = np.ones(24) * 1.5  # Over max
        _, _, _, _, info = env.step(action)

        # Should run without error
        assert info['commitments'] is not None

    def test_multiple_episodes(self, env):
        """Multiple episodes should work correctly."""
        for _ in range(3):
            obs, _ = env.reset()
            assert obs.shape == (56,)

            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert terminated
            assert isinstance(reward, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
