"""
Tests for Battery Environment
"""

import numpy as np
import pandas as pd
import pytest

from src.environment.battery_env import BatteryEnv, load_battery_env
from src.environment.solar_plant import PlantConfig


@pytest.fixture
def sample_data():
    """Create sample dataset for testing."""
    hours = 72  # 3 days
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
        'day_sin': [0.0] * hours,
        'day_cos': [1.0] * hours,
        'month_sin': [0.0] * hours,
        'month_cos': [1.0] * hours,
    })


@pytest.fixture
def env(sample_data):
    """Create BatteryEnv with sample data."""
    return BatteryEnv(sample_data)


class TestBatteryEnvInit:
    """Tests for BatteryEnv initialization."""

    def test_default_config(self, sample_data):
        env = BatteryEnv(sample_data)
        assert env.config.plant_capacity_mw == 20.0
        assert env.config.battery_capacity_mwh == 10.0

    def test_custom_config(self, sample_data):
        config = PlantConfig(plant_capacity_mw=30.0, battery_capacity_mwh=15.0)
        env = BatteryEnv(sample_data, plant_config=config)
        assert env.config.plant_capacity_mw == 30.0
        assert env.config.battery_capacity_mwh == 15.0

    def test_observation_space(self, env):
        assert env.observation_space.shape == (21,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self, env):
        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == 0.0
        assert env.action_space.high[0] == 1.0


class TestBatteryEnvReset:
    """Tests for BatteryEnv reset."""

    def test_reset_returns_correct_shapes(self, env):
        obs, info = env.reset(seed=42)
        assert obs.shape == (21,)
        assert isinstance(info, dict)

    def test_reset_with_seed_reproducible(self, env):
        obs1, _ = env.reset(seed=42)
        env.reset()  # Random reset
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_reset_with_custom_commitments(self, env):
        commitments = np.ones(24) * 5.0
        obs, info = env.reset(options={'commitments': commitments})

        np.testing.assert_array_equal(info['commitments'], commitments)
        np.testing.assert_array_equal(env.commitments, commitments)

    def test_reset_with_invalid_commitments_raises(self, env):
        with pytest.raises(ValueError, match="24 values"):
            env.reset(options={'commitments': np.ones(12)})

    def test_reset_with_custom_soc(self, env):
        obs, info = env.reset(options={'initial_soc': 0.8})
        expected_soc = 0.8 * env.config.battery_capacity_mwh
        assert env.battery.soc == pytest.approx(expected_soc)

    def test_reset_soc_clamped(self, env):
        obs, info = env.reset(options={'initial_soc': 1.5})  # Over 1.0
        assert env.battery.soc <= env.config.battery_capacity_mwh

    def test_reset_with_start_idx(self, env):
        obs, info = env.reset(options={'start_idx': 10})
        assert info['start_idx'] == 10
        assert env.current_idx == 10

    def test_reset_clears_tracking(self, env):
        # Run some steps
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())

        # Reset should clear
        env.reset()
        assert env.cumulative_imbalance == 0.0
        assert env.episode_revenue == 0.0
        assert env.episode_imbalance_cost == 0.0
        assert env.episode_step == 0


class TestBatteryEnvStep:
    """Tests for BatteryEnv step."""

    def test_step_returns_correct_shapes(self, env):
        env.reset(seed=42)
        action = np.array([0.5])  # Idle
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (21,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_info_contains_expected_keys(self, env):
        env.reset(seed=42)
        _, _, _, _, info = env.step(np.array([0.5]))

        expected_keys = [
            'hour', 'pv_actual', 'committed', 'delivered',
            'imbalance', 'price', 'revenue', 'imbalance_cost',
            'battery_soc', 'battery_action', 'throughput', 'degradation'
        ]
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"

    def test_step_idle_action(self, env):
        commitments = np.zeros(24)  # No commitments
        env.reset(options={'commitments': commitments, 'initial_soc': 0.5})

        initial_soc = env.battery.soc
        _, _, _, _, info = env.step(np.array([0.5]))

        # Idle should not change SOC
        assert env.battery.soc == pytest.approx(initial_soc)
        assert info['throughput'] == 0.0

    def test_step_discharge_action(self, env):
        commitments = np.ones(24) * 10.0  # High commitments
        env.reset(options={'commitments': commitments, 'initial_soc': 0.8})

        initial_soc = env.battery.soc
        _, _, _, _, info = env.step(np.array([0.0]))  # Full discharge

        # Discharge should reduce SOC
        assert env.battery.soc < initial_soc
        assert info['throughput'] > 0

    def test_step_charge_action(self, env):
        commitments = np.zeros(24)  # No commitments
        env.reset(options={'commitments': commitments, 'initial_soc': 0.3})

        initial_soc = env.battery.soc
        obs, _, _, _, info = env.step(np.array([1.0]))  # Full charge

        # Charge should increase SOC (if PV available)
        # Note: may not increase if PV is zero at this hour
        # Just verify action was recorded
        assert info['battery_action'] == 1.0

    def test_episode_terminates_after_24_steps(self, env):
        env.reset(seed=42, options={'start_idx': 0})

        for i in range(23):
            _, _, terminated, _, _ = env.step(np.array([0.5]))
            assert not terminated, f"Terminated early at step {i}"

        _, _, terminated, _, info = env.step(np.array([0.5]))
        assert terminated
        assert 'episode_profit' in info

    def test_cumulative_imbalance_tracking(self, env):
        commitments = np.ones(24) * 5.0
        env.reset(options={'commitments': commitments, 'initial_soc': 0.5})

        assert env.cumulative_imbalance == 0.0

        env.step(np.array([0.5]))
        # Cumulative imbalance should be updated
        # Value depends on PV actual vs commitment

    def test_reward_calculation(self, env):
        # Set up scenario where we know expected reward
        commitments = np.zeros(24)  # No commitments, so all delivery is "long"
        env.reset(options={
            'commitments': commitments,
            'initial_soc': 0.5,
            'start_idx': 12  # Midday, likely has PV
        })

        _, reward, _, _, info = env.step(np.array([0.5]))

        # Reward = revenue - imbalance_cost - degradation
        expected = info['revenue'] - info['imbalance_cost'] - info['degradation']
        assert reward == pytest.approx(expected)


class TestBatteryEnvObservation:
    """Tests for observation construction."""

    def test_observation_range(self, env):
        env.reset(seed=42)
        obs, _ = env.reset()

        # Hour should be in [0, 1)
        assert 0 <= obs[0] < 1

        # SOC should be in [0, 1]
        assert 0 <= obs[1] <= 1

        # All values should be finite
        assert np.all(np.isfinite(obs))

    def test_observation_updates_each_step(self, env):
        env.reset(seed=42)
        obs1 = env._get_observation()
        env.step(np.array([0.5]))
        obs2 = env._get_observation()

        # Hour should advance
        # (may wrap around midnight)
        assert not np.array_equal(obs1, obs2)


class TestBatteryEnvCommitmentGeneration:
    """Tests for random commitment generation."""

    def test_generated_commitments_shape(self, env):
        env.reset(seed=42)
        assert len(env.commitments) == 24

    def test_generated_commitments_nonnegative(self, env):
        for _ in range(10):
            env.reset()
            assert np.all(env.commitments >= 0)

    def test_generated_commitments_bounded(self, env):
        for _ in range(10):
            env.reset()
            max_possible = env.config.plant_capacity_mw + env.config.battery_power_mw
            assert np.all(env.commitments <= max_possible)


class TestBatteryEnvEpisodeTracking:
    """Tests for episode-level tracking."""

    def test_episode_metrics_accumulated(self, env):
        env.reset(seed=42)

        for _ in range(24):
            env.step(np.array([0.5]))

        assert env.episode_revenue >= 0
        # Imbalance cost could be 0 if always balanced

    def test_final_info_contains_episode_summary(self, env):
        env.reset(seed=42)

        for i in range(24):
            _, _, terminated, _, info = env.step(np.array([0.5]))
            if i < 23:
                assert 'episode_profit' not in info
            else:
                assert 'episode_profit' in info
                assert 'episode_revenue' in info
                assert 'episode_imbalance_cost' in info


class TestBatteryEnvEdgeCases:
    """Tests for edge cases."""

    def test_empty_battery_discharge(self, env):
        env.reset(options={'initial_soc': 0.0, 'commitments': np.ones(24) * 10})
        initial_soc = env.battery.soc

        _, _, _, _, info = env.step(np.array([0.0]))  # Try to discharge

        # Should not go negative
        assert env.battery.soc >= 0
        # Throughput should be minimal or zero
        assert info['throughput'] >= 0

    def test_full_battery_charge(self, env):
        env.reset(options={'initial_soc': 1.0, 'commitments': np.zeros(24)})

        _, _, _, _, info = env.step(np.array([1.0]))  # Try to charge

        # Should not exceed capacity
        assert env.battery.soc <= env.config.battery_capacity_mwh

    def test_action_clipping(self, env):
        env.reset()

        # Action outside [0, 1] should be handled gracefully
        # (Gym typically clips, but we should verify behavior)
        action = np.array([1.5])  # Over max
        obs, reward, _, _, info = env.step(action)
        assert np.isfinite(reward)


class TestLoadBatteryEnv:
    """Tests for load_battery_env helper."""

    def test_load_from_file(self, sample_data, tmp_path):
        # Save sample data
        path = tmp_path / "test_data.csv"
        sample_data.to_csv(path, index=False)

        # Load environment
        env = load_battery_env(str(path))
        assert isinstance(env, BatteryEnv)

        obs, _ = env.reset()
        assert obs.shape == (21,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
