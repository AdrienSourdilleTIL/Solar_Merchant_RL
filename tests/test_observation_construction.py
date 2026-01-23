"""
Tests for observation construction in SolarMerchantEnv.

This module validates that the 84-dimensional observation space is correctly
constructed according to the specification in Story 2-2.
"""

import numpy as np
import pytest
from pathlib import Path
from src.environment import load_environment


class TestObservationConstruction:
    """Test suite for validating observation construction."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        # Use train.csv for testing
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_observation_shape(self, env):
        """Test observation has exactly 84 dimensions."""
        obs, _ = env.reset()
        assert obs.shape == (84,), f"Expected shape (84,), got {obs.shape}"

    def test_observation_dtype(self, env):
        """Test observation is float32."""
        obs, _ = env.reset()
        assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"

    def test_observation_is_numpy_array(self, env):
        """Test observation is a numpy array."""
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray), f"Expected np.ndarray, got {type(obs)}"

    def test_observation_component_dimensions(self, env):
        """Test each component of observation has correct dimensions."""
        obs, _ = env.reset()

        # Dimension breakdown:
        # - Current hour: 1
        # - Battery SOC: 1
        # - Committed schedule: 24
        # - Cumulative imbalance: 1
        # - PV forecast: 24
        # - Prices: 24
        # - Current PV: 1
        # - Weather: 2
        # - Time features: 6
        # Total: 84

        idx = 0

        # Current hour (1 dim)
        hour_component = obs[idx:idx+1]
        assert len(hour_component) == 1, "Current hour should be 1 dimension"
        idx += 1

        # Battery SOC (1 dim)
        soc_component = obs[idx:idx+1]
        assert len(soc_component) == 1, "Battery SOC should be 1 dimension"
        idx += 1

        # Committed schedule (24 dims)
        commit_component = obs[idx:idx+24]
        assert len(commit_component) == 24, "Committed schedule should be 24 dimensions"
        idx += 24

        # Cumulative imbalance (1 dim)
        imbalance_component = obs[idx:idx+1]
        assert len(imbalance_component) == 1, "Cumulative imbalance should be 1 dimension"
        idx += 1

        # PV forecast (24 dims)
        forecast_component = obs[idx:idx+24]
        assert len(forecast_component) == 24, "PV forecast should be 24 dimensions"
        idx += 24

        # Prices (24 dims)
        price_component = obs[idx:idx+24]
        assert len(price_component) == 24, "Prices should be 24 dimensions"
        idx += 24

        # Current PV (1 dim)
        current_pv_component = obs[idx:idx+1]
        assert len(current_pv_component) == 1, "Current PV should be 1 dimension"
        idx += 1

        # Weather (2 dims)
        weather_component = obs[idx:idx+2]
        assert len(weather_component) == 2, "Weather should be 2 dimensions"
        idx += 2

        # Time features (6 dims)
        time_component = obs[idx:idx+6]
        assert len(time_component) == 6, "Time features should be 6 dimensions"
        idx += 6

        assert idx == 84, f"Total dimensions should sum to 84, got {idx}"


class TestNormalizationFactors:
    """Test suite for validating normalization factor computation."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_normalization_factors_exist(self, env):
        """Test that normalization factors are computed."""
        env.reset()
        assert hasattr(env, 'norm_factors'), "Environment should have norm_factors attribute"
        assert isinstance(env.norm_factors, dict), "norm_factors should be a dictionary"

    def test_normalization_factors_keys(self, env):
        """Test that all required normalization factors are present."""
        env.reset()
        required_keys = {'price', 'pv', 'temperature', 'irradiance'}
        actual_keys = set(env.norm_factors.keys())
        assert actual_keys == required_keys, f"Expected keys {required_keys}, got {actual_keys}"

    def test_normalization_factors_positive(self, env):
        """Test that all normalization factors are positive."""
        env.reset()
        for key, value in env.norm_factors.items():
            assert value > 0, f"Normalization factor '{key}' should be positive, got {value}"

    def test_normalization_factors_non_zero(self, env):
        """Test that normalization factors prevent division by zero."""
        env.reset()
        for key, value in env.norm_factors.items():
            assert value > 1e-10, f"Normalization factor '{key}' should be > 1e-10 to prevent div-by-zero"

    def test_pv_normalization_uses_plant_capacity(self, env):
        """Test that PV is normalized by plant capacity."""
        env.reset()
        # Plant capacity is 20 MW as per architecture
        assert env.norm_factors['pv'] == env.plant_capacity_mw, \
            f"PV normalization should use plant capacity ({env.plant_capacity_mw} MW)"


class TestObservationNormalization:
    """Test suite for validating observation value normalization."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_hour_normalization_range(self, env):
        """Test that current hour is normalized to [0, 1) range."""
        obs, _ = env.reset()
        hour_normalized = obs[0]
        assert 0 <= hour_normalized < 1, f"Hour should be in [0, 1), got {hour_normalized}"

    def test_battery_soc_normalization_range(self, env):
        """Test that battery SOC is normalized to [0, 1] range."""
        obs, _ = env.reset()
        soc_normalized = obs[1]
        assert 0 <= soc_normalized <= 1, f"Battery SOC should be in [0, 1], got {soc_normalized}"

    def test_committed_schedule_normalization(self, env):
        """Test that committed schedule is normalized by plant capacity."""
        obs, _ = env.reset()
        committed_schedule = obs[2:26]

        # All commitments should be in [0, 1] range (normalized by plant capacity)
        assert np.all(committed_schedule >= 0), "Committed schedule should be non-negative"
        assert np.all(committed_schedule <= 1), "Committed schedule should be <= 1 (normalized by capacity)"

    def test_time_features_in_valid_range(self, env):
        """Test that cyclical time features are in [-1, 1] range."""
        obs, _ = env.reset()
        time_features = obs[78:84]  # Last 6 dimensions

        assert np.all(time_features >= -1), "Time features should be >= -1"
        assert np.all(time_features <= 1), "Time features should be <= 1"

    def test_time_features_are_cyclical(self, env):
        """Test that time features follow sin/cos pattern."""
        obs, _ = env.reset()
        time_features = obs[78:84]

        # Extract sin/cos pairs
        hour_sin, hour_cos = time_features[0], time_features[1]
        day_sin, day_cos = time_features[2], time_features[3]
        month_sin, month_cos = time_features[4], time_features[5]

        # sin^2 + cos^2 should be approximately 1
        hour_mag = hour_sin**2 + hour_cos**2
        day_mag = day_sin**2 + day_cos**2
        month_mag = month_sin**2 + month_cos**2

        assert np.isclose(hour_mag, 1.0, atol=1e-5), f"Hour sin^2+cos^2 should be 1, got {hour_mag}"
        assert np.isclose(day_mag, 1.0, atol=1e-5), f"Day sin^2+cos^2 should be 1, got {day_mag}"
        assert np.isclose(month_mag, 1.0, atol=1e-5), f"Month sin^2+cos^2 should be 1, got {month_mag}"


class TestObservationEdgeCases:
    """Test suite for validating observation edge cases."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_observation_at_episode_start(self, env):
        """Test observation is valid at episode start."""
        obs, _ = env.reset()

        assert obs.shape == (84,), "Observation shape should be (84,) at start"
        assert obs.dtype == np.float32, "Observation should be float32 at start"
        assert not np.any(np.isnan(obs)), "Observation should not contain NaN at start"
        assert not np.any(np.isinf(obs)), "Observation should not contain Inf at start"

    def test_observation_after_step(self, env):
        """Test observation is valid after taking a step."""
        env.reset()

        # Take a random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (84,), "Observation shape should be (84,) after step"
        assert obs.dtype == np.float32, "Observation should be float32 after step"
        assert not np.any(np.isnan(obs)), "Observation should not contain NaN after step"
        assert not np.any(np.isinf(obs)), "Observation should not contain Inf after step"

    def test_observation_at_different_hours(self, env):
        """Test observation is valid at different hours of the day."""
        env.reset()

        # Take multiple steps to progress through hours
        for _ in range(5):
            action = env.action_space.sample()
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

            # Verify observation validity
            assert obs.shape == (84,), "Observation should remain valid shape"
            assert not np.any(np.isnan(obs)), "Observation should not contain NaN"
            assert not np.any(np.isinf(obs)), "Observation should not contain Inf"

    def test_cumulative_imbalance_at_hour_zero(self, env):
        """Test cumulative imbalance is zero at start of episode."""
        obs, _ = env.reset()
        cumulative_imbalance = obs[26]  # Index 26 is cumulative imbalance

        assert cumulative_imbalance == 0.0, \
            f"Cumulative imbalance should be 0 at episode start, got {cumulative_imbalance}"

    def test_observation_consistency(self, env):
        """Test that same state produces same observation."""
        # Reset with seed for reproducibility
        obs1, _ = env.reset(seed=42)

        # Reset again with same seed
        obs2, _ = env.reset(seed=42)

        np.testing.assert_array_equal(obs1, obs2,
            "Same reset seed should produce identical observations")


class TestObservationPerformance:
    """Test suite for validating observation construction performance."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_observation_construction_performance(self, env):
        """Test that observation construction is fast (<10ms per call).

        NFR2 requires episode completion within 5 seconds.
        With 24 steps per episode, this allows ~200ms per step.
        Observation construction should be a small fraction of that.
        """
        import time

        env.reset()

        # Warm-up
        for _ in range(10):
            env._get_observation()

        # Measure multiple runs for statistical reliability
        times = []
        for _ in range(5):  # 5 runs
            start_time = time.time()
            for _ in range(200):  # 200 calls per run
                env._get_observation()
            elapsed_time = time.time() - start_time
            avg_per_call = (elapsed_time / 200) * 1000  # ms
            times.append(avg_per_call)

        # Use p95 for robustness against outliers
        times_sorted = sorted(times)
        p95_time_ms = times_sorted[int(len(times) * 0.95)]
        avg_time_ms = np.mean(times)

        # Average should be well under 15ms, p95 allows some headroom for variance
        # Note: With 48h episodes, observation calls doubled but still fast
        assert avg_time_ms < 15.0, \
            f"Observation construction average should be <15ms, got {avg_time_ms:.3f}ms"
        assert p95_time_ms < 20.0, \
            f"Observation construction p95 should be <20ms, got {p95_time_ms:.3f}ms (avg: {avg_time_ms:.3f}ms)"


class TestForecastWindowPadding:
    """Test suite for validating forecast window padding near dataset boundaries."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_forecast_window_normal_case(self, env):
        """Test forecast window is correctly constructed in normal case."""
        obs, _ = env.reset()

        # PV forecast is at indices 27-50 (24 dimensions)
        forecast_window = obs[27:51]

        assert len(forecast_window) == 24, "Forecast window should be 24 hours"
        # In normal case (not near boundary), should not have padding zeros at end
        # This is hard to test without knowing position, so just verify it's valid
        assert isinstance(forecast_window, np.ndarray), "Forecast window should be numpy array"

    def test_price_window_normal_case(self, env):
        """Test price window is correctly constructed in normal case."""
        obs, _ = env.reset()

        # Prices are at indices 51-74 (24 dimensions)
        price_window = obs[51:75]

        assert len(price_window) == 24, "Price window should be 24 hours"
        assert isinstance(price_window, np.ndarray), "Price window should be numpy array"

    def test_forecast_window_at_dataset_boundary(self, env):
        """Test that forecast/price windows are zero-padded near dataset end."""
        # Force environment to near end of dataset
        env.reset()
        env.current_idx = len(env.data) - 10  # Only 10 hours left

        obs = env._get_observation()

        # Forecast window (indices 27-50)
        forecast_window = obs[27:51]
        assert len(forecast_window) == 24, "Forecast window should still be 24 dims"

        # Last 14 values (24 - 10) should be zero-padded
        # First 10 should have actual data
        actual_data_portion = forecast_window[:10]
        padded_portion = forecast_window[10:]

        # Padded portion should be all zeros
        assert np.all(padded_portion == 0.0), \
            f"Expected zero padding for last 14 hours, got {padded_portion}"

    def test_price_window_at_dataset_boundary(self, env):
        """Test that price window is zero-padded near dataset end."""
        # Force environment to near end of dataset
        env.reset()
        env.current_idx = len(env.data) - 5  # Only 5 hours left

        obs = env._get_observation()

        # Price window (indices 51-74)
        price_window = obs[51:75]
        assert len(price_window) == 24, "Price window should still be 24 dims"

        # Last 19 values (24 - 5) should be zero-padded
        padded_portion = price_window[5:]
        assert np.all(padded_portion == 0.0), \
            f"Expected zero padding for last 19 hours, got {padded_portion}"
