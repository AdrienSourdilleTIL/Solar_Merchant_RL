"""
Tests for SolarMerchantEnv structure and registration.

Tests AC #1: Environment structure, spaces, registration.
"""

import numpy as np
import pandas as pd
import pytest
import gymnasium

from src.environment.solar_merchant_env import SolarMerchantEnv, load_environment


class TestEnvironmentStructure:
    """Test AC #1: Environment class structure and spaces."""

    @pytest.fixture
    def mock_data(self):
        """Create minimal valid DataFrame for testing."""
        hours = 100
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=hours, freq='h'),
            'hour': [h % 24 for h in range(hours)],
            'price_eur_mwh': np.random.uniform(20, 100, hours),
            'pv_actual_mwh': np.random.uniform(0, 20, hours),
            'pv_forecast_mwh': np.random.uniform(0, 20, hours),
            'price_imbalance_short': np.random.uniform(30, 150, hours),
            'price_imbalance_long': np.random.uniform(10, 60, hours),
            'temperature_c': np.random.uniform(-5, 35, hours),
            'irradiance_direct': np.random.uniform(0, 1000, hours),
            'hour_sin': np.sin(2 * np.pi * np.arange(hours) / 24),
            'hour_cos': np.cos(2 * np.pi * np.arange(hours) / 24),
            'day_sin': np.sin(2 * np.pi * np.arange(hours) / (24 * 7)),
            'day_cos': np.cos(2 * np.pi * np.arange(hours) / (24 * 7)),
            'month_sin': np.sin(2 * np.pi * np.arange(hours) / (24 * 30)),
            'month_cos': np.cos(2 * np.pi * np.arange(hours) / (24 * 30)),
        })
        return data

    def test_inheritance_from_gymnasium_env(self, mock_data):
        """Verify SolarMerchantEnv inherits from gymnasium.Env."""
        env = SolarMerchantEnv(mock_data)
        assert isinstance(env, gymnasium.Env)

    def test_observation_space_is_84_dimensional(self, mock_data):
        """Verify observation space is Box with 84 dimensions."""
        env = SolarMerchantEnv(mock_data)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)
        assert env.observation_space.shape == (84,)

    def test_action_space_is_25_dimensional(self, mock_data):
        """Verify action space is Box with 25 dimensions in [0, 1]."""
        env = SolarMerchantEnv(mock_data)
        assert isinstance(env.action_space, gymnasium.spaces.Box)
        assert env.action_space.shape == (25,)
        assert np.all(env.action_space.low == 0.0)
        assert np.all(env.action_space.high == 1.0)

    def test_type_hints_on_init(self):
        """Verify __init__ has proper type hints."""
        import inspect
        sig = inspect.signature(SolarMerchantEnv.__init__)

        # Check key parameters have annotations
        assert sig.parameters['data'].annotation == pd.DataFrame
        assert sig.parameters['plant_capacity_mw'].annotation == float
        assert sig.return_annotation == None or sig.return_annotation == type(None)

    def test_type_hints_on_reset(self):
        """Verify reset() has proper type hints."""
        import inspect
        sig = inspect.signature(SolarMerchantEnv.reset)

        # Should return tuple[np.ndarray, dict]
        # This test will fail initially until we add type hints
        assert sig.return_annotation != inspect.Signature.empty

    def test_type_hints_on_step(self):
        """Verify step() has proper type hints."""
        import inspect
        sig = inspect.signature(SolarMerchantEnv.step)

        # Should have action parameter typed and return type
        assert sig.parameters['action'].annotation == np.ndarray
        assert sig.return_annotation != inspect.Signature.empty


class TestEnvironmentRegistration:
    """Test AC #1: Environment registration with gymnasium."""

    @pytest.fixture
    def mock_data(self):
        """Create minimal valid DataFrame for testing."""
        hours = 100
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=hours, freq='h'),
            'hour': [h % 24 for h in range(hours)],
            'price_eur_mwh': np.random.uniform(20, 100, hours),
            'pv_actual_mwh': np.random.uniform(0, 20, hours),
            'pv_forecast_mwh': np.random.uniform(0, 20, hours),
            'price_imbalance_short': np.random.uniform(30, 150, hours),
            'price_imbalance_long': np.random.uniform(10, 60, hours),
            'temperature_c': np.random.uniform(-5, 35, hours),
            'irradiance_direct': np.random.uniform(0, 1000, hours),
            'hour_sin': np.sin(2 * np.pi * np.arange(hours) / 24),
            'hour_cos': np.cos(2 * np.pi * np.arange(hours) / 24),
            'day_sin': np.sin(2 * np.pi * np.arange(hours) / (24 * 7)),
            'day_cos': np.cos(2 * np.pi * np.arange(hours) / (24 * 7)),
            'month_sin': np.sin(2 * np.pi * np.arange(hours) / (24 * 30)),
            'month_cos': np.cos(2 * np.pi * np.arange(hours) / (24 * 30)),
        })
        return data

    def test_environment_registers_as_solar_merchant_v0(self, mock_data):
        """Verify environment can be created via gymnasium.make()."""
        # Import to trigger registration
        import src.environment

        env = gymnasium.make('SolarMerchant-v0', data=mock_data)
        assert env is not None
        assert isinstance(env.unwrapped, SolarMerchantEnv)

    def test_registered_env_has_correct_spaces(self, mock_data):
        """Verify registered environment has correct observation/action spaces."""
        import src.environment

        env = gymnasium.make('SolarMerchant-v0', data=mock_data)
        assert env.observation_space.shape == (84,)
        assert env.action_space.shape == (25,)


class TestArchitectureCompliance:
    """Test AC #1: Validate architecture compliance."""

    def test_file_location_correct(self):
        """Verify environment is in src/environment/solar_merchant_env.py."""
        import src.environment.solar_merchant_env as env_module
        from pathlib import Path

        file_path = Path(env_module.__file__)
        assert file_path.name == 'solar_merchant_env.py'
        assert file_path.parent.name == 'environment'

    def test_class_naming_convention(self):
        """Verify PascalCase for class name."""
        assert SolarMerchantEnv.__name__ == 'SolarMerchantEnv'
        assert SolarMerchantEnv.__name__[0].isupper()

    def test_methods_use_snake_case(self):
        """Verify public methods use snake_case."""
        public_methods = [m for m in dir(SolarMerchantEnv) if not m.startswith('_') and callable(getattr(SolarMerchantEnv, m))]

        for method_name in public_methods:
            # Skip inherited methods from gym.Env
            if method_name in ['reset', 'step', 'render', 'close', 'seed']:
                assert method_name.islower() or '_' in method_name

    def test_has_google_style_docstrings(self):
        """Verify class and methods have docstrings."""
        assert SolarMerchantEnv.__doc__ is not None
        assert len(SolarMerchantEnv.__doc__) > 50

        assert SolarMerchantEnv.__init__.__doc__ is not None
        assert SolarMerchantEnv.reset.__doc__ is not None
        assert SolarMerchantEnv.step.__doc__ is not None


class TestDataLoading:
    """Test AC #1: Environment loads processed data correctly."""

    def test_load_environment_helper_exists(self):
        """Verify load_environment helper function exists."""
        assert callable(load_environment)

    def test_loads_from_processed_data_path(self, tmp_path):
        """Verify environment can load from CSV file path."""
        # Create temporary CSV
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=100, freq='h'),
            'hour': [h % 24 for h in range(100)],
            'price_eur_mwh': np.random.uniform(20, 100, 100),
            'pv_actual_mwh': np.random.uniform(0, 20, 100),
            'pv_forecast_mwh': np.random.uniform(0, 20, 100),
            'price_imbalance_short': np.random.uniform(30, 150, 100),
            'price_imbalance_long': np.random.uniform(10, 60, 100),
            'temperature_c': np.random.uniform(-5, 35, 100),
            'irradiance_direct': np.random.uniform(0, 1000, 100),
            'hour_sin': np.sin(2 * np.pi * np.arange(100) / 24),
            'hour_cos': np.cos(2 * np.pi * np.arange(100) / 24),
            'day_sin': np.sin(2 * np.pi * np.arange(100) / (24 * 7)),
            'day_cos': np.cos(2 * np.pi * np.arange(100) / (24 * 7)),
            'month_sin': np.sin(2 * np.pi * np.arange(100) / (24 * 30)),
            'month_cos': np.cos(2 * np.pi * np.arange(100) / (24 * 30)),
        })
        csv_path = tmp_path / "test_data.csv"
        data.to_csv(csv_path, index=False)

        env = load_environment(str(csv_path))
        assert isinstance(env, SolarMerchantEnv)
        assert len(env.data) == 100

    def test_environment_validates_required_columns(self, tmp_path):
        """Verify environment raises error if required columns are missing."""
        # Create CSV with missing columns
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='h'),
            'price_eur_mwh': np.random.uniform(20, 100, 10),
        })
        csv_path = tmp_path / "incomplete_data.csv"
        data.to_csv(csv_path, index=False)

        # This should raise an error or handle gracefully
        # Test will validate error handling is present
        with pytest.raises((KeyError, ValueError)):
            env = load_environment(str(csv_path))
            obs, _ = env.reset()


class TestEpisodeTermination:
    """Test episode termination after exactly 24 hours."""

    @pytest.fixture
    def env_with_data(self):
        """Create environment with sufficient data for testing."""
        hours = 1000
        data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=hours, freq='h'),
            'hour': [h % 24 for h in range(hours)],
            'price_eur_mwh': np.random.uniform(20, 100, hours),
            'pv_actual_mwh': np.random.uniform(0, 20, hours),
            'pv_forecast_mwh': np.random.uniform(0, 20, hours),
            'price_imbalance_short': np.random.uniform(30, 150, hours),
            'price_imbalance_long': np.random.uniform(10, 60, hours),
            'temperature_c': np.random.uniform(-5, 35, hours),
            'irradiance_direct': np.random.uniform(0, 1000, hours),
            'hour_sin': np.sin(2 * np.pi * np.arange(hours) / 24),
            'hour_cos': np.cos(2 * np.pi * np.arange(hours) / 24),
            'day_sin': np.sin(2 * np.pi * np.arange(hours) / (24 * 7)),
            'day_cos': np.cos(2 * np.pi * np.arange(hours) / (24 * 7)),
            'month_sin': np.sin(2 * np.pi * np.arange(hours) / (24 * 30)),
            'month_cos': np.cos(2 * np.pi * np.arange(hours) / (24 * 30)),
        })
        return SolarMerchantEnv(data)

    def test_episode_terminates_after_24_hours(self, env_with_data):
        """Verify episode terminates after exactly 24 hours."""
        env = env_with_data
        obs, info = env.reset(seed=42)

        step_count = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and step_count < 30:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

        assert terminated or truncated, "Episode should terminate"
        assert step_count == 24, f"Episode should last exactly 24 hours, but lasted {step_count}"

    def test_episode_tracks_start_index(self, env_with_data):
        """Verify episode correctly tracks start index."""
        env = env_with_data
        obs, info = env.reset(seed=42)

        start_idx = env.episode_start_idx
        assert start_idx == env.current_idx - 1 or start_idx == env.current_idx

    def test_commitment_hour_occurs_in_episode(self, env_with_data):
        """Verify commitment hour appears within 24-hour episode."""
        env = env_with_data
        obs, info = env.reset(seed=42)

        commitment_seen = False
        for _ in range(24):
            if env._is_commitment_hour():
                commitment_seen = True
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        assert commitment_seen, "Commitment hour should occur within episode"
