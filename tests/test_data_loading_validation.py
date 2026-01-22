"""
Tests for data loading validation (Task 4).

Tests that environment properly validates required DataFrame columns.
"""

import numpy as np
import pandas as pd
import pytest

from src.environment.solar_merchant_env import SolarMerchantEnv, load_environment


class TestDataValidation:
    """Test data loading validation."""

    def test_validates_required_columns_on_init(self):
        """Verify environment checks for required columns."""
        # Missing critical columns
        incomplete_data = pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=10, freq='h'),
            'price_eur_mwh': np.random.uniform(20, 100, 10),
        })

        with pytest.raises((KeyError, ValueError)) as exc_info:
            env = SolarMerchantEnv(incomplete_data)

        assert 'hour' in str(exc_info.value) or 'column' in str(exc_info.value).lower()

    def test_raises_informative_error_for_missing_file(self):
        """Verify clear error message when data file is missing."""
        with pytest.raises(FileNotFoundError) as exc_info:
            env = load_environment('data/does_not_exist.csv')

        assert 'does_not_exist.csv' in str(exc_info.value)

    def test_loads_actual_processed_train_data(self):
        """Verify environment loads actual train.csv successfully."""
        import os
        train_path = 'data/processed/train.csv'

        if not os.path.exists(train_path):
            pytest.skip(f"Train data not found at {train_path}")

        env = load_environment(train_path)
        assert len(env.data) > 1000  # Should have many hours
        obs, info = env.reset(seed=42)
        assert obs.shape == (84,)

    def test_loads_actual_processed_test_data(self):
        """Verify environment loads actual test.csv successfully."""
        import os
        test_path = 'data/processed/test.csv'

        if not os.path.exists(test_path):
            pytest.skip(f"Test data not found at {test_path}")

        env = load_environment(test_path)
        assert len(env.data) > 1000
        obs, info = env.reset(seed=42)
        assert obs.shape == (84,)
