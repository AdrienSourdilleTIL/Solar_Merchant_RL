"""
Pytest configuration and shared fixtures for Solar Merchant RL tests.

This module provides:
- Shared environment fixture for all test modules
- Warning filters for expected RuntimeWarnings during tests
"""

import pytest
from src.environment import load_environment


@pytest.fixture
def env():
    """Create environment instance for testing.

    This fixture is shared across all test modules to avoid code duplication.
    The environment uses the training dataset and is properly closed after tests.

    Yields:
        SolarMerchantEnv: Configured environment instance
    """
    data_path = 'data/processed/train.csv'
    env = load_environment(data_path)
    yield env
    env.close()


def pytest_configure(config):
    """Configure pytest with custom markers and warning filters."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# Suppress expected RuntimeWarnings that occur during normal test execution
# These warnings are informational and don't indicate test failures
def pytest_addoption(parser):
    """Add custom command line options."""
    pass


@pytest.fixture(autouse=True)
def suppress_expected_warnings():
    """Suppress expected RuntimeWarnings during tests.

    The cumulative imbalance warning is expected when running episodes
    with zero-action or random policies - this is normal behavior, not a bug.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Cumulative imbalance unusually large:.*",
            category=RuntimeWarning
        )
        warnings.filterwarnings(
            "ignore",
            message="Battery SOC outside expected range.*",
            category=RuntimeWarning
        )
        yield
