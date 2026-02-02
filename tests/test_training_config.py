"""Tests for SAC training configuration and seed management."""

import random
from pathlib import Path

import numpy as np
import pytest
import torch

TRAIN_DATA_PATH = Path('data/processed/train.csv')


class TestSeedManagement:
    """Tests for set_all_seeds reproducibility."""

    def test_numpy_deterministic(self):
        from src.training.train import set_all_seeds

        set_all_seeds(42)
        a = np.random.rand(5)
        set_all_seeds(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_torch_deterministic(self):
        from src.training.train import set_all_seeds

        set_all_seeds(42)
        a = torch.rand(5)
        set_all_seeds(42)
        b = torch.rand(5)
        assert torch.equal(a, b)

    def test_random_deterministic(self):
        from src.training.train import set_all_seeds

        set_all_seeds(42)
        a = [random.random() for _ in range(5)]
        set_all_seeds(42)
        b = [random.random() for _ in range(5)]
        assert a == b


class TestHyperparameterConstants:
    """Tests for training configuration constants."""

    def test_seed_is_int(self):
        from src.training.train import SEED

        assert isinstance(SEED, int)

    def test_learning_rate_type(self):
        from src.training.train import LEARNING_RATE

        assert isinstance(LEARNING_RATE, float)
        assert 0 < LEARNING_RATE < 1

    def test_batch_size_type(self):
        from src.training.train import BATCH_SIZE

        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0

    def test_buffer_size_type(self):
        from src.training.train import BUFFER_SIZE

        assert isinstance(BUFFER_SIZE, int)
        assert BUFFER_SIZE > 0

    def test_gamma_range(self):
        from src.training.train import GAMMA

        assert 0 < GAMMA <= 1

    def test_tau_range(self):
        from src.training.train import TAU

        assert isinstance(TAU, float)
        assert 0 < TAU < 1

    def test_train_freq_type(self):
        from src.training.train import TRAIN_FREQ

        assert isinstance(TRAIN_FREQ, int)
        assert TRAIN_FREQ > 0

    def test_gradient_steps_type(self):
        from src.training.train import GRADIENT_STEPS

        assert isinstance(GRADIENT_STEPS, int)
        assert GRADIENT_STEPS > 0

    def test_net_arch_is_list(self):
        from src.training.train import NET_ARCH

        assert isinstance(NET_ARCH, list)
        assert all(isinstance(x, int) for x in NET_ARCH)

    def test_plant_config_keys(self):
        from src.training.train import PLANT_CONFIG

        required = {
            'plant_capacity_mw',
            'battery_capacity_mwh',
            'battery_power_mw',
            'battery_efficiency',
            'battery_degradation_cost',
        }
        assert required.issubset(PLANT_CONFIG.keys())


class TestTrainingLoopConstants:
    """Tests for training loop configuration constants."""

    def test_checkpoint_freq_value(self):
        from src.training.train import CHECKPOINT_FREQ

        assert isinstance(CHECKPOINT_FREQ, int)
        assert CHECKPOINT_FREQ == 50_000

    def test_eval_freq_type(self):
        from src.training.train import EVAL_FREQ

        assert isinstance(EVAL_FREQ, int)
        assert EVAL_FREQ > 0

    def test_n_eval_episodes_type(self):
        from src.training.train import N_EVAL_EPISODES

        assert isinstance(N_EVAL_EPISODES, int)
        assert N_EVAL_EPISODES > 0

    def test_total_timesteps_value(self):
        from src.training.train import TOTAL_TIMESTEPS

        assert isinstance(TOTAL_TIMESTEPS, int)
        assert TOTAL_TIMESTEPS == 500_000

    def test_checkpoint_freq_divides_total(self):
        from src.training.train import CHECKPOINT_FREQ, TOTAL_TIMESTEPS

        assert TOTAL_TIMESTEPS % CHECKPOINT_FREQ == 0

    def test_eval_freq_divides_total(self):
        from src.training.train import EVAL_FREQ, TOTAL_TIMESTEPS

        assert TOTAL_TIMESTEPS % EVAL_FREQ == 0

    def test_time_module_available(self):
        """Verify time module is imported at module level for elapsed reporting."""
        import src.training.train as train_module

        assert hasattr(train_module, 'time')

    def test_checkpoint_and_eval_directories_defined(self):
        """Verify MODEL_PATH is defined for checkpoint/best subdirectory creation."""
        from src.training.train import MODEL_PATH

        assert MODEL_PATH is not None
        assert isinstance(MODEL_PATH, Path)


class TestCreateEnv:
    """Tests for environment creation utility."""

    @pytest.mark.skipif(
        not Path('data/processed/train.csv').exists(),
        reason="Training data not available",
    )
    def test_create_env_returns_env(self):
        from src.training.train import PLANT_CONFIG, create_env

        env = create_env(TRAIN_DATA_PATH, **PLANT_CONFIG)
        assert type(env).__name__ == 'SolarMerchantEnv'
        env.close()
