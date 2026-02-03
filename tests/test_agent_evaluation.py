"""Tests for RL agent evaluation script."""

from pathlib import Path

import numpy as np
import pytest


class MockModel:
    """Minimal mock that mimics SB3 model.predict()."""

    def predict(self, obs, deterministic=True):
        action = np.full(25, 0.5, dtype=np.float32)
        return action, None


class TestMakeAgentPolicy:
    """Tests for the make_agent_policy wrapper function."""

    def test_importable_and_callable(self):
        from src.evaluation.evaluate_agent import make_agent_policy

        assert callable(make_agent_policy)

    def test_returns_callable(self):
        from src.evaluation.evaluate_agent import make_agent_policy

        mock = MockModel()
        policy = make_agent_policy(mock)
        assert callable(policy)

    def test_policy_accepts_obs_returns_action(self):
        from src.evaluation.evaluate_agent import make_agent_policy

        mock = MockModel()
        policy = make_agent_policy(mock)
        obs = np.zeros(84, dtype=np.float32)
        action = policy(obs)
        assert isinstance(action, np.ndarray)
        assert action.shape == (25,)

    def test_policy_passes_deterministic_true(self):
        """Verify the wrapper calls predict with deterministic=True."""
        from src.evaluation.evaluate_agent import make_agent_policy

        class TrackingModel:
            def __init__(self):
                self.last_deterministic = None

            def predict(self, obs, deterministic=False):
                self.last_deterministic = deterministic
                return np.full(25, 0.5, dtype=np.float32), None

        tracking = TrackingModel()
        policy = make_agent_policy(tracking)
        policy(np.zeros(84, dtype=np.float32))
        assert tracking.last_deterministic is True


class TestMainImportable:
    """Tests for the main function."""

    def test_main_importable(self):
        from src.evaluation.evaluate_agent import main

        assert callable(main)


class TestEvaluationMetrics:
    """Tests for evaluation producing required metric keys."""

    @pytest.mark.skipif(
        not Path('data/processed/train.csv').exists(),
        reason="Training data not available",
    )
    def test_evaluate_produces_required_keys(self, env):
        from src.evaluation.evaluate import evaluate_policy
        from src.evaluation.evaluate_agent import make_agent_policy

        mock = MockModel()
        policy = make_agent_policy(mock)
        result = evaluate_policy(policy, env, n_episodes=1, seed=42)

        required_keys = [
            "revenue", "imbalance_cost", "net_profit", "degradation_cost",
            "total_reward", "delivered", "committed", "pv_produced",
            "delivery_ratio", "battery_cycles", "hours", "n_episodes",
        ]
        for key in required_keys:
            assert key in result, f"Missing required metric key: {key}"


class TestDeterminism:
    """Tests for evaluation determinism (AC #3)."""

    @pytest.mark.skipif(
        not Path('data/processed/train.csv').exists(),
        reason="Training data not available",
    )
    def test_deterministic_results(self, env):
        from src.evaluation.evaluate import evaluate_policy
        from src.evaluation.evaluate_agent import make_agent_policy

        mock = MockModel()
        policy = make_agent_policy(mock)
        r1 = evaluate_policy(policy, env, n_episodes=2, seed=42)
        r2 = evaluate_policy(policy, env, n_episodes=2, seed=42)
        for key in r1:
            assert r1[key] == r2[key], f"{key} differs: {r1[key]} vs {r2[key]}"


class TestCSVOutput:
    """Tests for CSV output (AC #4)."""

    @pytest.mark.skipif(
        not Path('data/processed/train.csv').exists(),
        reason="Training data not available",
    )
    def test_save_results_creates_csv(self, env, tmp_path):
        from src.evaluation.evaluate import evaluate_policy, save_results
        from src.evaluation.evaluate_agent import make_agent_policy

        mock = MockModel()
        policy = make_agent_policy(mock)
        result = evaluate_policy(policy, env, n_episodes=1, seed=42)
        result["policy"] = "RL Agent (SAC)"

        output_path = tmp_path / "agent_evaluation.csv"
        save_results([result], output_path)

        assert output_path.exists()
        import pandas as pd
        df = pd.read_csv(output_path)
        assert len(df) == 1
        assert "policy" in df.columns
        assert df["policy"].iloc[0] == "RL Agent (SAC)"
        assert "net_profit" in df.columns


class TestPathConstants:
    """Tests for script path constants and CLI defaults (AC #1, #4)."""

    def test_default_model_path(self):
        from src.evaluation.evaluate_agent import DEFAULT_MODEL

        assert isinstance(DEFAULT_MODEL, Path)
        assert DEFAULT_MODEL.name == 'solar_merchant_final.zip'
        assert 'models' in DEFAULT_MODEL.parts

    def test_results_path(self):
        from src.evaluation.evaluate_agent import RESULTS_PATH

        assert isinstance(RESULTS_PATH, Path)
        assert RESULTS_PATH.name == 'metrics'
        assert 'results' in RESULTS_PATH.parts

    def test_data_path(self):
        from src.evaluation.evaluate_agent import DATA_PATH

        assert isinstance(DATA_PATH, Path)
        assert DATA_PATH.name == 'processed'
        assert 'data' in DATA_PATH.parts
