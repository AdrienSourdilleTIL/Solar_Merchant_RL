"""Tests for multi-seed statistical evaluation."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.evaluation.evaluate_multi_seed import (
    aggregate_results,
    print_multi_seed_table,
    run_multi_seed,
)


class MockModel:
    """Minimal mock that mimics SB3 model.predict()."""

    def predict(self, obs, deterministic=True):
        action = np.full(25, 0.5, dtype=np.float32)
        return action, None


class TestRunMultiSeed:
    """Tests for run_multi_seed function."""

    @pytest.mark.skipif(
        not Path('data/processed/train.csv').exists(),
        reason="Training data not available",
    )
    def test_returns_list_with_one_dict_per_seed(self, env):
        """run_multi_seed returns a list with one dict per seed."""
        from src.evaluation.evaluate_agent import make_agent_policy

        mock = MockModel()
        policy = make_agent_policy(mock)
        seeds = [42, 123]

        def env_factory():
            from src.environment import load_environment
            return load_environment('data/processed/train.csv')

        results = run_multi_seed(policy, env_factory, seeds, n_episodes=1)

        assert isinstance(results, list)
        assert len(results) == len(seeds)
        for r in results:
            assert isinstance(r, dict)
            assert "net_profit" in r
            assert "revenue" in r
            assert "n_episodes" in r


class TestAggregateResults:
    """Tests for aggregate_results function (unit tests, no env needed)."""

    def test_computes_correct_mean_and_std(self):
        """aggregate_results computes correct mean and sample std for known values."""
        seed_results = [
            {"net_profit": 100.0, "revenue": 200.0, "n_episodes": 10.0},
            {"net_profit": 200.0, "revenue": 300.0, "n_episodes": 10.0},
            {"net_profit": 300.0, "revenue": 400.0, "n_episodes": 10.0},
        ]
        agg = aggregate_results(seed_results)

        assert agg["net_profit_mean"] == pytest.approx(200.0)
        # Sample std (ddof=1): sqrt(((100-200)^2+(200-200)^2+(300-200)^2)/2) = 100.0
        assert agg["net_profit_std"] == pytest.approx(100.0)
        assert agg["revenue_mean"] == pytest.approx(300.0)
        assert agg["revenue_std"] == pytest.approx(100.0)

    def test_output_contains_mean_and_std_suffixed_keys(self):
        """aggregate_results output contains _mean and _std suffixed keys."""
        seed_results = [
            {"net_profit": 10.0, "revenue": 20.0, "imbalance_cost": 5.0},
            {"net_profit": 20.0, "revenue": 30.0, "imbalance_cost": 10.0},
        ]
        agg = aggregate_results(seed_results)

        for key in ["net_profit", "revenue", "imbalance_cost"]:
            assert f"{key}_mean" in agg, f"Missing {key}_mean"
            assert f"{key}_std" in agg, f"Missing {key}_std"

    def test_includes_n_seeds(self):
        """aggregate_results includes n_seeds count."""
        seed_results = [
            {"net_profit": 10.0},
            {"net_profit": 20.0},
            {"net_profit": 30.0},
        ]
        agg = aggregate_results(seed_results)
        assert agg["n_seeds"] == 3

    def test_single_seed_has_zero_std(self):
        """A single seed should produce zero std."""
        seed_results = [{"net_profit": 100.0, "revenue": 200.0}]
        agg = aggregate_results(seed_results)
        assert agg["net_profit_std"] == pytest.approx(0.0)
        assert agg["revenue_std"] == pytest.approx(0.0)

    def test_empty_input_raises_value_error(self):
        """aggregate_results raises ValueError on empty input."""
        with pytest.raises(ValueError, match="seed_results must not be empty"):
            aggregate_results([])


class TestPrintMultiSeedTable:
    """Tests for print_multi_seed_table function."""

    def test_callable_with_mock_data(self, capsys):
        """print_multi_seed_table is callable with mock data."""
        mock_results = [
            {
                "policy": "RL Agent (SAC)",
                "net_profit_mean": 5000.0,
                "net_profit_std": 500.0,
                "imbalance_cost_mean": 1000.0,
                "imbalance_cost_std": 100.0,
                "n_seeds": 5,
                "n_episodes_mean": 10.0,
            },
            {
                "policy": "Conservative (80%)",
                "net_profit_mean": 3000.0,
                "net_profit_std": 300.0,
                "imbalance_cost_mean": 800.0,
                "imbalance_cost_std": 80.0,
                "n_seeds": 5,
                "n_episodes_mean": 10.0,
            },
        ]
        print_multi_seed_table(mock_results)
        captured = capsys.readouterr()
        assert "MULTI-SEED STATISTICAL EVALUATION" in captured.out
        assert "RL Agent (SAC)" in captured.out
        assert "Conservative (80%)" in captured.out
        assert "Best policy:" in captured.out


class TestMainImportable:
    """Tests for the main function."""

    def test_main_importable(self):
        from src.evaluation.evaluate_multi_seed import main

        assert callable(main)


class TestDeterminism:
    """Tests for determinism: same seeds produce identical aggregates."""

    @pytest.mark.skipif(
        not Path('data/processed/train.csv').exists(),
        reason="Training data not available",
    )
    def test_same_seeds_produce_identical_aggregates(self, env):
        """Two runs with same seeds produce identical aggregates."""
        from src.evaluation.evaluate_agent import make_agent_policy

        mock = MockModel()
        policy = make_agent_policy(mock)
        seeds = [42, 123]

        def env_factory():
            from src.environment import load_environment
            return load_environment('data/processed/train.csv')

        results1 = run_multi_seed(policy, env_factory, seeds, n_episodes=1)
        agg1 = aggregate_results(results1)

        results2 = run_multi_seed(policy, env_factory, seeds, n_episodes=1)
        agg2 = aggregate_results(results2)

        for key in agg1:
            assert agg1[key] == agg2[key], f"{key} differs: {agg1[key]} vs {agg2[key]}"


class TestCSVOutput:
    """Tests for CSV output with all 4 policies."""

    @pytest.mark.skipif(
        not Path('data/processed/train.csv').exists(),
        reason="Training data not available",
    )
    def test_csv_output_contains_all_policies(self, env, tmp_path):
        """CSV output is created with all 4 policies."""
        import pandas as pd
        from src.evaluation.evaluate import save_results
        from src.evaluation.evaluate_agent import make_agent_policy
        from src.baselines import conservative_policy, aggressive_policy, price_aware_policy

        mock = MockModel()
        seeds = [42]

        def env_factory():
            from src.environment import load_environment
            return load_environment('data/processed/train.csv')

        policies = [
            ("RL Agent (SAC)", make_agent_policy(mock)),
            ("Conservative (80%)", conservative_policy),
            ("Aggressive (100%)", aggressive_policy),
            ("Price-Aware", price_aware_policy),
        ]

        all_aggregates = []
        for name, policy in policies:
            seed_results = run_multi_seed(
                policy, env_factory, seeds, n_episodes=1
            )
            agg = aggregate_results(seed_results)
            agg["policy"] = name
            all_aggregates.append(agg)

        output_path = tmp_path / "multi_seed_evaluation.csv"
        save_results(all_aggregates, output_path)

        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 4
        assert set(df["policy"]) == {
            "RL Agent (SAC)", "Conservative (80%)",
            "Aggressive (100%)", "Price-Aware"
        }
