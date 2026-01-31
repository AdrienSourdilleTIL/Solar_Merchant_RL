"""Tests for the evaluation module.

Tests evaluate_policy, print_comparison, and save_results functions
using shared env fixture from conftest.py.
"""

import numpy as np
import pytest

from src.baselines import aggressive_policy, conservative_policy, price_aware_policy
from src.evaluation import evaluate_policy
from src.evaluation.evaluate import print_comparison, save_results


class TestEvaluatePolicyImport:
    """Tests for evaluation module imports."""

    def test_import_evaluate_policy(self):
        from src.evaluation import evaluate_policy
        assert callable(evaluate_policy)

    def test_import_print_comparison(self):
        from src.evaluation.evaluate import print_comparison
        assert callable(print_comparison)

    def test_import_save_results(self):
        from src.evaluation.evaluate import save_results
        assert callable(save_results)


class TestEvaluatePolicyOutput:
    """Tests for evaluate_policy return value."""

    def test_returns_dict(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert isinstance(result, dict)

    def test_required_keys(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        required_keys = {
            "revenue", "imbalance_cost", "net_profit", "degradation_cost",
            "total_reward", "delivered", "committed", "pv_produced",
            "delivery_ratio", "battery_cycles", "hours", "n_episodes",
        }
        assert required_keys.issubset(result.keys())

    def test_metrics_finite(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        for key, value in result.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_revenue_non_negative(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert result["revenue"] >= 0

    def test_imbalance_cost_non_negative(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert result["imbalance_cost"] >= 0

    def test_net_profit_equals_revenue_minus_cost(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        expected = result["revenue"] - result["imbalance_cost"]
        np.testing.assert_allclose(result["net_profit"], expected, atol=1e-6)

    def test_hours_positive(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert result["hours"] > 0

    def test_degradation_cost_non_negative(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert result["degradation_cost"] >= 0

    def test_total_reward_consistent(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        expected = result["net_profit"] - result["degradation_cost"]
        np.testing.assert_allclose(result["total_reward"], expected, atol=1e-6)

    def test_n_episodes_in_result(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=3)
        assert result["n_episodes"] == 3.0


class TestEvaluatePolicyMultiEpisode:
    """Tests for multi-episode evaluation."""

    def test_multi_episode(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=3)
        assert isinstance(result, dict)
        assert result["hours"] > 0

    def test_multi_episode_consistent(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=2)
        # With deterministic seeds, results should be finite and reasonable
        assert np.isfinite(result["net_profit"])
        assert result["hours"] > 0

    def test_single_episode(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert isinstance(result, dict)
        assert len(result) >= 9


class TestAllBaselines:
    """Tests for evaluating all three baseline policies."""

    def test_conservative(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=2)
        assert result["net_profit"] != 0

    def test_aggressive(self, env):
        result = evaluate_policy(aggressive_policy, env, n_episodes=2)
        assert result["net_profit"] != 0

    def test_price_aware(self, env):
        result = evaluate_policy(price_aware_policy, env, n_episodes=2)
        assert result["net_profit"] != 0


class TestPrintComparison:
    """Tests for print_comparison function."""

    def test_runs_without_error(self, env, capsys):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        result["policy"] = "Test Policy"
        print_comparison([result])
        captured = capsys.readouterr()
        assert "BASELINE POLICY COMPARISON" in captured.out
        assert "Test Policy" in captured.out

    def test_multiple_policies(self, env, capsys):
        results = []
        for name, policy in [("A", conservative_policy), ("B", aggressive_policy)]:
            r = evaluate_policy(policy, env, n_episodes=1)
            r["policy"] = name
            results.append(r)
        print_comparison(results)
        captured = capsys.readouterr()
        assert "Best policy:" in captured.out


class TestSaveResults:
    """Tests for save_results function."""

    def test_creates_csv(self, env, tmp_path):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        result["policy"] = "Test"
        output = tmp_path / "results.csv"
        save_results([result], output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_csv_content(self, env, tmp_path):
        import pandas as pd
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        result["policy"] = "Test"
        output = tmp_path / "subdir" / "results.csv"
        save_results([result], output)
        df = pd.read_csv(output)
        assert "policy" in df.columns
        assert "net_profit" in df.columns
        assert len(df) == 1

    def test_csv_column_order(self, env, tmp_path):
        import pandas as pd
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        result["policy"] = "Test"
        output = tmp_path / "results.csv"
        save_results([result], output)
        df = pd.read_csv(output)
        assert df.columns[0] == "policy"
        assert df.columns[1] == "net_profit"


class TestInputValidation:
    """Tests for input validation."""

    def test_n_episodes_zero_raises(self, env):
        with pytest.raises(ValueError, match="n_episodes must be >= 1"):
            evaluate_policy(conservative_policy, env, n_episodes=0)

    def test_n_episodes_negative_raises(self, env):
        with pytest.raises(ValueError, match="n_episodes must be >= 1"):
            evaluate_policy(conservative_policy, env, n_episodes=-1)


class TestEvaluateBaselinesScript:
    """Tests for evaluate_baselines.py script integration."""

    def test_script_importable(self):
        from src.evaluation import evaluate_baselines
        assert hasattr(evaluate_baselines, 'main')

    def test_evaluation_flow_with_all_baselines(self, env, tmp_path):
        """Integration test: evaluate all 3 baselines and produce comparison + CSV."""
        policies = [
            ("Conservative (80%)", conservative_policy),
            ("Aggressive (100%)", aggressive_policy),
            ("Price-Aware", price_aware_policy),
        ]
        results = []
        for name, policy in policies:
            result = evaluate_policy(policy, env, n_episodes=1)
            result["policy"] = name
            results.append(result)

        print_comparison(results)
        output = tmp_path / "baseline_comparison.csv"
        save_results(results, output)
        assert output.exists()

        import pandas as pd
        df = pd.read_csv(output)
        assert len(df) == 3
        assert set(df["policy"]) == {"Conservative (80%)", "Aggressive (100%)", "Price-Aware"}
