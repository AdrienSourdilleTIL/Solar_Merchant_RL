"""Tests for RL agent vs baselines comparison script."""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


class TestModuleImportable:
    """Test that the comparison module is importable."""

    def test_importable(self):
        from src.evaluation import compare_agent_baselines

        assert hasattr(compare_agent_baselines, 'load_results')
        assert hasattr(compare_agent_baselines, 'calculate_improvement')
        assert hasattr(compare_agent_baselines, 'identify_best_baseline')
        assert hasattr(compare_agent_baselines, 'determine_verdict')
        assert hasattr(compare_agent_baselines, 'main')


class TestLoadResults:
    """Tests for load_results() function."""

    def test_loads_valid_csv(self, tmp_path):
        from src.evaluation.compare_agent_baselines import load_results

        csv_path = tmp_path / "test_results.csv"
        df = pd.DataFrame([
            {"policy": "TestPolicy", "net_profit": 1000.0, "revenue": 5000.0},
            {"policy": "OtherPolicy", "net_profit": 2000.0, "revenue": 6000.0},
        ])
        df.to_csv(csv_path, index=False)

        results = load_results(csv_path)

        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0]['policy'] == 'TestPolicy'
        assert results[0]['net_profit'] == 1000.0
        assert results[1]['policy'] == 'OtherPolicy'

    def test_raises_file_not_found(self, tmp_path):
        from src.evaluation.compare_agent_baselines import load_results

        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "nonexistent.csv")


class TestCalculateImprovement:
    """Tests for calculate_improvement() function."""

    def test_positive_improvement(self):
        from src.evaluation.compare_agent_baselines import calculate_improvement

        # agent=5000, baseline=4000 -> (5000-4000)/4000*100 = 25%
        result = calculate_improvement(5000.0, 4000.0)
        assert abs(result - 25.0) < 1e-8

    def test_negative_baseline(self):
        """Test handling of negative baselines (e.g., Aggressive at -4132)."""
        from src.evaluation.compare_agent_baselines import calculate_improvement

        # agent=5000, baseline=-4132 -> (5000-(-4132))/abs(-4132)*100
        # = 9132/4132*100 = 221.0%
        result = calculate_improvement(5000.0, -4132.0)
        expected = (5000.0 - (-4132.0)) / abs(-4132.0) * 100
        assert abs(result - expected) < 1e-8

    def test_zero_baseline(self):
        from src.evaluation.compare_agent_baselines import calculate_improvement

        result = calculate_improvement(5000.0, 0.0)
        assert result == float('inf')

    def test_zero_baseline_negative_agent(self):
        from src.evaluation.compare_agent_baselines import calculate_improvement

        result = calculate_improvement(-100.0, 0.0)
        assert result == 0.0

    def test_agent_worse_than_baseline(self):
        from src.evaluation.compare_agent_baselines import calculate_improvement

        # agent=3000, baseline=4000 -> (3000-4000)/4000*100 = -25%
        result = calculate_improvement(3000.0, 4000.0)
        assert abs(result - (-25.0)) < 1e-8


class TestIdentifyBestBaseline:
    """Tests for identify_best_baseline() function."""

    def test_identifies_highest_profit(self):
        from src.evaluation.compare_agent_baselines import identify_best_baseline

        baselines = [
            {"policy": "Conservative", "net_profit": 3895.0},
            {"policy": "Aggressive", "net_profit": -4132.0},
            {"policy": "Price-Aware", "net_profit": 2096.0},
        ]
        best = identify_best_baseline(baselines)
        assert best['policy'] == 'Conservative'
        assert best['net_profit'] == 3895.0


class TestDetermineVerdict:
    """Tests for determine_verdict() function."""

    def test_pass_when_agent_beats_all(self):
        from src.evaluation.compare_agent_baselines import determine_verdict

        baselines = [
            {"net_profit": 3895.0},
            {"net_profit": -4132.0},
            {"net_profit": 2096.0},
        ]
        verdict = determine_verdict(5000.0, baselines)
        assert verdict == "PASS"

    def test_fail_some_when_agent_loses_to_any(self):
        from src.evaluation.compare_agent_baselines import determine_verdict

        baselines = [
            {"net_profit": 3895.0},
            {"net_profit": -4132.0},
            {"net_profit": 2096.0},
        ]
        # Agent at 3000 beats Aggressive and Price-Aware, loses to Conservative
        verdict = determine_verdict(3000.0, baselines)
        assert verdict == "FAIL_SOME"

    def test_fail_none_when_agent_equals_baseline(self):
        from src.evaluation.compare_agent_baselines import determine_verdict

        baselines = [{"net_profit": 3895.0}]
        # Equal is not greater, so beats none
        verdict = determine_verdict(3895.0, baselines)
        assert verdict == "FAIL_NONE"

    def test_fail_none_when_agent_loses_to_all(self):
        from src.evaluation.compare_agent_baselines import determine_verdict

        baselines = [
            {"net_profit": 3895.0},
            {"net_profit": 5000.0},
            {"net_profit": 2096.0},
        ]
        # Agent at 1000 loses to all
        verdict = determine_verdict(1000.0, baselines)
        assert verdict == "FAIL_NONE"


class TestCSVOutput:
    """Tests for CSV output with all 4 policies and improvement columns."""

    def test_csv_output_created(self, tmp_path):
        from src.evaluation.compare_agent_baselines import (
            calculate_improvement,
            load_results,
        )
        from src.evaluation.evaluate import save_results

        # Create mock agent CSV
        agent_csv = tmp_path / "agent.csv"
        pd.DataFrame([{
            "policy": "RL Agent (SAC)",
            "net_profit": 5000.0,
            "revenue": 20000.0,
            "imbalance_cost": 15000.0,
            "degradation_cost": 0.3,
            "total_reward": 4999.7,
            "delivered": 170.0,
            "committed": 150.0,
            "pv_produced": 168.0,
            "delivery_ratio": 1.133,
            "battery_cycles": 1.5,
            "hours": 48.0,
            "n_episodes": 10.0,
        }]).to_csv(agent_csv, index=False)

        # Create mock baseline CSV
        baseline_csv = tmp_path / "baselines.csv"
        pd.DataFrame([
            {"policy": "Conservative (80%)", "net_profit": 3895.0,
             "revenue": 20510.0, "imbalance_cost": 16615.0,
             "degradation_cost": 0.2, "total_reward": 3894.8,
             "delivered": 172.0, "committed": 151.5,
             "pv_produced": 168.0, "delivery_ratio": 1.174,
             "battery_cycles": 1.0, "hours": 48.0, "n_episodes": 10.0},
            {"policy": "Aggressive (100%)", "net_profit": -4132.0,
             "revenue": 20836.0, "imbalance_cost": 24969.0,
             "degradation_cost": 0.5, "total_reward": -4132.5,
             "delivered": 171.0, "committed": 189.0,
             "pv_produced": 168.0, "delivery_ratio": 0.935,
             "battery_cycles": 2.3, "hours": 48.0, "n_episodes": 10.0},
            {"policy": "Price-Aware", "net_profit": 2096.0,
             "revenue": 20776.0, "imbalance_cost": 18679.0,
             "degradation_cost": 0.5, "total_reward": 2095.6,
             "delivered": 171.0, "committed": 148.0,
             "pv_produced": 168.0, "delivery_ratio": 1.161,
             "battery_cycles": 2.3, "hours": 48.0, "n_episodes": 10.0},
        ]).to_csv(baseline_csv, index=False)

        # Load and combine
        agent_results = load_results(agent_csv)
        baseline_results = load_results(baseline_csv)

        # Add improvement columns
        agent_profit = agent_results[0]['net_profit']
        for b in baseline_results:
            b['improvement_pct'] = calculate_improvement(agent_profit, b['net_profit'])
        agent_results[0]['improvement_pct'] = 0.0

        all_results = agent_results + baseline_results
        output_path = tmp_path / "agent_vs_baselines.csv"
        save_results(all_results, output_path)

        # Verify output
        assert output_path.exists()
        df = pd.read_csv(output_path)
        assert len(df) == 4
        policies = df['policy'].tolist()
        assert "RL Agent (SAC)" in policies
        assert "Conservative (80%)" in policies
        assert "Aggressive (100%)" in policies
        assert "Price-Aware" in policies
        assert "improvement_pct" in df.columns


def _make_mock_csvs(tmp_path, agent_profit=5000.0):
    """Helper to create mock agent and baseline CSVs for main() tests."""
    agent_csv = tmp_path / "agent.csv"
    pd.DataFrame([{
        "policy": "RL Agent (SAC)", "net_profit": agent_profit,
        "revenue": 20000.0, "imbalance_cost": 15000.0,
        "degradation_cost": 0.3, "total_reward": agent_profit - 0.3,
        "delivered": 170.0, "committed": 150.0, "pv_produced": 168.0,
        "delivery_ratio": 1.133, "battery_cycles": 1.5,
        "hours": 48.0, "n_episodes": 10.0,
    }]).to_csv(agent_csv, index=False)

    baseline_csv = tmp_path / "baselines.csv"
    pd.DataFrame([
        {"policy": "Conservative (80%)", "net_profit": 3895.0,
         "revenue": 20510.0, "imbalance_cost": 16615.0,
         "degradation_cost": 0.2, "total_reward": 3894.8,
         "delivered": 172.0, "committed": 151.5, "pv_produced": 168.0,
         "delivery_ratio": 1.174, "battery_cycles": 1.0,
         "hours": 48.0, "n_episodes": 10.0},
        {"policy": "Aggressive (100%)", "net_profit": -4132.0,
         "revenue": 20836.0, "imbalance_cost": 24969.0,
         "degradation_cost": 0.5, "total_reward": -4132.5,
         "delivered": 171.0, "committed": 189.0, "pv_produced": 168.0,
         "delivery_ratio": 0.935, "battery_cycles": 2.3,
         "hours": 48.0, "n_episodes": 10.0},
        {"policy": "Price-Aware", "net_profit": 2096.0,
         "revenue": 20776.0, "imbalance_cost": 18679.0,
         "degradation_cost": 0.5, "total_reward": 2095.6,
         "delivered": 171.0, "committed": 148.0, "pv_produced": 168.0,
         "delivery_ratio": 1.161, "battery_cycles": 2.3,
         "hours": 48.0, "n_episodes": 10.0},
    ]).to_csv(baseline_csv, index=False)

    return agent_csv, baseline_csv


class TestMain:
    """Integration tests for main() CLI function."""

    def test_main_pass_verdict(self, tmp_path, capsys):
        from src.evaluation.compare_agent_baselines import main

        agent_csv, baseline_csv = _make_mock_csvs(tmp_path, agent_profit=5000.0)
        output_csv = tmp_path / "agent_vs_baselines.csv"

        with patch('src.evaluation.compare_agent_baselines.RESULTS_PATH', tmp_path):
            with patch('sys.argv', ['prog',
                                    '--agent-results', str(agent_csv),
                                    '--baseline-results', str(baseline_csv)]):
                main()

        captured = capsys.readouterr()
        assert "VERDICT: PASS" in captured.out
        assert "beats all 3 baselines" in captured.out
        assert output_csv.exists()

    def test_main_fail_some_verdict(self, tmp_path, capsys):
        from src.evaluation.compare_agent_baselines import main

        # Agent at 3000 beats Aggressive (-4132) and Price-Aware (2096) but not Conservative (3895)
        agent_csv, baseline_csv = _make_mock_csvs(tmp_path, agent_profit=3000.0)

        with patch('src.evaluation.compare_agent_baselines.RESULTS_PATH', tmp_path):
            with patch('sys.argv', ['prog',
                                    '--agent-results', str(agent_csv),
                                    '--baseline-results', str(baseline_csv)]):
                main()

        captured = capsys.readouterr()
        assert "VERDICT: FAIL" in captured.out
        assert "beats 2 of 3 baselines" in captured.out

    def test_main_exits_on_missing_agent_file(self, tmp_path):
        from src.evaluation.compare_agent_baselines import main

        with patch('sys.argv', ['prog',
                                '--agent-results', str(tmp_path / "missing.csv"),
                                '--baseline-results', str(tmp_path / "other.csv")]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1

    def test_main_exits_on_missing_baseline_file(self, tmp_path):
        from src.evaluation.compare_agent_baselines import main

        agent_csv, _ = _make_mock_csvs(tmp_path)

        with patch('sys.argv', ['prog',
                                '--agent-results', str(agent_csv),
                                '--baseline-results', str(tmp_path / "missing.csv")]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 1
