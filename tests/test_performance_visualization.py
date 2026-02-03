"""Tests for performance visualization module.

Tests the visualization pipeline: data loading, chart generation,
and error handling. Uses synthetic CSV data (no real evaluation required).
"""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def synthetic_multi_seed_csv(tmp_path):
    """Create a synthetic multi_seed_evaluation.csv for testing."""
    df = pd.DataFrame([
        {"policy": "RL Agent (SAC)", "net_profit_mean": 5000.0, "net_profit_std": 200.0},
        {"policy": "Conservative (80%)", "net_profit_mean": 3895.0, "net_profit_std": 150.0},
        {"policy": "Aggressive (100%)", "net_profit_mean": -4132.0, "net_profit_std": 300.0},
        {"policy": "Price-Aware", "net_profit_mean": 2096.0, "net_profit_std": 180.0},
    ])
    csv_path = tmp_path / "multi_seed_evaluation.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def synthetic_plain_csv(tmp_path):
    """Create synthetic agent_vs_baselines.csv without std columns."""
    df = pd.DataFrame([
        {"policy": "RL Agent (SAC)", "net_profit": 5000.0},
        {"policy": "Conservative (80%)", "net_profit": 3895.0},
        {"policy": "Aggressive (100%)", "net_profit": -4132.0},
        {"policy": "Price-Aware", "net_profit": 2096.0},
    ])
    csv_path = tmp_path / "agent_vs_baselines.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


@pytest.fixture
def synthetic_individual_csvs(tmp_path):
    """Create synthetic individual agent + baseline CSVs for fallback loading."""
    agent_df = pd.DataFrame([
        {"policy": "RL Agent (SAC)", "net_profit": 5000.0},
    ])
    baseline_df = pd.DataFrame([
        {"policy": "Conservative (80%)", "net_profit": 3895.0},
        {"policy": "Aggressive (100%)", "net_profit": -4132.0},
        {"policy": "Price-Aware", "net_profit": 2096.0},
    ])
    agent_df.to_csv(tmp_path / "agent_evaluation.csv", index=False)
    baseline_df.to_csv(tmp_path / "baseline_comparison.csv", index=False)
    return tmp_path


class TestLoadComparisonData:
    """Tests for load_comparison_data function."""

    def test_loads_multi_seed_csv(self, synthetic_multi_seed_csv):
        from src.evaluation.visualize import load_comparison_data
        df = load_comparison_data(synthetic_multi_seed_csv)
        assert isinstance(df, pd.DataFrame)
        assert "policy" in df.columns
        assert "net_profit_mean" in df.columns
        assert "net_profit_std" in df.columns
        assert len(df) == 4

    def test_loads_agent_vs_baselines_csv(self, synthetic_plain_csv):
        from src.evaluation.visualize import load_comparison_data
        df = load_comparison_data(synthetic_plain_csv)
        assert isinstance(df, pd.DataFrame)
        assert "policy" in df.columns
        assert "net_profit" in df.columns
        assert len(df) == 4

    def test_loads_individual_csvs_fallback(self, synthetic_individual_csvs):
        from src.evaluation.visualize import load_comparison_data
        df = load_comparison_data(synthetic_individual_csvs)
        assert isinstance(df, pd.DataFrame)
        assert "policy" in df.columns
        assert len(df) == 4

    def test_raises_file_not_found_when_no_csvs(self, tmp_path):
        from src.evaluation.visualize import load_comparison_data
        with pytest.raises(FileNotFoundError, match="No evaluation CSV"):
            load_comparison_data(tmp_path)

    def test_prefers_multi_seed_over_plain(self, tmp_path):
        """When both multi_seed and agent_vs_baselines exist, prefer multi_seed."""
        from src.evaluation.visualize import load_comparison_data
        # Create both files
        multi_df = pd.DataFrame([
            {"policy": "RL Agent (SAC)", "net_profit_mean": 5000.0, "net_profit_std": 200.0},
        ])
        plain_df = pd.DataFrame([
            {"policy": "RL Agent (SAC)", "net_profit": 9999.0},
        ])
        multi_df.to_csv(tmp_path / "multi_seed_evaluation.csv", index=False)
        plain_df.to_csv(tmp_path / "agent_vs_baselines.csv", index=False)

        df = load_comparison_data(tmp_path)
        assert "net_profit_mean" in df.columns


class TestPlotNetProfitComparison:
    """Tests for plot_net_profit_comparison function."""

    def test_creates_png_file(self, synthetic_multi_seed_csv, tmp_path):
        from src.evaluation.visualize import load_comparison_data, plot_net_profit_comparison
        df = load_comparison_data(synthetic_multi_seed_csv)
        out = tmp_path / "figures" / "test.png"
        plot_net_profit_comparison(df, out)
        assert out.exists()

    def test_png_file_is_non_empty(self, synthetic_multi_seed_csv, tmp_path):
        from src.evaluation.visualize import load_comparison_data, plot_net_profit_comparison
        df = load_comparison_data(synthetic_multi_seed_csv)
        out = tmp_path / "figures" / "test.png"
        plot_net_profit_comparison(df, out)
        assert out.stat().st_size > 0

    def test_creates_output_directory(self, synthetic_multi_seed_csv, tmp_path):
        from src.evaluation.visualize import load_comparison_data, plot_net_profit_comparison
        df = load_comparison_data(synthetic_multi_seed_csv)
        out = tmp_path / "nested" / "dir" / "test.png"
        plot_net_profit_comparison(df, out)
        assert out.parent.exists()
        assert out.exists()

    def test_works_without_std_columns(self, synthetic_plain_csv, tmp_path):
        from src.evaluation.visualize import load_comparison_data, plot_net_profit_comparison
        df = load_comparison_data(synthetic_plain_csv)
        out = tmp_path / "test_no_std.png"
        plot_net_profit_comparison(df, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_works_with_std_columns(self, synthetic_multi_seed_csv, tmp_path):
        from src.evaluation.visualize import load_comparison_data, plot_net_profit_comparison
        df = load_comparison_data(synthetic_multi_seed_csv)
        out = tmp_path / "test_with_std.png"
        plot_net_profit_comparison(df, out)
        assert out.exists()
        assert out.stat().st_size > 0


class TestMain:
    """Tests for main function importability."""

    def test_main_is_importable(self):
        from src.evaluation.visualize import main
        assert callable(main)
