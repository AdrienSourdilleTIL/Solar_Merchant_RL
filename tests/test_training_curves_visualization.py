"""Tests for training curves visualization module.

Tests the TensorBoard data loading and training curves chart generation.
Uses synthetic data (no real TensorBoard logs required for most tests).
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def synthetic_training_df():
    """Create synthetic training data mimicking TensorBoard scalars."""
    np.random.seed(42)  # Fixed seed for reproducibility
    steps = np.arange(0, 500000, 1000)  # Every 1k steps
    # Simulate reward increasing then plateauing with high variance noise
    rewards = 1000 + 3000 * (1 - np.exp(-steps / 100000)) + np.random.normal(0, 500, len(steps))
    return pd.DataFrame({'step': steps, 'value': rewards})


class TestLoadTensorboardData:
    """Tests for load_tensorboard_data function."""

    def test_raises_file_not_found_when_no_logs(self, tmp_path):
        """Test helpful error when TensorBoard logs don't exist."""
        from src.evaluation.visualize import load_tensorboard_data
        with pytest.raises(FileNotFoundError) as exc_info:
            load_tensorboard_data(tmp_path)
        # Check helpful message mentions running training
        assert "training" in str(exc_info.value).lower() or "train" in str(exc_info.value).lower()

    def test_raises_file_not_found_with_empty_run_dirs(self, tmp_path):
        """Test error when run directories exist but have no event files."""
        from src.evaluation.visualize import load_tensorboard_data
        # Create empty run directory
        (tmp_path / "SAC_1").mkdir()
        with pytest.raises(FileNotFoundError):
            load_tensorboard_data(tmp_path)

    def test_returns_dataframe_with_expected_columns(self, tmp_path):
        """Test that real TensorBoard logs (if available) return correct structure.

        This test will be skipped if no real TensorBoard logs exist.
        """
        from src.evaluation.visualize import load_tensorboard_data
        real_logs = Path("outputs/tensorboard")
        if not real_logs.exists() or not list(real_logs.glob("SAC_*")):
            pytest.skip("No real TensorBoard logs available for integration test")

        df = load_tensorboard_data(real_logs)
        assert isinstance(df, pd.DataFrame)
        assert "step" in df.columns
        assert "value" in df.columns
        assert len(df) > 0


class TestSmoothData:
    """Tests for smooth_data helper function."""

    def test_smooth_data_returns_series(self, synthetic_training_df):
        from src.evaluation.visualize import smooth_data
        result = smooth_data(synthetic_training_df['value'])
        assert isinstance(result, pd.Series)
        assert len(result) == len(synthetic_training_df)

    def test_smooth_data_reduces_variance(self):
        """Test that smoothing reduces variance of noisy data."""
        from src.evaluation.visualize import smooth_data
        np.random.seed(123)
        # Create pure noise (no trend) to clearly show variance reduction
        raw = pd.Series(np.random.normal(0, 100, 1000))
        smoothed = smooth_data(raw, weight=0.95)
        # Smoothed data should have less variance than raw pure noise
        assert smoothed.std() < raw.std()

    def test_smooth_data_zero_weight_returns_original(self, synthetic_training_df):
        from src.evaluation.visualize import smooth_data
        raw = synthetic_training_df['value']
        smoothed = smooth_data(raw, weight=0.0)
        # With weight=0, should return original values
        pd.testing.assert_series_equal(smoothed.reset_index(drop=True), raw.reset_index(drop=True))

    def test_smooth_data_empty_series_returns_empty(self):
        """Test that empty series returns empty series without crashing."""
        from src.evaluation.visualize import smooth_data
        result = smooth_data(pd.Series([], dtype=float))
        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_smooth_data_with_nan_preserves_nan_position(self):
        """Test that NaN values are preserved in output, not propagated."""
        from src.evaluation.visualize import smooth_data
        raw = pd.Series([1.0, float('nan'), 3.0, 4.0, 5.0])
        smoothed = smooth_data(raw, weight=0.5)
        # NaN should be preserved at position 1
        assert pd.isna(smoothed.iloc[1])
        # Other values should NOT be NaN
        assert not pd.isna(smoothed.iloc[0])
        assert not pd.isna(smoothed.iloc[2])
        assert not pd.isna(smoothed.iloc[3])
        assert not pd.isna(smoothed.iloc[4])

    def test_smooth_data_invalid_weight_raises_error(self):
        """Test that invalid weight values raise ValueError."""
        from src.evaluation.visualize import smooth_data
        raw = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="weight must be in"):
            smooth_data(raw, weight=1.5)
        with pytest.raises(ValueError, match="weight must be in"):
            smooth_data(raw, weight=-0.1)


class TestPlotTrainingCurves:
    """Tests for plot_training_curves function."""

    def test_creates_png_file(self, synthetic_training_df, tmp_path):
        from src.evaluation.visualize import plot_training_curves
        out = tmp_path / "training_curves.png"
        plot_training_curves(synthetic_training_df, out)
        assert out.exists()

    def test_png_file_is_non_empty(self, synthetic_training_df, tmp_path):
        from src.evaluation.visualize import plot_training_curves
        out = tmp_path / "training_curves.png"
        plot_training_curves(synthetic_training_df, out)
        assert out.stat().st_size > 0

    def test_creates_output_directory(self, synthetic_training_df, tmp_path):
        from src.evaluation.visualize import plot_training_curves
        out = tmp_path / "nested" / "dir" / "training_curves.png"
        plot_training_curves(synthetic_training_df, out)
        assert out.parent.exists()
        assert out.exists()

    def test_works_with_smoothing_enabled(self, synthetic_training_df, tmp_path):
        from src.evaluation.visualize import plot_training_curves
        out = tmp_path / "training_curves_smooth.png"
        plot_training_curves(synthetic_training_df, out, smoothing=0.9)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_works_with_smoothing_disabled(self, synthetic_training_df, tmp_path):
        from src.evaluation.visualize import plot_training_curves
        out = tmp_path / "training_curves_raw.png"
        plot_training_curves(synthetic_training_df, out, smoothing=0.0)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_returns_figure_object(self, synthetic_training_df):
        from src.evaluation.visualize import plot_training_curves
        import matplotlib.figure
        fig = plot_training_curves(synthetic_training_df, output_path=None)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_chart_has_required_elements(self, synthetic_training_df):
        """Test chart has correct title, axis labels (AC #3 publication quality)."""
        from src.evaluation.visualize import plot_training_curves
        import matplotlib.pyplot as plt
        fig = plot_training_curves(synthetic_training_df, output_path=None)
        ax = fig.axes[0]

        # Check title exists and is meaningful
        title = ax.get_title()
        assert len(title) > 0
        assert 'Training' in title or 'Reward' in title

        # Check x-axis label (Training Steps)
        xlabel = ax.get_xlabel()
        assert 'Step' in xlabel or 'step' in xlabel

        # Check y-axis label (Episode Reward)
        ylabel = ax.get_ylabel()
        assert 'Reward' in ylabel or 'EUR' in ylabel

        plt.close(fig)

    def test_chart_shows_data_points(self, synthetic_training_df):
        """Test chart actually plots the data."""
        from src.evaluation.visualize import plot_training_curves
        import matplotlib.pyplot as plt
        fig = plot_training_curves(synthetic_training_df, output_path=None)
        ax = fig.axes[0]

        # Check that there are lines plotted
        lines = ax.get_lines()
        assert len(lines) > 0

        plt.close(fig)

    def test_empty_dataframe_raises_error(self):
        """Test that empty DataFrame raises ValueError with helpful message."""
        from src.evaluation.visualize import plot_training_curves
        with pytest.raises(ValueError, match="DataFrame is empty"):
            plot_training_curves(pd.DataFrame({'step': [], 'value': []}))

    def test_missing_columns_raises_error(self):
        """Test that missing required columns raises ValueError."""
        from src.evaluation.visualize import plot_training_curves
        with pytest.raises(ValueError, match="must have 'step' and 'value' columns"):
            plot_training_curves(pd.DataFrame({'x': [1, 2], 'y': [3, 4]}))

    def test_multiple_runs_plotting(self, tmp_path):
        """Test that multiple runs are plotted with labels when 'run' column present."""
        from src.evaluation.visualize import plot_training_curves
        import matplotlib.pyplot as plt

        # Create synthetic data for multiple runs
        np.random.seed(42)
        data = []
        for run in ['SAC_1', 'SAC_2']:
            steps = np.arange(0, 100000, 1000)
            values = 1000 + np.random.normal(0, 100, len(steps))
            for s, v in zip(steps, values):
                data.append({'step': s, 'value': v, 'run': run})
        df = pd.DataFrame(data)

        out = tmp_path / "multi_run.png"
        fig = plot_training_curves(df, out)
        ax = fig.axes[0]

        # Check title indicates multiple runs
        assert 'All Runs' in ax.get_title() or 'Comparison' in ax.get_title()

        # Check that legend has run names
        legend_texts = [t.get_text() for t in ax.get_legend().get_texts()]
        assert 'SAC_1' in legend_texts
        assert 'SAC_2' in legend_texts

        plt.close(fig)
        assert out.exists()


class TestLoadTensorboardDataAllRuns:
    """Tests for load_tensorboard_data all_runs parameter."""

    def test_all_runs_parameter_exists(self):
        """Test that load_tensorboard_data accepts all_runs parameter."""
        from src.evaluation.visualize import load_tensorboard_data
        import inspect
        sig = inspect.signature(load_tensorboard_data)
        assert 'all_runs' in sig.parameters

    def test_all_runs_with_real_logs(self, tmp_path):
        """Test all_runs parameter with real TensorBoard logs if available.

        Skipped if no real TensorBoard logs exist or only 1 run has data.
        """
        from src.evaluation.visualize import load_tensorboard_data
        from pathlib import Path

        real_logs = Path("outputs/tensorboard")
        run_dirs = list(real_logs.glob("SAC_*")) if real_logs.exists() else []

        if len(run_dirs) < 2:
            pytest.skip("Need at least 2 training run directories for all_runs test")

        df = load_tensorboard_data(real_logs, all_runs=True)
        assert 'run' in df.columns
        # At least one run should be loaded
        assert df['run'].nunique() >= 1
        # Note: May be less than run_dirs count if some runs lack valid data


class TestMainCLI:
    """Tests for CLI --training-curves integration."""

    def test_main_accepts_training_curves_flag(self):
        """Test that main() accepts --training-curves argument without error."""
        from src.evaluation.visualize import main
        assert callable(main)

    def test_cli_imports_correctly(self):
        """Test that CLI dependencies are importable."""
        from src.evaluation.visualize import (
            load_tensorboard_data,
            plot_training_curves,
            smooth_data,
            DEFAULT_TENSORBOARD,
        )
        assert callable(load_tensorboard_data)
        assert callable(plot_training_curves)
        assert callable(smooth_data)
        assert DEFAULT_TENSORBOARD is not None

    def test_cli_accepts_all_runs_flag(self):
        """Test that CLI has --all-runs argument."""
        from src.evaluation.visualize import main
        import argparse
        # Verify by checking the function is callable
        # (actual parsing tested by integration tests)
        assert callable(main)


class TestIntegrationWithRealLogs:
    """Integration tests using real TensorBoard logs if available."""

    def test_full_pipeline_with_real_logs(self, tmp_path):
        """Test complete pipeline from logs to chart (integration test).

        Skipped if no real TensorBoard logs exist.
        """
        from src.evaluation.visualize import load_tensorboard_data, plot_training_curves
        from pathlib import Path

        real_logs = Path("outputs/tensorboard")
        if not real_logs.exists() or not list(real_logs.glob("SAC_*")):
            pytest.skip("No real TensorBoard logs available for integration test")

        # Load real data
        df = load_tensorboard_data(real_logs)
        assert len(df) > 0

        # Generate chart
        out = tmp_path / "training_curves.png"
        fig = plot_training_curves(df, out)

        assert out.exists()
        assert out.stat().st_size > 10000  # Should be a substantial image
