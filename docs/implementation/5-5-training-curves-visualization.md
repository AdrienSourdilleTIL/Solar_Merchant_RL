# Story 5.5: Training Curves Visualization

Status: done

## Story

As a developer,
I want to generate training reward curves from TensorBoard logs,
so that the README shows learning progress and convergence behavior.

## Acceptance Criteria

1. **Given** TensorBoard logs exist in `outputs/tensorboard/`
   **When** the training curves visualization function is called
   **Then** episode reward over training steps is plotted
   **And** the chart shows convergence behavior (reward improvement over time)

2. **Given** the visualization is generated
   **When** saved to disk
   **Then** chart is saved as PNG to `results/figures/training_curves.png`
   **And** the directory is created automatically if missing

3. **Given** the chart is rendered
   **When** viewed
   **Then** it is publication-quality: clear axis labels, title, readable font sizes
   **And** x-axis shows training timesteps
   **And** y-axis shows episode reward
   **And** line is smoothed to show trend (optional raw data as light background)

4. **Given** the script is run
   **When** no TensorBoard logs exist yet
   **Then** a user-friendly error message is printed directing the user to run training first

5. **Given** multiple training runs exist (SAC_1, SAC_2, etc.)
   **When** visualization is generated
   **Then** the most recent run is used by default OR all runs are plotted with labels

## Tasks / Subtasks

- [x] Task 1: Add `load_tensorboard_data(log_dir: Path) -> pd.DataFrame` to `src/evaluation/visualize.py` (AC: #1, #4)
  - [x] Implement TensorBoard event file reading using `tensorboard.backend.event_processing.event_accumulator` or `tbparse`
  - [x] Extract `rollout/ep_rew_mean` scalar (SB3 default metric) from event files
  - [x] Return DataFrame with columns: `step`, `value` (and optionally `wall_time`, `run`)
  - [x] Handle missing logs with `FileNotFoundError` and helpful message
  - [x] Support multiple run directories (glob `SAC_*` or `*/events.*`)

- [x] Task 2: Add `plot_training_curves(df: pd.DataFrame, output_path: Path, backend: str = 'Agg') -> Figure` to `src/evaluation/visualize.py` (AC: #1, #2, #3)
  - [x] Follow existing `plot_net_profit_comparison` pattern (backend parameter, deferred imports)
  - [x] Plot episode reward vs training steps
  - [x] Add optional smoothing (rolling average) with raw data as light alpha background
  - [x] Set publication-quality labels: "Training Steps" (x), "Episode Reward (EUR)" (y), title
  - [x] Include legend if multiple runs
  - [x] Save PNG to output_path with `dpi=150`

- [x] Task 3: Extend `main()` CLI with `--training-curves` flag (AC: #1, #2, #4)
  - [x] Add `--tensorboard-dir` argument (default `outputs/tensorboard`)
  - [x] Add `--training-curves` action flag to generate training curves chart
  - [x] Integrate with existing CLI structure (keep `--metrics-dir`, `--output-dir`, `--show`)

- [x] Task 4: Write tests in `tests/test_training_curves_visualization.py` (AC: #1, #2, #3, #4)
  - [x] Test `load_tensorboard_data` returns DataFrame with expected columns from synthetic event data
  - [x] Test `plot_training_curves` creates a PNG file at the specified path
  - [x] Test `plot_training_curves` works with smoothing enabled/disabled
  - [x] Test `load_tensorboard_data` raises `FileNotFoundError` with helpful message when no logs exist
  - [x] Test output PNG file is non-empty
  - [x] Test chart has correct axis labels and title

- [x] Task 5: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` -- all existing 335 tests must pass plus new tests
  - [x] Verify no regressions in existing test files

## Dev Notes

### CRITICAL: File Location -- Architecture Mandated

Architecture specifies `src/evaluation/visualize.py` as the visualization module. ADD functions to this existing file. Do NOT create a new file.

[Source: docs/architecture.md] -- `visualize.py` consumes metrics, generates `results/figures/`

### CRITICAL: Extend Existing Module -- Do NOT Create New File

Story 5-4 created `src/evaluation/visualize.py`. This story adds training curve functions to the SAME file. Follow the exact same patterns:
- `matplotlib.use(backend)` deferred inside function
- `backend: str = 'Agg'` parameter pattern
- Google-style docstrings with Args/Returns/Raises
- Return Figure object for testing

### TensorBoard Log Structure

SB3 logs TensorBoard events to `outputs/tensorboard/`. Current runs:
- `outputs/tensorboard/SAC_1/events.out.tfevents.*`
- `outputs/tensorboard/SAC_2/events.out.tfevents.*`

Key scalar to extract: `rollout/ep_rew_mean` (average episode reward during training)

Other potentially useful scalars (optional enhancement):
- `rollout/ep_len_mean` - episode length
- `train/loss` - training loss
- `train/entropy_loss` - entropy loss

### TensorBoard Reading Approach

Two options for reading TensorBoard event files:

**Option A: tensorboard.backend.event_processing (built-in with tensorboard)**
```python
from tensorboard.backend.event_processing import event_accumulator

def load_tensorboard_data(log_dir: Path) -> pd.DataFrame:
    """Load training metrics from TensorBoard event files.

    Args:
        log_dir: Path to tensorboard log directory (e.g., outputs/tensorboard).

    Returns:
        DataFrame with columns: step, value (episode reward).

    Raises:
        FileNotFoundError: If no TensorBoard logs found.
    """
    # Find most recent run directory
    run_dirs = sorted(log_dir.glob('SAC_*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise FileNotFoundError(
            f"No TensorBoard logs found in {log_dir}.\n"
            "Run training first:\n"
            "  python src/training/train.py"
        )

    ea = event_accumulator.EventAccumulator(str(run_dirs[0]))
    ea.Reload()

    scalars = ea.Scalars('rollout/ep_rew_mean')
    df = pd.DataFrame([{'step': s.step, 'value': s.value} for s in scalars])
    return df
```

**Option B: tbparse (cleaner API, may need to add dependency)**
```python
from tbparse import SummaryReader

def load_tensorboard_data(log_dir: Path) -> pd.DataFrame:
    reader = SummaryReader(log_dir)
    df = reader.scalars
    df = df[df['tag'] == 'rollout/ep_rew_mean'][['step', 'value']]
    return df
```

**Recommendation:** Use Option A (`event_accumulator`) since tensorboard is already a dependency. Avoids adding new dependencies.

### Smoothing Implementation

Apply exponential moving average (EMA) or rolling mean for smooth line:

```python
def smooth_data(values: pd.Series, weight: float = 0.9) -> pd.Series:
    """Apply exponential moving average smoothing.

    Args:
        values: Raw values to smooth.
        weight: Smoothing weight (0-1, higher = smoother). Default 0.9.

    Returns:
        Smoothed values.
    """
    smoothed = []
    last = values.iloc[0]
    for v in values:
        smoothed_val = last * weight + (1 - weight) * v
        smoothed.append(smoothed_val)
        last = smoothed_val
    return pd.Series(smoothed)
```

### Plot Structure

```python
def plot_training_curves(
    df: pd.DataFrame,
    output_path: Path | None = None,
    backend: str = 'Agg',
    smoothing: float = 0.9,
) -> matplotlib.figure.Figure:
    """Generate training reward curve chart.

    Args:
        df: DataFrame with 'step' and 'value' columns.
        output_path: Path to save PNG. If None, not saved.
        backend: Matplotlib backend. Default 'Agg' for headless.
        smoothing: EMA smoothing weight (0 = no smoothing, 1 = full). Default 0.9.

    Returns:
        Matplotlib Figure object.
    """
    import matplotlib
    matplotlib.use(backend)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot raw data as light background
    ax.plot(df['step'], df['value'], alpha=0.3, color='#2196F3', linewidth=0.5)

    # Plot smoothed line
    if smoothing > 0:
        smoothed = smooth_data(df['value'], smoothing)
        ax.plot(df['step'], smoothed, color='#2196F3', linewidth=2, label='Episode Reward')
    else:
        ax.plot(df['step'], df['value'], color='#2196F3', linewidth=2, label='Episode Reward')

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Episode Reward (EUR)', fontsize=12)
    ax.set_title('SAC Training Progress — Episode Reward over Time', fontsize=14)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    return fig
```

### CLI Extension

Extend existing `main()` to support both chart types:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description='Generate performance charts')
    parser.add_argument('--metrics-dir', type=str, default=str(DEFAULT_METRICS))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_FIGURES))
    parser.add_argument('--tensorboard-dir', type=str, default=str(Path('outputs/tensorboard')))
    parser.add_argument('--show', action='store_true', help='Display chart interactively')
    parser.add_argument('--training-curves', action='store_true', help='Generate training curves chart')
    parser.add_argument('--performance', action='store_true', help='Generate performance comparison chart')
    parser.add_argument('--all', action='store_true', help='Generate all charts')
    args = parser.parse_args()

    # Default: generate performance comparison if no specific flag
    if not args.training_curves and not args.performance and not args.all:
        args.performance = True

    backend = 'TkAgg' if args.show else 'Agg'

    if args.performance or args.all:
        # Existing performance comparison logic...
        pass

    if args.training_curves or args.all:
        tb_dir = Path(args.tensorboard_dir)
        df = load_tensorboard_data(tb_dir)
        output_path = Path(args.output_dir) / 'training_curves.png'
        fig = plot_training_curves(df, output_path, backend=backend)
        # ...
```

### Test Approach

Tests should NOT require real TensorBoard logs. Create synthetic event-like data:

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def synthetic_training_df():
    """Create synthetic training data mimicking TensorBoard scalars."""
    steps = np.arange(0, 500000, 1000)  # Every 1k steps
    # Simulate reward increasing then plateauing
    rewards = 1000 + 3000 * (1 - np.exp(-steps / 100000)) + np.random.normal(0, 200, len(steps))
    return pd.DataFrame({'step': steps, 'value': rewards})


def test_plot_training_curves_creates_png(synthetic_training_df, tmp_path):
    from src.evaluation.visualize import plot_training_curves
    out = tmp_path / "training_curves.png"
    plot_training_curves(synthetic_training_df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_training_curves_no_smoothing(synthetic_training_df, tmp_path):
    from src.evaluation.visualize import plot_training_curves
    out = tmp_path / "training_curves_raw.png"
    plot_training_curves(synthetic_training_df, out, smoothing=0)
    assert out.exists()
```

For `load_tensorboard_data` tests, mock the event_accumulator or create minimal event files in tmp_path.

### What NOT to Do

- Do NOT create a new visualization file -- extend existing `src/evaluation/visualize.py`
- Do NOT import SB3, torch, or SolarMerchantEnv -- this is pure visualization
- Do NOT run any training or evaluation -- load from TensorBoard logs only
- Do NOT use `plt.show()` by default -- use `Agg` backend
- Do NOT break existing CLI functionality -- extend it with new flags
- Do NOT hardcode run names -- discover dynamically with glob
- Do NOT add dependencies without checking if tensorboard already provides the functionality

### Scope Boundary

This story creates ONE chart: **training reward curves**. Keep it focused:
- Story 5-6 handles final README export and sizing
- The chart should look ready for README but 5-6 may adjust sizing

### Project Structure Notes

| Requirement | Location |
|-------------|----------|
| Modified script | `src/evaluation/visualize.py` (EXTEND existing) |
| New tests | `tests/test_training_curves_visualization.py` (NEW) |
| Chart output | `results/figures/training_curves.png` |
| Data input | `outputs/tensorboard/SAC_*/events.*` (existing training logs) |

### References

- [Source: docs/epics.md#Story-5.5] -- FR36: Generate training reward curves
- [Source: docs/architecture.md] -- `visualize.py` location, matplotlib/tensorboard dependencies
- [Source: docs/architecture.md#Module-Boundaries] -- `visualize` module consumes metrics, outputs figures
- [Source: src/evaluation/visualize.py] -- Existing module with `plot_net_profit_comparison` pattern
- [Source: outputs/tensorboard/] -- TensorBoard event files from training runs

### Previous Story Intelligence

**From Story 5-4 (Performance Visualization):**
- 335 tests passing after code review. Do not regress.
- `plot_net_profit_comparison` pattern: `backend` param, deferred matplotlib import, return Figure object.
- CLI structure: `--metrics-dir`, `--output-dir`, `--show` flags.
- Backend handling: choose before first matplotlib import, `TkAgg` for interactive, `Agg` for headless.
- Docstring style: Google format with Args/Returns/Raises sections.
- Path handling: `output_path.parent.mkdir(parents=True, exist_ok=True)` pattern.
- Test pattern: synthetic data fixtures, no real data required, test PNG exists and is non-empty.

**From Story 4-3 (TensorBoard Logging):**
- TensorBoard logs saved to `outputs/tensorboard/` directory.
- SB3 logs `rollout/ep_rew_mean` scalar by default.
- Multiple runs create `SAC_1/`, `SAC_2/` subdirectories.

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

No issues encountered during implementation.

### Completion Notes List

- Implemented `load_tensorboard_data()` function using tensorboard.backend.event_processing.event_accumulator (Option A from Dev Notes)
- Implemented `smooth_data()` helper for exponential moving average smoothing
- Implemented `plot_training_curves()` function following existing `plot_net_profit_comparison` pattern
- Extended CLI with `--training-curves`, `--tensorboard-dir`, `--performance`, and `--all` flags
- Created comprehensive test suite with 17 tests covering all acceptance criteria
- All 355 tests pass (335 existing + 17 new + 3 integration tests)
- Successfully generated `results/figures/training_curves.png` (163KB publication-quality chart)

### Change Log

- 2026-02-06: Implemented Story 5.5 - Training Curves Visualization
  - Added TensorBoard data loading and training curves chart generation
  - Extended CLI with new flags for training curves visualization
  - Created new test file with 17 comprehensive tests
- 2026-02-06: Code Review Fixes (Senior Developer Review)
  - Fixed smooth_data() crash on empty Series
  - Fixed NaN propagation bug in smooth_data() - NaN values now preserved, not propagated
  - Added weight parameter validation (must be in [0, 1] range)
  - Fixed plot_training_curves() crash on empty DataFrame with helpful error message
  - Added multiple runs support with labels (AC #5 completion) - `--all-runs` CLI flag
  - Added 9 new edge case tests (total: 26 tests in file, 364 tests total)

### File List

**Modified:**
- `src/evaluation/visualize.py` — Added load_tensorboard_data(), smooth_data(), plot_training_curves() functions and extended main() CLI. Code review: added edge case handling, weight validation, multiple runs support.

**Created:**
- `tests/test_training_curves_visualization.py` — 26 tests for training curves visualization (17 original + 9 edge case tests)
- `results/figures/training_curves.png` — Generated training curves chart

## Senior Developer Review (AI)

**Review Date:** 2026-02-06
**Reviewer:** Claude Opus 4.5 (Adversarial Code Review)
**Outcome:** ✅ APPROVED (after fixes)

### Issues Found and Fixed

| # | Severity | Issue | Resolution |
|---|----------|-------|------------|
| 1 | HIGH | AC #5 incomplete - no "all runs with labels" option | Added `--all-runs` CLI flag, `all_runs` parameter to `load_tensorboard_data()`, multi-run plotting in `plot_training_curves()` |
| 2 | HIGH | Empty DataFrame crashes `plot_training_curves()` | Added validation with helpful error message |
| 3 | MEDIUM | `smooth_data()` crashes on empty Series | Added early return for empty input |
| 4 | MEDIUM | NaN propagation bug in `smooth_data()` | Refactored to preserve NaN, not propagate |
| 5 | MEDIUM | No weight parameter validation | Added range check with ValueError |
| 6 | MEDIUM | Missing edge case tests | Added 9 new tests |
| 7 | LOW | Chart shows decreasing reward | Not a code bug - training data shows this behavior |

### Test Results

- **Before review:** 355 tests passing
- **After review:** 364 tests passing (+9 new edge case tests)
- **No regressions**
