# Story 5.4: Performance Visualization

Status: review

## Story

As a developer,
I want to generate performance comparison charts for the RL agent and baselines,
so that the README showcases results visually with publication-quality figures.

## Acceptance Criteria

1. **Given** evaluation metrics exist in `results/metrics/` (from Stories 5-1, 5-2, 5-3)
   **When** the visualization script is run
   **Then** a bar chart comparing net profit across all 4 policies is generated
   **And** the chart includes error bars from multi-seed std when available

2. **Given** the visualization is generated
   **When** saved to disk
   **Then** chart is saved as PNG to `results/figures/performance_comparison.png`
   **And** the `results/figures/` directory is created automatically if missing

3. **Given** the chart is rendered
   **When** viewed
   **Then** it is publication-quality: clear axis labels, legend, title, readable font sizes
   **And** policies are visually distinguished (distinct bar colors)
   **And** the RL agent bar is visually emphasized (e.g., different color or annotation)

4. **Given** the script is run
   **When** no evaluation CSVs exist yet
   **Then** a user-friendly error message is printed directing the user to run the prerequisite evaluation scripts

## Tasks / Subtasks

- [x] Task 1: Create `src/evaluation/visualize.py` script (AC: #1, #2, #3, #4)
  - [x] Create `load_comparison_data(metrics_dir: Path) -> pd.DataFrame` that loads `multi_seed_evaluation.csv` (preferred) or falls back to `agent_vs_baselines.csv` / `baseline_comparison.csv` + `agent_evaluation.csv`
  - [x] Create `plot_net_profit_comparison(df: pd.DataFrame, output_path: Path) -> None` that renders a grouped bar chart of net profit per policy with error bars (std) if available
  - [x] Create `main()` with CLI args: `--metrics-dir` (default `results/metrics`), `--output-dir` (default `results/figures`), `--show` flag to display interactively
  - [x] Ensure `results/figures/` directory is created via `output_dir.mkdir(parents=True, exist_ok=True)`
  - [x] Add user-friendly `FileNotFoundError` handling if no CSVs found

- [x] Task 2: Write tests in `tests/test_performance_visualization.py` (AC: #1, #2, #3, #4)
  - [x] Test `load_comparison_data` returns DataFrame with expected columns from a synthetic CSV
  - [x] Test `plot_net_profit_comparison` creates a PNG file at the specified path
  - [x] Test `plot_net_profit_comparison` works with and without std columns (error bars optional)
  - [x] Test `main` is importable
  - [x] Test `load_comparison_data` raises `FileNotFoundError` with helpful message when no CSVs exist
  - [x] Test output PNG file is non-empty

- [x] Task 3: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` -- all existing 321 tests must pass plus new tests
  - [x] Verify no regressions in existing test files

## Dev Notes

### CRITICAL: File Location -- Architecture Mandated

Architecture specifies `src/evaluation/visualize.py` as the visualization module. Do NOT create it elsewhere.

[Source: docs/architecture.md] -- `visualize.py` consumes metrics, generates `results/figures/`

### CRITICAL: Reuse Existing Data -- Do NOT Re-Evaluate

All evaluation data already exists as CSVs. This script loads pre-computed results and visualizes them. Do NOT import SB3, do NOT import SolarMerchantEnv, do NOT run any evaluation loops.

**Data sources (load from CSV only):**
- `results/metrics/multi_seed_evaluation.csv` -- Has `{metric}_mean` and `{metric}_std` columns (preferred, from 5-3)
- `results/metrics/agent_vs_baselines.csv` -- Combined agent + baseline results with improvement_pct (from 5-2)
- `results/metrics/baseline_comparison.csv` -- 3 baseline rows (from 3-4)
- `results/metrics/agent_evaluation.csv` -- 1 agent row (from 5-1)

### Data Loading Strategy

Multi-seed data is the richest source (has mean + std for error bars). Fallback chain:

```python
def load_comparison_data(metrics_dir: Path) -> pd.DataFrame:
    """Load evaluation results for visualization.

    Tries multi_seed_evaluation.csv first (has std for error bars),
    falls back to agent_vs_baselines.csv, then individual files.

    Args:
        metrics_dir: Path to results/metrics/ directory.

    Returns:
        DataFrame with columns: policy, net_profit (or net_profit_mean),
        and optionally net_profit_std for error bars.

    Raises:
        FileNotFoundError: If no evaluation CSV files found.
    """
    multi_seed = metrics_dir / 'multi_seed_evaluation.csv'
    if multi_seed.exists():
        return pd.read_csv(multi_seed)

    combined = metrics_dir / 'agent_vs_baselines.csv'
    if combined.exists():
        return pd.read_csv(combined)

    # Fallback: merge individual files
    agent_f = metrics_dir / 'agent_evaluation.csv'
    baseline_f = metrics_dir / 'baseline_comparison.csv'
    if agent_f.exists() and baseline_f.exists():
        return pd.concat([pd.read_csv(agent_f), pd.read_csv(baseline_f)], ignore_index=True)

    raise FileNotFoundError(
        f"No evaluation CSV files found in {metrics_dir}.\n"
        "Run evaluation scripts first:\n"
        "  python src/evaluation/evaluate_baselines.py\n"
        "  python src/evaluation/evaluate_agent.py\n"
        "  python src/evaluation/evaluate_multi_seed.py"
    )
```

### Multi-Seed CSV Column Names

The `multi_seed_evaluation.csv` uses `{metric}_mean` and `{metric}_std` suffixes:

```
policy, net_profit_mean, net_profit_std, revenue_mean, revenue_std, imbalance_cost_mean, imbalance_cost_std, ...
```

The non-multi-seed CSVs use plain column names:

```
policy, net_profit, revenue, imbalance_cost, ...
```

The plotting function must handle both naming conventions:

```python
# Detect column naming
if 'net_profit_mean' in df.columns:
    profit_col = 'net_profit_mean'
    std_col = 'net_profit_std' if 'net_profit_std' in df.columns else None
else:
    profit_col = 'net_profit'
    std_col = None
```

### Plotting Function

```python
def plot_net_profit_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Generate bar chart comparing net profit across policies.

    Args:
        df: DataFrame with policy names and net_profit values.
            Supports both plain columns (net_profit) and
            multi-seed columns (net_profit_mean, net_profit_std).
        output_path: Path to save PNG file.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for CI/headless
    import matplotlib.pyplot as plt

    # Detect columns
    if 'net_profit_mean' in df.columns:
        values = df['net_profit_mean']
        errors = df['net_profit_std'] if 'net_profit_std' in df.columns else None
    else:
        values = df['net_profit']
        errors = None

    policies = df['policy']

    # Color: highlight RL agent differently
    colors = []
    for p in policies:
        if 'rl' in p.lower() or 'sac' in p.lower() or 'agent' in p.lower():
            colors.append('#2196F3')  # Blue for RL agent
        else:
            colors.append('#9E9E9E')  # Grey for baselines

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(
        range(len(policies)),
        values,
        yerr=errors,
        capsize=5,
        color=colors,
        edgecolor='black',
        linewidth=0.8,
    )

    ax.set_xticks(range(len(policies)))
    ax.set_xticklabels(policies, rotation=15, ha='right', fontsize=11)
    ax.set_ylabel('Net Profit (EUR / episode)', fontsize=12)
    ax.set_title('RL Agent vs Baseline Policies — Net Profit Comparison', fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
    ax.grid(axis='y', alpha=0.3)

    # Value labels on bars
    for bar, val in zip(bars, values):
        y_pos = bar.get_height()
        offset = 50 if y_pos >= 0 else -150
        ax.annotate(
            f'EUR {val:,.0f}',
            xy=(bar.get_x() + bar.get_width() / 2, y_pos),
            xytext=(0, offset if y_pos >= 0 else -offset),
            textcoords='offset points',
            ha='center', va='bottom' if y_pos >= 0 else 'top',
            fontsize=9, fontweight='bold',
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {output_path}")
```

### Main Script Structure

```python
"""Visualize evaluation results for Solar Merchant RL.

Generates performance comparison charts from pre-computed evaluation CSVs.
Outputs publication-quality PNG figures for README embedding.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Paths
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results'
DEFAULT_METRICS = RESULTS_PATH / 'metrics'
DEFAULT_FIGURES = RESULTS_PATH / 'figures'


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate performance charts')
    parser.add_argument('--metrics-dir', type=str, default=str(DEFAULT_METRICS))
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_FIGURES))
    parser.add_argument('--show', action='store_true', help='Display chart interactively')
    args = parser.parse_args()

    metrics_dir = Path(args.metrics_dir)
    output_dir = Path(args.output_dir)

    df = load_comparison_data(metrics_dir)
    output_path = output_dir / 'performance_comparison.png'
    plot_net_profit_comparison(df, output_path)

    if args.show:
        import matplotlib.pyplot as plt
        # Re-render for interactive display (use default backend)
        ...


if __name__ == '__main__':
    main()
```

### Import Pattern

Follow project convention:
- `sys.path.insert(0, str(Path(__file__).parent.parent.parent))` at top
- Module-level imports: argparse, sys, pathlib only
- Deferred matplotlib import inside the plotting function (heavy dep)
- pandas imported at module level (lightweight enough, needed by load function)
- Do NOT import SB3, torch, SolarMerchantEnv, or any training/evaluation code

### matplotlib Backend

Use `matplotlib.use('Agg')` inside the plotting function before importing pyplot. This ensures headless operation (CI, SSH). The `--show` flag can override for interactive use.

### Test Approach

Tests should NOT require real evaluation CSVs. Create synthetic data:

```python
import pytest
import pandas as pd
import numpy as np
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
    """Create synthetic CSVs without std columns."""
    df = pd.DataFrame([
        {"policy": "RL Agent (SAC)", "net_profit": 5000.0},
        {"policy": "Conservative (80%)", "net_profit": 3895.0},
        {"policy": "Aggressive (100%)", "net_profit": -4132.0},
        {"policy": "Price-Aware", "net_profit": 2096.0},
    ])
    csv_path = tmp_path / "agent_vs_baselines.csv"
    df.to_csv(csv_path, index=False)
    return tmp_path


def test_plot_creates_png(synthetic_multi_seed_csv, tmp_path):
    from src.evaluation.visualize import load_comparison_data, plot_net_profit_comparison
    df = load_comparison_data(synthetic_multi_seed_csv)
    out = tmp_path / "figures" / "test.png"
    plot_net_profit_comparison(df, out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_plot_no_std(synthetic_plain_csv, tmp_path):
    from src.evaluation.visualize import load_comparison_data, plot_net_profit_comparison
    df = load_comparison_data(synthetic_plain_csv)
    out = tmp_path / "test_no_std.png"
    plot_net_profit_comparison(df, out)
    assert out.exists()


def test_missing_csvs_error(tmp_path):
    from src.evaluation.visualize import load_comparison_data
    with pytest.raises(FileNotFoundError, match="No evaluation CSV"):
        load_comparison_data(tmp_path)
```

### What NOT to Do

- Do NOT modify any existing files (`evaluate.py`, `evaluate_baselines.py`, `evaluate_agent.py`, `evaluate_multi_seed.py`, `compare_agent_baselines.py`)
- Do NOT import SB3, torch, or SolarMerchantEnv -- this is a pure visualization script
- Do NOT run any evaluations -- load from CSV only
- Do NOT use `plt.show()` by default -- use `Agg` backend for headless operation
- Do NOT create the file anywhere other than `src/evaluation/visualize.py` -- architecture mandates this location
- Do NOT hardcode metric values -- load dynamically from CSV
- Do NOT add additional chart types in this story -- Story 5-5 handles training curves, Story 5-6 handles README export

### Scope Boundary

This story creates ONE chart: **net profit bar chart comparison**. Keep it focused:
- Story 5-5 will add training reward curves from TensorBoard logs
- Story 5-6 will handle final export sizing and README embedding
- The `visualize.py` module should be structured so 5-5 can add functions to it

### Project Structure Notes

| Requirement | Location |
|-------------|----------|
| New script | `src/evaluation/visualize.py` (NEW) |
| New tests | `tests/test_performance_visualization.py` (NEW) |
| Chart output | `results/figures/performance_comparison.png` |
| Data input | `results/metrics/*.csv` (existing, from Stories 3-4, 5-1, 5-2, 5-3) |

Naming follows architecture conventions: snake_case files, snake_case functions, Google-style docstrings with Args/Returns/Raises.

### References

- [Source: docs/epics.md#Story-5.4] -- FR35: Generate performance comparison charts
- [Source: docs/architecture.md] -- `visualize.py` location, matplotlib dependency, `results/figures/` output
- [Source: docs/architecture.md#Module-Boundaries] -- `visualize` module: Metrics -> PNG figures, matplotlib
- [Source: docs/architecture.md#Data-Flow] -- `visualize.py` consumes metrics, generates `results/figures/`
- [Source: src/evaluation/evaluate_multi_seed.py] -- Produces `multi_seed_evaluation.csv` with `_mean`/`_std` columns
- [Source: src/evaluation/compare_agent_baselines.py] -- Produces `agent_vs_baselines.csv`
- [Source: src/evaluation/evaluate.py] -- `save_results()` CSV format, column ordering
- [Source: src/evaluation/evaluate_agent.py] -- Produces `agent_evaluation.csv`
- [Source: src/evaluation/evaluate_baselines.py] -- Produces `baseline_comparison.csv`

### Previous Story Intelligence

**From Story 5-3 (Multi-Seed Statistical Evaluation):**
- 321 tests passing after code review. Do not regress.
- `multi_seed_evaluation.csv` has `{metric}_mean` and `{metric}_std` suffixed columns plus `n_seeds` count.
- `multi_seed_per_seed.csv` has per-seed detail rows with `seed` column.
- `aggregate_results()` uses `ddof=1` for sample std.
- `_order_aggregate_keys()` groups metric pairs (mean, std) in CSV output.

**From Story 5-2 (Baseline Comparison):**
- `agent_vs_baselines.csv` has combined results with `improvement_pct` column.
- `load_results()` in `compare_agent_baselines.py` loads CSV -> list of dicts with 'policy' key.
- `determine_verdict()` returns "PASS" / "FAIL_SOME" / "FAIL_NONE".

**From Story 5-1 (Agent Evaluation):**
- Agent policy name is `"RL Agent (SAC)"`.
- Deferred imports pattern for heavy dependencies inside `main()`.

**Known Baseline Values (from CSV -- do not hardcode):**
- Conservative: EUR 3,895 net profit/episode
- Aggressive: EUR -4,132 net profit/episode
- Price-Aware: EUR 2,096 net profit/episode

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None - clean implementation, no debug issues encountered.

### Completion Notes List

- Implemented `src/evaluation/visualize.py` with three functions: `load_comparison_data()`, `plot_net_profit_comparison()`, and `main()`.
- Data loading follows the specified fallback chain: multi_seed_evaluation.csv -> agent_vs_baselines.csv -> individual CSVs (agent_evaluation.csv + baseline_comparison.csv).
- Plotting handles both `net_profit_mean`/`net_profit_std` (multi-seed) and plain `net_profit` column naming conventions.
- RL agent bar highlighted in blue (#2196F3), baselines in grey (#9E9E9E), with EUR value annotations on each bar.
- Uses `matplotlib.use('Agg')` for headless operation; `--show` flag available for interactive display.
- CLI supports `--metrics-dir`, `--output-dir`, and `--show` arguments with sensible defaults.
- User-friendly `FileNotFoundError` with guidance on which evaluation scripts to run.
- 11 tests written using synthetic CSV fixtures (no real evaluation data required).
- Full test suite: 332 passed (321 existing + 11 new), 0 failed, no regressions.

### Change Log

- 2026-02-03: Implemented Story 5-4 - Performance visualization with net profit bar chart, data loading with fallback chain, and 11 tests.

### File List

- `src/evaluation/visualize.py` (NEW) - Performance visualization script
- `tests/test_performance_visualization.py` (NEW) - 11 tests for visualization module
