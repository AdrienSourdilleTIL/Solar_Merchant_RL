# Story 5.2: Baseline Comparison

Status: done

## Story

As a developer,
I want to compare the trained RL agent against all baseline policies,
so that I can determine if RL beats rule-based strategies and quantify the improvement.

## Acceptance Criteria

1. **Given** agent evaluation results (`results/metrics/agent_evaluation.csv`) and baseline results (`results/metrics/baseline_comparison.csv`)
   **When** the comparison script is run
   **Then** all four policies (RL Agent + 3 baselines) are loaded and compared side-by-side

2. **Given** all four policy results are loaded
   **When** comparison metrics are computed
   **Then** percentage improvement of the RL agent over each baseline is calculated for net_profit
   **And** percentage improvement uses the formula: `(agent - baseline) / abs(baseline) * 100`

3. **Given** comparison results
   **When** the best baseline is identified
   **Then** the best-performing baseline (highest net_profit) is identified
   **And** the comparison clearly states whether the agent beats all baselines, some, or none

4. **Given** comparison results
   **When** output is generated
   **Then** a formatted comparison table prints to stdout (all 4 policies, key metrics, improvement %)
   **And** results are saved to `results/metrics/agent_vs_baselines.csv`
   **And** a summary pass/fail line states whether the agent beats all baselines

## Tasks / Subtasks

- [x] Task 1: Create `src/evaluation/compare_agent_baselines.py` script (AC: #1, #2, #3, #4)
  - [x] Load agent results from `results/metrics/agent_evaluation.csv` using pandas
  - [x] Load baseline results from `results/metrics/baseline_comparison.csv` using pandas
  - [x] Validate both files exist with user-friendly error messages if not
  - [x] Combine into unified results list (dicts with 'policy' key)
  - [x] Calculate percentage improvement of agent over each baseline for net_profit
  - [x] Identify the best baseline (highest net_profit)
  - [x] Print formatted comparison table using `print_comparison()` from `evaluate.py`
  - [x] Print additional improvement summary section (% over each baseline)
  - [x] Print pass/fail verdict (agent beats all baselines = PASS)
  - [x] Save combined results to `results/metrics/agent_vs_baselines.csv` using `save_results()`
  - [x] Accept `--agent-results` CLI arg (default: `results/metrics/agent_evaluation.csv`)
  - [x] Accept `--baseline-results` CLI arg (default: `results/metrics/baseline_comparison.csv`)

- [x] Task 2: Write tests in `tests/test_baseline_comparison.py` (AC: #1, #2, #3, #4)
  - [x] Test script is importable (`compare_agent_baselines` module)
  - [x] Test `load_results()` loads a valid CSV and returns list of dicts with 'policy' key
  - [x] Test `calculate_improvement()` returns correct percentage for known values (e.g., agent=5000, baseline=4000 → 25%)
  - [x] Test `calculate_improvement()` handles negative baselines correctly (e.g., Aggressive at -4132)
  - [x] Test `identify_best_baseline()` returns the baseline with highest net_profit
  - [x] Test `determine_verdict()` returns "PASS" when agent beats all baselines
  - [x] Test `determine_verdict()` returns "FAIL" when agent loses to any baseline
  - [x] Test CSV output is created with all 4 policies and improvement columns

- [x] Task 3: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` -- all existing 293 tests must pass (306 total: 293 existing + 13 new)
  - [x] Run new tests -- all 13 pass

## Dev Notes

### CRITICAL: Reuse Existing Code -- Do NOT Reinvent

The evaluation infrastructure is already built. This script loads pre-computed results CSVs and adds comparison logic.

**REUSE these existing functions (DO NOT rewrite):**
- `print_comparison()` from `src/evaluation/evaluate.py` -- formatted table output
- `save_results()` from `src/evaluation/evaluate.py` -- saves to CSV with correct column order

### Script Structure: `src/evaluation/compare_agent_baselines.py`

```python
"""Compare trained RL agent against baseline policies.

Loads pre-computed evaluation results for the agent and baselines,
calculates percentage improvement, and determines if RL beats rule-based strategies.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd

from src.evaluation.evaluate import print_comparison, save_results

# Paths
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results' / 'metrics'
DEFAULT_AGENT_RESULTS = RESULTS_PATH / 'agent_evaluation.csv'
DEFAULT_BASELINE_RESULTS = RESULTS_PATH / 'baseline_comparison.csv'


def load_results(csv_path: Path) -> list[dict[str, float]]:
    """Load evaluation results from CSV.

    Args:
        csv_path: Path to CSV file with policy results.

    Returns:
        List of result dicts, each with 'policy' key and metric values.

    Raises:
        FileNotFoundError: If CSV file does not exist.
    """
    ...


def calculate_improvement(agent_value: float, baseline_value: float) -> float:
    """Calculate percentage improvement of agent over baseline.

    Args:
        agent_value: Agent metric value.
        baseline_value: Baseline metric value.

    Returns:
        Percentage improvement: (agent - baseline) / abs(baseline) * 100.
    """
    ...


def identify_best_baseline(baselines: list[dict[str, float]]) -> dict[str, float]:
    """Find the baseline with the highest net_profit.

    Args:
        baselines: List of baseline result dicts.

    Returns:
        The best-performing baseline dict.
    """
    ...


def determine_verdict(agent_profit: float, baselines: list[dict[str, float]]) -> str:
    """Determine if agent beats all baselines.

    Args:
        agent_profit: Agent net_profit.
        baselines: List of baseline result dicts with net_profit.

    Returns:
        "PASS" if agent beats all baselines, "FAIL" otherwise.
    """
    ...


def main() -> None:
    """Load results, compare, and print verdict."""
    parser = argparse.ArgumentParser(description='Compare RL agent vs baselines')
    parser.add_argument('--agent-results', type=str, default=str(DEFAULT_AGENT_RESULTS))
    parser.add_argument('--baseline-results', type=str, default=str(DEFAULT_BASELINE_RESULTS))
    args = parser.parse_args()
    # Load, compare, print, save
```

### Improvement Calculation

```python
def calculate_improvement(agent_value: float, baseline_value: float) -> float:
    if abs(baseline_value) < 1e-8:
        return float('inf') if agent_value > 0 else 0.0
    return (agent_value - baseline_value) / abs(baseline_value) * 100
```

Note: `abs(baseline_value)` in denominator handles the Aggressive baseline which has **negative** net_profit (-4,132 EUR). Without `abs()`, the percentage would be inverted.

### Expected Output Format

```
================================================================================
POLICY COMPARISON: RL AGENT vs BASELINES
================================================================================

Policy                          Net Profit   Imb Cost  Deliv %  Episodes
--------------------------------------------------------------------------------
RL Agent (SAC)                  EUR X,XXX  EUR X,XXX   XX.X%        10
Conservative (80%)              EUR 3,895  EUR 16,615  117.4%        10
Aggressive (100%)               EUR -4,132  EUR 24,969   93.5%        10
Price-Aware                     EUR 2,096  EUR 18,679  116.1%        10
--------------------------------------------------------------------------------

Best policy: [winner]
Mean net profit: EUR X,XXX per episode

IMPROVEMENT OVER BASELINES:
  vs Conservative (80%):    +XX.X%
  vs Aggressive (100%):     +XXX.X%
  vs Price-Aware:           +XX.X%

VERDICT: [PASS/FAIL] - Agent [beats/does not beat] all baselines.
Best baseline: Conservative (80%) at EUR 3,895
```

### Baseline Reference Values (from existing CSV)

| Policy | Net Profit (EUR) | Imbalance Cost | Delivery % |
|--------|------------------|----------------|------------|
| Conservative (80%) | 3,895 | 16,615 | 117.4% |
| Aggressive (100%) | -4,132 | 24,969 | 93.5% |
| Price-Aware | 2,096 | 18,679 | 116.1% |

Best baseline is **Conservative at EUR 3,895**. Agent must beat this to PASS.

### What NOT to Do

- Do NOT modify `evaluate.py`, `evaluate_baselines.py`, `evaluate_agent.py`, or baseline_policies.py
- Do NOT re-run evaluations -- load from existing CSVs
- Do NOT import SB3 or torch -- this script only needs pandas and the evaluate.py helpers
- Do NOT create a new evaluation loop -- this is purely a comparison/reporting script
- Do NOT hardcode baseline values -- load them dynamically from CSV

### Project Structure Notes

| Requirement | Location |
|-------------|----------|
| New script | `src/evaluation/compare_agent_baselines.py` (NEW) |
| New tests | `tests/test_baseline_comparison.py` (NEW) |
| Combined output | `results/metrics/agent_vs_baselines.csv` |
| Agent input | `results/metrics/agent_evaluation.csv` (from 5-1) |
| Baseline input | `results/metrics/baseline_comparison.csv` (from 3-4) |

Naming follows architecture.md conventions: snake_case files, snake_case functions, Google-style docstrings.

### References

- [Source: docs/epics.md#Story-5.2](../../docs/epics.md) -- FR31: Compare against baselines, FR32: Calculate metrics, FR33: Report improvement percentage
- [Source: docs/architecture.md#Evaluation-Architecture](../../docs/architecture.md) -- Evaluation module: Model + Env -> Metrics dict
- [Source: docs/architecture.md#Module-Boundaries](../../docs/architecture.md) -- evaluation module input/output
- [Source: src/evaluation/evaluate.py](../../src/evaluation/evaluate.py) -- print_comparison(), save_results() -- REUSE
- [Source: src/evaluation/evaluate_baselines.py](../../src/evaluation/evaluate_baselines.py) -- Produces baseline_comparison.csv
- [Source: src/evaluation/evaluate_agent.py](../../src/evaluation/evaluate_agent.py) -- Produces agent_evaluation.csv
- [Source: results/metrics/baseline_comparison.csv](../../results/metrics/baseline_comparison.csv) -- Existing baseline results
- [Source: docs/implementation/5-1-agent-evaluation-on-test-set.md](../../docs/implementation/5-1-agent-evaluation-on-test-set.md) -- Story 5-1: agent_evaluation.csv format and location

### Previous Story Intelligence

**From Story 5-1 (Agent Evaluation on Test Set):**
- 293 tests passing. Do not regress.
- `agent_evaluation.csv` saved to `results/metrics/` with columns: policy, net_profit, revenue, imbalance_cost, etc.
- Agent policy name is "RL Agent (SAC)".
- Uses same `save_results()` function, so CSV column order is consistent with baseline_comparison.csv.

**From Story 3-4 (Baseline Evaluation Framework):**
- `baseline_comparison.csv` has 3 rows (Conservative, Aggressive, Price-Aware).
- Column order: policy, net_profit, revenue, imbalance_cost, degradation_cost, total_reward, delivered, committed, pv_produced, delivery_ratio, battery_cycles, hours, n_episodes.
- `print_comparison()` works with any list of result dicts containing 'policy' key.

**From Baseline Results:**
- Conservative: EUR 3,895 net profit/episode (best baseline)
- Aggressive: EUR -4,132 net profit/episode (negative -- loses money)
- Price-Aware: EUR 2,096 net profit/episode

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

None - clean implementation with no issues encountered.

### Completion Notes List

- Created `src/evaluation/compare_agent_baselines.py` implementing all 4 core functions (`load_results`, `calculate_improvement`, `identify_best_baseline`, `determine_verdict`) plus `main()` with CLI args
- Reused `print_comparison()` and `save_results()` from `src/evaluation/evaluate.py` as specified
- `calculate_improvement()` uses `abs(baseline_value)` in denominator to correctly handle negative baselines (Aggressive at -4132 EUR)
- Script loads pre-computed CSVs only, no SB3/torch imports, no re-evaluation
- User-friendly error messages when CSV files are missing (directs user to run the appropriate evaluation script)
- Output includes: formatted comparison table, improvement percentages, pass/fail verdict, best baseline identification
- Saves combined results with `improvement_pct` column to `results/metrics/agent_vs_baselines.csv`
- 13 new tests covering all functions, edge cases (negative baselines, zero baselines, equal values), and CSV output
- Full test suite: 306 passed (293 existing + 13 new), 0 failures, no regressions

### Change Log

- 2026-02-03: Story 5-2 implementation complete. Created comparison script and tests.
- 2026-02-03: Code review fixes applied (4 MEDIUM issues):
  - M1: `determine_verdict()` now returns PASS/FAIL_SOME/FAIL_NONE; verdict line prints "beats X of Y baselines"
  - M2: Added 5 new tests (4 integration tests for `main()` + 1 new verdict test); total now 18 tests
  - M3: `main()` now calls `sys.exit(1)` on missing files instead of silent return
  - M4: Eliminated in-place mutation of result dicts; `main()` builds new dicts for CSV saving

### File List

- `src/evaluation/compare_agent_baselines.py` (NEW, REVIEW-FIXED) - Agent vs baselines comparison script
- `tests/test_baseline_comparison.py` (NEW, REVIEW-FIXED) - 18 tests for comparison module
- `docs/implementation/5-2-baseline-comparison.md` (MODIFIED) - Story status updated to done
- `docs/implementation/sprint-status.yaml` (MODIFIED) - Story status updated to done
