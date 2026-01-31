# Story 3.4: Baseline Evaluation Framework

Status: done

## Story

As a developer,
I want to evaluate any policy and report metrics,
so that I can compare different strategies fairly.

## Acceptance Criteria

1. **Given** a policy function and environment
   **When** `evaluate_policy(policy, env, n_episodes)` is called
   **Then** policy runs for `n_episodes` complete episodes
   **And** returns dict with keys: `revenue`, `imbalance_cost`, `net_profit`
   **And** metrics are averaged across episodes
   **And** function follows the interface `def evaluate_policy(policy, env, n_episodes: int) -> dict[str, float]`

2. **Given** evaluation results
   **When** results are printed
   **Then** output includes a formatted comparison table with policy name, net profit, imbalance cost, and delivery ratio
   **And** results are optionally saved to CSV file

3. **Given** the evaluation module
   **When** it is imported
   **Then** `src/evaluation/__init__.py` exports `evaluate_policy`
   **And** `src/evaluation/evaluate.py` contains the implementation
   **And** type hints are present on all public functions (NFR4)
   **And** Google-style docstrings with Args/Returns sections are present (NFR14)

4. **Given** all three baseline policies and the evaluation framework
   **When** `evaluate_baselines.py` is run as a script
   **Then** all three baselines (conservative, aggressive, price-aware) are evaluated on test data
   **And** a comparison table is printed
   **And** results CSV is saved to `results/metrics/`

## Tasks / Subtasks

- [x] Task 1: Create `src/evaluation/` module structure (AC: #3)
  - [x] Create `src/evaluation/__init__.py` with `evaluate_policy` export
  - [x] Create `src/evaluation/evaluate.py` with module docstring

- [x] Task 2: Implement `evaluate_policy` function (AC: #1)
  - [x] Implement `evaluate_policy(policy, env, n_episodes: int, seed: int = 42) -> dict[str, float]`
  - [x] Loop: reset env with seed, run policy until episode terminates, collect per-episode metrics from `info` dict
  - [x] Increment seed per episode for diversity (`seed + i` for episode `i`)
  - [x] Accumulate per-episode: revenue, imbalance_cost, net_profit, delivered, committed, pv_actual, battery_throughput
  - [x] Compute averages across episodes for all metrics
  - [x] Return dict with averaged metrics (see Metrics Dict section below)

- [x] Task 3: Implement `print_comparison` function (AC: #2)
  - [x] Implement `print_comparison(results: list[dict[str, float]]) -> None`
  - [x] Print formatted table: policy name, net profit, imbalance cost, delivery ratio
  - [x] Identify and highlight best policy by net profit

- [x] Task 4: Implement `save_results` function (AC: #2)
  - [x] Implement `save_results(results: list[dict[str, float]], output_path: Path) -> None`
  - [x] Save to CSV using pandas DataFrame
  - [x] Create output directory if it doesn't exist

- [x] Task 5: Create `evaluate_baselines.py` script (AC: #4)
  - [x] Create `src/evaluation/evaluate_baselines.py` with `main()` entry point
  - [x] Load test data from `data/processed/test.csv`
  - [x] Import all three baselines from `src.baselines`
  - [x] Evaluate each on the test environment
  - [x] Print comparison and save to `results/metrics/baseline_comparison.csv`

- [x] Task 6: Write tests (AC: #1, #2, #3)
  - [x] Create `tests/test_evaluation.py`
  - [x] Test `evaluate_policy` returns correct dict keys
  - [x] Test metrics are finite and reasonable (non-negative revenue, non-negative imbalance_cost)
  - [x] Test multi-episode averaging produces consistent results
  - [x] Test n_episodes=1 works
  - [x] Test with each baseline policy (conservative, aggressive, price_aware)
  - [x] Test `print_comparison` runs without error
  - [x] Test `save_results` creates CSV file
  - [x] Test module import of `evaluate_policy`

- [x] Task 7: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` — all existing 221+ tests must pass
  - [x] Run new evaluation tests

## Dev Notes

### CRITICAL: This is a NEW module

`src/evaluation/` does NOT exist yet. Create it from scratch following the architecture spec. This is the evaluation module defined in the architecture for FR30-34, and this story lays its foundation with FR24.

### Architecture Placement

Per architecture, evaluation lives in `src/evaluation/`:
```
src/evaluation/
├── __init__.py           # Export evaluate_policy
├── evaluate.py           # FR24: Generic policy evaluation (THIS STORY)
└── evaluate_baselines.py # Script to run all baselines
```

Future stories (Epic 5) will add:
- `evaluate.py` additions for agent evaluation (FR30-34)
- `visualize.py` for charts (FR35-37)

### Policy Interface

The function-based baseline policies follow this interface:
```python
def policy(obs: np.ndarray) -> np.ndarray
```

**Input:** 84-dimensional observation vector (normalized)
**Output:** 25-dimensional action array (float32, in [0, 1])

The `evaluate_policy` function must work with ANY callable matching this signature. Use `Callable[[np.ndarray], np.ndarray]` as the type hint for the policy parameter.

### DO NOT Use Class-Based Policies

There is a legacy `src/training/baseline_rules.py` with class-based policies (`ConservativePolicy`, etc.) that have a different interface: `policy.predict(obs, env)`. **Ignore this entirely.** Story 3-1 through 3-3 created function-based policies in `src/baselines/baseline_policies.py` — those are the canonical implementations.

### DO NOT Reuse `src/training/evaluate.py`

There is a legacy `src/training/evaluate.py` with `evaluate_agent()` for SB3 models. It uses `model.predict(obs, deterministic=True)` — different interface. **Do not import from or depend on this file.** Build the evaluation module fresh.

### `evaluate_policy` Function Signature

```python
from typing import Callable
from pathlib import Path
import numpy as np

def evaluate_policy(
    policy: Callable[[np.ndarray], np.ndarray],
    env: gym.Env,
    n_episodes: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate a policy over multiple episodes and return averaged metrics.

    Runs the policy for n_episodes complete episodes, collecting financial
    and operational metrics from the environment's info dict at each step.

    Args:
        policy: Function mapping observation to action array.
        env: Gymnasium environment instance.
        n_episodes: Number of episodes to evaluate.
        seed: Base random seed (incremented per episode).

    Returns:
        Dict with averaged metrics:
            - revenue: Mean total revenue per episode (EUR)
            - imbalance_cost: Mean total imbalance cost per episode (EUR)
            - net_profit: Mean net profit per episode (EUR)
            - delivered: Mean total delivered energy per episode (MWh)
            - committed: Mean total committed energy per episode (MWh)
            - pv_produced: Mean total PV produced per episode (MWh)
            - delivery_ratio: Mean delivered/committed ratio
            - battery_cycles: Mean battery cycles per episode
            - hours: Mean episode length in hours
    """
```

### Metrics Dict — Required Keys

The return dict MUST include at minimum:
```python
{
    "revenue": float,          # Mean total revenue per episode
    "imbalance_cost": float,   # Mean total imbalance cost per episode
    "net_profit": float,       # Mean net profit per episode (revenue - imbalance_cost)
    "delivered": float,        # Mean total MWh delivered per episode
    "committed": float,        # Mean total MWh committed per episode
    "pv_produced": float,      # Mean total PV production per episode
    "delivery_ratio": float,   # Mean delivered/committed ratio
    "battery_cycles": float,   # Mean battery full-equivalent cycles per episode
    "hours": float,            # Mean episode length
}
```

### Episode Loop Structure

```python
for ep in range(n_episodes):
    obs, _ = env.reset(seed=seed + ep)
    episode_revenue = 0.0
    episode_imbalance = 0.0
    # ... etc

    done = False
    steps = 0
    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_revenue += info['revenue']
        episode_imbalance += info['imbalance_cost']
        # ... accumulate from info dict
        steps += 1

    # Store episode totals
    # ...

# Average across episodes
```

### Environment `info` Dict Keys

The environment's `step()` returns an `info` dict with these keys (all available every step):
```python
info = {
    'revenue': float,           # Revenue this hour (EUR)
    'imbalance_cost': float,    # Imbalance cost this hour (EUR)
    'delivered': float,         # Energy delivered this hour (MWh)
    'committed': float,         # Energy committed this hour (MWh)
    'pv_actual': float,         # Actual PV production this hour (MWh)
    'battery_throughput': float,# Battery throughput this hour (MWh)
    'hour': int,                # Current hour of day
    'price': float,             # Day-ahead price (EUR/MWh)
    'imbalance': float,         # Imbalance this hour (MWh, + = long, - = short)
    'battery_soc': float,       # Battery state of charge (MWh)
}
```

### Battery Cycles Calculation

```python
battery_cycles = total_battery_throughput / (2 * BATTERY_CAPACITY_MWH)
```
Where `BATTERY_CAPACITY_MWH = 10.0`. Throughput counts both charge and discharge, so 2× capacity = 1 full cycle.

### `print_comparison` Output Format

Follow the pattern from `src/training/baseline_rules.py:print_comparison()` but adapted for function-based policies:

```
================================================================================
BASELINE POLICY COMPARISON
================================================================================

Policy                          Net Profit   Imb Cost  Deliv %  Episodes
--------------------------------------------------------------------------------
Conservative (80%)              EUR  12,345  EUR 1,234   95.2%        10
Aggressive (100%)               EUR  15,678  EUR 3,456   88.7%        10
Price-Aware                     EUR  14,567  EUR 2,345   91.3%        10
--------------------------------------------------------------------------------

Best policy: Aggressive (100%)
Mean net profit: EUR 15,678 per episode
```

Policy names: Use a `name` parameter on `evaluate_policy` OR derive from function `__name__`. Prefer passing explicit names in `evaluate_baselines.py` to keep it clean.

### `evaluate_baselines.py` Script Structure

```python
def main():
    # Load test data
    test_df = pd.read_csv(DATA_PATH / 'test.csv', parse_dates=['datetime'])

    # Import policies
    from src.baselines import conservative_policy, aggressive_policy, price_aware_policy

    policies = [
        ("Conservative (80%)", conservative_policy),
        ("Aggressive (100%)", aggressive_policy),
        ("Price-Aware", price_aware_policy),
    ]

    results = []
    for name, policy in policies:
        env = SolarMerchantEnv(test_df.copy(), **PLANT_CONFIG)
        result = evaluate_policy(policy, env, n_episodes=10)
        result["policy"] = name
        results.append(result)
        env.close()

    print_comparison(results)
    save_results(results, RESULTS_PATH / 'baseline_comparison.csv')
```

### Path Constants

```python
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results' / 'metrics'
```

### Import Pattern

Follow architecture import conventions:
```python
# Standard library
from pathlib import Path
from typing import Callable

# Third-party
import numpy as np
import pandas as pd
import gymnasium as gym

# Local
from src.environment import load_environment
from src.baselines import conservative_policy, aggressive_policy, price_aware_policy
```

For the script (`evaluate_baselines.py`), use `sys.path` manipulation like existing scripts in `src/training/`:
```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Testing Pattern

Create `tests/test_evaluation.py` following the same patterns as `tests/test_baseline_policies.py`:

```python
import pytest
import numpy as np
from src.evaluation import evaluate_policy
from src.baselines import conservative_policy, aggressive_policy, price_aware_policy


class TestEvaluatePolicyImport:
    """Tests for evaluation module imports."""

    def test_import_evaluate_policy(self):
        from src.evaluation import evaluate_policy
        assert callable(evaluate_policy)


class TestEvaluatePolicyOutput:
    """Tests for evaluate_policy return value."""

    def test_returns_dict(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert isinstance(result, dict)

    def test_required_keys(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        required_keys = {"revenue", "imbalance_cost", "net_profit", "delivered",
                         "committed", "pv_produced", "delivery_ratio",
                         "battery_cycles", "hours"}
        assert required_keys.issubset(result.keys())

    def test_metrics_finite(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        for key, value in result.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"

    def test_revenue_non_negative(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        assert result["revenue"] >= 0

    def test_net_profit_equals_revenue_minus_cost(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=1)
        expected = result["revenue"] - result["imbalance_cost"]
        np.testing.assert_allclose(result["net_profit"], expected, atol=1e-6)


class TestEvaluatePolicyMultiEpisode:
    """Tests for multi-episode evaluation."""

    def test_multi_episode(self, env):
        result = evaluate_policy(conservative_policy, env, n_episodes=3)
        assert isinstance(result, dict)
        assert result["hours"] > 0

    def test_all_baselines(self, env):
        for policy in [conservative_policy, aggressive_policy, price_aware_policy]:
            result = evaluate_policy(policy, env, n_episodes=2)
            assert result["net_profit"] != 0  # Should have some activity
```

Use the shared `env` fixture from `tests/conftest.py`.

### Architecture Compliance

| Requirement | How to Satisfy |
|-------------|----------------|
| File location | `src/evaluation/evaluate.py` (NEW module) |
| Script location | `src/evaluation/evaluate_baselines.py` (NEW) |
| Naming | snake_case functions, UPPER_SNAKE_CASE constants |
| Type hints | All public functions (Python 3.10+ `dict[str, float]`, `Callable`) |
| Docstrings | Google style with Args, Returns |
| Interface | `def evaluate_policy(policy, env, n_episodes) -> dict[str, float]` |
| Module export | `src/evaluation/__init__.py` exports `evaluate_policy` |
| Results output | `results/metrics/` directory |

### Previous Story Intelligence

**From Story 3-3 (Price-Aware Baseline):**
- All three function-based baselines are complete and tested (221 tests passing).
- Baselines exported from `src/baselines`: `conservative_policy`, `aggressive_policy`, `price_aware_policy`.
- Each returns 25-dim float32 action array in [0, 1].
- `_parse_observation()` helper handles observation decoding — not needed here (evaluation doesn't parse obs).

**From Story 3-2 code review:**
- Float32 precision: use `np.testing.assert_allclose(atol=1e-6)` for metric comparisons.
- 221 tests currently passing. Do not regress.

**From Story 3-1:**
- `conftest.py` provides shared `env` fixture — reuse it for evaluation tests.
- Environment `step()` returns proper `info` dict with all needed metrics.
- 48-hour episodes: each episode is 48 hours (commitment at hour 11, consequences visible before termination).

**From Epic 2 (Environment):**
- `env.reset(seed=N)` starts at a deterministic position given the seed.
- `terminated or truncated` signals episode end.
- `load_environment('data/processed/train.csv')` creates a ready env instance.

**From existing legacy code:**
- `src/training/baseline_rules.py` has a `print_comparison` function — use as reference for formatting, but implement independently in the new module.
- `src/training/evaluate.py` has `evaluate_agent` — different interface, don't import.

### Edge Cases

1. **n_episodes=1**: Should work correctly, returning metrics from a single episode (no averaging needed, but code path should handle it).
2. **Short episodes**: If an episode terminates very quickly (unlikely but possible), delivery_ratio could be 0/0. Guard: `delivery_ratio = delivered / max(committed, 1e-8)`.
3. **Zero PV hours**: Nighttime hours have zero PV — revenue and delivery will be zero. This is normal.

### Project Structure Notes

Files to create:
- `src/evaluation/__init__.py` — NEW: Module init, export `evaluate_policy`
- `src/evaluation/evaluate.py` — NEW: `evaluate_policy`, `print_comparison`, `save_results`
- `src/evaluation/evaluate_baselines.py` — NEW: Script to evaluate all baselines
- `tests/test_evaluation.py` — NEW: Test suite for evaluation module

No existing files should be modified (except sprint-status.yaml).

### References

- [Source: docs/epics.md#Story-3.4](../../docs/epics.md) — Story requirements: evaluate_policy(policy, env, n_episodes) -> dict
- [Source: docs/architecture.md#Key-Interfaces](../../docs/architecture.md) — Evaluation interface: `def evaluate_policy(policy, env, n_episodes: int) -> dict[str, float]`
- [Source: docs/architecture.md#Module-Boundaries](../../docs/architecture.md) — evaluation module: Input: Model + Env, Output: Metrics dict
- [Source: docs/architecture.md#Project-Structure](../../docs/architecture.md) — `src/evaluation/evaluate.py` for FR30-34
- [Source: src/baselines/baseline_policies.py](../../src/baselines/baseline_policies.py) — Three function-based policies to evaluate
- [Source: src/baselines/__init__.py](../../src/baselines/__init__.py) — Exports: conservative_policy, aggressive_policy, price_aware_policy
- [Source: src/environment/solar_merchant_env.py](../../src/environment/solar_merchant_env.py) — Environment step() info dict with revenue, imbalance_cost, delivered, committed, pv_actual, battery_throughput
- [Source: src/training/baseline_rules.py](../../src/training/baseline_rules.py) — Legacy evaluate_policy and print_comparison (reference only, do NOT import)
- [Source: docs/implementation/3-3-price-aware-baseline-policy.md](../../docs/implementation/3-3-price-aware-baseline-policy.md) — 221 tests passing, all three baselines complete
- [Source: docs/implementation/3-1-conservative-baseline-policy.md](../../docs/implementation/3-1-conservative-baseline-policy.md) — conftest.py env fixture, 48h episodes, float precision lesson
- [Source: CLAUDE.md#Architecture](../../CLAUDE.md) — Evaluation interface and module structure

## Change Log

- 2026-01-31: Implemented baseline evaluation framework — all 7 tasks completed, 20 new tests added, 241 total tests passing (0 regressions)
- 2026-01-31: Code review — 5 issues fixed (1 HIGH, 4 MEDIUM), 8 new tests added, 249 total tests passing (0 regressions)

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

No issues encountered during implementation.

### Completion Notes List

- Task 1: Created `src/evaluation/` module with `__init__.py` exporting `evaluate_policy` and `evaluate.py` with module docstring.
- Task 2: Implemented `evaluate_policy(policy, env, n_episodes, seed)` — loops over episodes with incremented seeds, accumulates revenue/imbalance_cost/delivered/committed/pv_actual/battery_throughput from env info dict, computes delivery_ratio and battery_cycles, returns averaged metrics dict with all 9 required keys.
- Task 3: Implemented `print_comparison(results)` — formatted table with policy name, net profit, imbalance cost, delivery %, episode count. Identifies and highlights best policy by net profit.
- Task 4: Implemented `save_results(results, output_path)` — saves to CSV via pandas DataFrame, creates output directory if needed.
- Task 5: Created `evaluate_baselines.py` script with `main()` entry point — loads test.csv, imports all 3 baselines from `src.baselines`, evaluates each with 10 episodes, prints comparison, saves CSV to `results/metrics/baseline_comparison.csv`.
- Task 6: Created `tests/test_evaluation.py` with 20 tests across 6 test classes covering imports, output keys, metric validity, multi-episode averaging, all 3 baselines, print_comparison output, and save_results CSV creation.
- Task 7: Full test suite: 241 passed (221 existing + 20 new), 0 failures, 0 regressions.

### File List

- `src/evaluation/__init__.py` — NEW: Module init, exports evaluate_policy
- `src/evaluation/evaluate.py` — NEW: evaluate_policy, print_comparison, save_results (MODIFIED in review: added input validation, degradation_cost/total_reward/n_episodes tracking, fixed print_comparison episodes, fixed CSV column order)
- `src/evaluation/evaluate_baselines.py` — NEW: Script to evaluate all baselines (MODIFIED in review: removed manual episodes injection)
- `tests/test_evaluation.py` — NEW: 28 tests for evaluation module (8 added in review: input validation, degradation tracking, CSV column order, baselines script integration)
- `docs/implementation/sprint-status.yaml` — MODIFIED: 3-4 status updated to done
- `docs/implementation/3-4-baseline-evaluation-framework.md` — MODIFIED: Tasks marked complete, review notes added

## Senior Developer Review (AI)

**Reviewer:** Claude Opus 4.5 (code-review workflow)
**Date:** 2026-01-31
**Outcome:** Approved (after fixes)

### Issues Found: 1 HIGH, 4 MEDIUM, 3 LOW

All HIGH and MEDIUM issues were auto-fixed. LOW issues deferred.

| ID | Severity | Description | Status |
|----|----------|-------------|--------|
| H1 | HIGH | `n_episodes=0` crashes with IndexError — added input validation | FIXED |
| M1 | MEDIUM | `reward` from env.step() discarded — added `total_reward` and `degradation_cost` tracking | FIXED |
| M2 | MEDIUM | `print_comparison` episodes column used fragile mixed-type fallback — added `n_episodes` to return dict | FIXED |
| M3 | MEDIUM | `evaluate_baselines.py` had zero test coverage — added import + integration tests | FIXED |
| M4 | MEDIUM | `save_results` didn't control CSV column order — added preferred column ordering | FIXED |
| L1 | LOW | `print_comparison` and `save_results` not exported from `__init__.py` | DEFERRED |
| L2 | LOW | No `__all__` defined in `evaluate.py` | DEFERRED |
| L3 | LOW | `delivery_ratio` averages mean-of-ratios instead of ratio-of-means | DEFERRED |

### Test Results After Fixes

- **249 tests passed** (221 existing + 28 evaluation)
- **0 failures, 0 regressions**
