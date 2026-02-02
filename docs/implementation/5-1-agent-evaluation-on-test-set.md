# Story 5.1: Agent Evaluation on Test Set

Status: ready-for-dev

## Story

As a developer,
I want to evaluate the trained RL agent on the test dataset,
so that I can measure out-of-sample performance with deterministic, reproducible metrics.

## Acceptance Criteria

1. **Given** a trained SAC model at `models/solar_merchant_final.zip` (or configurable path)
   **When** the evaluation script is run
   **Then** agent performance is measured on the 2022-2023 test data (`data/processed/test.csv`)
   **And** the agent uses `model.predict(obs, deterministic=True)` for all decisions

2. **Given** evaluation completes
   **When** results are reported
   **Then** revenue, imbalance_cost, net_profit are calculated per episode and averaged
   **And** additional metrics include: degradation_cost, total_reward, delivered, committed, pv_produced, delivery_ratio, battery_cycles, hours

3. **Given** a fixed seed (default 42)
   **When** evaluation is run twice with the same model and seed
   **Then** results are identical (deterministic)

4. **Given** evaluation results
   **When** output is generated
   **Then** a formatted summary prints to stdout
   **And** results are saved to `results/metrics/agent_evaluation.csv`

## Tasks / Subtasks

- [ ] Task 1: Create `src/evaluation/evaluate_agent.py` script (AC: #1, #2, #4)
  - [ ] Create `make_agent_policy(model) -> Callable` wrapper that adapts `model.predict(obs, deterministic=True)` to the `policy(obs) -> action` interface expected by `evaluate_policy()`
  - [ ] Create `main()` function that: loads model via `load_model()`, creates test env via `create_env()`, wraps model, calls `evaluate_policy()`, prints results, saves CSV
  - [ ] Accept `--model` CLI argument for checkpoint path (default: `models/solar_merchant_final.zip`)
  - [ ] Accept `--episodes` CLI argument (default: 10)
  - [ ] Accept `--seed` CLI argument (default: 42)
  - [ ] Print agent evaluation summary to stdout (name, metrics, episodes, seed)
  - [ ] Save results to `results/metrics/agent_evaluation.csv` using `save_results()`

- [ ] Task 2: Write tests in `tests/test_agent_evaluation.py` (AC: #1, #2, #3)
  - [ ] Test `make_agent_policy` is importable and callable
  - [ ] Test `make_agent_policy` returns a callable that accepts obs (np.ndarray) and returns action (np.ndarray) with shape (25,)
  - [ ] Test `main` function is importable
  - [ ] Test evaluation produces required metric keys (revenue, imbalance_cost, net_profit, etc.)
  - [ ] Test determinism: two runs with same seed produce identical results (using a mock or lightweight model)

- [ ] Task 3: Run full test suite to verify no regressions (AC: all)
  - [ ] Run `pytest tests/` -- all existing 282+ tests must pass
  - [ ] Run new tests -- all pass

## Dev Notes

### CRITICAL: Reuse Existing Code -- Do NOT Reinvent

The evaluation infrastructure is already built. Your job is to create a thin script that connects the trained model to the existing evaluation framework.

**REUSE these existing functions (DO NOT rewrite):**
- `evaluate_policy()` from `src/evaluation/evaluate.py` -- runs episodes, collects metrics, returns averaged dict
- `print_comparison()` from `src/evaluation/evaluate.py` -- formatted table output
- `save_results()` from `src/evaluation/evaluate.py` -- saves to CSV with correct column order
- `load_model()` from `src/training/train.py` -- loads SAC checkpoint, prints info
- `create_env()` from `src/training/train.py` -- creates env from CSV with plant config

### The One Key Piece: Policy Interface Adapter

`evaluate_policy()` expects: `policy: Callable[[np.ndarray], np.ndarray]`

SB3 model gives: `model.predict(obs, deterministic=True) -> tuple[np.ndarray, Any]`

You need a wrapper:

```python
def make_agent_policy(model) -> Callable[[np.ndarray], np.ndarray]:
    """Wrap an SB3 model into the policy interface for evaluate_policy().

    Args:
        model: Trained SB3 model with predict() method.

    Returns:
        Policy function: obs -> action.
    """
    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy
```

### Follow the evaluate_baselines.py Pattern

`src/evaluation/evaluate_baselines.py` is the exact template. Your script does the same thing but with the RL agent instead of rule-based baselines.

Key pattern from `evaluate_baselines.py`:
```python
env = SolarMerchantEnv(test_df.copy())
result = evaluate_policy(policy, env, n_episodes=10)
result["policy"] = name
save_results([result], output_path)
env.close()
```

### Script Structure: `src/evaluation/evaluate_agent.py`

```python
"""Evaluate trained RL agent on test dataset.

Loads a trained SAC model, evaluates on 2022-2023 test data,
prints metrics, and saves results to CSV.
"""
import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd

from src.environment import SolarMerchantEnv
from src.evaluation.evaluate import evaluate_policy, save_results
from src.training.train import load_model, PLANT_CONFIG

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
RESULTS_PATH = Path(__file__).parent.parent.parent / 'results' / 'metrics'
DEFAULT_MODEL = Path(__file__).parent.parent.parent / 'models' / 'solar_merchant_final.zip'


def make_agent_policy(model) -> Callable[[np.ndarray], np.ndarray]:
    # ... wrapper as above


def main() -> None:
    parser = argparse.ArgumentParser(...)
    # --model, --episodes, --seed
    # Load model, create test env, evaluate, print, save
```

### Import Patterns

Follow the project convention:
- `sys.path.insert` at top for project root (same as `evaluate_baselines.py` and `train.py`)
- Import from `src.evaluation.evaluate` for evaluation functions
- Import from `src.training.train` for `load_model` and `PLANT_CONFIG`
- Import `SolarMerchantEnv` from `src.environment`

### Test Approach

For testing `make_agent_policy` without needing a real trained SB3 model, create a mock:

```python
class MockModel:
    """Minimal mock that mimics SB3 model.predict()."""
    def predict(self, obs, deterministic=True):
        action = np.full(25, 0.5, dtype=np.float32)
        return action, None
```

This lets you test the wrapper and evaluate_policy integration without SB3 or a trained checkpoint.

### Test for Determinism (AC #3)

```python
def test_determinism(env):
    mock = MockModel()
    policy = make_agent_policy(mock)
    r1 = evaluate_policy(policy, env, n_episodes=2, seed=42)
    r2 = evaluate_policy(policy, env, n_episodes=2, seed=42)
    for key in r1:
        assert r1[key] == r2[key], f"{key} differs: {r1[key]} vs {r2[key]}"
```

### Output CSV Location

Save to `results/metrics/agent_evaluation.csv` (alongside existing `baseline_comparison.csv`).
Story 5-2 will later load both files for comparison.

### What NOT to Do

- Do NOT modify `evaluate.py`, `evaluate_baselines.py`, `train.py`, or baseline_policies.py
- Do NOT import SB3 at module level (follow deferred import pattern in `load_model`)
- Do NOT add VecNormalize wrapping (observations normalized internally by env)
- Do NOT create a new evaluation loop -- use `evaluate_policy()` as-is
- Do NOT run baselines in this script -- that's `evaluate_baselines.py` (and Story 5-2)

### Project Structure Notes

| Requirement | Location |
|-------------|----------|
| New script | `src/evaluation/evaluate_agent.py` (NEW) |
| New tests | `tests/test_agent_evaluation.py` (NEW) |
| Results output | `results/metrics/agent_evaluation.csv` |
| Model input | `models/solar_merchant_final.zip` (default) |
| Test data | `data/processed/test.csv` |

Naming follows architecture.md conventions: snake_case files, snake_case functions.

### References

- [Source: docs/epics.md#Story-5.1](../../docs/epics.md) -- FR30: Evaluate agent on test set, FR32: Calculate metrics
- [Source: docs/architecture.md#Evaluation-Architecture](../../docs/architecture.md) -- matplotlib visualization, mean+std over seeds, baseline interface
- [Source: docs/architecture.md#Cross-Component-Dependencies](../../docs/architecture.md) -- "Training saves -> Evaluation loads"
- [Source: docs/architecture.md#Module-Boundaries](../../docs/architecture.md) -- evaluation module: Model + Env -> Metrics dict
- [Source: src/evaluation/evaluate.py](../../src/evaluation/evaluate.py) -- evaluate_policy(), print_comparison(), save_results() -- REUSE
- [Source: src/evaluation/evaluate_baselines.py](../../src/evaluation/evaluate_baselines.py) -- Pattern template: load data, create env, run policies, save CSV
- [Source: src/training/train.py:88-111](../../src/training/train.py) -- load_model() helper, PLANT_CONFIG, create_env()
- [Source: docs/implementation/4-4-model-loading-and-resumption.md](../../docs/implementation/4-4-model-loading-and-resumption.md) -- 282 tests passing, load_model API, replay buffer not needed for eval
- [Source: results/metrics/baseline_comparison.csv](../../results/metrics/baseline_comparison.csv) -- Existing baseline results (Story 5-2 will compare against these)
- [Source: tests/conftest.py](../../tests/conftest.py) -- Shared env fixture using train.csv, suppress_expected_warnings

### Previous Story Intelligence

**From Story 4-4 (Model Loading and Resumption):**
- 282 tests passing. Do not regress.
- `load_model(checkpoint_path)` returns SAC model ready for `.predict()`. Import from `src.training.train`.
- SB3 import is deferred inside `load_model()` -- no module-level SB3 dependency.
- `PLANT_CONFIG` dict available for `create_env()` calls.
- Model saved at `models/solar_merchant_final.zip`.
- Code review found that `load_model()` docstring clarifies it loads for evaluation (predict) only.

**From Story 3-4 (Baseline Evaluation Framework):**
- `evaluate_policy()` is battle-tested with all 3 baselines.
- Returns dict with keys: revenue, imbalance_cost, net_profit, degradation_cost, total_reward, delivered, committed, pv_produced, delivery_ratio, battery_cycles, hours, n_episodes.
- `save_results()` enforces column order with policy name first.
- `print_comparison()` formats table and identifies best policy.

**From Baseline Results (existing CSV):**
- Conservative: EUR 3,895 net profit/episode
- Aggressive: EUR -4,132 net profit/episode (negative!)
- Price-Aware: EUR 2,096 net profit/episode
- Best baseline is Conservative at EUR 3,895

## Dev Agent Record

### Agent Model Used

{{agent_model_name_version}}

### Debug Log References

### Completion Notes List

### File List
