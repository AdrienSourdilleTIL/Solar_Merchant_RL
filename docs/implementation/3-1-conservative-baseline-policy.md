# Story 3.1: Conservative Baseline Policy

Status: review

## Story

As a developer,
I want a conservative baseline that commits a fraction of forecast,
so that I have a low-risk benchmark strategy.

## Acceptance Criteria

1. **Given** an observation from the environment
   **When** the conservative policy is called
   **Then** it commits 80% of forecast for each hour
   **And** battery is used to fill delivery gaps
   **And** action array is compatible with environment `step()` (25-dim float32 in [0, 1])
   **And** function follows the interface `def conservative_policy(obs: np.ndarray) -> np.ndarray`

2. **Given** the conservative policy is evaluated on the environment
   **When** a full 48-hour episode completes
   **Then** all commitment values are 80% of forecast
   **And** battery discharges when delivery falls short of commitment
   **And** battery charges when PV surplus exists above commitment
   **And** returned metrics (revenue, imbalance_cost, net_profit) are finite and reasonable

3. **Given** the baselines module
   **When** it is imported
   **Then** `src/baselines/__init__.py` exports `conservative_policy`
   **And** `src/baselines/baseline_policies.py` contains the implementation
   **And** type hints are present on all public functions (NFR4)
   **And** Google-style docstrings with Args/Returns sections are present (NFR14)

## Tasks / Subtasks

- [x] Task 1: Create baselines module structure (AC: #3)
  - [x] Create `src/baselines/__init__.py` with exports
  - [x] Create `src/baselines/baseline_policies.py` with module docstring and plant constants

- [x] Task 2: Implement `conservative_policy` function (AC: #1)
  - [x] Parse observation vector to extract PV forecast (indices 27-50) and battery SOC (index 1)
  - [x] Set commitment fractions to 0.8 for all 24 hours
  - [x] Implement battery logic: discharge (action[24] < 0.5) when delivery gap, charge (action[24] > 0.5) when surplus
  - [x] Return 25-dim np.float32 array in [0, 1]

- [x] Task 3: Create observation parsing helper (AC: #1, #2)
  - [x] Create `_parse_observation(obs: np.ndarray) -> dict` to extract named fields from 84-dim vector
  - [x] Extract: hour (idx 0), soc (idx 1), commitments (idx 2-25), imbalance (idx 26), forecast (idx 27-50), prices (idx 51-74), actual_pv (idx 75), weather (idx 76-77), time_features (idx 78-83)
  - [x] Private helper — not part of the public baseline interface

- [x] Task 4: Write tests for conservative policy (AC: #1, #2, #3)
  - [x] Test action shape is (25,) float32
  - [x] Test all action values in [0, 1] range
  - [x] Test commitment fractions are 0.8 (action[0:24] == 0.8)
  - [x] Test battery action responds to delivery gap (discharge when short)
  - [x] Test battery action responds to surplus (charge when excess PV)
  - [x] Test full episode runs without errors
  - [x] Test module imports correctly

- [x] Task 5: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` — all existing 141+ tests must pass (184 passed)
  - [x] Run new baseline tests (15 passed)

## Dev Notes

### CRITICAL: Greenfield Implementation

**This is a NEW module.** There is no existing `src/baselines/` directory. You must create it from scratch following the architecture spec exactly.

### Action Space Mapping

The environment expects a 25-dimensional action in [0, 1]:

| Index | Meaning | Conservative Strategy |
|-------|---------|----------------------|
| 0-23  | Commitment fractions per hour | **0.8** for all hours |
| 24    | Battery action | 0.0 = full discharge, 0.5 = idle, 1.0 = full charge |

**Battery action mapping in the environment:**
```python
battery_action = (action[24] - 0.5) * 2  # Converts [0,1] → [-1,1]
battery_target = battery_action * self.battery_power_mw  # ±5 MW
```

So:
- `action[24] = 0.0` → discharge at 5 MW
- `action[24] = 0.5` → idle
- `action[24] = 1.0` → charge at 5 MW

### Observation Parsing

The policy receives raw 84-dim observations. Key indices for conservative policy:

| Index | Dims | Field | Normalization |
|-------|------|-------|---------------|
| 0 | 1 | Hour | hour / 24.0 |
| 1 | 1 | Battery SOC | soc / 10.0 (capacity) |
| 2-25 | 24 | Today's commitments | commit / 20.0 (plant_cap) |
| 26 | 1 | Cumulative imbalance | imbalance / 20.0 |
| 27-50 | 24 | PV forecast 24h | forecast / 20.0 |
| 51-74 | 24 | Prices 24h | price / max_abs_price |
| 75 | 1 | Current actual PV | pv / 20.0 |

**Denormalization for the conservative policy is NOT needed.** The commitment fractions (action[0:24]) are interpreted by the environment as multipliers against `forecast + battery_power`. The policy simply outputs 0.8 for all hours — the environment handles the rest.

### Battery Heuristic for Conservative Policy

The battery logic should be simple:
1. **During commitment hours (hour == 11):** Just set commitment fractions to 0.8. The environment multiplies by `forecast + battery_power` to get the actual MWh commitment.
2. **Every hour:** Look at current imbalance and SOC to decide battery action:
   - If cumulative imbalance < 0 (under-delivering) AND SOC > 0: **discharge** (action[24] < 0.5)
   - If cumulative imbalance > 0 (over-delivering) OR surplus PV: **charge** (action[24] > 0.5)
   - Otherwise: **idle** (action[24] = 0.5)

**Key insight:** The commitment fractions (action[0:24]) are ONLY read by the environment at the commitment hour (hour 11). At all other hours, action[0:24] are ignored. But we still must output valid values. Output 0.8 every step to keep it simple.

### Function Signature

```python
def conservative_policy(obs: np.ndarray) -> np.ndarray:
    """Conservative baseline: commit 80% of forecast, use battery to fill gaps.

    Args:
        obs: 84-dimensional observation from SolarMerchantEnv.

    Returns:
        25-dimensional action array in [0, 1] range.
    """
```

### Plant Constants (for reference only — DO NOT hardcode denormalization)

```python
PLANT_CAPACITY_MW = 20.0
BATTERY_CAPACITY_MWH = 10.0
BATTERY_POWER_MW = 5.0
COMMITMENT_FRACTION = 0.8
```

### Module Structure

```
src/baselines/
├── __init__.py              # Export conservative_policy
└── baseline_policies.py     # Implementation of all baseline policies
```

`__init__.py` should export:
```python
from .baseline_policies import conservative_policy
```

Story 3.2 and 3.3 will add `aggressive_policy` and `price_aware_policy` to the same file. Design the module to accommodate future additions cleanly but do NOT implement them now.

### Testing Pattern

Follow the existing test pattern from Epic 2. Create `tests/test_baseline_policies.py`:

```python
import pytest
import numpy as np
from src.baselines import conservative_policy

class TestConservativePolicy:
    """Tests for conservative baseline policy."""

    def test_action_shape(self, env):
        obs, _ = env.reset(seed=42)
        action = conservative_policy(obs)
        assert action.shape == (25,)

    def test_action_range(self, env):
        obs, _ = env.reset(seed=42)
        action = conservative_policy(obs)
        assert np.all(action >= 0.0)
        assert np.all(action <= 1.0)

    # ... etc
```

Use the shared `env` fixture from `tests/conftest.py` — it creates a `SolarMerchantEnv` from training data.

### Architecture Compliance

| Requirement | How to Satisfy |
|-------------|----------------|
| File location | `src/baselines/baseline_policies.py` |
| Naming | snake_case functions, UPPER_SNAKE_CASE constants |
| Type hints | All public functions (Python 3.10+ style) |
| Docstrings | Google style with Args, Returns |
| Interface | `def conservative_policy(obs: np.ndarray) -> np.ndarray` |
| Import style | stdlib → third-party → local |
| No hardcoded paths | Not applicable (no file I/O in baselines) |

### Previous Story Intelligence

**From Story 2-5 (last completed story):**
- 35 tests, all passing. Full suite: 141+ tests.
- `conftest.py` provides shared `env` fixture — reuse it.
- Environment's `step()` correctly processes commitment at hour 11 and battery every hour.
- 48-hour episodes ensure agent sees commitment consequences.

**From Epic 2 general learnings:**
- Tests should be comprehensive but focused on behavior, not implementation details.
- The environment is stable and well-tested — trust its interfaces.
- `load_environment('data/processed/train.csv')` creates a ready-to-use env instance.

### Project Structure Notes

- `src/baselines/` does NOT exist yet — create it
- No conflicts with existing modules
- Follows architecture's module boundary: baselines depends only on numpy, takes observations as input, returns actions as output

### References

- [Source: docs/epics.md#Story-3.1](../../docs/epics.md) — Story requirements and acceptance criteria
- [Source: docs/architecture.md#Evaluation-Architecture](../../docs/architecture.md) — Baseline interface: `def conservative_policy(obs: np.ndarray) -> np.ndarray`
- [Source: docs/architecture.md#Module-Boundaries](../../docs/architecture.md) — Module: baselines, Input: Observations, Output: Actions, Dependencies: numpy
- [Source: docs/architecture.md#Project-Structure](../../docs/architecture.md) — `src/baselines/baseline_policies.py` for FR21-24
- [Source: src/environment/solar_merchant_env.py](../../src/environment/solar_merchant_env.py) — Action processing at commitment hour, battery mechanics
- [Source: docs/implementation/2-5-episode-flow-and-reset.md](../../docs/implementation/2-5-episode-flow-and-reset.md) — Episode flow, commitment hour, midnight transition
- [Source: CLAUDE.md#Action-Space](../../CLAUDE.md) — Action[0:24] commitment fractions, Action[24] battery

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Float32 precision: `assert_array_equal` failed for float32 0.8 vs float64 0.8 comparison — switched to `assert_allclose(atol=1e-7)`.

### Completion Notes List

- Created `src/baselines/` module from scratch (greenfield implementation).
- Implemented `conservative_policy()`: commits 80% of forecast every hour, uses battery heuristic (discharge when under-delivering with SOC > 0, charge when over-delivering, idle otherwise).
- Implemented `_parse_observation()` helper to extract named fields from 84-dim observation vector.
- 15 new tests covering: module imports, observation parsing, action shape/dtype/range, commitment fractions, battery heuristic (4 scenarios), full episode execution, and episode metrics.
- All 184 tests pass (169 existing + 15 new). Zero regressions.

### Change Log

- 2026-01-29: Story 3-1 implemented — conservative baseline policy with full test coverage.

### File List

- `src/baselines/__init__.py` (NEW) — Module init, exports `conservative_policy`
- `src/baselines/baseline_policies.py` (NEW) — Implementation of `conservative_policy`, `_parse_observation`, and plant constants
- `tests/test_baseline_policies.py` (NEW) — 15 tests for conservative policy
- `docs/implementation/3-1-conservative-baseline-policy.md` (MODIFIED) — Story status and task checkboxes updated
- `docs/implementation/sprint-status.yaml` (MODIFIED) — Story status updated
