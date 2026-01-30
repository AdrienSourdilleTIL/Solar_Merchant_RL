# Story 3.2: Aggressive Baseline Policy

Status: done

## Story

As a developer,
I want an aggressive baseline that maximizes commitment,
so that I have a high-risk/high-reward benchmark.

## Acceptance Criteria

1. **Given** an observation from the environment
   **When** the aggressive policy is called
   **Then** it commits 100% of forecast plus battery discharge capacity
   **And** action array is compatible with environment `step()` (25-dim float32 in [0, 1])
   **And** function follows the interface `def aggressive_policy(obs: np.ndarray) -> np.ndarray`

2. **Given** the aggressive policy is evaluated on the environment
   **When** a full 48-hour episode completes
   **Then** all commitment values represent maximum possible delivery (fraction = 1.0)
   **And** battery discharges aggressively to meet commitments
   **And** battery charges only when PV surplus exists and SOC is not full
   **And** returned metrics (revenue, imbalance_cost, net_profit) are finite and reasonable

3. **Given** the baselines module
   **When** it is imported
   **Then** `src/baselines/__init__.py` exports `aggressive_policy`
   **And** `src/baselines/baseline_policies.py` contains the implementation
   **And** type hints are present on all public functions (NFR4)
   **And** Google-style docstrings with Args/Returns sections are present (NFR14)

## Tasks / Subtasks

- [x] Task 1: Implement `aggressive_policy` function in `src/baselines/baseline_policies.py` (AC: #1)
  - [x] Add `AGGRESSIVE_FRACTION = 1.0` constant alongside existing `COMMITMENT_FRACTION`
  - [x] Implement `aggressive_policy(obs: np.ndarray) -> np.ndarray` function
  - [x] Set commitment fractions to 1.0 for all 24 hours (maximum delivery)
  - [x] Implement battery logic: discharge aggressively to meet commitments, charge only on PV surplus
  - [x] Return 25-dim np.float32 array in [0, 1]

- [x] Task 2: Update module exports (AC: #3)
  - [x] Add `aggressive_policy` import in `src/baselines/__init__.py`
  - [x] Add to `__all__` list

- [x] Task 3: Write tests for aggressive policy (AC: #1, #2, #3)
  - [x] Add `TestAggressivePolicy` class(es) to `tests/test_baseline_policies.py`
  - [x] Test action shape is (25,) float32
  - [x] Test all action values in [0, 1] range
  - [x] Test commitment fractions are 1.0 (action[0:24] == 1.0)
  - [x] Test battery action: discharges by default (action[24] near 0.0) to maximize delivery
  - [x] Test battery charges when PV surplus exists
  - [x] Test full 48-hour episode runs without errors
  - [x] Test module import of `aggressive_policy`

- [x] Task 4: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` — all existing 186+ tests must still pass
  - [x] Run new aggressive policy tests

## Dev Notes

### CRITICAL: This is an ADDITION to existing module

**DO NOT** create new files. `src/baselines/baseline_policies.py` and `src/baselines/__init__.py` already exist from Story 3-1. You are ADDING a function to the existing file.

### Action Space Mapping (same as conservative)

The environment expects a 25-dimensional action in [0, 1]:

| Index | Meaning | Aggressive Strategy |
|-------|---------|---------------------|
| 0-23  | Commitment fractions per hour | **1.0** for all hours |
| 24    | Battery action | 0.0 = full discharge, 0.5 = idle, 1.0 = full charge |

**Battery action mapping in the environment:**
```python
battery_action = (action[24] - 0.5) * 2  # Converts [0,1] -> [-1,1]
battery_target = battery_action * self.battery_power_mw  # +/-5 MW
```

### Aggressive Strategy Logic

The aggressive policy maximizes revenue by committing the maximum possible:

1. **Commitment fractions = 1.0** for all 24 hours. The environment computes actual commitment as `fraction * (forecast + battery_power)`, so 1.0 commits the maximum — all forecast plus full battery discharge capacity.

2. **Battery heuristic — prioritize discharge to meet commitments:**
   - Default: **discharge** (action[24] = 0.0) — the aggressive strategy assumes battery will be needed to meet the high commitment
   - If PV surplus exists above commitment AND SOC is not full: **charge** (action[24] = 1.0) — store surplus for later
   - If SOC is empty (soc == 0): **idle** (action[24] = 0.5) — can't discharge an empty battery

   This is the OPPOSITE of conservative: conservative defaults to idle and only discharges on shortfall. Aggressive defaults to discharge and only charges on surplus.

### Reuse `_parse_observation` Helper

The `_parse_observation()` helper already exists in `baseline_policies.py` from Story 3-1. Reuse it directly — DO NOT duplicate or recreate it.

### Observation Parsing Reference

Same as conservative (from `_parse_observation`):

| Index | Dims | Field | Normalization |
|-------|------|-------|---------------|
| 0 | 1 | Hour | hour / 24.0 |
| 1 | 1 | Battery SOC | soc / 10.0 (capacity) |
| 2-25 | 24 | Today's commitments | commit / 20.0 (plant_cap) |
| 26 | 1 | Cumulative imbalance | imbalance / 20.0 |
| 27-50 | 24 | PV forecast 24h | forecast / 20.0 |
| 51-74 | 24 | Prices 24h | price / max_abs_price |
| 75 | 1 | Current actual PV | pv / 20.0 |

**No denormalization needed.** Commitment fractions are interpreted by the environment as multipliers against `forecast + battery_power`.

### Function Signature

```python
def aggressive_policy(obs: np.ndarray) -> np.ndarray:
    """Aggressive baseline: commit 100% of max capacity, discharge battery aggressively.

    Args:
        obs: 84-dimensional observation from SolarMerchantEnv.

    Returns:
        25-dimensional action array in [0, 1] range (float32).
    """
```

### Key Difference from Conservative

| Aspect | Conservative (3-1) | Aggressive (3-2) |
|--------|-------------------|------------------|
| Commitment | 0.8 (80% of max) | 1.0 (100% of max) |
| Battery default | Idle (0.5) | Discharge (0.0) |
| Battery discharge trigger | Under-delivering (imbalance < 0) | Always (default) |
| Battery charge trigger | Over-delivering OR PV surplus | PV surplus only |
| Risk profile | Low risk, lower revenue | High risk, higher revenue |

### Testing Pattern

Follow the EXACT same pattern as `TestConservativePolicy*` classes in `tests/test_baseline_policies.py`. Add new classes in the SAME file:

```python
class TestAggressivePolicyAction:
    """Tests for aggressive policy action output."""

    def test_action_shape(self, env):
        obs, _ = env.reset(seed=42)
        action = aggressive_policy(obs)
        assert action.shape == (25,)

    # ... etc
```

Use the shared `env` fixture from `tests/conftest.py`.

### Architecture Compliance

| Requirement | How to Satisfy |
|-------------|----------------|
| File location | ADD to existing `src/baselines/baseline_policies.py` |
| Naming | snake_case function, UPPER_SNAKE_CASE constant |
| Type hints | All public functions (Python 3.10+ style) |
| Docstrings | Google style with Args, Returns |
| Interface | `def aggressive_policy(obs: np.ndarray) -> np.ndarray` |
| Module export | Add to `src/baselines/__init__.py` and `__all__` |

### Previous Story Intelligence

**From Story 3-1 (Conservative Baseline):**
- `_parse_observation()` helper is already implemented — reuse it.
- Plant constants (`PLANT_CAPACITY_MW`, `BATTERY_CAPACITY_MWH`, `BATTERY_POWER_MW`) are already defined — reuse them.
- Float32 precision: use `np.testing.assert_allclose(atol=1e-7)` instead of `assert_array_equal` for float comparisons (lesson learned in 3-1).
- 186 tests currently passing (169 Epic 2 + 17 baseline tests). Do not regress.
- The `env` fixture from `tests/conftest.py` is shared — use it for episode tests.

**From Story 3-1 code review:**
- Battery heuristic must handle the "SOC is empty" edge case (can't discharge from empty battery).
- PV surplus detection pattern: `parsed["actual_pv"] > parsed["commitments"][hour_idx]`.
- Hour index computation: `int(round(parsed["hour"] * 24)) % 24`.

### Project Structure Notes

- `src/baselines/baseline_policies.py` — ADD `aggressive_policy` function here (below `conservative_policy`)
- `src/baselines/__init__.py` — ADD import and `__all__` entry
- `tests/test_baseline_policies.py` — ADD test classes (do NOT create a new test file)
- No new files should be created

### References

- [Source: docs/epics.md#Story-3.2](../../docs/epics.md) — Story requirements: commits 100% of forecast + battery discharge capacity
- [Source: docs/architecture.md#Key-Interfaces](../../docs/architecture.md) — Baseline interface: `def policy(obs) -> action`
- [Source: docs/architecture.md#Module-Boundaries](../../docs/architecture.md) — baselines module: Input: Observations, Output: Actions, Dependencies: numpy
- [Source: src/baselines/baseline_policies.py](../../src/baselines/baseline_policies.py) — Existing module with conservative_policy, _parse_observation, plant constants
- [Source: docs/implementation/3-1-conservative-baseline-policy.md](../../docs/implementation/3-1-conservative-baseline-policy.md) — Previous story learnings, float32 precision, battery edge cases
- [Source: CLAUDE.md#Action-Space](../../CLAUDE.md) — Action[0:24] commitment fractions, Action[24] battery

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

No debug issues encountered.

### Completion Notes List

- Implemented `aggressive_policy` in existing `baseline_policies.py`, reusing `_parse_observation` helper and plant constants from Story 3-1.
- Added `AGGRESSIVE_FRACTION = 1.0` constant.
- Battery heuristic: discharge by default (action[24]=0.0), charge on PV surplus when SOC < 1.0, idle when SOC == 0.
- Updated `__init__.py` to export `aggressive_policy` in `__all__`.
- Added 15 new tests across 4 test classes (TestAggressivePolicyImport, TestAggressivePolicyAction, TestAggressivePolicyBattery, TestAggressivePolicyEpisode).
- All 202 tests pass (186 existing + 16 new), zero regressions.
- Type hints and Google-style docstrings present on all public functions.

### Change Log

- 2026-01-30: Implemented aggressive baseline policy (Story 3-2) — all 4 tasks complete, 201/201 tests passing.
- 2026-01-30: Code review fixes — Fixed battery logic ordering bug (SOC=0 blocked charging on surplus), replaced float equality with tolerance, added missing edge case test, updated test docstring, updated File List.

### File List

- `src/baselines/baseline_policies.py` (modified) — Added `AGGRESSIVE_FRACTION` constant and `aggressive_policy` function
- `src/baselines/__init__.py` (modified) — Added `aggressive_policy` import and `__all__` entry
- `tests/test_baseline_policies.py` (modified) — Added 16 tests in 4 new test classes for aggressive policy
- `docs/implementation/sprint-status.yaml` (modified) — Updated story status tracking
