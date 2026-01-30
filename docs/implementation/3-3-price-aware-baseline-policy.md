# Story 3.3: Price-Aware Baseline Policy

Status: done

## Story

As a developer,
I want a price-aware baseline that adjusts commitment based on price levels,
so that I have a smarter rule-based benchmark.

## Acceptance Criteria

1. **Given** an observation from the environment
   **When** the price-aware policy is called
   **Then** commitment fractions vary per hour based on price level relative to the 24h price window
   **And** hours with above-median prices get high commitment (1.0)
   **And** hours with below-median prices get low commitment (0.5)
   **And** action array is compatible with environment `step()` (25-dim float32 in [0, 1])
   **And** function follows the interface `def price_aware_policy(obs: np.ndarray) -> np.ndarray`

2. **Given** the price-aware policy is evaluated on the environment
   **When** a full 48-hour episode completes
   **Then** commitment fractions are NOT uniform across all 24 hours (unlike conservative/aggressive)
   **And** battery discharges during high-price hours to maximize revenue
   **And** battery charges during low-price hours to store energy
   **And** returned metrics (revenue, imbalance_cost, net_profit) are finite and reasonable

3. **Given** the baselines module
   **When** it is imported
   **Then** `src/baselines/__init__.py` exports `price_aware_policy`
   **And** `src/baselines/baseline_policies.py` contains the implementation
   **And** type hints are present on all public functions (NFR4)
   **And** Google-style docstrings with Args/Returns sections are present (NFR14)

## Tasks / Subtasks

- [x] Task 1: Implement `price_aware_policy` function in `src/baselines/baseline_policies.py` (AC: #1)
  - [x] Add `PRICE_AWARE_HIGH_FRACTION = 1.0` and `PRICE_AWARE_LOW_FRACTION = 0.5` constants
  - [x] Implement `price_aware_policy(obs: np.ndarray) -> np.ndarray` function
  - [x] Compute median of 24h price window from observation (indices 51-74)
  - [x] Set per-hour commitment fractions: 1.0 for hours above median, 0.5 for hours at/below median
  - [x] Implement battery logic: discharge during high-price hours, charge during low-price hours
  - [x] Return 25-dim np.float32 array in [0, 1]

- [x] Task 2: Update module exports (AC: #3)
  - [x] Add `price_aware_policy` import in `src/baselines/__init__.py`
  - [x] Add to `__all__` list

- [x] Task 3: Write tests for price-aware policy (AC: #1, #2, #3)
  - [x] Add `TestPriceAwarePolicyImport` class to `tests/test_baseline_policies.py`
  - [x] Add `TestPriceAwarePolicyAction` class: shape, dtype, range
  - [x] Test commitment fractions vary across hours (NOT all identical)
  - [x] Test high-price hours get fraction 1.0
  - [x] Test low-price hours get fraction 0.5
  - [x] Add `TestPriceAwarePolicyBattery` class: discharge on high price, charge on low price
  - [x] Add `TestPriceAwarePolicyEpisode` class: full 48h episode, metrics finite, commitment variation
  - [x] Test module import of `price_aware_policy`

- [x] Task 4: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` — all existing 202+ tests must still pass
  - [x] Run new price-aware policy tests

## Dev Notes

### CRITICAL: This is an ADDITION to existing module

**DO NOT** create new files. `src/baselines/baseline_policies.py` and `src/baselines/__init__.py` already exist from Stories 3-1 and 3-2. You are ADDING a function to the existing file.

### Action Space Mapping (same as conservative/aggressive)

The environment expects a 25-dimensional action in [0, 1]:

| Index | Meaning | Price-Aware Strategy |
|-------|---------|---------------------|
| 0-23  | Commitment fractions per hour | **Variable**: 1.0 (high price) or 0.5 (low price) |
| 24    | Battery action | 0.0 = full discharge, 0.5 = idle, 1.0 = full charge |

**Battery action mapping in the environment:**
```python
battery_action = (action[24] - 0.5) * 2  # Converts [0,1] -> [-1,1]
battery_target = battery_action * self.battery_power_mw  # +/-5 MW
```

### Price-Aware Strategy Logic

The price-aware policy is the "smart" baseline — it adjusts BOTH commitment AND battery based on prices:

1. **Per-hour commitment fractions based on price level:**
   - Extract 24h price window from observation (indices 51-74, already normalized by `max_abs_price`)
   - Compute the **median** of the 24h price window as the threshold
   - Hours with price **above** median: fraction = 1.0 (commit aggressively — high revenue hours)
   - Hours with price **at or below** median: fraction = 0.5 (commit conservatively — low revenue hours)

2. **Battery heuristic — price-driven with PV surplus awareness:**
   - Determine current hour's price: `parsed["prices"][hour_idx]`
   - Compute median of 24h prices as threshold
   - Check PV surplus: `parsed["actual_pv"] > parsed["commitments"][hour_idx]`
   - If current price **above** median AND SOC > 0: **discharge** (action[24] = 0.0) — sell stored energy at high price
   - If current price **strictly below** median AND PV surplus exists AND SOC < 1.0: **charge** (action[24] = 1.0) — store surplus during cheap hours
   - Otherwise (SOC boundary, no surplus, or price at median): **idle** (action[24] = 0.5)

### Key Difference from Conservative and Aggressive

| Aspect | Conservative (3-1) | Aggressive (3-2) | Price-Aware (3-3) |
|--------|-------------------|------------------|-------------------|
| Commitment | Uniform 0.8 | Uniform 1.0 | **Variable**: 1.0 (high) / 0.5 (low) |
| Commitment trigger | None (static) | None (static) | **Per-hour price level** |
| Battery default | Idle (0.5) | Discharge (0.0) | **Price-driven** |
| Battery strategy | Imbalance reactive | Always discharge | Charge low/discharge high |
| Risk profile | Low risk | High risk | **Smart risk** |

### Reuse `_parse_observation` Helper

The `_parse_observation()` helper already exists in `baseline_policies.py` from Story 3-1. Reuse it directly — DO NOT duplicate or recreate it.

### Observation Parsing Reference

Same as conservative/aggressive (from `_parse_observation`):

| Index | Dims | Field | Normalization |
|-------|------|-------|---------------|
| 0 | 1 | Hour | hour / 24.0 |
| 1 | 1 | Battery SOC | soc / 10.0 (capacity) |
| 2-25 | 24 | Today's commitments | commit / 20.0 (plant_cap) |
| 26 | 1 | Cumulative imbalance | imbalance / 20.0 |
| 27-50 | 24 | PV forecast 24h | forecast / 20.0 |
| 51-74 | 24 | Prices 24h | price / max_abs_price |
| 75 | 1 | Current actual PV | pv / 20.0 |

**Price values in observation:** Normalized by `max_abs_price` (the absolute maximum price in the dataset). Values can be negative (negative wholesale prices do occur). The median comparison works directly on normalized values — no denormalization needed since relative ranking is preserved.

### Function Signature

```python
def price_aware_policy(obs: np.ndarray) -> np.ndarray:
    """Price-aware baseline: adjust commitment and battery based on price levels.

    Strategy:
        - Commitment: Set per-hour fractions based on price relative to 24h median.
          High-price hours get 1.0, low-price hours get 0.5.
        - Battery: Discharge during high-price hours, charge during low-price hours.

    Args:
        obs: 84-dimensional observation from SolarMerchantEnv.

    Returns:
        25-dimensional action array in [0, 1] range (float32).
    """
```

### Edge Cases

1. **All prices identical** (e.g., flat price day): Median equals all prices. All hours at/below median → all fractions = 0.5. Battery idles (price equals median, no strategic advantage to charge/discharge). Degrades gracefully to a conservative-like strategy.
2. **Negative prices**: Normalized prices can be negative. The median comparison still works — hours below median get low commitment (good — avoid selling at negative prices).
3. **SOC boundaries**: Check `soc > 1e-6` for discharge (not exactly 0 due to float), check `soc < 1.0 - 1e-6` for charge (not exactly full).

### Testing Pattern

Follow the EXACT same pattern as `TestConservativePolicy*` and `TestAggressivePolicy*` classes in `tests/test_baseline_policies.py`. Add new classes in the SAME file:

```python
class TestPriceAwarePolicyAction:
    """Tests for price-aware policy action output."""

    def test_action_shape(self, env):
        obs, _ = env.reset(seed=42)
        action = price_aware_policy(obs)
        assert action.shape == (25,)

    def test_commitment_fractions_vary(self, env):
        """Commitment fractions should NOT be uniform (unlike conservative/aggressive)."""
        obs, _ = env.reset(seed=42)
        action = price_aware_policy(obs)
        unique_fractions = set(action[0:24].tolist())
        assert len(unique_fractions) > 1  # Should have both high and low fractions

    # ... etc
```

**Key test for price-aware that differs from 3-1/3-2:** Test that commitment fractions are NOT all identical. This is the defining characteristic — price-aware produces variable commitments based on price levels.

**Unit test for price-based commitment:**
```python
def test_high_price_gets_high_commitment(self):
    """Hours with above-median price should get fraction 1.0."""
    obs = np.zeros(84, dtype=np.float32)
    # Set up prices: hours 0-11 = 0.1 (low), hours 12-23 = 0.9 (high)
    obs[51:63] = 0.1   # Low prices
    obs[63:75] = 0.9   # High prices
    action = price_aware_policy(obs)
    # Median = 0.5 → hours 12-23 (high) should get 1.0, hours 0-11 (low) should get 0.5
    np.testing.assert_allclose(action[0:12], 0.5, atol=1e-7)  # Low price hours
    np.testing.assert_allclose(action[12:24], 1.0, atol=1e-7)  # High price hours
```

Use the shared `env` fixture from `tests/conftest.py`.

### Architecture Compliance

| Requirement | How to Satisfy |
|-------------|----------------|
| File location | ADD to existing `src/baselines/baseline_policies.py` |
| Naming | snake_case function, UPPER_SNAKE_CASE constants |
| Type hints | All public functions (Python 3.10+ style) |
| Docstrings | Google style with Args, Returns |
| Interface | `def price_aware_policy(obs: np.ndarray) -> np.ndarray` |
| Module export | Add to `src/baselines/__init__.py` and `__all__` |

### Previous Story Intelligence

**From Story 3-2 (Aggressive Baseline):**
- `_parse_observation()` helper is stable and tested — reuse it.
- Plant constants (`PLANT_CAPACITY_MW`, `BATTERY_CAPACITY_MWH`, `BATTERY_POWER_MW`) are defined — reuse them.
- Float32 precision: use `np.testing.assert_allclose(atol=1e-7)` for float comparisons.
- 202 tests currently passing. Do not regress.
- The `env` fixture from `tests/conftest.py` is shared — use it for episode tests.

**From Story 3-2 code review:**
- Battery heuristic must handle SOC boundary edge cases (can't discharge empty, can't charge full).
- PV surplus detection pattern: `parsed["actual_pv"] > parsed["commitments"][hour_idx]`.
- Hour index computation: `int(round(parsed["hour"] * 24)) % 24`.

**From Story 3-1 code review:**
- Use `np.testing.assert_allclose(atol=1e-7)` instead of `assert_array_equal` for float32 comparisons.
- Battery edge case: SOC=0 → can't discharge. SOC=1.0 → can't charge.

### Project Structure Notes

- `src/baselines/baseline_policies.py` — ADD `price_aware_policy` function (below `aggressive_policy`)
- `src/baselines/__init__.py` — ADD import and `__all__` entry
- `tests/test_baseline_policies.py` — ADD test classes (do NOT create a new test file)
- No new files should be created

### References

- [Source: docs/epics.md#Story-3.3](../../docs/epics.md) — Story requirements: adjusts commitment based on price levels
- [Source: docs/architecture.md#Key-Interfaces](../../docs/architecture.md) — Baseline interface: `def policy(obs) -> action`
- [Source: docs/architecture.md#Module-Boundaries](../../docs/architecture.md) — baselines module: Input: Observations, Output: Actions, Dependencies: numpy
- [Source: src/baselines/baseline_policies.py](../../src/baselines/baseline_policies.py) — Existing module with conservative_policy, aggressive_policy, _parse_observation, plant constants
- [Source: src/environment/solar_merchant_env.py#L206-L230](../../src/environment/solar_merchant_env.py) — Price normalization: `price / max_abs_price`
- [Source: docs/implementation/3-1-conservative-baseline-policy.md](../../docs/implementation/3-1-conservative-baseline-policy.md) — Float32 precision lesson, battery edge cases
- [Source: docs/implementation/3-2-aggressive-baseline-policy.md](../../docs/implementation/3-2-aggressive-baseline-policy.md) — Battery logic ordering, SOC boundary handling
- [Source: CLAUDE.md#Action-Space](../../CLAUDE.md) — Action[0:24] commitment fractions, Action[24] battery

## Change Log

- 2026-01-30: Implemented price-aware baseline policy with price-driven commitment and battery heuristics. Added 16 tests covering imports, action output, battery logic, and full episode execution. All 218 tests pass.
- 2026-01-30: [Code Review] Fixed battery heuristic: (1) flat-price battery now idles instead of charging (strict < instead of <=), (2) added PV surplus check for charging consistency with other baselines, (3) vectorized commitment loop with np.where. Added 3 new tests (flat-price idle, no-PV-surplus idle, negative prices). Updated existing test to require PV surplus for charge. All 221 tests pass.

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

No issues encountered during implementation.

### Completion Notes List

- Implemented `price_aware_policy` function in `src/baselines/baseline_policies.py` following existing patterns from conservative and aggressive policies
- Added `PRICE_AWARE_HIGH_FRACTION = 1.0` and `PRICE_AWARE_LOW_FRACTION = 0.5` constants
- Policy computes median of 24h price window and sets per-hour commitment: 1.0 for above-median, 0.5 for at/below-median
- Battery heuristic: discharge during high-price hours (SOC > 0), charge during low-price hours (SOC < full), idle at SOC boundaries
- Edge cases handled: all-prices-identical degrades to conservative-like 0.5, SOC boundary checks use 1e-6 tolerance
- Reused `_parse_observation` helper and plant constants from Stories 3-1/3-2
- Updated `src/baselines/__init__.py` with `price_aware_policy` import and `__all__` entry
- Added 16 tests in 4 classes: `TestPriceAwarePolicyImport` (2), `TestPriceAwarePolicyAction` (5), `TestPriceAwarePolicyBattery` (4), `TestPriceAwarePolicyEpisode` (4)
- Full test suite: 218 tests passed (202 existing + 16 new), zero regressions

### File List

- `src/baselines/baseline_policies.py` — Modified: added `PRICE_AWARE_HIGH_FRACTION`, `PRICE_AWARE_LOW_FRACTION` constants and `price_aware_policy()` function. [Review] Fixed battery heuristic (strict <, PV surplus check, np.where vectorization).
- `src/baselines/__init__.py` — Modified: added `price_aware_policy` import and `__all__` entry
- `tests/test_baseline_policies.py` — Modified: added 4 test classes with 19 tests for price-aware policy (16 original + 3 added in review). [Review] Updated `test_battery_charge_on_low_price` to include PV surplus, added battery assertion to `test_low_price_gets_low_commitment`.
- `docs/implementation/sprint-status.yaml` — Modified: set `3-3-price-aware-baseline-policy` status to `review`
