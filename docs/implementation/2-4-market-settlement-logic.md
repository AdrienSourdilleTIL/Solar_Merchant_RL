# Story 2.4: Market Settlement Logic

Status: ready-for-dev

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want the environment to calculate revenue and imbalance costs correctly,
So that the reward signal reflects real market economics.

## Acceptance Criteria

1. **Given** commitments have been made and energy delivered
   **When** settlement occurs at each hour
   **Then** revenue = delivered energy x day-ahead price
   **And** short positions (under-delivery) pay 1.5x day-ahead price
   **And** long positions (over-delivery) receive 0.6x day-ahead price
   **And** total reward = revenue - imbalance_cost - degradation_cost

## Tasks / Subtasks

- [ ] Task 1: Validate existing market settlement implementation (AC: #1)
  - [ ] Verify revenue calculation: delivered * price_eur_mwh
  - [ ] Verify short imbalance: uses price_imbalance_short (expected: 1.5x DA price)
  - [ ] Verify long imbalance: uses price_imbalance_long (expected: 0.6x DA price)
  - [ ] Verify reward formula: revenue - imbalance_cost - degradation_cost
  - [ ] Trace settlement logic through step() method (lines 493-529)

- [ ] Task 2: Test revenue calculation (AC: #1)
  - [ ] Test revenue with zero delivery (0 MWh * price = 0 EUR)
  - [ ] Test revenue with typical delivery (10 MWh * 50 EUR = 500 EUR)
  - [ ] Test revenue with maximum delivery (plant capacity * price)
  - [ ] Test revenue with negative prices (delivered * negative_price = negative revenue)
  - [ ] Verify revenue accumulates in episode_revenue

- [ ] Task 3: Test short imbalance settlement (AC: #1)
  - [ ] Test under-delivery penalty calculation
  - [ ] Test short penalty rate is 1.5x day-ahead price
  - [ ] Test partial shortage (delivered < committed but > 0)
  - [ ] Test complete shortage (delivered = 0, committed > 0)
  - [ ] Verify imbalance_cost is positive for short positions

- [ ] Task 4: Test long imbalance settlement (AC: #1)
  - [ ] Test over-delivery value calculation
  - [ ] Test long rate is 0.6x day-ahead price
  - [ ] Test excess calculation: delivered - committed
  - [ ] Test opportunity cost: excess * (price - price_long)
  - [ ] Verify imbalance_cost reflects lost revenue opportunity

- [ ] Task 5: Test reward composition (AC: #1)
  - [ ] Test reward = revenue - imbalance_cost - degradation_cost
  - [ ] Test reward with zero imbalance (balanced delivery)
  - [ ] Test reward with positive imbalance (over-delivery)
  - [ ] Test reward with negative imbalance (under-delivery)
  - [ ] Test reward accumulation across episode

- [ ] Task 6: Test edge cases and economics (AC: #1)
  - [ ] Test settlement at hour 0 (day transition)
  - [ ] Test settlement with zero commitment (no penalty)
  - [ ] Test asymmetric risk: verify short penalty > long opportunity cost
  - [ ] Test with extreme prices (near-zero, high prices)
  - [ ] Test settlement when no commitment has been made (todays_commitments = zeros)

## Dev Notes

### CRITICAL: Brownfield Validation Task

**Existing Code Status:** Market settlement logic is **implemented** in [src/environment/solar_merchant_env.py](../../src/environment/solar_merchant_env.py).

**Key Implementation Sections:**
- Lines 493-529: Revenue, imbalance, and reward calculation in `step()` method
- Lines 500-501: Revenue = delivered * price
- Lines 507-517: Imbalance cost calculation (short vs long)
- Lines 522-523: Degradation cost calculation
- Lines 528-529: Final reward composition

**This story focuses on:**
1. **Validating** the existing market settlement logic is correct
2. **Testing** comprehensive settlement scenarios (short, long, balanced)
3. **Verifying** economic incentives align with real market behavior
4. **Documenting** settlement mechanics for future reference

**DO NOT:**
- Rewrite settlement logic wholesale unless critical bugs are found
- Change price multipliers without architectural justification
- Add unnecessary complexity to working settlement simulation
- Break existing functionality from Stories 2-1, 2-2, and 2-3

### Existing Implementation Analysis

**Revenue Calculation:** [solar_merchant_env.py:499-501](../../src/environment/solar_merchant_env.py#L499-L501)

```python
# Revenue for delivered energy at day-ahead price
revenue = delivered * price
self.episode_revenue += revenue
```

**Rationale:** The solar merchant receives day-ahead price for all energy delivered, regardless of commitment. Settlement then adjusts for imbalance.

**Imbalance Settlement:** [solar_merchant_env.py:503-517](../../src/environment/solar_merchant_env.py#L503-L517)

```python
# Imbalance settlement
imbalance = delivered - committed
if imbalance < 0:
    # Short: under-delivered, pay penalty at short price
    imbalance_cost = abs(imbalance) * price_short
else:
    # Long: over-delivered, receive long price instead of DA price for excess
    # Since we already counted delivered * price in revenue,
    # we need to subtract the excess that should have been at long price
    imbalance_cost = imbalance * (price - price_long)
```

**Key Design Decisions:**

1. **Revenue first, then adjust:** Revenue is calculated for all delivered energy at DA price. Imbalance cost then adjusts the economics.

2. **Short penalty (imbalance < 0):**
   - Must buy shortfall at imbalance short price
   - `imbalance_cost = abs(imbalance) * price_short`
   - Example: Committed 10 MWh, delivered 8 MWh, price_short = 75 EUR/MWh
   - Cost = 2 MWh * 75 EUR = 150 EUR

3. **Long opportunity cost (imbalance > 0):**
   - Excess energy receives long price instead of DA price
   - Already got DA price in revenue, so we subtract the difference
   - `imbalance_cost = excess * (price - price_long)`
   - Example: Committed 10 MWh, delivered 12 MWh, price = 50 EUR, price_long = 30 EUR
   - Revenue already includes 12 * 50 = 600 EUR
   - But excess 2 MWh should have been at 30 EUR, not 50 EUR
   - Opportunity cost = 2 * (50 - 30) = 40 EUR

**Reward Composition:** [solar_merchant_env.py:528-529](../../src/environment/solar_merchant_env.py#L528-L529)

```python
# Reward = revenue - imbalance cost - degradation
reward = revenue - imbalance_cost - degradation
```

### Imbalance Price Generation

**From Data Processing Pipeline:**
- `price_imbalance_short = price_eur_mwh * 1.5` (short penalty multiplier)
- `price_imbalance_long = price_eur_mwh * 0.6` (long recovery rate)

**Asymmetric Risk:**
- Short penalty: 1.5x DA price (50% premium for buying at imbalance)
- Long recovery: 0.6x DA price (40% haircut for selling excess)
- This creates asymmetric risk that the agent must learn to manage
- Conservative strategies favor under-commitment to avoid short penalties

### Economic Incentives

**Optimal Behavior:**
1. **Perfect matching:** Deliver exactly what committed = no imbalance cost
2. **Conservative (under-commit):** Over-deliver → opportunity cost of (price - price_long)
3. **Aggressive (over-commit):** Under-deliver → penalty of abs(shortage) * price_short

**Agent Learning Objectives:**
- Learn when forecast uncertainty justifies conservative commitment
- Learn to use battery to fill delivery gaps
- Balance revenue maximization vs imbalance risk

### Settlement Example Walkthrough

**Scenario:** Hour 14, committed 10 MWh, delivered 8 MWh (short by 2 MWh)
- `price = 50.0 EUR/MWh` (day-ahead price)
- `price_short = 75.0 EUR/MWh` (1.5x penalty)
- `price_long = 30.0 EUR/MWh` (0.6x recovery)

**Calculation:**
1. Revenue: `8 MWh * 50 EUR = 400 EUR`
2. Imbalance: `8 - 10 = -2 MWh` (short)
3. Imbalance cost: `abs(-2) * 75 = 150 EUR`
4. Degradation: `0 EUR` (assuming no battery use)
5. Reward: `400 - 150 - 0 = 250 EUR`

**Comparison to perfect delivery:**
- Perfect: `10 * 50 - 0 - 0 = 500 EUR`
- Shortage cost the agent 250 EUR (the 100 EUR of missing revenue + 150 EUR penalty)

### Previous Story Intelligence (Story 2-3)

**Key Learnings Applied:**

1. **Validation over Rewriting**: Story 2-3 successfully validated battery mechanics with tests. Apply same approach to settlement logic.

2. **Comprehensive Testing**: Story 2-3 created 30 tests for battery. Create similar coverage for settlement:
   - Unit tests for revenue calculation
   - Unit tests for short/long imbalance
   - Integration tests for full episode economics
   - Edge case tests for boundaries

3. **Documentation Pattern**: Story 2-3 added detailed Dev Notes explaining battery physics. Document settlement economics similarly.

4. **Files from Story 2-3:**
   - `tests/test_battery_mechanics.py` (30 tests)
   - Pattern: create `tests/test_market_settlement.py`

### Architecture Compliance

**From Architecture Document:** [docs/architecture.md#Environment-Architecture](../../docs/architecture.md#environment-architecture)

| Requirement | Status | Implementation Notes |
|-------------|--------|---------------------|
| **Revenue = delivered * DA price** | Implemented | Line 500 |
| **Short = 1.5x DA price** | Implemented | Via price_imbalance_short from data |
| **Long = 0.6x DA price** | Implemented | Via price_imbalance_long from data |
| **Reward = revenue - imbalance - degradation** | Implemented | Line 529 |

**From PRD:** [docs/prd.md#Reward-Function](../../docs/prd.md#reward-function)

```
reward = revenue - imbalance_cost - battery_degradation_cost
```
Where:
- Revenue = delivered energy x day-ahead price
- Imbalance cost = penalty for over/under delivery (asymmetric)
- Degradation = small cost per MWh battery throughput

### Testing Requirements

**From Architecture - NFR2:**
> Single episode evaluation completes within 5 seconds

**Settlement calculations should be fast (<1ms per step)**

**Test Plan for Story 2-4:**

1. **Unit Tests - Revenue Calculation**:
   - Zero delivery
   - Typical delivery
   - Maximum delivery (plant capacity)
   - Negative prices

2. **Unit Tests - Short Settlement**:
   - Various shortage amounts
   - Penalty rate verification (1.5x)
   - Cost accumulation

3. **Unit Tests - Long Settlement**:
   - Various excess amounts
   - Recovery rate verification (0.6x)
   - Opportunity cost calculation

4. **Integration Tests - Reward Composition**:
   - Balanced delivery (zero imbalance)
   - Mixed scenarios across episode
   - Full 48-hour episode economics

5. **Edge Case Tests**:
   - Zero commitment (no penalty)
   - Day transition (hour 0)
   - Extreme prices
   - No commitment made yet

6. **Economic Verification Tests**:
   - Asymmetric risk (short penalty > long cost)
   - Agent incentive alignment
   - Perfect delivery vs conservative strategy

### Project Structure Notes

**Current Test Structure:**
```
tests/
├── test_environment.py                # General environment tests
├── test_observation_construction.py   # Observation tests (Story 2-2)
├── test_battery_mechanics.py          # Battery tests (Story 2-3)
└── test_market_settlement.py          # NEW - Story 2-4 focused tests
```

**Settlement Logic Location:**
```
src/environment/solar_merchant_env.py
├── step() [lines 392-554]
│   ├── Revenue calculation [lines 499-501]
│   ├── Imbalance settlement [lines 503-519]
│   ├── Degradation cost [lines 521-523]
│   └── Reward composition [lines 528-529]
```

### Implementation Checklist

**Primary Task: Validate Settlement Implementation**
- [ ] Read and analyze revenue calculation (lines 499-501)
- [ ] Read and analyze short imbalance logic (lines 507-511)
- [ ] Read and analyze long imbalance logic (lines 512-517)
- [ ] Verify reward composition is correct (line 529)
- [ ] Verify imbalance prices from data (1.5x and 0.6x multipliers)
- [ ] Test with known inputs to verify economics

**Secondary Task: Test Settlement Mechanics**
- [ ] Test revenue scenarios (zero, typical, max, negative prices)
- [ ] Test short settlement (various shortage amounts)
- [ ] Test long settlement (various excess amounts)
- [ ] Test edge cases (zero commitment, day transition)
- [ ] Test asymmetric risk economics

**Documentation Task**
- [ ] Document settlement economics clearly (Dev Notes section)
- [ ] Add examples of settlement calculations
- [ ] Document economic incentives for agent
- [ ] Document design decisions

### Data Requirements

**Required columns for settlement:**
- `price_eur_mwh`: Day-ahead price for revenue
- `price_imbalance_short`: Penalty price for under-delivery
- `price_imbalance_long`: Recovery price for over-delivery
- `pv_actual_mwh`: Actual PV production (contributes to delivery)

**Generated in data processing:**
- `price_imbalance_short = price_eur_mwh * 1.5`
- `price_imbalance_long = price_eur_mwh * 0.6`

### References

- [Source: docs/epics.md#Story-2.4](../../docs/epics.md#story-24-market-settlement-logic)
- [Source: docs/architecture.md#Environment-Architecture](../../docs/architecture.md#environment-architecture)
- [Source: docs/prd.md#Reward-Function](../../docs/prd.md#reward-function)
- [Source: src/environment/solar_merchant_env.py:493-529](../../src/environment/solar_merchant_env.py#L493-L529) - Settlement logic
- [Source: docs/implementation/2-3-battery-mechanics.md](../../docs/implementation/2-3-battery-mechanics.md) - Previous story pattern
- [Source: CLAUDE.md#Imbalance-Settlement](../../CLAUDE.md#imbalance-settlement) - Project guidance

## Change Log

- **2026-01-26 (Initial)**: Story 2-4 created by SM agent (Bob). Comprehensive context gathered from epics, architecture, PRD, existing code, and Story 2-3 patterns. Status -> ready-for-dev.

## Dev Agent Record

### Agent Model Used

_To be filled by dev agent_

### Debug Log References

_To be filled by dev agent_

### Completion Notes List

_To be filled by dev agent_

### File List

**New Files:**
- tests/test_market_settlement.py (to be created)

**Modified Files:**
- docs/implementation/2-4-market-settlement-logic.md (this story file)
- docs/implementation/sprint-status.yaml (status update)
