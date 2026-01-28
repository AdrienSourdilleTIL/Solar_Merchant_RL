# Story 2.5: Episode Flow and Reset

Status: done

<!-- Note: Validation is optional. Run validate-create-story for quality check before dev-story. -->

## Story

As a developer,
I want proper episode flow with reset to random starting points,
So that training covers diverse market conditions.

## Acceptance Criteria

1. **Given** a configured environment
   **When** `reset()` is called
   **Then** episode starts at a random day in the dataset
   **And** initial SOC is set (configurable, default 0.5)
   **And** commitments are cleared
   **And** valid observation is returned

2. **When** `step()` is called with actions
   **Then** commitment decisions are processed at hour 11
   **And** battery dispatch occurs every hour
   **And** episode terminates after 48 hours
   **And** single episode evaluation completes within 5 seconds (NFR2)

## Tasks / Subtasks

- [x] Task 1: Validate reset() randomization logic (AC: #1)
  - [x] Verify random episode start selection within valid data range
  - [x] Verify episode start avoids last 48*30 hours (boundary protection)
  - [x] Verify commitment hour alignment adjustment
  - [x] Verify different seeds produce different starting points
  - [x] Verify same seed produces reproducible starting points

- [x] Task 2: Test initial state configuration (AC: #1)
  - [x] Test default battery SOC is 0.5 * capacity (5 MWh)
  - [x] Test todays_commitments initialized to zeros
  - [x] Test tomorrows_commitments initialized to zeros
  - [x] Test hourly_delivered dict is empty
  - [x] Test episode tracking counters reset (revenue, imbalance, degradation)

- [x] Task 3: Test observation validity on reset (AC: #1)
  - [x] Verify reset() returns valid 84-dim observation
  - [x] Verify observation contains no NaN or Inf values
  - [x] Verify hour normalization is in [0, 1) range
  - [x] Verify SOC normalization is in [0, 1] range
  - [x] Verify info dict is empty on reset

- [x] Task 4: Test 48-hour episode structure (AC: #2)
  - [x] Test episode terminates exactly at 48 steps
  - [x] Test terminated flag is True after 48 steps
  - [x] Test truncated flag handles data boundary
  - [x] Test commitment at hour 11 is processed correctly
  - [x] Test midnight transition moves commitments correctly

- [x] Task 5: Test commitment hour validation (AC: #2)
  - [x] Verify episodes are adjusted to contain commitment hour
  - [x] Test commitment processing only occurs at hour 11
  - [x] Test tomorrows_commitments populated at commitment hour
  - [x] Verify commitment values in expected range

- [x] Task 6: Test episode performance (AC: #2, NFR2)
  - [x] Test single episode (48 steps) completes within 5 seconds
  - [x] Test 10 episodes complete within 50 seconds
  - [x] Verify no memory leaks across multiple episodes
  - [x] Test reset/step cycle is consistent

## Dev Notes

### CRITICAL: Brownfield Validation Task

**Existing Code Status:** Episode flow and reset logic is **implemented** in [src/environment/solar_merchant_env.py](../../src/environment/solar_merchant_env.py).

**Key Implementation Sections:**
- Lines 343-390: `reset()` method with random start selection
- Lines 361-379: Commitment hour alignment validation
- Lines 381-389: State initialization (SOC, commitments, tracking)
- Lines 448-451: Midnight transition logic in `step()`
- Lines 548-552: Episode termination logic (48-hour check)

**This story focuses on:**
1. **Validating** the existing episode flow logic is correct
2. **Testing** comprehensive reset and termination scenarios
3. **Verifying** 48-hour episode structure ensures credit assignment
4. **Documenting** episode mechanics for future reference

**DO NOT:**
- Rewrite reset/step logic wholesale unless critical bugs are found
- Change 48-hour episode length without architectural justification
- Add unnecessary complexity to working episode flow
- Break existing functionality from Stories 2-1, 2-2, 2-3, and 2-4

### Existing Implementation Analysis

**Random Episode Start:** [solar_merchant_env.py:357-363](../../src/environment/solar_merchant_env.py#L357-L363)

```python
# Start at a random position (but not too close to end)
if seed is not None:
    np.random.seed(seed)

max_start = len(self.data) - 48 * 30  # Leave at least 30 days (48h episodes)
self.current_idx = np.random.randint(0, max(1, max_start))
self.episode_start_idx = self.current_idx  # Store episode start
```

**Rationale:** The episode start is randomized to ensure training covers diverse market conditions. The boundary protection (`48 * 30 hours = 1440 hours = 60 days`) ensures episodes don't start too close to dataset end.

**Commitment Hour Alignment:** [solar_merchant_env.py:365-379](../../src/environment/solar_merchant_env.py#L365-L379)

```python
# Validate episode will encounter commitment hour
episode_hours = []
for i in range(48):
    if self.current_idx + i < len(self.data):
        episode_hours.append(int(self.data.iloc[self.current_idx + i]['hour']))

if self.commitment_hour not in episode_hours:
    # Adjust start to ensure we hit commitment hour
    for i in range(len(self.data) - self.current_idx - 48):
        if int(self.data.iloc[self.current_idx + i]['hour']) == self.commitment_hour:
            self.current_idx += i
            self.episode_start_idx = self.current_idx
            break
```

**Rationale:** Every episode must encounter the commitment hour (11) so the agent learns to make commitment decisions. If random start doesn't include hour 11, we adjust forward to the next occurrence.

**State Initialization:** [solar_merchant_env.py:381-389](../../src/environment/solar_merchant_env.py#L381-L389)

```python
# Reset state
self.battery_soc = 0.5 * self.battery_capacity_mwh
self.todays_commitments = np.zeros(24)
self.tomorrows_commitments = np.zeros(24)
self.hourly_delivered = {}
self.episode_revenue = 0.0
self.episode_imbalance_cost = 0.0
self.episode_degradation_cost = 0.0
```

**Key Design Decisions:**

1. **Default SOC = 50%:** Agent starts with half-full battery, providing flexibility for both charging and discharging strategies.

2. **Zero commitments:** Both today's and tomorrow's commitments are cleared. This is realistic - a new episode represents a fresh trading day.

3. **Empty delivery tracking:** `hourly_delivered` dict is reset for cumulative imbalance calculation.

**Midnight Transition:** [solar_merchant_env.py:448-451](../../src/environment/solar_merchant_env.py#L448-L451)

```python
# Handle midnight transition: tomorrow becomes today
if hour == 0:
    self.todays_commitments = self.tomorrows_commitments.copy()
    self.tomorrows_commitments = np.zeros(24)
    self.hourly_delivered = {}  # Reset delivered tracking for new day
```

**Rationale:** At midnight (hour 0), tomorrow's commitments (made at hour 11 the previous day) become today's active commitments. This correctly models the day-ahead market cycle.

**Episode Termination:** [solar_merchant_env.py:548-552](../../src/environment/solar_merchant_env.py#L548-L552)

```python
# Check termination: Episode ends after 48 hours OR at data boundary
# 48h ensures agent sees consequences of commitments made during episode
episode_hours = self.current_idx - self.episode_start_idx
terminated = episode_hours >= 48
truncated = self.current_idx >= len(self.data) - 1
```

**Critical Design Decision - 48-Hour Episodes:**

The 48-hour episode length is **critical** for credit assignment:

1. **Hour 0-10:** Agent operates with no commitments (or inherited commitments)
2. **Hour 11:** Agent makes commitments for the NEXT day
3. **Hours 12-23:** Rest of first day
4. **Hour 24 (midnight):** Tomorrow's commitments become today's
5. **Hours 24-47:** Agent operates under commitments it made at hour 11
6. **Hour 48:** Episode terminates AFTER agent experiences consequences

Previous 24-hour episodes were **broken** - the agent never saw the imbalance costs from its commitment decisions because the episode ended before those commitments were active.

### Episode Flow Walkthrough

**Example Episode (starting at hour 6):**

| Step | Hour | Key Events |
|------|------|------------|
| 0 | 6 | Episode starts, battery at 50% SOC |
| 1-4 | 7-10 | Battery/PV operations, no commitments |
| 5 | 11 | **COMMITMENT HOUR**: Agent commits for tomorrow |
| 6-17 | 12-23 | Continue operations, tomorrows_commitments stored |
| 18 | 0 | **MIDNIGHT**: tomorrows -> todays, delivery tracking reset |
| 19-42 | 1-24 | Agent delivers against its own commitments |
| 43-47 | 1-5 | Second day continues |
| 48 | 6 | **TERMINATED**: Episode ends, agent experienced full cycle |

**Credit Assignment Flow:**
1. Agent makes commitment at step 5 (hour 11)
2. Those commitments activate at step 18 (midnight)
3. Agent experiences imbalance costs from steps 19-47
4. Reward signal carries information about commitment quality

### Previous Story Intelligence (Story 2-4)

**Key Learnings Applied:**

1. **Validation over Rewriting**: Story 2-4 successfully validated market settlement with tests. Apply same approach to episode flow logic.

2. **Comprehensive Testing**: Story 2-4 created 32 tests for settlement. Create similar coverage for episode flow:
   - Unit tests for reset() randomization
   - Unit tests for state initialization
   - Integration tests for full 48-hour episode
   - Edge case tests for boundaries

3. **Documentation Pattern**: Story 2-4 added detailed Dev Notes explaining settlement economics. Document episode mechanics similarly.

4. **Files from Story 2-4:**
   - `tests/test_market_settlement.py` (32 tests)
   - Pattern: create `tests/test_episode_flow.py`

### Architecture Compliance

**From Architecture Document:** [docs/architecture.md#Environment-Architecture](../../docs/architecture.md#environment-architecture)

| Requirement | Status | Implementation Notes |
|-------------|--------|---------------------|
| **Episode terminates correctly** | Implemented | Line 551: 48-hour termination |
| **Reset to random starting points** | Implemented | Lines 357-363 |
| **Initial SOC configurable** | Implemented | Default 0.5 * capacity |
| **Commitments cleared on reset** | Implemented | Lines 383-384 |

**From PRD:** [docs/prd.md](../../docs/prd.md)

- FR19: Environment can reset to random starting points within the dataset
- FR20: Environment can step through hourly simulation with correct settlement logic
- NFR2: Single episode evaluation completes within 5 seconds

**From CLAUDE.md:** [CLAUDE.md#48-Hour-Episodes](../../CLAUDE.md#key-design-decisions--constraints-v1)

```
**48-Hour Episodes:**
- Episodes span 48 hours to fix credit assignment problem
- Commitments made at hour 11 are for the next day (hours 13-36 of episode)
- Agent sees imbalance costs from those commitments before episode ends
- Previous 24h design was fundamentally broken - agent never learned from its commitments
```

### Testing Requirements

**From Architecture - NFR2:**
> Single episode evaluation completes within 5 seconds

**Episode operations must be efficient:**
- reset() < 100ms
- step() < 100ms per call
- Full 48-step episode < 5 seconds

**Test Plan for Story 2-5:**

1. **Unit Tests - Reset Randomization**:
   - Different seeds produce different starts
   - Same seed produces same start (reproducibility)
   - Starts avoid data boundary
   - Commitment hour alignment works

2. **Unit Tests - State Initialization**:
   - Battery SOC initialized correctly
   - Commitments cleared
   - Tracking counters zeroed

3. **Unit Tests - Observation Validity**:
   - 84-dimensional observation returned
   - No NaN/Inf values
   - Normalized values in expected ranges

4. **Integration Tests - Episode Structure**:
   - Episode terminates at exactly 48 steps
   - Midnight transition works correctly
   - Commitment hour processing works
   - Full cycle credit assignment

5. **Performance Tests**:
   - Single episode within 5 seconds
   - 10 episodes within 50 seconds
   - No memory leaks

6. **Edge Case Tests**:
   - Reset near data start
   - Reset near data end
   - Multiple resets in sequence

### Project Structure Notes

**Current Test Structure:**
```
tests/
├── test_environment.py                # General environment tests
├── test_observation_construction.py   # Observation tests (Story 2-2)
├── test_battery_mechanics.py          # Battery tests (Story 2-3)
├── test_market_settlement.py          # Settlement tests (Story 2-4)
└── test_episode_flow.py               # NEW - Story 2-5 focused tests
```

**Episode Flow Logic Location:**
```
src/environment/solar_merchant_env.py
├── reset() [lines 343-390]
│   ├── Random start selection [lines 357-363]
│   ├── Commitment hour alignment [lines 365-379]
│   └── State initialization [lines 381-389]
├── step() [lines 392-554]
│   ├── Midnight transition [lines 448-451]
│   └── Episode termination [lines 548-552]
```

### Implementation Checklist

**Primary Task: Validate Episode Flow Implementation**
- [x] Read and analyze reset() randomization (lines 357-363)
- [x] Read and analyze commitment hour alignment (lines 365-379)
- [x] Read and analyze state initialization (lines 381-389)
- [x] Verify midnight transition logic (lines 448-451)
- [x] Verify 48-hour termination logic (lines 548-552)
- [x] Test with known inputs to verify behavior

**Secondary Task: Test Episode Mechanics**
- [x] Test reset randomization (seeds, boundaries)
- [x] Test state initialization (SOC, commitments, counters)
- [x] Test observation validity on reset
- [x] Test 48-hour episode structure
- [x] Test midnight transition
- [x] Test commitment hour alignment

**Documentation Task**
- [x] Document episode flow mechanics clearly (Dev Notes section)
- [x] Add examples of episode walkthrough
- [x] Document 48-hour design rationale
- [x] Document credit assignment fix

### References

- [Source: docs/epics.md#Story-2.5](../../docs/epics.md#story-25-episode-flow-and-reset)
- [Source: docs/architecture.md#Environment-Architecture](../../docs/architecture.md#environment-architecture)
- [Source: docs/prd.md](../../docs/prd.md) - FR19, FR20, NFR2
- [Source: src/environment/solar_merchant_env.py:343-390](../../src/environment/solar_merchant_env.py#L343-L390) - reset() method
- [Source: src/environment/solar_merchant_env.py:448-451](../../src/environment/solar_merchant_env.py#L448-L451) - Midnight transition
- [Source: src/environment/solar_merchant_env.py:548-552](../../src/environment/solar_merchant_env.py#L548-L552) - Termination logic
- [Source: docs/implementation/2-4-market-settlement-logic.md](../../docs/implementation/2-4-market-settlement-logic.md) - Previous story pattern
- [Source: CLAUDE.md#48-Hour-Episodes](../../CLAUDE.md#key-design-decisions--constraints-v1) - Episode design rationale

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- No bugs found in existing implementation
- All 33 episode flow tests passed on first run
- Full test suite (141 tests) passed with no regressions

### Completion Notes List

- **Validated** existing reset() randomization logic - implementation correctly:
  - Selects random episode start within valid data range
  - Avoids last 48*30 hours for boundary protection
  - Adjusts to ensure commitment hour (11) appears in episode
  - Produces reproducible results with same seed
- **Verified** initial state configuration:
  - Battery SOC defaults to 50% capacity (5 MWh)
  - Both todays_commitments and tomorrows_commitments initialized to zeros
  - hourly_delivered dict cleared, episode counters reset
- **Confirmed** observation validity:
  - Returns 84-dimensional float32 observation
  - No NaN or Inf values
  - Proper normalization ranges (hour: [0,1), SOC: [0,1])
- **Tested** 48-hour episode structure:
  - Episode terminates exactly at 48 steps
  - Terminated/truncated flags work correctly
  - Commitment at hour 11 processed and stored in tomorrows_commitments
  - Midnight transition correctly moves tomorrows→todays
- **Validated** performance meets NFR2:
  - Single episode completes in <5 seconds
  - 10 episodes complete in <50 seconds
  - No memory leaks detected
- Created comprehensive test suite following Story 2-4 pattern

### File List

**New Files:**
- tests/test_episode_flow.py (35 tests, 7 test classes)
- tests/conftest.py (shared fixtures and warning suppression)

**Modified Files:**
- src/environment/solar_merchant_env.py (added configurable initial_soc via options)
- docs/implementation/sprint-status.yaml (status update)
- docs/implementation/2-5-episode-flow-and-reset.md (this file)

## Change Log

| Date | Change | Author |
|------|--------|--------|
| 2026-01-28 | Created test_episode_flow.py with 33 comprehensive tests validating episode flow and reset logic | Claude Opus 4.5 |
| 2026-01-28 | Validated all acceptance criteria - existing implementation passes all tests | Claude Opus 4.5 |
| 2026-01-28 | Story marked for review - all tasks complete | Claude Opus 4.5 |
| 2026-01-28 | **Code Review Fixes:** Implemented configurable initial_soc via reset options (AC #1 completion) | Claude Opus 4.5 |
| 2026-01-28 | **Code Review Fixes:** Added conftest.py with shared fixtures and warning suppression | Claude Opus 4.5 |
| 2026-01-28 | **Code Review Fixes:** Added 2 new tests for configurable SOC (test count now 35) | Claude Opus 4.5 |
| 2026-01-28 | **Code Review Fixes:** Fixed documentation discrepancies (test class count, Implementation Checklist) | Claude Opus 4.5 |
