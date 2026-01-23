# Critical Architectural Fixes (2026-01-22)

**Status:** Completed
**Story:** 2-2-observation-construction
**Severity:** CRITICAL - Would have broken RL learning entirely

## Summary

Post-code-review architectural analysis revealed 2 fundamental design flaws that would have prevented the RL agent from learning effective commitment strategies. Both issues have been fixed.

---

## Issue #1: Episode/Commitment Credit Assignment Problem

### The Problem

**Original Design:**
- Episodes were 24 hours long
- Commitments made at hour 11 were for "tomorrow" (next 24 hours)
- Episodes ended after 24 hours

**Why This Breaks Learning:**
```
Timeline Example:
Hour 0:  Episode starts
Hour 11: Agent commits for tomorrow (hours 13-36 in absolute timeline)
Hour 24: Episode ENDS
Hours 25-36: Tomorrow's imbalances occur BUT EPISODE IS OVER

Result: Agent NEVER sees the rewards/penalties from its commitments!
```

The agent was making blind commitments with zero feedback. This is catastrophic for credit assignment in RL - the agent can't learn what works because it never sees the consequences.

### The Fix

**Extended episodes to 48 hours:**
```
Timeline Example:
Hour 0:  Episode starts
Hour 11: Agent commits for tomorrow (hours 13-36)
Hour 24: Midnight - tomorrow becomes today
Hours 25-36: Agent experiences imbalance costs from its commitments
Hour 48: Episode ends (AFTER seeing consequences)

Result: Agent gets full reward signal from its decisions!
```

### Files Modified
- `src/environment/solar_merchant_env.py`:
  - Episode termination changed from 24h to 48h
  - `reset()` updated to check 48h window for commitment hour
  - Comments added explaining the credit assignment rationale

- `tests/test_environment.py`:
  - `test_episode_terminates_after_48_hours` (updated from 24h)
  - Loop limits increased from 30 → 60 steps
  - Commitment hour test updated for 48h window

---

## Issue #2: Commitment Indexing Bug

### The Problem

**Original Design:**
```python
self.committed_schedule = np.zeros(24)  # Single array for "today"

# At commitment hour (11:00):
self.committed_schedule = new_commitments  # For TOMORROW

# During hourly step:
committed = self.committed_schedule[hour]  # Which day is this?!
```

**The Bug:**
- `committed_schedule` was a 24-element array indexed by hour-of-day (0-23)
- No tracking of WHICH DAY these commitments were for
- After midnight, the indexing became misaligned

**Failure Scenario:**
```
Episode starts at hour 15 on Day N
Hour 11 on Day N+1: Set committed_schedule for Day N+2
Hour 0 on Day N+2:  Access committed_schedule[0]
  → BUG: Gets commitment from Day N+1, not Day N+2!
```

### The Fix

**Separate today/tomorrow tracking:**
```python
self.todays_commitments = np.zeros(24)    # Being delivered against NOW
self.tomorrows_commitments = np.zeros(24) # Will become today at midnight

# At commitment hour:
self.tomorrows_commitments = new_commitments

# At midnight (hour == 0):
self.todays_commitments = self.tomorrows_commitments.copy()
self.tomorrows_commitments = np.zeros(24)

# During hourly step:
committed = self.todays_commitments[hour]  # Always correct!
```

### Files Modified
- `src/environment/solar_merchant_env.py`:
  - Replaced `committed_schedule` with `todays_commitments` + `tomorrows_commitments`
  - Added midnight transition logic in `step()`
  - Updated all references in `_get_observation()` and reward calculation
  - Fixed observation construction to use `todays_commitments`

---

## Testing

**All tests passing:**
- 24 observation construction tests ✅
- 18 environment structure tests ✅
- 3 episode termination tests (updated for 48h) ✅
- 1 performance test (thresholds adjusted) ✅

**Total: 46/46 tests passing**

---

## Impact Assessment

### What Would Have Happened Without These Fixes

1. **Episode/Commitment Mismatch:**
   - Agent would make random commitments (no learning signal)
   - Training would show no improvement over random baseline
   - Might converge to conservative "commit nothing" strategy
   - Wasted compute on training that can't learn

2. **Commitment Indexing Bug:**
   - Imbalance calculations would be wrong after midnight
   - Rewards would be nonsensical
   - Agent might learn perverse strategies exploiting the bug
   - Results would be completely invalid

### Why The Code Review Missed These

These are **architectural/domain-specific** issues, not code quality issues:
- Code was syntactically correct
- No crashes or obvious bugs
- Tests passed (but weren't testing the right thing)
- Required deep understanding of:
  - RL credit assignment principles
  - Day-ahead market mechanics
  - Episode/commitment timing relationships

---

## Documentation Updates

- ✅ CLAUDE.md - Added "Key Design Decisions & Constraints" section
- ✅ Story 2-2 change log - Added critical fixes entry
- ✅ Test updates - All episode length references updated
- ✅ Performance thresholds - Adjusted for 48h episodes

---

## Lessons Learned

**For Future Stories:**

1. **Episode Length Matters:** Always validate that episode length allows agent to see consequences of all actions taken during episode

2. **State Indexing:** When using arrays indexed by cyclical values (hour-of-day), need explicit day tracking if episodes span multiple cycles

3. **Domain Review:** Code review should include domain expert to catch timing/causality issues

4. **Test What Matters:** Tests should validate business logic (credit assignment), not just code execution

---

## Credit

Issues identified by: Adrien Sourdille (User)
Fixed by: Claude Sonnet 4.5
Date: 2026-01-22

**Status: RESOLVED ✅**
