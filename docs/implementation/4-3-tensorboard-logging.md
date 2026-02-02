# Story 4.3: TensorBoard Logging

Status: done

## Story

As a developer,
I want training metrics logged to TensorBoard,
so that I can monitor learning progress.

## Acceptance Criteria

1. **Given** training is in progress
   **When** metrics are generated
   **Then** episode rewards (mean/std) are logged to TensorBoard
   **And** this happens automatically via SB3's built-in logger + Monitor wrapper

2. **Given** training is in progress
   **When** SAC updates occur
   **Then** policy loss, critic loss, and entropy coefficient are logged
   **And** this happens automatically via SB3 when `tensorboard_log` is set

3. **Given** the TensorBoard log directory
   **When** logs are written
   **Then** logs are saved to `outputs/tensorboard/` (configurable via `TENSORBOARD_LOG_DIR` constant)
   **And** the directory is created before training starts
   **Note:** Epics say `runs/` but CLAUDE.md and existing code use `outputs/tensorboard/`. Keep current convention.

4. **Given** training has produced TensorBoard logs
   **When** the user runs `tensorboard --logdir outputs/tensorboard`
   **Then** TensorBoard visualizes episode reward curves, loss curves, and entropy
   **And** the launch command is printed at training start for discoverability

## Tasks / Subtasks

- [x] Task 1: Extract TensorBoard log directory to named constant (AC: #3)
  - [x] Add `TENSORBOARD_LOG_DIR = OUTPUT_PATH / 'tensorboard'` constant at module level (near other path constants)
  - [x] Replace inline `str(OUTPUT_PATH / 'tensorboard')` in SAC constructor with `str(TENSORBOARD_LOG_DIR)`
  - [x] Ensure `TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)` is called in `main()` (in the directory creation block)

- [x] Task 2: Add TensorBoard discoverability to training output (AC: #4)
  - [x] Print `TensorBoard log dir: {TENSORBOARD_LOG_DIR}` in the initialization summary block
  - [x] Print `tensorboard --logdir outputs/tensorboard` launch command after "Starting training..." line
  - [x] Confirm CLAUDE.md already documents `tensorboard --logdir outputs/tensorboard` (it does — no change needed)

- [x] Task 3: Verify SB3 automatic logging covers all ACs (AC: #1, #2)
  - [x] Verify `tensorboard_log=str(TENSORBOARD_LOG_DIR)` is passed to SAC constructor
  - [x] Verify Monitor wrapper is applied to `train_env` (enables episode reward/length logging)
  - [x] Verify `verbose=1` is set on SAC (enables console + logger output)
  - [x] Add a code comment above the SAC constructor's `tensorboard_log` parameter listing what SB3 logs automatically:
    ```
    # SB3 automatically logs to TensorBoard: ep_rew_mean, ep_len_mean,
    # actor_loss, critic_loss, ent_coef, ent_coef_loss, learning_rate
    ```

- [x] Task 4: Write tests for TensorBoard configuration (AC: #3)
  - [x] Add `TestTensorBoardConfig` class to `tests/test_training_config.py`
  - [x] Test `TENSORBOARD_LOG_DIR` is a `Path` object
  - [x] Test `TENSORBOARD_LOG_DIR` path ends with `tensorboard`
  - [x] Test `TENSORBOARD_LOG_DIR` is under `OUTPUT_PATH`

- [x] Task 5: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` — all existing 268+ tests must pass (274 passed)
  - [x] Run new tests — all must pass (3/3 passed)

## Dev Notes

### CRITICAL: TensorBoard logging ALREADY WORKS

The current `train.py` already passes `tensorboard_log=str(OUTPUT_PATH / 'tensorboard')` to the SAC constructor (line 183). SB3 automatically logs episode rewards, policy/value losses, and entropy to TensorBoard when this parameter is set. **The core functionality exists — your job is to formalize it with a named constant, add discoverability, and write tests.**

Do NOT add custom TensorBoard callbacks, custom logging code, or SummaryWriter usage. SB3 handles everything the ACs require.

### What This Story OWNS vs What It Does NOT

**IN SCOPE (this story):**
- `TENSORBOARD_LOG_DIR` constant (extract from inline)
- TensorBoard directory creation
- Print TensorBoard path and launch command
- Code comment documenting what SB3 logs
- Tests for TensorBoard config

**OUT OF SCOPE (do NOT change):**
- SAC hyperparameters, seed management -> Story 4-1 (done)
- Training loop, checkpoints, callbacks -> Story 4-2 (done)
- `model.save()` / `model.load()` -> Story 4-4
- Custom domain-specific TensorBoard metrics (revenue, imbalance_cost, etc.) -> Future enhancement, not in AC
- `model.learn()` call parameters (don't add `log_interval` — SB3 default of 1 is appropriate for SAC)

### What SB3 Logs Automatically

When `tensorboard_log` is set on SAC and the env is wrapped with Monitor, SB3 logs:

| Metric | Source | Description |
|--------|--------|-------------|
| `rollout/ep_rew_mean` | Monitor wrapper | Mean episode reward |
| `rollout/ep_len_mean` | Monitor wrapper | Mean episode length |
| `train/actor_loss` | SAC update | Policy network loss |
| `train/critic_loss` | SAC update | Q-network loss |
| `train/ent_coef` | SAC update | Entropy coefficient (auto-tuned) |
| `train/ent_coef_loss` | SAC update | Entropy coefficient loss |
| `train/learning_rate` | SAC update | Current learning rate |

This fully satisfies AC #1 (episode rewards) and AC #2 (policy/value losses). No custom logging needed.

### Environment Info Dict (NOT logged by SB3 — future reference)

The environment returns per-step info with domain metrics: `pv_actual`, `committed`, `delivered`, `imbalance`, `price`, `revenue`, `imbalance_cost`, `battery_soc`, `battery_throughput`. These are NOT logged to TensorBoard by SB3's default logger. If you need these in TensorBoard, that would require a custom callback — but that's **out of scope** for this story.

### Current Code to Modify

```python
# Line 183 — BEFORE (inline path):
        tensorboard_log=str(OUTPUT_PATH / 'tensorboard')

# AFTER (named constant):
        tensorboard_log=str(TENSORBOARD_LOG_DIR)
```

```python
# New constant (near other path constants, lines 22-25):
TENSORBOARD_LOG_DIR = OUTPUT_PATH / 'tensorboard'
```

```python
# Directory creation (in main(), after existing mkdir block):
TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
```

### Print Output Additions

Add to the initialization summary block (after `print(f"  Eval episodes: {N_EVAL_EPISODES}")`):
```python
    print(f"  TensorBoard log: {TENSORBOARD_LOG_DIR}")
```

Add after `print("Starting training...")`:
```python
    print("Monitor with: tensorboard --logdir outputs/tensorboard")
```

### Test Pattern

Add to `tests/test_training_config.py`:

```python
class TestTensorBoardConfig:
    """Tests for TensorBoard logging configuration."""

    def test_tensorboard_log_dir_is_path(self):
        from src.training.train import TENSORBOARD_LOG_DIR
        assert isinstance(TENSORBOARD_LOG_DIR, Path)

    def test_tensorboard_log_dir_name(self):
        from src.training.train import TENSORBOARD_LOG_DIR
        assert TENSORBOARD_LOG_DIR.name == 'tensorboard'

    def test_tensorboard_log_dir_under_output(self):
        from src.training.train import OUTPUT_PATH, TENSORBOARD_LOG_DIR
        assert TENSORBOARD_LOG_DIR.parent == OUTPUT_PATH
```

### Architecture Compliance

| Requirement | How to Satisfy |
|-------------|----------------|
| File location | `src/training/train.py` (MODIFY existing) |
| Test location | `tests/test_training_config.py` (EXTEND existing) |
| Naming | `TENSORBOARD_LOG_DIR` — UPPER_SNAKE_CASE |
| Logging | TensorBoard via SB3 built-in — no custom SummaryWriter |
| Architecture decision | TensorBoard for training metrics (docs/architecture.md) |
| CLAUDE.md | `tensorboard --logdir outputs/tensorboard` (already documented) |

### Previous Story Intelligence

**From Story 4-2 (Training Loop with Checkpoints):**
- 268 tests passing. Do not regress.
- Constants pattern: UPPER_SNAKE_CASE at module level, imported in tests without SB3.
- SB3 imports deferred to `main()`. New Path constants stay at module level (no SB3 dependency).
- Story 4-2 explicitly deferred TensorBoard config to this story (see Dev Notes "OUT OF SCOPE").
- The `tensorboard_log` param on SAC constructor is the only touchpoint — SB3 handles everything else.

**From Story 4-1 (SAC Agent Configuration):**
- Monitor wrapping is in place on `train_env` (line 119) — this is what enables `ep_rew_mean` logging.
- `verbose=1` set on SAC — enables console and TensorBoard logger output.

**From Architecture:**
- `tensorboard>=2.14.0` in dependencies (architecture.md line 120).
- "TensorBoard for training metrics" is the architectural decision (architecture.md line 174).
- "TensorBoard for training metrics" in Logging Pattern section (architecture.md line 229).

### References

- [Source: docs/epics.md#Story-4.3](../../docs/epics.md) — AC: episode rewards, losses, configurable log dir, TensorBoard visualization
- [Source: docs/architecture.md#Training-Architecture](../../docs/architecture.md) — TensorBoard for logging
- [Source: docs/architecture.md#Reproducibility-Patterns](../../docs/architecture.md) — TensorBoard for training metrics
- [Source: docs/prd.md#FR28](../../docs/prd.md) — FR28: Log training metrics to TensorBoard
- [Source: CLAUDE.md#Monitor-Training](../../CLAUDE.md) — `tensorboard --logdir outputs/tensorboard`
- [Source: src/training/train.py:183](../../src/training/train.py) — Existing `tensorboard_log` parameter on SAC constructor
- [Source: docs/implementation/4-2-training-loop-with-checkpoints.md](../../docs/implementation/4-2-training-loop-with-checkpoints.md) — 268 tests passing, deferred TensorBoard to this story
- [Source: docs/implementation/4-1-sac-agent-configuration.md](../../docs/implementation/4-1-sac-agent-configuration.md) — Monitor wrapping, verbose=1, deferred TensorBoard to this story

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

No issues encountered. All tasks completed on first pass.

### Completion Notes List

- Task 1: Extracted `TENSORBOARD_LOG_DIR = OUTPUT_PATH / 'tensorboard'` as module-level constant (line 26). Replaced inline path in SAC constructor. Added `TENSORBOARD_LOG_DIR.mkdir()` to directory creation block in `main()`.
- Task 2: Added TensorBoard log path to initialization summary and `tensorboard --logdir outputs/tensorboard` launch command after "Starting training..." line.
- Task 3: Verified SB3 automatic logging: `tensorboard_log` param set on SAC, Monitor wrapper on `train_env`, `verbose=1` on SAC. Added code comment documenting what SB3 logs automatically.
- Task 4: Added `TestTensorBoardConfig` class with 3 tests to `tests/test_training_config.py`. All pass.
- Task 5: Full regression suite: 274 tests passed, 0 failures.

### Implementation Plan

Minimal refactoring story — extracted inline TensorBoard path to a named constant, added discoverability print statements, added a documentation comment, and wrote configuration tests. No custom TensorBoard callbacks or SummaryWriter code added (SB3 handles all AC requirements natively).

### File List

- `src/training/train.py` — Modified: added `TENSORBOARD_LOG_DIR` constant, `mkdir` call, replaced inline path, added TensorBoard print statements, added SB3 logging comment
- `tests/test_training_config.py` — Modified: added `TestTensorBoardConfig` class with 3 tests
- `docs/implementation/4-3-tensorboard-logging.md` — Modified: task checkboxes, status, Dev Agent Record
- `docs/implementation/sprint-status.yaml` — Modified: story status updated

### Change Log

- 2026-02-02: Implemented Story 4.3 — TensorBoard Logging. Extracted `TENSORBOARD_LOG_DIR` constant, added discoverability output, verified SB3 automatic logging, added 3 tests. 274/274 tests pass.
- 2026-02-02: Code Review (AI) — 4 issues found (1M, 3L), all fixed: deleted `nul` artifact file, replaced hardcoded path in TensorBoard launch command with constant, improved print phrasing from "Monitor with:" to "Launch TensorBoard:", added TensorBoard metric path prefixes to SB3 logging comment. 274/274 tests still pass.
