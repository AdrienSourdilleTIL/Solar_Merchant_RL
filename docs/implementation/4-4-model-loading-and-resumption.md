# Story 4.4: Model Loading and Resumption

Status: done

## Story

As a developer,
I want to load saved checkpoints for evaluation or continued training,
so that I can resume work and evaluate past models.

## Acceptance Criteria

1. **Given** a saved checkpoint exists (final model, periodic checkpoint, or best model)
   **When** loading is requested via `--resume <path>` CLI argument
   **Then** model weights are restored from the `.zip` file
   **And** the loaded model is used instead of creating a new SAC agent
   **And** a summary prints the checkpoint path and restored timestep count

2. **Given** a model is loaded for continued training
   **When** `model.learn()` is called
   **Then** training continues with `reset_num_timesteps=False` so timestep counter resumes
   **And** replay buffer is restored if a matching `.pkl` file exists alongside the checkpoint
   **And** checkpoints and TensorBoard logging continue from the resumed state

3. **Given** the architecture decision on VecNormalize
   **When** a checkpoint is loaded
   **Then** NO VecNormalize statistics are saved or loaded
   **And** this is documented in a code comment — observations are normalized internally by `SolarMerchantEnv`
   **Note:** This AC is satisfied by documentation. No VecNormalize code needed.

4. **Given** a saved model exists
   **When** it is loaded for evaluation (by `src/evaluation/` or any script)
   **Then** `SAC.load(path)` restores a functioning model
   **And** `model.predict(obs, deterministic=True)` returns valid actions
   **And** a `load_model(checkpoint_path)` helper function is available in `src/training/train.py`

5. **Given** training completes or is interrupted
   **When** the final model is saved
   **Then** the replay buffer is saved alongside the model for future resumption
   **And** the replay buffer path follows the pattern `{model_path}_replay_buffer.pkl`

## Tasks / Subtasks

- [x] Task 1: Add `load_model` helper function to `train.py` (AC: #4)
  - [x] Add `load_model(checkpoint_path: Path) -> SAC` function after `create_env()`
  - [x] Function calls `SAC.load(str(checkpoint_path))` and returns the model
  - [x] Add Google-style docstring with Args/Returns
  - [x] Print loaded model info (path, `model.num_timesteps`)

- [x] Task 2: Add `--resume` CLI argument to `main()` (AC: #1, #2)
  - [x] Add `argparse` import and argument parsing at top of `main()`
  - [x] Accept `--resume CHECKPOINT_PATH` (optional positional or flag argument)
  - [x] When `--resume` is provided: validate path exists, load model with `SAC.load(path, env=train_env)`
  - [x] When `--resume` is NOT provided: create new SAC model (existing behavior)
  - [x] Print resume summary: checkpoint path, restored timesteps, replay buffer status

- [x] Task 3: Continue training with `reset_num_timesteps=False` (AC: #2)
  - [x] When resuming, pass `reset_num_timesteps=False` to `model.learn()` so timestep counter continues
  - [x] When NOT resuming (new model), keep `reset_num_timesteps=True` (default, current behavior)
  - [x] Attempt to load replay buffer from `{checkpoint_path}_replay_buffer.pkl` if it exists
  - [x] Print whether replay buffer was restored or training starts with empty buffer

- [x] Task 4: Save replay buffer alongside final model (AC: #5)
  - [x] After `model.save(final_model_path)`, call `model.save_replay_buffer(str(replay_buffer_path))`
  - [x] Replay buffer path: `MODEL_PATH / 'solar_merchant_final_replay_buffer.pkl'`
  - [x] Print replay buffer save path
  - [x] Add VecNormalize exclusion comment near save block (AC: #3)

- [x] Task 5: Write tests for model loading configuration (AC: all)
  - [x] Add `TestModelLoading` class to `tests/test_training_config.py`
  - [x] Test `load_model` is importable from `src.training.train`
  - [x] Test `load_model` has correct type hints (accepts Path, returns object)
  - [x] Test `MODEL_PATH` points to expected directory
  - [x] Test argparse setup doesn't break when no args provided (simulate `sys.argv`)

- [x] Task 6: Run full test suite to verify no regressions (AC: all)
  - [x] Run `pytest tests/` — all existing 274+ tests must pass (279 passed)
  - [x] Run new tests — all 5 new tests pass

## Dev Notes

### CRITICAL: This story ADDS to an existing file

`src/training/train.py` already has a complete training pipeline (stories 4-1, 4-2, 4-3). You are ADDING model loading and CLI resumption. **Do NOT restructure or move existing code.** Read the full file before making changes.

### What This Story OWNS vs What It Does NOT

**IN SCOPE (this story):**
- `load_model()` helper function
- `--resume` CLI argument with `argparse`
- Replay buffer save/load for training continuity
- `reset_num_timesteps=False` when resuming
- VecNormalize exclusion documentation near load/save code
- Tests for loading configuration

**OUT OF SCOPE (do NOT change):**
- SAC hyperparameters, seed management -> Story 4-1 (done)
- Training loop, checkpoints, callbacks -> Story 4-2 (done)
- TensorBoard configuration -> Story 4-3 (done)
- Evaluation scripts (`src/evaluation/evaluate.py`) -> Epic 5
- `create_env()` function -> Story 4-1 (done)

### SB3 Model Loading API

```python
# Load for evaluation (no env needed for predict):
model = SAC.load("path/to/model.zip")
action, _states = model.predict(obs, deterministic=True)

# Load for continued training (env required):
model = SAC.load("path/to/model.zip", env=train_env)
model.learn(total_timesteps=N, reset_num_timesteps=False)

# Replay buffer save/load (SAC is off-policy, buffer matters):
model.save_replay_buffer("path/to/buffer.pkl")
model.load_replay_buffer("path/to/buffer.pkl")
```

Key details:
- `SAC.load()` restores model weights, hyperparameters, and optimizer state
- `reset_num_timesteps=False` continues the timestep counter from where it stopped
- Replay buffer is NOT saved with `model.save()` — must use `save_replay_buffer()` separately
- No VecNormalize stats to save/load (we don't use VecNormalize)

### CLI Argument Pattern

Use `argparse` for clean CLI handling:

```python
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description='Train Solar Merchant RL agent')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint .zip file to resume training from')
    args = parser.parse_args()
```

This keeps backward compatibility — running `python train.py` with no args works exactly as before.

### Replay Buffer Save Location

Follow the SB3 convention of saving replay buffer alongside the model:

```python
# Final model save (existing):
model.save(str(MODEL_PATH / 'solar_merchant_final'))

# NEW: Replay buffer save:
replay_path = MODEL_PATH / 'solar_merchant_final_replay_buffer.pkl'
model.save_replay_buffer(str(replay_path))
```

Note: `CheckpointCallback` can also save replay buffers by passing `save_replay_buffer=True`. However, replay buffers for SAC with 100k buffer size can be large (~hundreds of MB). For V1, only save the replay buffer with the final model to avoid disk bloat. Future versions could add periodic replay buffer saves.

### load_model Helper Function

```python
def load_model(checkpoint_path: Path) -> "SAC":
    """Load a trained SAC model from a checkpoint file.

    Args:
        checkpoint_path: Path to the saved model .zip file.

    Returns:
        Loaded SAC model ready for evaluation or continued training.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
    """
    from stable_baselines3 import SAC

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = SAC.load(str(checkpoint_path))
    print(f"Loaded model from {checkpoint_path}")
    print(f"  Timesteps trained: {model.num_timesteps:,}")
    return model
```

Note: SB3 import is inside the function (same pattern as `main()`) to keep module-level constants importable without SB3.

### Resume Flow in main()

```python
if args.resume:
    checkpoint_path = Path(args.resume)
    print(f"\nResuming training from: {checkpoint_path}")
    model = SAC.load(str(checkpoint_path), env=train_env,
                     tensorboard_log=str(TENSORBOARD_LOG_DIR))
    # Attempt replay buffer restore
    replay_path = Path(str(checkpoint_path).replace('.zip', '') + '_replay_buffer.pkl')
    if replay_path.exists():
        model.load_replay_buffer(str(replay_path))
        print(f"  Replay buffer restored: {model.replay_buffer.size()} transitions")
    else:
        print(f"  No replay buffer found, starting with empty buffer")
    print(f"  Resuming from timestep: {model.num_timesteps:,}")
    is_resuming = True
else:
    # ... existing new model creation code ...
    is_resuming = False
```

Then in learn() call:
```python
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=not is_resuming,
)
```

### Replay Buffer Path Convention

For a checkpoint at `models/checkpoints/solar_merchant_100000_steps.zip`, the replay buffer would be at `models/checkpoints/solar_merchant_100000_steps_replay_buffer.pkl`. SB3's `save_replay_buffer` adds `.pkl` automatically if not present.

For the final model at `models/solar_merchant_final.zip`, the replay buffer is at `models/solar_merchant_final_replay_buffer.pkl`.

### Test Pattern

Add to `tests/test_training_config.py`:

```python
class TestModelLoading:
    """Tests for model loading and resumption configuration."""

    def test_load_model_importable(self):
        from src.training.train import load_model
        assert callable(load_model)

    def test_load_model_raises_on_missing_file(self):
        from src.training.train import load_model
        with pytest.raises(FileNotFoundError):
            load_model(Path('nonexistent_model.zip'))

    def test_model_path_defined(self):
        from src.training.train import MODEL_PATH
        assert isinstance(MODEL_PATH, Path)
        assert MODEL_PATH.name == 'models'

    def test_argparse_no_args(self):
        """Verify train.py can be imported without CLI args breaking."""
        import src.training.train as train_module
        # Module imports fine — argparse is only in main()
        assert hasattr(train_module, 'main')
```

### Architecture Compliance

| Requirement | How to Satisfy |
|-------------|----------------|
| File location | `src/training/train.py` (MODIFY existing) |
| Test location | `tests/test_training_config.py` (EXTEND existing) |
| Naming | `load_model` — snake_case function, `--resume` CLI flag |
| Type hints | `load_model(checkpoint_path: Path) -> SAC` |
| Docstrings | Google style with Args, Returns, Raises |
| VecNormalize | NOT used — document near save/load code |
| FR29 | Load saved model checkpoints for evaluation or continued training |

### Previous Story Intelligence

**From Story 4-3 (TensorBoard Logging):**
- 274 tests passing. Do not regress.
- `TENSORBOARD_LOG_DIR` constant available for passing to `SAC.load(..., tensorboard_log=...)`.
- SB3 imports deferred to `main()`. Keep `load_model` SB3 import deferred too.

**From Story 4-2 (Training Loop with Checkpoints):**
- `CheckpointCallback` saves to `models/checkpoints/solar_merchant_{N}_steps.zip`.
- `EvalCallback` saves best model to `models/best/best_model.zip`.
- Final model saved to `models/solar_merchant_final.zip`.
- `KeyboardInterrupt` handler saves final model — add replay buffer save here too.
- Elapsed time printing happens after try/except — no changes needed there.

**From Story 4-1 (SAC Agent Configuration):**
- `set_all_seeds(SEED)` called at top of `main()` — keep this before load.
- SB3 imports deferred to `main()`. Pattern: stdlib/numpy/torch at module level, SB3 inside functions.
- `PLANT_CONFIG`, `create_env()` available for environment setup before loading.

**From Architecture:**
- Checkpoints saved to `models/` directory (architecture.md).
- "Training saves -> Evaluation loads" data flow (architecture.md line 189).
- FR29: "System can load saved model checkpoints for evaluation or continued training."
- No VecNormalize (internal normalization decision).

### References

- [Source: docs/epics.md#Story-4.4](../../docs/epics.md) — AC: load checkpoints, restore VecNormalize (N/A), continue training, evaluation
- [Source: docs/architecture.md#Training-Architecture](../../docs/architecture.md) — SAC primary, checkpoints to models/
- [Source: docs/architecture.md#Cross-Component-Dependencies](../../docs/architecture.md) — "Training saves → Evaluation loads"
- [Source: docs/prd.md#FR29](../../docs/prd.md) — FR29: Load saved model checkpoints
- [Source: CLAUDE.md#Training](../../CLAUDE.md) — Model saved to models/
- [Source: src/training/train.py](../../src/training/train.py) — Existing training pipeline (MODIFY, add load/resume)
- [Source: docs/implementation/4-3-tensorboard-logging.md](../../docs/implementation/4-3-tensorboard-logging.md) — 274 tests passing, TENSORBOARD_LOG_DIR constant
- [Source: docs/implementation/4-2-training-loop-with-checkpoints.md](../../docs/implementation/4-2-training-loop-with-checkpoints.md) — Checkpoint paths, final model save, KeyboardInterrupt handler
- [Source: docs/implementation/4-1-sac-agent-configuration.md](../../docs/implementation/4-1-sac-agent-configuration.md) — Deferred SB3 imports, set_all_seeds, create_env
- [Source: SB3 docs — SAC save/load](https://stable-baselines3.readthedocs.io/) — SAC.load(), save_replay_buffer(), reset_num_timesteps

## Senior Developer Review (AI)

**Reviewer:** Adrien on 2026-02-02
**Model:** Claude Opus 4.5 (claude-opus-4-5-20251101)
**Outcome:** Approve (with fixes applied)

### Issues Found: 2 High, 4 Medium, 1 Low

| ID | Severity | Issue | Fix Applied |
|----|----------|-------|-------------|
| H1 | HIGH | `load_model()` docstring claimed "ready for evaluation or continued training" but loads without env — only evaluation works | Docstring updated to clarify evaluation-only; notes `set_env()` for training |
| H2 | HIGH | Replay buffer path derived via `str.replace('.zip', '')` — replaces ALL `.zip` occurrences in path, breaking on paths like `C:\zip_models\model.zip` | Replaced with `checkpoint_path.parent / (checkpoint_path.stem + '_replay_buffer.pkl')` |
| M1 | MEDIUM | No test coverage for replay buffer path derivation logic | Added 3 parameterized tests for final model, checkpoint, and best model paths |
| M2 | MEDIUM | `test_argparse_no_args` only checked `hasattr(main)` — proved nothing beyond module import | Renamed to `test_module_importable_without_cli_args`, added `load_model` check |
| M3 | MEDIUM | No feedback on remaining steps when resuming (500K total with 300K done = only 200K more) | Added remaining timesteps print with target |
| M4 | MEDIUM | `save_replay_buffer()` failure (disk full) unhandled — masks successful model save | Wrapped in try/except with warning message |
| L1 | LOW | Unnecessary f-string with no interpolation on line 199 | Changed to plain string |

### Test Results After Fixes

- **282 passed** (274 existing + 5 original new + 3 review-added), 0 failed, 1 warning
- No regressions

### Verdict

Implementation is solid. All ACs satisfied. The two HIGH issues were a misleading docstring (could cause downstream bugs in Epic 5 evaluation scripts) and a fragile string manipulation that would silently produce wrong paths on certain OS path layouts. Both are fixed. Approved.

## Change Log

- 2026-02-02: Implemented model loading, CLI resumption, replay buffer save/load, and tests (Story 4.4 complete)
- 2026-02-02: Code review — 7 issues found and fixed (2H, 4M, 1L). Tests: 274 → 282 passed

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Initial test run: 1 failure in `test_load_model_raises_on_missing_file` — SB3 import fired before FileNotFoundError check. Fixed by moving existence check before the deferred import.
- Final test run: 279 passed, 0 failed, 1 warning (gymnasium env override)

### Completion Notes List

- Task 1: Added `load_model(checkpoint_path: Path) -> SAC` helper function after `create_env()`. Uses deferred SB3 import pattern. Checks file existence before importing SB3 to allow tests without SB3 installed. Google-style docstring with Args/Returns/Raises.
- Task 2: Added `argparse` with `--resume` flag at top of `main()`. When resuming: validates path, loads model via `SAC.load()` with env and tensorboard_log, attempts replay buffer restore. When not resuming: existing new model creation unchanged. Full backward compatibility.
- Task 3: Added `reset_num_timesteps=not is_resuming` to `model.learn()` call. Resuming continues timestep counter; new training resets (default behavior preserved).
- Task 4: Added `model.save_replay_buffer()` call after final model save. Path: `models/solar_merchant_final_replay_buffer.pkl`. Added VecNormalize exclusion comment near save block. Both normal completion and KeyboardInterrupt paths save the replay buffer.
- Task 5: Added `TestModelLoading` class with 5 tests: importability, FileNotFoundError on missing file, MODEL_PATH validation, argparse compatibility, type hint verification.
- Task 6: Full test suite: 279 passed (274 existing + 5 new), 0 failed, no regressions.

### File List

- `src/training/train.py` (modified) — Added `argparse` import, `load_model()` function, `--resume` CLI argument, resume/new model branching, `reset_num_timesteps` logic, replay buffer save
- `tests/test_training_config.py` (modified) — Added `TestModelLoading` class with 5 tests
- `docs/implementation/sprint-status.yaml` (modified) — Story status: ready-for-dev → in-progress → review
- `docs/implementation/4-4-model-loading-and-resumption.md` (modified) — Tasks marked complete, Dev Agent Record, Change Log, File List, Status updated
