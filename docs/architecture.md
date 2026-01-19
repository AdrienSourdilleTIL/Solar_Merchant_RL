---
stepsCompleted:
  - step-01-init
  - step-02-context
  - step-03-starter
  - step-04-decisions
  - step-05-patterns
  - step-06-structure
  - step-07-validation
  - step-08-complete
status: 'complete'
completedAt: '2026-01-19'
inputDocuments:
  - docs/prd.md
  - CLAUDE.md
workflowType: 'architecture'
project_name: 'Solar Merchant RL'
user_name: 'Adrien'
date: '2026-01-19'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements (37 total):**
- Data Processing (9): Load, clean, align, scale, generate forecasts, derive imbalance prices
- Trading Environment (11): Gym env with commitment, battery, settlement mechanics
- Baseline Policies (4): Conservative, Aggressive, Price-Aware implementations
- RL Training (5): SAC configuration, training loop, checkpointing, TensorBoard
- Evaluation (5): Agent vs baseline comparison, metrics, statistical reporting
- Visualization (3): Performance charts, training curves, README export

**Non-Functional Requirements (14 total):**
- Performance: CPU-trainable (<24h for 500k steps), fast evaluation
- Code Quality: Type hints, PEP 8, modular structure
- Reproducibility: Deterministic seeds, documented hyperparameters
- Documentation: Clear README, docstrings

**Scale & Complexity:**
- Primary domain: ML/RL Training System
- Complexity level: Medium
- Estimated architectural components: 5 modules

### Technical Constraints & Dependencies

- Python 3.10+ with Stable Baselines3
- CPU-only training (Dell XPS 13)
- File-based data I/O (CSV)
- No external API dependencies in MVP

### Cross-Cutting Concerns Identified

- **Configuration Management**: Hyperparameters, file paths, plant parameters
- **Random Seed Control**: Must be consistent across data processing, environment, training
- **Type Hints**: Required throughout all modules
- **Logging/Monitoring**: TensorBoard integration for training

## Technical Foundation

### Project Structure Decision

**Selected Structure:** Simple Flat Layout

```
solar_merchant_rl/
├── src/
│   ├── data_processing/    # FR1-9: Data pipeline
│   │   ├── __init__.py
│   │   └── prepare_dataset.py
│   ├── environment/        # FR10-20: Gym environment
│   │   ├── __init__.py
│   │   └── solar_merchant_env.py
│   ├── baselines/          # FR21-24: Rule-based policies
│   │   ├── __init__.py
│   │   └── baseline_policies.py
│   ├── training/           # FR25-29: RL training
│   │   ├── __init__.py
│   │   └── train.py
│   └── evaluation/         # FR30-37: Evaluation & visualization
│       ├── __init__.py
│       ├── evaluate.py
│       └── visualize.py
├── data/
│   ├── raw/                # Original price/weather CSVs
│   └── processed/          # Generated train/test sets
├── models/                 # Saved checkpoints
├── tests/                  # Unit tests (optional for MVP)
├── requirements.txt
├── README.md
└── CLAUDE.md
```

**Rationale:**
- Maps directly to FR capability areas
- Simple navigation for portfolio reviewers
- No packaging overhead
- Matches structure already outlined in CLAUDE.md

### Dependency Management

**Selected Approach:** requirements.txt with pinned versions

```
# Core ML/RL
stable-baselines3>=2.0.0
gymnasium>=0.29.0
torch>=2.0.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Visualization
matplotlib>=3.7.0
tensorboard>=2.14.0

# Type Checking (dev)
mypy>=1.0.0
```

**Rationale:**
- Simple and universally understood
- Pinned versions ensure reproducibility (NFR8-11)
- No poetry/conda complexity for a solo project

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
- Gymnasium as environment API (affects all env code)
- Direct SB3 usage (affects all training code)
- CSV for data I/O (affects data pipeline)

**Important Decisions (Shape Architecture):**
- Episode structure (fixed 24-hour)
- Observation normalization (VecNormalize)
- Checkpoint strategy (periodic 50k)

**Deferred Decisions (Post-MVP):**
- Hyperparameter config files (can add later)
- Advanced logging (can add later)
- Parquet for faster I/O (can add later)

### Data Architecture

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Storage format** | CSV | Human-readable, matches existing data, simple |
| **Data validation** | Assert statements | Sufficient for single-developer project |
| **Configuration** | Python constants | No config parsing overhead, easy to modify |

### Environment Architecture

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Gym API** | Gymnasium >=0.29 | Modern API, actively maintained |
| **Episode length** | Fixed 24 hours | Matches day-ahead market cycle |
| **Observation normalization** | SB3 VecNormalize | Automatic, saves running statistics |
| **Action handling** | Raw [0,1] range | Environment interprets as fractions |

### Training Architecture

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **RL library** | Stable Baselines3 >=2.0 | Well-documented, reliable |
| **Algorithm** | SAC (primary), PPO/TD3 (fallback) | Per PRD risk mitigation |
| **Checkpoints** | Every 50k steps | Per PRD spec |
| **Logging** | TensorBoard | Industry standard for RL |

### Evaluation Architecture

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Baseline interface** | Simple functions | No class overhead for rule-based |
| **Metrics** | Per-episode + aggregated | Both for analysis |
| **Visualization** | matplotlib | Standard, no extra deps |
| **Statistical reporting** | Mean ± std over seeds | Per PRD NFR requirements |

### Cross-Component Dependencies

- Data pipeline outputs → Environment loads
- Environment → Training wraps with VecNormalize
- Training saves → Evaluation loads
- Evaluation → Visualization consumes metrics

## Implementation Patterns & Consistency Rules

### Naming Patterns

| Category | Convention | Example |
|----------|------------|---------|
| **Files** | `snake_case.py` | `prepare_dataset.py`, `solar_merchant_env.py` |
| **Functions** | `snake_case` | `load_price_data()`, `calculate_reward()` |
| **Classes** | `PascalCase` | `SolarMerchantEnv`, `ConservativePolicy` |
| **Constants** | `UPPER_SNAKE_CASE` | `BATTERY_CAPACITY_MWH`, `SEED` |
| **Variables** | `snake_case` | `pv_forecast`, `day_ahead_price` |
| **DataFrame columns** | `snake_case` | `price_eur_mwh`, `pv_production_mw` |

### Code Structure Patterns

**Import Organization:**
1. Standard library imports
2. Third-party imports (numpy, pandas, torch, sb3)
3. Local imports (from src.*)

**Docstring Format:** Google style with type hints
- All public functions must have docstrings
- Args, Returns, Raises sections as needed

**Constants Placement:**
- Module-level constants at top of relevant file
- No separate config files for MVP
- Group by category with comments

### Reproducibility Patterns

**Seed Management:**
- Single `SEED` constant defined at entry point
- `set_all_seeds(seed)` function for consistent seeding
- Environment seeds passed via `reset(seed=...)`

**Logging Pattern:**
- TensorBoard for training metrics
- Minimal `print()` for progress (not logging module)
- Results saved to CSV for post-analysis

### Type Hint Patterns

**Function Signatures:**
```python
def function_name(param: Type, optional: Type | None = None) -> ReturnType:
```

**Common Types:**
- `np.ndarray` for arrays
- `pd.DataFrame` for dataframes
- `dict[str, float]` for metrics
- `Path` for file paths (not strings)

### Anti-Patterns to Avoid

- Hardcoded file paths (use relative or Path objects)
- camelCase for Python variables
- Magic numbers without named constants
- Scattered seed setting
- Missing type hints on public functions

## Project Structure & Boundaries

### Complete Directory Structure

```
solar_merchant_rl/
├── README.md                    # Domain explanation, results
├── CLAUDE.md                    # AI agent guidance
├── requirements.txt             # Dependencies
├── .gitignore
├── data/
│   ├── raw/                     # Input CSVs (gitignored)
│   └── processed/               # Generated datasets (gitignored)
├── models/                      # Checkpoints (gitignored)
├── results/
│   ├── figures/                 # Generated plots
│   └── metrics/                 # CSV results
├── src/
│   ├── __init__.py
│   ├── data_processing/
│   │   ├── __init__.py
│   │   └── prepare_dataset.py   # FR1-9
│   ├── environment/
│   │   ├── __init__.py
│   │   └── solar_merchant_env.py  # FR10-20
│   ├── baselines/
│   │   ├── __init__.py
│   │   └── baseline_policies.py   # FR21-24
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py             # FR25-29
│   └── evaluation/
│       ├── __init__.py
│       ├── evaluate.py          # FR30-34
│       └── visualize.py         # FR35-37
└── tests/                       # Optional
    └── test_environment.py
```

### Module Boundaries

| Module | Input | Output | Dependencies |
|--------|-------|--------|--------------|
| `data_processing` | Raw CSVs | Processed train/test CSVs | pandas, numpy |
| `environment` | Processed CSVs | Gym env instance | gymnasium |
| `baselines` | Observations | Actions | numpy |
| `training` | Env instance | Model checkpoints | sb3, torch |
| `evaluation` | Model + Env | Metrics dict | sb3, numpy |
| `visualize` | Metrics | PNG figures | matplotlib |

### Data Flow

1. `prepare_dataset.py` → `data/processed/`
2. `train.py` loads processed data → creates env → trains → saves to `models/`
3. `evaluate.py` loads model + baselines → runs episodes → outputs metrics
4. `visualize.py` consumes metrics → generates `results/figures/`

### Key Interfaces

**Environment Registration:**
```python
gymnasium.register(id='SolarMerchant-v0', entry_point='src.environment:SolarMerchantEnv')
```

**Baseline Interface:**
```python
def conservative_policy(obs: np.ndarray) -> np.ndarray:
    """Returns action array compatible with env.step()"""
```

**Evaluation Interface:**
```python
def evaluate_policy(policy, env, n_episodes: int) -> dict[str, float]:
    """Returns metrics: revenue, imbalance_cost, net_profit"""
```

## Architecture Validation Results

### Coherence Validation ✅

All architectural decisions are compatible and coherent:
- Technology stack (Python 3.10+, SB3, Gymnasium) is well-tested combination
- Patterns align with Python ecosystem conventions
- Structure supports all defined boundaries and data flows

### Requirements Coverage ✅

**All 37 Functional Requirements covered:**
- Data Processing (FR1-9): `src/data_processing/`
- Environment (FR10-20): `src/environment/`
- Baselines (FR21-24): `src/baselines/`
- Training (FR25-29): `src/training/`
- Evaluation (FR30-34): `src/evaluation/`
- Visualization (FR35-37): `src/evaluation/visualize.py`

**All 14 Non-Functional Requirements addressed:**
- Performance, Code Quality, Reproducibility, Documentation all have architectural support

### Implementation Readiness ✅

Architecture is ready for AI agent implementation:
- All decisions documented with versions
- Project structure is complete and specific
- Patterns prevent implementation conflicts
- Interfaces defined with type signatures

### Architecture Completeness Checklist

- [x] Project context analyzed
- [x] Technical constraints identified
- [x] All technology decisions made with versions
- [x] Naming conventions established
- [x] Structure patterns defined
- [x] Complete directory structure
- [x] Module boundaries established
- [x] Data flow documented

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION

**Confidence Level:** High

**Key Strengths:**
- Simple, maintainable structure
- Clear FR-to-module mapping
- Standard Python/ML conventions
- Well-defined interfaces

**Implementation Order:**
1. Data processing pipeline
2. Environment implementation
3. Baseline policies
4. Training script
5. Evaluation and visualization

## Architecture Completion Summary

### Workflow Completion

**Architecture Decision Workflow:** COMPLETED ✅
**Total Steps Completed:** 8
**Date Completed:** 2026-01-19
**Document Location:** docs/architecture.md

### Final Architecture Deliverables

**Complete Architecture Document**
- All architectural decisions documented with specific versions
- Implementation patterns ensuring AI agent consistency
- Complete project structure with all files and directories
- Requirements to architecture mapping
- Validation confirming coherence and completeness

**Implementation Ready Foundation**
- 12 architectural decisions made (data, environment, training, evaluation categories)
- 6 implementation pattern categories defined (naming, code structure, reproducibility, type hints)
- 5 architectural modules specified
- 37 functional + 14 non-functional requirements fully supported

**AI Agent Implementation Guide**
- Technology stack with verified versions (SB3 >=2.0, Gymnasium >=0.29, PyTorch >=2.0)
- Consistency rules that prevent implementation conflicts
- Project structure with clear boundaries
- Integration patterns and communication standards

### Implementation Handoff

**For AI Agents:**
This architecture document is your complete guide for implementing Solar Merchant RL. Follow all decisions, patterns, and structures exactly as documented.

**First Implementation Priority:**
1. Create project directory structure as specified
2. Create `requirements.txt` with pinned dependencies
3. Implement `src/data_processing/prepare_dataset.py` (FR1-9)

**Development Sequence:**
1. Initialize project structure per architecture
2. Implement data processing pipeline
3. Build Gymnasium environment
4. Create baseline policies
5. Implement SAC training script
6. Build evaluation and visualization

### Quality Assurance Checklist

**Architecture Coherence**
- [x] All decisions work together without conflicts
- [x] Technology choices are compatible
- [x] Patterns support the architectural decisions
- [x] Structure aligns with all choices

**Requirements Coverage**
- [x] All 37 functional requirements are supported
- [x] All 14 non-functional requirements are addressed
- [x] Cross-cutting concerns are handled (config, seeds, logging)
- [x] Integration points are defined

**Implementation Readiness**
- [x] Decisions are specific and actionable
- [x] Patterns prevent agent conflicts
- [x] Structure is complete and unambiguous
- [x] Examples and interfaces are provided

---

**Architecture Status:** READY FOR IMPLEMENTATION ✅

**Next Phase:** Begin implementation using the architectural decisions and patterns documented herein.

**Document Maintenance:** Update this architecture when major technical decisions are made during implementation.
