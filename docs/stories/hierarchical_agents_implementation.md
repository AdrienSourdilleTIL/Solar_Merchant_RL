# Hierarchical Agents Implementation Plan

## Status: COMPLETE

All 7 stories have been implemented and tested.

## Overview

Replace the monolithic 25-action SAC agent with two specialized agents:
1. **Commitment Agent (High-Level)**: Daily decisions about energy commitments
2. **Battery Agent (Low-Level)**: Hourly battery charge/discharge decisions

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMMITMENT AGENT (High-Level)                │
│  Runs: Once per day at hour 11                                  │
│  Input: ~50-dim obs (forecasts, prices, battery SOC, weather)   │
│  Output: 24-dim action (commitment fractions for tomorrow)      │
│  Reward: End-of-day P&L from those commitments                  │
└─────────────────────────────────────────────────────────────────┘
                              │
            Commitments passed down as context
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BATTERY AGENT (Low-Level)                    │
│  Runs: Every hour                                               │
│  Input: ~30-dim obs (SOC, commitment, actual PV, imbalance)     │
│  Output: 1-dim action (charge/discharge)                        │
│  Reward: Immediate (minimize imbalance cost this hour)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Story 1: Create CommitmentEnv

### Description
Create a new Gymnasium environment `CommitmentEnv` for the high-level commitment agent.

### Acceptance Criteria
- [ ] Environment runs once per day at commitment hour (11:00)
- [ ] Observation space (~55 dimensions):
  - PV forecast for tomorrow (24 hours)
  - Day-ahead prices for tomorrow (24 hours)
  - Current battery SOC (1)
  - Weather features (2: temperature, irradiance)
  - Time features (4: day_sin, day_cos, month_sin, month_cos)
- [ ] Action space: 24-dimensional continuous [0, 1] (commitment fractions)
- [ ] Episode structure:
  - Starts at hour 11, agent makes commitment
  - Environment simulates 24 hours with a battery policy (heuristic or trained agent)
  - Returns reward = total revenue - imbalance costs - degradation
  - Episode terminates after receiving the reward
- [ ] Configurable battery policy (heuristic for initial training)
- [ ] Unit tests verifying observation/action shapes and episode flow

### Technical Notes
- The battery execution during the episode should use a simple heuristic initially:
  - Discharge when short (delivered < committed)
  - Charge when long (delivered > committed) and have excess PV
  - Idle otherwise
- Later, can swap in trained battery agent

### Files to Create
- `src/environment/commitment_env.py`
- `tests/test_commitment_env.py`

---

## Story 2: Create BatteryEnv

### Description
Create a new Gymnasium environment `BatteryEnv` for the low-level battery control agent.

### Acceptance Criteria
- [ ] Environment runs every hour for a 24-hour episode
- [ ] Observation space (~20 dimensions):
  - Current hour (1)
  - Battery SOC (1)
  - Current hour's commitment (1)
  - Current hour's actual PV (1)
  - Remaining hours' commitments (23) - or rolling window
  - Cumulative imbalance so far (1)
  - Current price (1)
  - Actual vs forecast error so far (1)
- [ ] Action space: 1-dimensional continuous [0, 1] (0=discharge, 0.5=idle, 1=charge)
- [ ] Reward: Immediate reward each step
  - Negative of imbalance cost for this hour
  - Small penalty for battery degradation
- [ ] Commitment schedule provided at episode start via reset options
- [ ] Episode: 24 hours (one full day of battery management)
- [ ] Unit tests verifying observation/action shapes and reward calculation

### Technical Notes
- Commitments are fixed for the episode (passed from commitment agent)
- PV forecast error is realized hour by hour
- Agent sees actual PV only for current hour

### Files to Create
- `src/environment/battery_env.py`
- `tests/test_battery_env.py`

---

## Story 3: Create Shared Data Module

### Description
Extract common data loading and simulation logic into a shared module used by both environments.

### Acceptance Criteria
- [ ] Create `src/environment/solar_plant.py` with:
  - `SolarPlant` class encapsulating physical parameters
  - Battery charge/discharge physics with efficiency losses
  - Revenue and imbalance cost calculations
- [ ] Both CommitmentEnv and BatteryEnv use this shared module
- [ ] Unit tests for physics calculations

### Files to Create
- `src/environment/solar_plant.py`
- `tests/test_solar_plant.py`

---

## Story 4: Train Battery Agent

### Description
Create training script for the battery agent with appropriate hyperparameters.

### Acceptance Criteria
- [ ] Training script `src/training/train_battery.py`
- [ ] Uses SAC or PPO algorithm
- [ ] Generates random commitment schedules for training diversity
- [ ] Saves checkpoints and best model
- [ ] TensorBoard logging
- [ ] Evaluation against heuristic battery policies

### Hyperparameter Considerations
- Shorter episodes (24 steps) = can use lower discount factor
- Single action dimension = simpler exploration
- Immediate rewards = faster learning

### Files to Create
- `src/training/train_battery.py`

---

## Story 5: Train Commitment Agent

### Description
Create training script for the commitment agent.

### Acceptance Criteria
- [ ] Training script `src/training/train_commitment.py`
- [ ] Uses trained battery agent for episode simulation
- [ ] Falls back to heuristic if no trained battery agent available
- [ ] Saves checkpoints and best model
- [ ] TensorBoard logging
- [ ] Evaluation against baseline commitment policies

### Training Options
1. **With heuristic battery**: Simpler, faster initial training
2. **With trained battery**: Better once battery agent is trained
3. **End-to-end**: Future option to fine-tune both together

### Files to Create
- `src/training/train_commitment.py`

---

## Story 6: Create Hierarchical Orchestrator

### Description
Create an orchestrator that combines both trained agents for evaluation and deployment.

### Acceptance Criteria
- [ ] `src/environment/hierarchical_orchestrator.py`:
  - Loads both trained agents
  - At hour 11: runs commitment agent
  - Every hour: runs battery agent
  - Tracks full episode metrics
- [ ] Compatible with existing evaluation scripts
- [ ] Comparison metrics vs monolithic agent

### Files to Create
- `src/environment/hierarchical_orchestrator.py`
- `src/evaluation/evaluate_hierarchical.py`

---

## Story 7: Baseline Policies for Hierarchical System

### Description
Create baseline heuristic policies for comparison.

### Acceptance Criteria
- [ ] Commitment baselines:
  - Conservative: 80% of forecast
  - Aggressive: 100% + battery capacity
  - Price-aware: Commit more on high-price hours
- [ ] Battery baselines:
  - Greedy: Always try to meet commitment
  - Conservative: Keep SOC buffer
  - Do-nothing: No battery usage
- [ ] Evaluation comparison script

### Files to Create
- `src/baselines/commitment_policies.py`
- `src/baselines/battery_policies.py`
- `src/evaluation/evaluate_hierarchical_baselines.py`

---

## Implementation Order

```
Story 3 (Shared Module)
    │
    ├──► Story 2 (BatteryEnv)
    │        │
    │        └──► Story 4 (Train Battery)
    │
    └──► Story 1 (CommitmentEnv)
             │
             └──► Story 5 (Train Commitment)
                      │
                      └──► Story 6 (Orchestrator)
                               │
                               └──► Story 7 (Baselines & Eval)
```

**Recommended sequence:**
1. Story 3 (Shared Module) - Foundation
2. Story 2 (BatteryEnv) - Simpler, can test independently
3. Story 4 (Train Battery) - Get battery agent working first
4. Story 1 (CommitmentEnv) - Can use trained battery agent
5. Story 5 (Train Commitment) - Uses battery agent
6. Story 6 (Orchestrator) - Combine both
7. Story 7 (Baselines) - Evaluation comparison

---

## Success Metrics

1. **Learning Speed**: Both agents should show clear learning curves
2. **Sample Efficiency**: Fewer total timesteps than monolithic agent
3. **Final Performance**: Combined agents should match or exceed monolithic agent
4. **Interpretability**: Can analyze each agent's strategy independently

---

## Notes

- Keep the original `SolarMerchantEnv` for comparison
- All new environments should follow the same API patterns
- Use consistent normalization across environments
- Document any deviations from original environment logic
