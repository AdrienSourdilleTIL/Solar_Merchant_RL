---
stepsCompleted:
  - step-01-init
  - step-02-discovery
  - step-03-success
  - step-04-journeys
  - step-05-domain
  - step-06-innovation
  - step-07-project-type
  - step-08-scoping
  - step-09-functional
  - step-10-nonfunctional
  - step-11-polish
inputDocuments:
  - CLAUDE.md
  - README.md
workflowType: 'prd'
projectType: 'greenfield'
documentCounts:
  briefs: 0
  research: 0
  brainstorming: 0
  projectDocs: 2
classification:
  projectType: 'ML/RL Training System (portfolio/demo)'
  domain: 'Energy Trading / Electricity Markets'
  complexity: 'Medium-High'
  projectContext: 'greenfield'
  targetAudience: 'Potential employers, energy/RL enthusiasts'
  successCriteria: 'RL agent beats rule-based baselines'
  v1Scope: 'Day-ahead market only, single plant'
---

# Product Requirements Document - Solar Merchant RL

**Author:** Adrien
**Date:** 2026-01-18

## Success Criteria

### User Success
- Viewers (employers, enthusiasts) immediately understand the problem and why it's interesting
- Clear demonstration of energy market domain knowledge
- Visualizations in README showcase agent performance vs baselines
- Write-up explains the approach and how the agent navigates market dynamics

### Technical Success
- RL agent beats **all three baselines** (Conservative, Aggressive, Price-Aware) on test set (2022-2023)
- Performance quantified as **% revenue increase** vs best baseline
- Clean, modular code structure
- Type hints throughout codebase

### Measurable Outcomes
- Agent outperforms all baselines on test set (pass/fail)
- Revenue improvement % documented and visualized
- README contains complete write-up with visualizations

## Product Scope

**MVP**: Day-ahead market trading environment with SAC agent beating three rule-based baselines. See [Project Scoping & Phased Development](#project-scoping--phased-development) for detailed breakdown.

**Growth**: Reproducibility for external users, generalization testing.

**Vision**: Live API integration, intraday market, multi-plant optimization.

## User Journeys

### Developer Workflow: Building the RL Trading Agent

**Adrien** - ML engineer with energy sector interest, building a portfolio piece to demonstrate RL and domain expertise.

#### Phase 1: Data Foundation

*Opening Scene:* Adrien has raw price and weather data from a previous project. Before writing any RL code, he needs to validate it's fit for purpose.

| Step | Action | Output |
|------|--------|--------|
| 1.1 | Validate price data | Confirm wholesale day-ahead prices, check for anomalies |
| 1.2 | Clean price data | Remove early 2015 flat period, drop unused columns, convert to EUR/MWh |
| 1.3 | Validate PV data | Confirm irradiance-based simulation, check coverage |
| 1.4 | Align datasets | Match date ranges, ensure hourly alignment |
| 1.5 | Scale PV to utility | 5 kW → 20 MW (4000× linear scale) |
| 1.6 | Generate synthetic forecasts | Day-ahead PV forecasts with ~15% RMSE error |
| 1.7 | Derive imbalance prices | Short = 1.5× DA, Long = 0.6× DA |
| 1.8 | Create train/test split | Train: 2015-2021, Test: 2022-2023 |
| 1.9 | Output processed dataset | `data/processed/{train.csv, test.csv}` |

**Risk:** Data quality issues surface late → validate early, document assumptions.

#### Phase 2: Environment

*Rising Action:* With clean data, build the Gym environment that simulates the trading problem.

| Step | Action | Output |
|------|--------|--------|
| 2.1 | Define state space | 84-dim observation (hour, SOC, commitments, forecasts, prices, weather) |
| 2.2 | Define action space | 25-dim continuous (24 hourly commitments + battery action) |
| 2.3 | Implement reward function | Revenue - imbalance cost - battery degradation |
| 2.4 | Implement step logic | Commitment at 11:00, hourly battery dispatch, settlement |
| 2.5 | Implement reset logic | Random episode start, initial SOC |
| 2.6 | Test environment | Verify mechanics with random actions |

**Risk:** Subtle bugs in reward/settlement logic → thorough manual testing.

#### Phase 3: Baselines

*Establishing the bar:* Before RL, implement simple policies to set performance benchmarks.

| Step | Action | Output |
|------|--------|--------|
| 3.1 | Implement Conservative policy | Commit 80% of forecast, battery fills gaps |
| 3.2 | Implement Aggressive policy | Commit 100% + battery capacity |
| 3.3 | Implement Price-Aware policy | Adjust commitment based on price levels |
| 3.4 | Evaluate all baselines | Revenue, imbalance costs, net profit on test set |
| 3.5 | Document baseline results | Benchmark numbers for comparison |

**Output:** Clear performance bar the RL agent must beat.

#### Phase 4: RL Agent

*Climax:* Train the SAC agent and see if it can outperform the baselines.

| Step | Action | Output |
|------|--------|--------|
| 4.1 | Configure SAC | Hyperparameters, network architecture |
| 4.2 | Train agent | 500k timesteps, checkpoint every 50k |
| 4.3 | Monitor training | TensorBoard for reward curves, losses |
| 4.4 | Evaluate on test set | Compare against all three baselines |
| 4.5 | Iterate if needed | Tune hyperparameters, reward shaping |

**Success:** Agent beats all three baselines on test set.

#### Phase 5: Documentation & Polish

*Resolution:* Package the results for portfolio presentation.

| Step | Action | Output |
|------|--------|--------|
| 5.1 | Generate visualizations | Performance charts, training curves |
| 5.2 | Write README | Problem explanation, approach, results |
| 5.3 | Add type hints | Throughout codebase |
| 5.4 | Code cleanup | Consistent structure, clear module organization |
| 5.5 | Final review | End-to-end check |

**Output:** Portfolio-ready project demonstrating RL + energy domain expertise.

### Journey Requirements Summary

| Capability | Revealed By |
|------------|-------------|
| Data processing pipeline | Phase 1 |
| Gym environment | Phase 2 |
| Baseline implementations | Phase 3 |
| SAC training pipeline | Phase 4 |
| Evaluation framework | Phases 3-4 |
| Visualization generation | Phase 5 |
| Documentation | Phase 5 |

## Domain-Specific Requirements

### Market Mechanics

| Aspect | Implementation | Rationale |
|--------|----------------|-----------|
| **Market** | French day-ahead (EPEX SPOT) | Real wholesale prices, hourly resolution |
| **Gate closure** | 12:00 CET | Matches actual EPEX SPOT timing |
| **Imbalance prices** | Real RTE data if available, else synthetic (Short=1.5×DA, Long=0.6×DA) | Realism preferred, synthetic acceptable |
| **Intraday corrections** | Excluded (V2 feature) | Scope control for MVP |

### Physical Constraints

| Constraint | Implementation | Rationale |
|------------|----------------|-----------|
| **Grid charging** | Allowed | Realistic arbitrage opportunity, adds strategic depth |
| **Curtailment** | Allowed with penalty | Physically realistic, discourages waste |
| **Grid export** | Unconstrained | Reasonable assumption for 20 MW scale |
| **Battery efficiency** | 92% round-trip | Industry standard for Li-ion |
| **Battery degradation** | Cost per MWh throughput | Captures real operational cost |

### Documented Simplifications

- **Synthetic imbalance prices** (if real data unavailable): 1.5× day-ahead for short positions, 0.6× for long positions
- **Synthetic forecasts**: Day-ahead PV forecasts with ~15% RMSE, temporally correlated errors
- **No transmission constraints**: Assumes unconstrained grid connection (realistic for plant of this scale)
- **Single price zone**: National French price, no locational marginal pricing

### Data Sources

| Data | Source | Coverage |
|------|--------|----------|
| Day-ahead prices | EPEX SPOT (via existing dataset) | 2015-2025 |
| PV production | PVGIS simulation (Île de Ré) | 2015-2023 |
| Imbalance prices | RTE (to source) or synthetic | TBD |

## ML/RL Training System Requirements

### Algorithm & Architecture

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Primary algorithm** | SAC (Soft Actor-Critic) | Handles continuous action space well, sample-efficient |
| **Exploration** | Consider PPO, TD3 as alternatives | If SAC struggles, have fallback options |
| **Framework** | Stable Baselines3 | Well-maintained, good documentation |

### Observation Space (84 dimensions)

| Component | Dimensions | Notes |
|-----------|------------|-------|
| Current hour | 1 | Time of day |
| Battery SOC | 1 | State of charge |
| Today's commitments | 24 | Hourly commitment schedule |
| Cumulative imbalance | 1 | Running total for the day |
| PV forecast (24h) | 24 | Day-ahead production forecast |
| Price forecast (24h) | 24 | Day-ahead market prices |
| Current actual PV | 1 | Real-time production |
| Weather features | 2 | Temperature, irradiance |
| Cyclical time features | 6 | sin/cos encoding of hour, day, month |

*Open to refinement during implementation if dimensions prove problematic.*

### Action Space (25 dimensions)

| Component | Range | Notes |
|-----------|-------|-------|
| Hourly commitments (24) | [0, 1] | Fraction of max possible delivery |
| Battery action (1) | [0, 1] | 0=full discharge, 0.5=idle, 1=full charge |

*Action masking may be needed if agent struggles with invalid actions.*

### Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Timesteps** | 500k | ~1.4 passes through 7 years training data |
| **Checkpoints** | Every 50k | For recovery and analysis |
| **Evaluation seeds** | 3-5 | Report mean ± std for statistical validity |
| **Hardware** | Dell XPS 13 (CPU) | Training must be feasible on this machine |

### Reward Function

```
reward = revenue - imbalance_cost - battery_degradation_cost
```

Where:
- Revenue = delivered energy × day-ahead price
- Imbalance cost = penalty for over/under delivery (asymmetric)
- Degradation = small cost per MWh battery throughput

*Reward shaping may be explored if training proves difficult.*

### Implementation Considerations

- **Battery stays in V1**: Essential for making the problem non-trivial and demonstrating value
- **Type hints throughout**: Required for code quality
- **Modular structure**: Separate environment, training, evaluation, baselines
- **TensorBoard integration**: For training monitoring and visualization

## Project Scoping & Phased Development

### MVP Strategy

**Approach:** Learning-focused MVP - the goal is validated learning about whether RL can outperform rule-based strategies in this domain. Success is defined by the experiment, not a shipped product.

**Timeline:** Open-ended until agent beats baselines or definitive learnings are captured.

### MVP Feature Set (Phase 1)

**Core Deliverables:**
- Data processing pipeline (clean, align, scale, generate forecasts)
- Gym environment with correct market mechanics
- Three rule-based baselines with documented performance
- SAC training pipeline with TensorBoard monitoring
- Evaluation framework comparing agent vs baselines
- README with domain explanation, approach, and results

**All items in Phase 1-5 of Developer Workflow are essential - no deferrals.**

### Risk Mitigation: Agent Doesn't Beat Baselines

| Escalation Level | Action | Continue if... |
|------------------|--------|----------------|
| 1. Hyperparameter tuning | Grid search learning rate, batch size, network size | Improvement trend visible |
| 2. Algorithm switch | Try PPO, TD3 | New algorithm shows promise |
| 3. Action space simplification | Reduce from 25-dim to simpler formulation | Simpler version learns |
| 4. Reward shaping | Add intermediate rewards for good behavior | Training stabilizes |
| 5. Problem reformulation | Reconsider observation/action design | Clear hypothesis for why |

**Acceptable "failure" outcome:** If after reasonable effort the agent doesn't beat baselines, document *why* - this is still valuable portfolio content showing rigorous methodology and honest analysis.

### Post-MVP Features

**Phase 2 (Growth):**
- Dependency management for reproducibility (requirements.txt, README instructions)
- Generalization testing on unseen market conditions
- Additional visualization types

**Phase 3 (Vision):**
- Live API integration with real daily prices
- Agent making daily decisions as live demo
- Intraday market participation
- Multi-plant portfolio optimization

## Functional Requirements

### Data Processing

- FR1: System can load and validate raw price data from CSV files
- FR2: System can load and validate raw PV production data from CSV files
- FR3: System can clean price data (remove anomalies, drop unused columns, convert units)
- FR4: System can align price and PV datasets to matching date ranges and hourly resolution
- FR5: System can scale PV production from residential (5 kW) to utility scale (20 MW)
- FR6: System can generate synthetic day-ahead PV forecasts with configurable error characteristics
- FR7: System can derive imbalance prices from day-ahead prices using configurable multipliers
- FR8: System can split data into train/test sets by date range
- FR9: System can output processed datasets to CSV files

### Trading Environment

- FR10: Environment can simulate day-ahead market commitment decisions at configurable gate closure time
- FR11: Environment can accept 24-hour commitment schedules as agent actions
- FR12: Environment can simulate hourly battery charge/discharge decisions
- FR13: Environment can track battery state of charge with configurable capacity and power limits
- FR14: Environment can apply round-trip efficiency losses to battery operations
- FR15: Environment can calculate revenue from energy delivery at day-ahead prices
- FR16: Environment can calculate imbalance costs for over/under delivery
- FR17: Environment can calculate battery degradation costs based on throughput
- FR18: Environment can provide observations including forecasts, prices, SOC, and time features
- FR19: Environment can reset to random starting points within the dataset
- FR20: Environment can step through hourly simulation with correct settlement logic

### Baseline Policies

- FR21: System can implement Conservative baseline (commit fraction of forecast)
- FR22: System can implement Aggressive baseline (commit forecast plus battery capacity)
- FR23: System can implement Price-Aware baseline (adjust commitment based on price levels)
- FR24: System can evaluate any policy on a dataset and report performance metrics

### RL Training

- FR25: System can configure and instantiate SAC agent with customizable hyperparameters
- FR26: System can train agent for configurable number of timesteps
- FR27: System can save model checkpoints at configurable intervals
- FR28: System can log training metrics to TensorBoard
- FR29: System can load saved model checkpoints for evaluation or continued training

### Evaluation & Comparison

- FR30: System can evaluate trained agent on test dataset
- FR31: System can compare agent performance against all baseline policies
- FR32: System can calculate revenue, imbalance costs, and net profit metrics
- FR33: System can report percentage improvement over baselines
- FR34: System can run evaluation across multiple seeds and report mean ± std

### Visualization & Documentation

- FR35: System can generate performance comparison charts (agent vs baselines)
- FR36: System can generate training reward curves
- FR37: System can export visualizations as image files for README

## Non-Functional Requirements

### Performance

- NFR1: Training 500k timesteps completes within 24 hours on Dell XPS 13 CPU
- NFR2: Single episode evaluation completes within 5 seconds
- NFR3: Data processing pipeline completes within 5 minutes for full dataset

### Code Quality

- NFR4: All Python files include type hints for function signatures
- NFR5: Code follows consistent style (PEP 8 compliance)
- NFR6: Modules have clear single responsibilities (data, environment, training, evaluation, baselines)
- NFR7: No hardcoded paths - all file paths configurable or relative

### Reproducibility

- NFR8: Training runs with same seed produce identical results
- NFR9: All hyperparameters documented in configuration files or code
- NFR10: Random seeds explicitly set and logged for all stochastic processes
- NFR11: Data processing is deterministic given same input files

### Documentation

- NFR12: README explains problem domain clearly for non-experts
- NFR13: README includes visualizations showing agent vs baseline performance
- NFR14: Code includes docstrings for all public functions
