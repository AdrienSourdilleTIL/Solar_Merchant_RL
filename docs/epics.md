---
stepsCompleted:
  - step-01-validate-prerequisites
  - step-02-design-epics
  - step-03-create-stories
  - step-04-final-validation
status: 'complete'
completedAt: '2026-01-19'
inputDocuments:
  - docs/prd.md
  - docs/architecture.md
workflowType: 'epics-and-stories'
project_name: 'Solar Merchant RL'
user_name: 'Adrien'
date: '2026-01-19'
---

# Solar Merchant RL - Epic Breakdown

## Overview

This document provides the complete epic and story breakdown for Solar Merchant RL, decomposing the requirements from the PRD and Architecture into implementable stories.

## Requirements Inventory

### Functional Requirements

**Data Processing (FR1-FR9):**
- FR1: System can load and validate raw price data from CSV files
- FR2: System can load and validate raw PV production data from CSV files
- FR3: System can clean price data (remove anomalies, drop unused columns, convert units)
- FR4: System can align price and PV datasets to matching date ranges and hourly resolution
- FR5: System can scale PV production from residential (5 kW) to utility scale (20 MW)
- FR6: System can generate synthetic day-ahead PV forecasts with configurable error characteristics
- FR7: System can derive imbalance prices from day-ahead prices using configurable multipliers
- FR8: System can split data into train/test sets by date range
- FR9: System can output processed datasets to CSV files

**Trading Environment (FR10-FR20):**
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

**Baseline Policies (FR21-FR24):**
- FR21: System can implement Conservative baseline (commit fraction of forecast)
- FR22: System can implement Aggressive baseline (commit forecast plus battery capacity)
- FR23: System can implement Price-Aware baseline (adjust commitment based on price levels)
- FR24: System can evaluate any policy on a dataset and report performance metrics

**RL Training (FR25-FR29):**
- FR25: System can configure and instantiate SAC agent with customizable hyperparameters
- FR26: System can train agent for configurable number of timesteps
- FR27: System can save model checkpoints at configurable intervals
- FR28: System can log training metrics to TensorBoard
- FR29: System can load saved model checkpoints for evaluation or continued training

**Evaluation & Comparison (FR30-FR34):**
- FR30: System can evaluate trained agent on test dataset
- FR31: System can compare agent performance against all baseline policies
- FR32: System can calculate revenue, imbalance costs, and net profit metrics
- FR33: System can report percentage improvement over baselines
- FR34: System can run evaluation across multiple seeds and report mean ± std

**Visualization & Documentation (FR35-FR37):**
- FR35: System can generate performance comparison charts (agent vs baselines)
- FR36: System can generate training reward curves
- FR37: System can export visualizations as image files for README

### NonFunctional Requirements

**Performance:**
- NFR1: Training 500k timesteps completes within 24 hours on Dell XPS 13 CPU
- NFR2: Single episode evaluation completes within 5 seconds
- NFR3: Data processing pipeline completes within 5 minutes for full dataset

**Code Quality:**
- NFR4: All Python files include type hints for function signatures
- NFR5: Code follows consistent style (PEP 8 compliance)
- NFR6: Modules have clear single responsibilities (data, environment, training, evaluation, baselines)
- NFR7: No hardcoded paths - all file paths configurable or relative

**Reproducibility:**
- NFR8: Training runs with same seed produce identical results
- NFR9: All hyperparameters documented in configuration files or code
- NFR10: Random seeds explicitly set and logged for all stochastic processes
- NFR11: Data processing is deterministic given same input files

**Documentation:**
- NFR12: README explains problem domain clearly for non-experts
- NFR13: README includes visualizations showing agent vs baseline performance
- NFR14: Code includes docstrings for all public functions

### Additional Requirements

**From Architecture:**

- **Project Structure**: Simple flat layout with 5 modules (data_processing, environment, baselines, training, evaluation)
- **Dependencies**: requirements.txt with pinned versions (SB3 >=2.0, Gymnasium >=0.29, PyTorch >=2.0, pandas >=2.0, numpy >=1.24, matplotlib >=3.7, tensorboard >=2.14)
- **Gymnasium API**: Modern Gymnasium >=0.29 API for environment implementation
- **Episode Structure**: Fixed 24-hour episodes matching day-ahead market cycle
- **Observation Normalization**: SB3 VecNormalize for automatic observation scaling
- **Checkpoint Strategy**: Save every 50k steps to models/ directory
- **Seed Management**: Single SEED constant with set_all_seeds() function
- **Naming Conventions**: snake_case for files/functions/variables, PascalCase for classes, UPPER_SNAKE_CASE for constants
- **Type Hints**: Required on all public functions using modern Python 3.10+ syntax
- **Docstrings**: Google style format with Args, Returns, Raises sections

**Key Interfaces Defined:**
- Environment registration: `gymnasium.register(id='SolarMerchant-v0', entry_point='src.environment:SolarMerchantEnv')`
- Baseline interface: `def conservative_policy(obs: np.ndarray) -> np.ndarray`
- Evaluation interface: `def evaluate_policy(policy, env, n_episodes: int) -> dict[str, float]`

### FR Coverage Map

| FR | Epic | Description |
|----|------|-------------|
| FR1 | Epic 1 | Load and validate raw price data |
| FR2 | Epic 1 | Load and validate raw PV production data |
| FR3 | Epic 1 | Clean price data |
| FR4 | Epic 1 | Align price and PV datasets |
| FR5 | Epic 1 | Scale PV to utility scale |
| FR6 | Epic 1 | Generate synthetic forecasts |
| FR7 | Epic 1 | Derive imbalance prices |
| FR8 | Epic 1 | Split train/test sets |
| FR9 | Epic 1 | Output processed datasets |
| FR10 | Epic 2 | Day-ahead commitment simulation |
| FR11 | Epic 2 | Accept 24-hour commitment schedules |
| FR12 | Epic 2 | Battery charge/discharge simulation |
| FR13 | Epic 2 | Track battery SOC |
| FR14 | Epic 2 | Apply battery efficiency losses |
| FR15 | Epic 2 | Calculate revenue |
| FR16 | Epic 2 | Calculate imbalance costs |
| FR17 | Epic 2 | Calculate degradation costs |
| FR18 | Epic 2 | Provide observations |
| FR19 | Epic 2 | Reset to random starting points |
| FR20 | Epic 2 | Step through hourly simulation |
| FR21 | Epic 3 | Conservative baseline policy |
| FR22 | Epic 3 | Aggressive baseline policy |
| FR23 | Epic 3 | Price-Aware baseline policy |
| FR24 | Epic 3 | Evaluate any policy |
| FR25 | Epic 4 | Configure SAC agent |
| FR26 | Epic 4 | Train agent |
| FR27 | Epic 4 | Save checkpoints |
| FR28 | Epic 4 | Log to TensorBoard |
| FR29 | Epic 4 | Load checkpoints |
| FR30 | Epic 5 | Evaluate agent on test set |
| FR31 | Epic 5 | Compare against baselines |
| FR32 | Epic 5 | Calculate metrics |
| FR33 | Epic 5 | Report improvement percentage |
| FR34 | Epic 5 | Run multi-seed evaluation |
| FR35 | Epic 5 | Generate comparison charts |
| FR36 | Epic 5 | Generate training curves |
| FR37 | Epic 5 | Export visualizations |

## Epic List

### Epic 1: Data Foundation
**Goal:** Developer can process raw data into a validated, training-ready dataset with forecasts and derived prices.
**FRs covered:** FR1, FR2, FR3, FR4, FR5, FR6, FR7, FR8, FR9
**User Value:** Complete data pipeline that transforms raw CSVs into train/test datasets ready for RL training.

### Epic 2: Trading Environment
**Goal:** Developer can simulate day-ahead market trading with battery storage in a Gymnasium-compatible environment.
**FRs covered:** FR10, FR11, FR12, FR13, FR14, FR15, FR16, FR17, FR18, FR19, FR20
**User Value:** Working Gym environment that accurately models the solar merchant trading problem with correct market mechanics.

### Epic 3: Baseline Benchmarks
**Goal:** Developer can evaluate rule-based trading strategies to establish performance benchmarks.
**FRs covered:** FR21, FR22, FR23, FR24
**User Value:** Three documented baseline policies with measured performance on test data, setting the bar the RL agent must beat.

### Epic 4: RL Training Pipeline
**Goal:** Developer can train, checkpoint, and monitor a SAC agent learning to trade.
**FRs covered:** FR25, FR26, FR27, FR28, FR29
**User Value:** Complete training pipeline that produces a trained agent with TensorBoard monitoring and saved checkpoints.

### Epic 5: Evaluation & Showcase
**Goal:** Developer can compare agent vs baselines and generate portfolio-ready visualizations.
**FRs covered:** FR30, FR31, FR32, FR33, FR34, FR35, FR36, FR37
**User Value:** Quantified performance comparison with visualizations ready for README, demonstrating whether RL beats baselines.

---

## Epic 1: Data Foundation

**Goal:** Developer can process raw data into a validated, training-ready dataset with forecasts and derived prices.

### Story 1.1: Load and Validate Raw Data

As a developer,
I want to load and validate raw price and PV data from CSV files,
So that I can verify data quality before processing.

**Acceptance Criteria:**

**Given** raw CSV files exist in `data/raw/`
**When** the data loading functions are called
**Then** price data is loaded as a DataFrame with datetime index
**And** PV production data is loaded as a DataFrame with datetime index
**And** basic validation checks pass (no nulls in key columns, reasonable value ranges)
**And** informative errors are raised for missing or malformed files
**And** type hints are included on all public functions (NFR4)

### Story 1.2: Clean and Align Datasets

As a developer,
I want to clean price data and align it with PV data,
So that I have consistent hourly data for the overlapping period.

**Acceptance Criteria:**

**Given** raw data has been loaded
**When** cleaning and alignment functions are called
**Then** price anomalies (early 2015 flat period) are removed
**And** unused columns are dropped
**And** prices are in EUR/MWh
**And** datasets are aligned to matching hourly timestamps
**And** the overlapping date range is 2015-2023

### Story 1.3: Scale PV and Generate Forecasts

As a developer,
I want to scale PV production to utility scale and generate day-ahead forecasts,
So that I have realistic 20 MW plant data with forecast uncertainty.

**Acceptance Criteria:**

**Given** cleaned and aligned data exists
**When** scaling and forecast generation functions are called
**Then** PV production is scaled from 5 kW to 20 MW (4000× factor)
**And** synthetic forecasts have ~15% RMSE error
**And** forecast errors are temporally correlated (AR(1) with ρ≈0.8)
**And** forecast error is proportional to production level

### Story 1.4: Derive Prices and Output Dataset

As a developer,
I want to derive imbalance prices and create train/test splits,
So that I have complete datasets ready for training and evaluation.

**Acceptance Criteria:**

**Given** scaled PV data with forecasts exists
**When** derivation and splitting functions are called
**Then** imbalance prices are derived (Short=1.5×DA, Long=0.6×DA)
**And** data is split into train (2015-2021) and test (2022-2023)
**And** processed data is saved to `data/processed/{train.csv, test.csv}`
**And** full pipeline completes within 5 minutes (NFR3)
**And** running with same inputs produces identical outputs (NFR11)

---

## Epic 2: Trading Environment

**Goal:** Developer can simulate day-ahead market trading with battery storage in a Gymnasium-compatible environment.

### Story 2.1: Environment Structure and Registration

As a developer,
I want a Gymnasium-compatible environment class with proper spaces defined,
So that I can use standard RL training tools.

**Acceptance Criteria:**

**Given** the environment module exists
**When** the environment is imported and registered
**Then** `SolarMerchantEnv` class inherits from `gymnasium.Env`
**And** observation space is `Box` with 84 dimensions
**And** action space is `Box` with 25 dimensions in [0, 1] range
**And** environment registers as `SolarMerchant-v0`
**And** environment loads processed data from `data/processed/`

### Story 2.2: Observation Construction

As a developer,
I want the environment to construct proper observations,
So that the agent has all information needed for decision making.

**Acceptance Criteria:**

**Given** the environment is initialized with data
**When** observations are requested
**Then** observation includes current hour (1 dim)
**And** observation includes battery SOC (1 dim)
**And** observation includes today's commitment schedule (24 dims)
**And** observation includes cumulative imbalance (1 dim)
**And** observation includes PV forecast for next 24h (24 dims)
**And** observation includes prices for next 24h (24 dims)
**And** observation includes current actual PV (1 dim)
**And** observation includes weather features (2 dims)
**And** observation includes cyclical time features (6 dims)

### Story 2.3: Battery Mechanics

As a developer,
I want the environment to simulate battery charge/discharge with efficiency losses,
So that battery operations are physically realistic.

**Acceptance Criteria:**

**Given** battery parameters (10 MWh capacity, 5 MW power, 92% efficiency)
**When** battery actions are executed
**Then** SOC is tracked and bounded [0, 1]
**And** charge/discharge respects power limits (5 MW max)
**And** round-trip efficiency of 92% is applied
**And** degradation cost is calculated per MWh throughput
**And** invalid actions are clipped to valid range

### Story 2.4: Market Settlement Logic

As a developer,
I want the environment to calculate revenue and imbalance costs correctly,
So that the reward signal reflects real market economics.

**Acceptance Criteria:**

**Given** commitments have been made and energy delivered
**When** settlement occurs at each hour
**Then** revenue = delivered energy × day-ahead price
**And** short positions (under-delivery) pay 1.5× day-ahead price
**And** long positions (over-delivery) receive 0.6× day-ahead price
**And** total reward = revenue - imbalance_cost - degradation_cost

### Story 2.5: Episode Flow and Reset

As a developer,
I want proper episode flow with reset to random starting points,
So that training covers diverse market conditions.

**Acceptance Criteria:**

**Given** a configured environment
**When** `reset()` is called
**Then** episode starts at a random day in the dataset
**And** initial SOC is set (configurable, default 0.5)
**And** commitments are cleared
**And** valid observation is returned
**When** `step()` is called with actions
**Then** commitment decisions are processed at hour 11
**And** battery dispatch occurs every hour
**And** episode terminates after 24 hours
**And** single episode evaluation completes within 5 seconds (NFR2)

---

## Epic 3: Baseline Benchmarks

**Goal:** Developer can evaluate rule-based trading strategies to establish performance benchmarks.

### Story 3.1: Conservative Baseline Policy

As a developer,
I want a conservative baseline that commits a fraction of forecast,
So that I have a low-risk benchmark strategy.

**Acceptance Criteria:**

**Given** an observation from the environment
**When** the conservative policy is called
**Then** it commits 80% of forecast for each hour
**And** battery is used to fill delivery gaps
**And** action array is compatible with environment step()
**And** function follows the interface `def conservative_policy(obs: np.ndarray) -> np.ndarray`

### Story 3.2: Aggressive Baseline Policy

As a developer,
I want an aggressive baseline that maximizes commitment,
So that I have a high-risk/high-reward benchmark.

**Acceptance Criteria:**

**Given** an observation from the environment
**When** the aggressive policy is called
**Then** it commits 100% of forecast plus battery discharge capacity
**And** action array is compatible with environment step()
**And** function follows the baseline interface

### Story 3.3: Price-Aware Baseline Policy

As a developer,
I want a price-aware baseline that adjusts based on price levels,
So that I have a smarter rule-based benchmark.

**Acceptance Criteria:**

**Given** an observation from the environment
**When** the price-aware policy is called
**Then** commitment increases when prices are high
**And** commitment decreases when prices are low
**And** battery charges during low prices, discharges during high prices
**And** action array is compatible with environment step()

### Story 3.4: Baseline Evaluation Framework

As a developer,
I want to evaluate any policy and report metrics,
So that I can compare different strategies fairly.

**Acceptance Criteria:**

**Given** a policy function and environment
**When** `evaluate_policy(policy, env, n_episodes)` is called
**Then** policy runs for n_episodes
**And** returns dict with revenue, imbalance_cost, net_profit
**And** metrics are averaged across episodes
**And** results are printed and optionally saved to CSV

---

## Epic 4: RL Training Pipeline

**Goal:** Developer can train, checkpoint, and monitor a SAC agent learning to trade.

### Story 4.1: SAC Agent Configuration

As a developer,
I want to configure a SAC agent with customizable hyperparameters,
So that I can tune the learning process.

**Acceptance Criteria:**

**Given** the environment is available
**When** SAC is configured
**Then** agent is instantiated with SB3 SAC class
**And** hyperparameters are documented (learning rate, batch size, etc.)
**And** network architecture is configurable
**And** environment is wrapped with VecNormalize for observation scaling
**And** all hyperparameters satisfy NFR9 (documented)

### Story 4.2: Training Loop with Checkpoints

As a developer,
I want to train the agent with periodic checkpoints,
So that I can recover from interruptions and analyze progress.

**Acceptance Criteria:**

**Given** a configured SAC agent
**When** training is started
**Then** agent trains for configurable timesteps (default 500k)
**And** checkpoints are saved every 50k steps to `models/`
**And** VecNormalize statistics are saved with checkpoints
**And** training completes within 24 hours on CPU (NFR1)
**And** random seed is set for reproducibility (NFR8, NFR10)

### Story 4.3: TensorBoard Logging

As a developer,
I want training metrics logged to TensorBoard,
So that I can monitor learning progress.

**Acceptance Criteria:**

**Given** training is in progress
**When** metrics are generated
**Then** episode rewards are logged
**And** policy/value losses are logged
**And** logs are saved to `runs/` or configurable directory
**And** TensorBoard can visualize training curves

### Story 4.4: Model Loading and Resumption

As a developer,
I want to load saved checkpoints for evaluation or continued training,
So that I can resume work and evaluate past models.

**Acceptance Criteria:**

**Given** a saved checkpoint exists
**When** loading is requested
**Then** model weights are restored
**And** VecNormalize statistics are restored
**And** agent can continue training from checkpoint
**And** agent can be used for evaluation

---

## Epic 5: Evaluation & Showcase

**Goal:** Developer can compare agent vs baselines and generate portfolio-ready visualizations.

### Story 5.1: Agent Evaluation on Test Set

As a developer,
I want to evaluate the trained agent on the test dataset,
So that I can measure out-of-sample performance.

**Acceptance Criteria:**

**Given** a trained agent and test environment
**When** evaluation is run
**Then** agent performance is measured on 2022-2023 data
**And** revenue, imbalance_cost, net_profit are calculated
**And** results are deterministic with fixed seed

### Story 5.2: Baseline Comparison

As a developer,
I want to compare agent against all baselines,
So that I can determine if RL beats rule-based strategies.

**Acceptance Criteria:**

**Given** agent and baseline evaluation results
**When** comparison is performed
**Then** all four policies (agent + 3 baselines) are compared
**And** percentage improvement over each baseline is calculated
**And** best baseline is identified
**And** pass/fail determined (agent must beat all baselines)

### Story 5.3: Multi-Seed Statistical Evaluation

As a developer,
I want to run evaluation across multiple seeds,
So that I can report statistically valid results.

**Acceptance Criteria:**

**Given** agent and baselines
**When** multi-seed evaluation is run
**Then** 3-5 seeds are used (configurable)
**And** mean ± std is calculated for all metrics
**And** results table shows statistical summary
**And** satisfies NFR for statistical reporting

### Story 5.4: Performance Visualization

As a developer,
I want to generate performance comparison charts,
So that README showcases results visually.

**Acceptance Criteria:**

**Given** evaluation metrics for agent and baselines
**When** visualization is generated
**Then** bar chart compares net profit across policies
**And** chart is saved to `results/figures/`
**And** chart is publication-quality (clear labels, legend)

### Story 5.5: Training Curves Visualization

As a developer,
I want to generate training reward curves,
So that README shows learning progress.

**Acceptance Criteria:**

**Given** TensorBoard logs exist
**When** training curve visualization is generated
**Then** episode reward over time is plotted
**And** chart shows convergence behavior
**And** chart is saved to `results/figures/`

### Story 5.6: Export for README

As a developer,
I want all visualizations exported as image files,
So that they can be embedded in README.

**Acceptance Criteria:**

**Given** generated visualizations
**When** export is performed
**Then** PNG files are saved to `results/figures/`
**And** file names are descriptive (e.g., `performance_comparison.png`)
**And** images are sized appropriately for README display
