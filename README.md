# Solar Merchant RL

Reinforcement learning agent for utility-scale solar farm + battery trading on the day-ahead electricity market.

## The Problem

A 20 MW solar farm with 10 MWh battery storage must:
1. **Each day at 11:00**: Commit how much energy to deliver each hour tomorrow
2. **Each hour**: Decide battery charge/discharge to meet commitments
3. **Goal**: Maximize revenue while minimizing imbalance penalties

This is how real merchant solar plants operate in European electricity markets.

## Why RL?

Unlike simple battery arbitrage with fixed prices, this problem has:

| Challenge | Why RL Helps |
|-----------|--------------|
| **Price volatility** | €0-300+/MWh creates complex arbitrage patterns |
| **Forecast uncertainty** | 15-25% error requires learning risk management |
| **Asymmetric penalties** | Short costs 1.5×, long recovers 0.6× - must balance |
| **Sequential decisions** | Today's commitment constrains tomorrow's options |

Simple rules can't capture: "On cloudy autumn days when prices spike at 6 PM, commit conservatively at noon but reserve battery for peak hours."

## Quick Start

```bash
# 1. Prepare data
cd src/data_processing
python prepare_dataset.py

# 2. Evaluate baselines (to set the bar)
cd ../training
python baseline_rules.py

# 3. Train RL agent
python train.py

# 4. Evaluate agent
python evaluate.py
```

## Project Structure

```
Solar_Merchant_RL/
├── data/
│   ├── prices/           # Day-ahead wholesale prices (France)
│   ├── weather/          # Irradiance, temperature, wind
│   └── processed/        # Merged train/test datasets
├── src/
│   ├── environment/      # Gym environment for trading
│   ├── data_processing/  # Data preparation scripts
│   └── training/         # Train, evaluate, baselines
├── models/               # Saved trained models
├── outputs/              # Evaluation results
└── CLAUDE.md             # Developer guide
```

## The Environment

**Actions** (25 continuous values):
- 24 hourly commitment fractions (what to promise delivering tomorrow)
- 1 battery action (charge/discharge for current hour)

**Observations** (84 values):
- Current state (hour, battery SOC, today's commitments)
- Forecasts (PV and prices for next 24 hours)
- Context (weather, time features)

**Reward**:
```
profit = revenue - imbalance_cost - battery_degradation
```

## Plant Specifications

| Parameter | Value |
|-----------|-------|
| Solar capacity | 20 MW |
| Battery capacity | 10 MWh |
| Battery power | 5 MW |
| Round-trip efficiency | 92% |

## Data

Uses real French day-ahead electricity prices (2015-2023) and weather-derived solar production. Forecasts are simulated with realistic error patterns.

- **Training**: 2015-2021 (7 years)
- **Testing**: 2022-2023 (2 years)

## Baseline Policies

| Policy | Strategy |
|--------|----------|
| Conservative | Commit 80% of forecast, buffer with battery |
| Aggressive | Commit 100% + battery, maximize revenue |
| Price-Aware | Adjust commitment based on price levels |

The RL agent should learn to outperform all baselines by discovering non-obvious patterns in price/weather correlations.

## Requirements

- Python 3.10+
- gymnasium
- stable-baselines3
- pandas, numpy
- torch (for SAC)

## Context

This project follows from a residential solar battery project that showed **simple rules beat RL** when prices are static. By moving to:
- Dynamic wholesale prices
- Day-ahead commitment decisions
- Forecast uncertainty

We create a problem complex enough for RL to demonstrate value.

## License

MIT
