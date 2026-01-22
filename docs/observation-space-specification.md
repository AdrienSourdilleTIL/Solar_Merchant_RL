# Observation Space Specification

**Environment:** SolarMerchant-v0
**Total Dimensions:** 84
**Dtype:** float32
**Update:** Story 2-2 (2026-01-22)

## Overview

The observation space provides the agent with all information needed to make trading and battery management decisions. It combines current state, historical data (commitments, imbalances), and future lookahead (forecasts, prices) into a single 84-dimensional vector.

## Complete Dimension Breakdown

| Index  | Dims | Component              | Value Range   | Normalization Method         | Description                                    |
|--------|------|------------------------|---------------|------------------------------|------------------------------------------------|
| 0      | 1    | Current hour           | [0, 0.958]    | `hour / 24.0`                | Hour of day (0-23), normalized to [0, 1)      |
| 1      | 1    | Battery SOC            | [0, 1]        | `soc / capacity_mwh`         | State of charge (0-10 MWh) normalized         |
| 2-25   | 24   | Today's commitments    | [0, 1]        | `commit / plant_cap_mw`      | Hourly delivery commitments for today         |
| 26     | 1    | Cumulative imbalance   | ~[-1, 1]      | `imbalance / plant_cap_mw`   | Running sum of (delivered - committed)        |
| 27-50  | 24   | PV forecast (next 24h) | [0, 1]        | `forecast / plant_cap_mw`    | Solar production forecast for next 24 hours   |
| 51-74  | 24   | Prices (next 24h)      | normalized    | `price / max_abs_price`      | Day-ahead market prices for next 24 hours     |
| 75     | 1    | Current actual PV      | [0, 1]        | `pv_actual / plant_cap_mw`   | Real-time solar production                    |
| 76     | 1    | Temperature            | normalized    | `temp / max_abs_temp`        | Current temperature (°C)                      |
| 77     | 1    | Irradiance             | normalized    | `irr / max_irradiance`       | Current direct irradiance (W/m²)              |
| 78-83  | 6    | Cyclical time features | [-1, 1]       | sin/cos encoding             | Hour, day, month encoded as sine/cosine pairs |

**Total:** 1 + 1 + 24 + 1 + 24 + 24 + 1 + 2 + 6 = **84 dimensions** ✅

## Detailed Component Descriptions

### 1. Current Hour (Index 0)

**Purpose:** Tells the agent what hour of the day it is (0-23).

**Normalization:** `hour / 24.0`
- Hour 0 (midnight) → 0.0
- Hour 11 (commitment deadline) → 0.458
- Hour 23 (11 PM) → 0.958

**Why it matters:**
- Agent needs to know when commitment hour (11:00) arrives
- Different trading strategies may apply at different hours
- Helps agent understand time remaining in current day

### 2. Battery State of Charge (Index 1)

**Purpose:** Current energy stored in the battery (0-10 MWh).

**Normalization:** `soc / battery_capacity_mwh`
- Empty battery (0 MWh) → 0.0
- Half charged (5 MWh) → 0.5
- Full battery (10 MWh) → 1.0

**Why it matters:**
- Determines available flexibility for meeting commitments
- Influences arbitrage opportunities
- Affects risk of under-delivery penalties

### 3. Today's Committed Schedule (Indices 2-25)

**Purpose:** Shows how much energy was committed for each hour of the current day.

**Structure:** 24-dimensional vector, one value per hour (0-23).

**Normalization:** `commitment / plant_capacity_mw`
- No commitment → 0.0
- Maximum commitment (20 MW) → 1.0
- Typical commitment (e.g., 10 MW) → 0.5

**Why it matters:**
- Agent must deliver energy according to these commitments
- Deviations incur imbalance penalties
- Battery must compensate when actual PV differs from forecast

**Special cases:**
- Before commitment hour: May be zeros or previous day's commitments
- After commitment: Fixed for the rest of the day
- During first few hours: Early commitments may already be fulfilled

### 4. Cumulative Imbalance (Index 26)

**Purpose:** Running sum of (delivered - committed) for hours already passed today.

**Calculation:**
```python
for h in range(current_hour):
    delivered = hourly_delivered[h]
    committed = committed_schedule[h]
    cumulative_imbalance += (delivered - committed)
```

**Normalization:** `imbalance / plant_capacity_mw`
- Balanced delivery → 0.0
- Over-delivered by 5 MWh → 0.25
- Under-delivered by 10 MWh → -0.5

**Why it matters:**
- Indicates if agent is ahead or behind on commitments
- Positive: Over-delivered (long position, may receive lower prices)
- Negative: Under-delivered (short position, must pay penalty prices)
- Agent can adjust battery strategy to minimize further imbalance

### 5. PV Forecast Next 24 Hours (Indices 27-50)

**Purpose:** Predicted solar production for the next 24 hours.

**Structure:** 24-dimensional vector, starting from current_idx in dataset.

**Normalization:** `pv_forecast / plant_capacity_mw`
- No production (night) → 0.0
- Maximum production (20 MW) → 1.0
- Typical midday production (15 MW) → 0.75

**Forecast characteristics:**
- Based on weather data with ~15% RMSE error
- Temporally correlated noise (AR(1) with ρ=0.8)
- Slight positive bias (forecasts tend to be optimistic)

**Why it matters:**
- Primary input for deciding commitments at hour 11
- Agent must account for forecast uncertainty
- Battery can compensate for forecast errors

**Edge case:** When `current_idx + 24 >= len(data)`, remaining slots are zero-padded.

### 6. Prices Next 24 Hours (Indices 51-74)

**Purpose:** Day-ahead market prices for the next 24 hours (EUR/MWh).

**Structure:** 24-dimensional vector, starting from current_idx in dataset.

**Normalization:** `price / max_abs_price_in_dataset`
- Typical range in dataset: €0-300/MWh
- After normalization: approximately [0, 1] for positive prices
- Negative prices possible (rare): normalized to negative values

**Why it matters:**
- Determines revenue from delivered energy
- Influences commitment strategy (commit more when prices high)
- Used to calculate imbalance penalties (1.5× for short, 0.6× for long)
- Battery arbitrage opportunities (charge at low prices, discharge at high)

**Edge case:** Same zero-padding as PV forecast near dataset boundary.

### 7. Current Actual PV (Index 75)

**Purpose:** Real-time solar production right now (MWh).

**Normalization:** `pv_actual / plant_capacity_mw`
- No production (night) → 0.0
- Peak production (20 MW) → 1.0
- Cloudy day (8 MW) → 0.4

**Why it matters:**
- Shows actual vs forecast performance
- Agent can compare to committed amount
- Determines how much battery adjustment needed

### 8-9. Weather Features (Indices 76-77)

**Purpose:** Current weather conditions affecting solar production.

#### Temperature (Index 76)
**Normalization:** `temperature / max_abs_temperature_in_dataset`
- Dataset range: typically -10°C to +40°C
- Affects PV panel efficiency (higher temp → slightly lower efficiency)

#### Irradiance (Index 77)
**Normalization:** `irradiance / max_irradiance_in_dataset`
- Direct irradiance in W/m²
- Primary driver of solar production
- Clear sky → high values (~1.0)
- Cloudy/night → low values (~0.0)

**Why it matters:**
- Provides context for current PV production
- Helps agent understand forecast reliability
- May indicate weather patterns affecting future hours

### 10. Cyclical Time Features (Indices 78-83)

**Purpose:** Encode periodic time patterns that influence solar production and prices.

**Structure:** 6 dimensions as sine/cosine pairs:

| Feature     | Indices | Calculation                        | Period   |
|-------------|---------|-------------------------------------|----------|
| Hour of day | 78, 79  | sin(2π·hour/24), cos(2π·hour/24)   | 24 hours |
| Day of year | 80, 81  | sin(2π·day/365), cos(2π·day/365)   | 365 days |
| Month       | 82, 83  | sin(2π·month/12), cos(2π·month/12) | 12 months|

**Why sin/cos encoding?**
- Captures cyclical nature (hour 23 is close to hour 0)
- Smooth continuous representation
- Both components needed to uniquely identify position in cycle
- Already in [-1, 1] range (no additional normalization needed)

**Example values:**
- Midnight (hour 0): sin=0, cos=1
- Noon (hour 12): sin=0, cos=-1
- 6 AM (hour 6): sin=1, cos=0
- 6 PM (hour 18): sin=-1, cos=0

**Why it matters:**
- Solar production has strong diurnal pattern
- Electricity prices often have daily patterns
- Seasonal variations affect both production and demand
- Agent can learn time-dependent strategies

## Normalization Factors

All normalization factors are computed once during environment initialization in `_compute_normalization_factors()`:

```python
self.norm_factors = {
    'price': data['price_eur_mwh'].abs().max() + 1e-8,
    'pv': plant_capacity_mw,  # 20 MW
    'temperature': max(abs(data['temperature_c'].min()),
                      abs(data['temperature_c'].max())) + 1e-8,
    'irradiance': data['irradiance_direct'].max() + 1e-8,
}
```

**Design rationale:**
- **Price:** Uses max absolute value to handle potential negative prices
- **PV:** Uses plant capacity as the physical maximum (20 MW)
- **Temperature:** Uses symmetric max to center around 0
- **Irradiance:** Uses dataset max as practical upper bound
- **Epsilon (1e-8):** Prevents division by zero for edge cases

## Edge Cases and Special Handling

### 1. Dataset Boundary (End of Data)

**Scenario:** When `current_idx + 24 >= len(data)`, there aren't enough future hours for full 24-hour windows.

**Handling:**
```python
if self.current_idx + i < len(self.data):
    forecast_window.append(normalized_forecast)
    price_window.append(normalized_price)
else:
    forecast_window.append(0.0)  # Zero padding
    price_window.append(0.0)     # Zero padding
```

**Impact:** Episodes naturally end before this becomes an issue (24-hour episodes), but the environment remains robust.

### 2. Cumulative Imbalance at Episode Start

**Scenario:** At hour 0 of a new episode, no deliveries have occurred yet.

**Handling:** The loop `for h in range(int(hour))` runs zero iterations when `hour=0`, so `cumulative_imbalance=0.0`.

**Verification:** Test confirms `obs[26] == 0.0` at episode start.

### 3. Missing hourly_delivered Data

**Scenario:** On first observation before any `step()` has been called.

**Handling:**
```python
if hasattr(self, 'hourly_delivered'):
    delivered = self.hourly_delivered.get(h, 0.0)
```

Uses `hasattr()` check and `.get()` with default 0.0 to safely handle missing data.

### 4. Commitment Hour (Hour 11)

**Scenario:** At hour 11, agent must provide commitments for next day.

**Observation behavior:**
- Observation includes forecasts and prices for next 24 hours
- Current day's commitments still shown (indices 2-25)
- Next day's commitments will be set by the action taken this step

### 5. Night Hours (No Solar Production)

**Scenario:** Hours 0-6 and 20-23 typically have zero or near-zero solar production.

**Observation:**
- PV forecast components will be 0.0
- Actual PV will be 0.0
- Battery becomes the only tool for meeting commitments
- Agent must rely solely on stored energy

## Performance Characteristics

**Measured Performance:**
- Average observation construction time: ~4ms per call
- Budget: <10ms per call (well within 200ms/step for 5-second episode target)
- Performance verified by `test_observation_construction_performance`

**Performance factors:**
- `np.concatenate()` is efficient for this size (~84 elements)
- Normalization factors pre-computed during `__init__()`
- Window construction uses simple list iteration (24 iterations)
- No heavy computation or external API calls

## Testing Coverage

Comprehensive test suite in `tests/test_observation_construction.py`:

1. **Structure Tests** (4 tests)
   - Shape validation (84 dimensions)
   - Dtype validation (float32)
   - Type validation (numpy array)
   - Component dimension breakdown

2. **Normalization Factor Tests** (5 tests)
   - Factor existence and keys
   - Positive and non-zero values
   - PV uses plant capacity

3. **Normalization Value Tests** (5 tests)
   - Hour range [0, 1)
   - Battery SOC range [0, 1]
   - Committed schedule range
   - Time features in [-1, 1]
   - Cyclical property (sin²+cos²=1)

4. **Edge Case Tests** (5 tests)
   - Episode start validity
   - Post-step validity
   - Multi-hour progression
   - Zero cumulative imbalance at start
   - Observation consistency with seed

5. **Performance Tests** (1 test)
   - Construction time <10ms

6. **Window Padding Tests** (2 tests)
   - Forecast window structure
   - Price window structure

**Total:** 22 tests, all passing ✅

## Usage Examples

### Example 1: Inspecting an Observation

```python
from src.environment import load_environment

env = load_environment('data/processed/train.csv')
obs, info = env.reset(seed=42)

print(f"Observation shape: {obs.shape}")  # (84,)
print(f"Observation dtype: {obs.dtype}")  # float32

# Decode specific components
current_hour = obs[0] * 24
battery_soc_mwh = obs[1] * env.battery_capacity_mwh
today_commitments = obs[2:26] * env.plant_capacity_mw
cumulative_imbalance_mwh = obs[26] * env.plant_capacity_mw

print(f"Current hour: {current_hour:.0f}")
print(f"Battery SOC: {battery_soc_mwh:.1f} MWh")
print(f"Cumulative imbalance: {cumulative_imbalance_mwh:.2f} MWh")

# Check forecast for next 6 hours
pv_forecast_next_6h = obs[27:33] * env.plant_capacity_mw
prices_next_6h = obs[51:57] * env.norm_factors['price']

print(f"PV forecast (next 6h): {pv_forecast_next_6h}")
print(f"Prices (next 6h): {prices_next_6h}")

# Cyclical time features
hour_sin, hour_cos = obs[78], obs[79]
print(f"Time encoding - sin: {hour_sin:.3f}, cos: {hour_cos:.3f}")
```

### Example 2: Monitoring Observation Changes

```python
env = load_environment('data/processed/train.csv')
obs, info = env.reset()

for step in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    hour = obs[0] * 24
    battery_soc = obs[1] * env.battery_capacity_mwh
    imbalance = obs[26] * env.plant_capacity_mw

    print(f"Step {step}: hour={hour:.0f}, "
          f"battery={battery_soc:.1f} MWh, "
          f"imbalance={imbalance:.2f} MWh")

    if terminated or truncated:
        break
```

### Example 3: Validating Observation Ranges

```python
import numpy as np

env = load_environment('data/processed/train.csv')
obs, info = env.reset()

# Check all values are finite
assert not np.any(np.isnan(obs)), "Observation contains NaN"
assert not np.any(np.isinf(obs)), "Observation contains Inf"

# Check hour is in valid range
assert 0 <= obs[0] < 1, f"Hour {obs[0]} outside [0, 1)"

# Check battery SOC is in valid range
assert 0 <= obs[1] <= 1, f"Battery SOC {obs[1]} outside [0, 1]"

# Check time features are in valid range
time_features = obs[78:84]
assert np.all(time_features >= -1) and np.all(time_features <= 1), \
    "Time features outside [-1, 1]"

# Verify sin^2 + cos^2 = 1 for cyclical features
for i in range(0, 6, 2):
    sin_val, cos_val = obs[78 + i], obs[78 + i + 1]
    magnitude = sin_val**2 + cos_val**2
    assert np.isclose(magnitude, 1.0, atol=1e-5), \
        f"Cyclical pair {i//2} doesn't satisfy sin^2+cos^2=1"

print("✓ All observation validations passed")
```

## Architecture Compliance

This observation space implementation complies with all requirements from [docs/architecture.md#Environment-Architecture](../docs/architecture.md#environment-architecture):

| Requirement                        | Status | Implementation                           |
|------------------------------------|--------|------------------------------------------|
| 84-dimensional observation space   | ✅      | Defined in `__init__` (line 170)       |
| Box space with continuous values   | ✅      | `spaces.Box(shape=(84,), dtype=float32)`|
| Normalized observations            | ✅      | Via `_compute_normalization_factors()`  |
| Float32 dtype                      | ✅      | `.astype(np.float32)` in construction   |
| Includes forecast lookahead        | ✅      | 24-hour PV forecast (indices 27-50)     |
| Includes price lookahead           | ✅      | 24-hour prices (indices 51-74)          |
| Supports Gymnasium API             | ✅      | Returns np.ndarray from reset()/step()  |

## References

- **Source Code:** [src/environment/solar_merchant_env.py:188-289](../src/environment/solar_merchant_env.py#L188-L289)
- **Architecture:** [docs/architecture.md#Environment-Architecture](../docs/architecture.md#environment-architecture)
- **Story Documentation:** [docs/implementation/2-2-observation-construction.md](../docs/implementation/2-2-observation-construction.md)
- **Test Suite:** [tests/test_observation_construction.py](../tests/test_observation_construction.py)
- **Project Context:** [CLAUDE.md#Environment](../CLAUDE.md#environment-srcenvironmentsolar_merchant_envpy)

---

**Last Updated:** 2026-01-22 (Story 2-2)
**Version:** 1.0
**Status:** Validated ✅
