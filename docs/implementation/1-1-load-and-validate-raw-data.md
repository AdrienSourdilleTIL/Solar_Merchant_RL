# Story 1.1: Load and Validate Raw Data

Status: done

## Story

As a developer,
I want to load and validate raw price and PV data from CSV files,
So that I can verify data quality before processing.

## Acceptance Criteria

1. **Given** raw CSV files exist in `data/raw/` (Note: actual paths are `data/prices/` and `data/weather/`)
   **When** the data loading functions are called
   **Then** price data is loaded as a DataFrame with datetime index
   **And** PV production data is loaded as a DataFrame with datetime index

2. **Given** data is loaded
   **When** validation is performed
   **Then** basic validation checks pass (no nulls in key columns, reasonable value ranges)
   **And** informative errors are raised for missing or malformed files

3. **Given** implementation is complete
   **When** code is reviewed
   **Then** type hints are included on all public functions (NFR4)

## Tasks / Subtasks

- [x] Task 1: Validate existing `load_price_data()` function (AC: #1, #2)
  - [x] Verify datetime parsing works correctly
  - [x] Add explicit validation for nulls in key columns
  - [x] Add validation for reasonable price ranges (e.g., -500 to 3000 EUR/MWh)
  - [x] Add informative error messages for missing files
  - [x] Add return type hints

- [x] Task 2: Validate existing `load_weather_data()` function (AC: #1, #2)
  - [x] Verify datetime parsing works correctly
  - [x] Add explicit validation for nulls in key columns
  - [x] Add validation for reasonable PV production values (0 to max capacity)
  - [x] Add informative error messages for missing files
  - [x] Add return type hints

- [x] Task 3: Add comprehensive type hints (AC: #3)
  - [x] Add type hints to all public functions in `prepare_dataset.py`
  - [x] Use `Path` type for file path parameters
  - [x] Use `pd.DataFrame` return types

- [x] Task 4: Add validation utility function
  - [x] Create `validate_dataframe()` helper
  - [x] Check for nulls in critical columns
  - [x] Check for reasonable value ranges
  - [x] Raise `ValueError` with descriptive messages

- [x] Task 5: Update data paths if needed
  - [x] Verify actual data locations match code expectations
  - [x] Current: `data/prices/France_clean.csv` and `data/weather/PV_production_2015_2023.csv`
  - [x] Document any path changes needed

## Dev Notes

### CRITICAL: Brownfield Implementation

**Existing Code Status:** `src/data_processing/prepare_dataset.py` already exists with working implementation.

**This story focuses on:**
1. Validating the existing implementation meets all acceptance criteria
2. Adding explicit data validation that may be missing
3. Ensuring type hints comply with NFR4
4. Adding informative error handling

**DO NOT:**
- Rewrite working code unnecessarily
- Change the existing function signatures (unless required for type hints)
- Modify the core logic that already works

### Architecture Compliance

**Source:** [docs/architecture.md](../architecture.md)

| Requirement | Status | Notes |
|-------------|--------|-------|
| File location | ✅ Compliant | `src/data_processing/prepare_dataset.py` |
| Naming convention | ✅ Compliant | snake_case for functions and variables |
| Type hints | ⚠️ Partial | Need to verify all public functions have hints |
| Docstrings | ✅ Compliant | Google style already present |

### Data File Locations

| Data Type | Expected Path | Actual Path | Status |
|-----------|---------------|-------------|--------|
| Price data | `data/raw/` | `data/prices/France_clean.csv` | ⚠️ Different folder |
| Weather data | `data/raw/` | `data/weather/PV_production_2015_2023.csv` | ⚠️ Different folder |

**Note:** The acceptance criteria mention `data/raw/` but actual data is in separate `data/prices/` and `data/weather/` folders. The existing code correctly handles this - no change needed.

### Existing Function Analysis

**`load_price_data(price_path: Path) -> pd.DataFrame`**
- ✅ Takes Path parameter
- ✅ Returns DataFrame
- ⚠️ No explicit null checking
- ⚠️ No range validation
- ⚠️ No error handling for missing file

**`load_weather_data(weather_path: Path) -> pd.DataFrame`**
- ✅ Takes Path parameter
- ✅ Returns DataFrame
- ⚠️ No explicit null checking
- ⚠️ No range validation for PV production
- ⚠️ No error handling for missing file

### Project Structure Notes

**Alignment with Architecture:**
- Module: `src/data_processing/` - correct
- Main file: `prepare_dataset.py` - correct
- Follows FR1-FR9 responsibility area

**Current Constants (already defined):**
```python
PLANT_CAPACITY_MW = 20.0
ORIGINAL_CAPACITY_KW = 5.0
SCALE_FACTOR = 4000
FORECAST_ERROR_STD = 0.15
IMBALANCE_SHORT_MULTIPLIER = 1.5
IMBALANCE_LONG_MULTIPLIER = 0.6
```

### References

- [Source: docs/architecture.md#Data-Architecture](../architecture.md#data-architecture)
- [Source: docs/architecture.md#Type-Hint-Patterns](../architecture.md#type-hint-patterns)
- [Source: docs/prd.md#FR1-FR2](../prd.md#functional-requirements)
- [Source: docs/epics.md#Story-1.1](../epics.md#story-11-load-and-validate-raw-data)

### Technical Requirements

**From Architecture:**
- Python 3.10+ type hint syntax: `param: Type | None = None`
- Use `Path` for file paths (not strings)
- Google-style docstrings with Args, Returns, Raises sections
- Assert statements sufficient for validation (per architecture decision)

**Validation Ranges:**
- Price: -500 to 3000 EUR/MWh (allows negative prices during oversupply)
- PV production: 0 to PLANT_CAPACITY_MW (20 MW)
- Temperature: -40 to 50°C
- Wind speed: 0 to 50 m/s

### Testing Approach

1. Run existing `prepare_dataset.py` to verify it works
2. Add validation tests for edge cases
3. Test error handling with missing/malformed files
4. Verify output matches expected format

## Dev Agent Record

### Agent Model Used

Claude Opus 4.5 (claude-opus-4-5-20251101)

### Debug Log References

- Tested data loading with validation successfully
- Tested FileNotFoundError handling for missing files
- Ran full pipeline (prepare_dataset.py main) - completed successfully

### Completion Notes List

1. **validate_dataframe()** - Created reusable validation function that checks:
   - Required columns exist
   - No nulls in required columns
   - Values within specified ranges
   - Descriptive error messages for all failures

2. **load_price_data()** - Enhanced with:
   - FileNotFoundError for missing files with helpful message
   - Automatic removal of rows with null datetime (trailing empty rows)
   - Validation for price range (-500 to 3000 EUR/MWh)
   - Full type hints and docstrings

3. **load_weather_data()** - Enhanced with:
   - FileNotFoundError for missing files with helpful message
   - Automatic removal of rows with null datetime (13 rows dropped)
   - Validation for PV production, temperature, wind speed ranges
   - Full type hints and docstrings

4. **Type hints** - Added type annotations to all constants and validation utility function using Python 3.10+ syntax

5. **Data paths verified** - Confirmed actual locations:
   - `data/prices/France_clean.csv` (93,224 rows)
   - `data/weather/PV_production_2015_2023.csv` (78,888 rows after null removal)

### Change Log

- Story created: 2026-01-19
- Status: in-progress
- 2026-01-19: Implemented all tasks - validation, type hints, error handling

### File List

**Files modified:**
- `src/data_processing/prepare_dataset.py` - Added validation, type hints, error handling

**Files verified:**
- `data/prices/France_clean.csv` - ✅ Exists, 93,224 rows, valid format
- `data/weather/PV_production_2015_2023.csv` - ✅ Exists, 78,888 valid rows (13 null rows cleaned)
- `data/processed/train.csv` - ✅ Generated, 61,367 rows
- `data/processed/test.csv` - ✅ Generated, 17,520 rows
- `data/processed/full_dataset.csv` - ✅ Generated, 78,887 rows
