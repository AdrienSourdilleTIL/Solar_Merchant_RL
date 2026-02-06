"""
Data Processing for Solar Merchant RL

Prepares the dataset for a utility-scale solar farm + battery trading on
the day-ahead market.

Inputs:
- Day-ahead wholesale prices (France)
- Weather/irradiance data (for PV production calculation)

Outputs:
- Processed dataset with:
  - Hourly day-ahead prices
  - Scaled PV production (20 MW plant)
  - Simulated day-ahead PV forecasts (with realistic error)
  - Derived imbalance prices
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
PLANT_CAPACITY_MW: float = 20.0  # 20 MW solar farm
ORIGINAL_CAPACITY_KW: float = 5.0  # Original data was ~5 kW residential
SCALE_FACTOR: float = (PLANT_CAPACITY_MW * 1000) / ORIGINAL_CAPACITY_KW  # 4000x

# Forecast error parameters (standard deviations as fraction of capacity)
# Day-ahead forecast error is typically 15-25% RMSE for solar
FORECAST_ERROR_STD: float = 0.15  # 15% of actual production

# Imbalance price parameters
# When short (under-delivered): pay premium to buy balancing energy
# When long (over-delivered): receive discount for excess energy
IMBALANCE_SHORT_MULTIPLIER: float = 1.5  # Pay 1.5x day-ahead price when short
IMBALANCE_LONG_MULTIPLIER: float = 0.3   # Receive 0.3x day-ahead price when long (realistic for solar peaks)

# Validation range constants
PRICE_MIN_EUR_MWH: float = -500.0  # Allow negative prices during oversupply
PRICE_MAX_EUR_MWH: float = 3000.0  # Maximum reasonable price
TEMPERATURE_MIN_C: float = -40.0
TEMPERATURE_MAX_C: float = 50.0
WIND_SPEED_MIN_MS: float = 0.0
WIND_SPEED_MAX_MS: float = 50.0


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str],
    column_ranges: dict[str, tuple[float, float]] | None = None,
    name: str = "DataFrame"
) -> None:
    """
    Validate a DataFrame for nulls and value ranges.

    Args:
        df: DataFrame to validate.
        required_columns: List of column names that must exist and have no nulls.
        column_ranges: Dict mapping column names to (min, max) valid value tuples.
        name: Descriptive name for error messages.

    Raises:
        ValueError: If validation fails with descriptive message.
    """
    # Check for missing required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{name}: Missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # Check for nulls in required columns
    for col in required_columns:
        null_count = df[col].isna().sum()
        if null_count > 0:
            raise ValueError(
                f"{name}: Column '{col}' contains {null_count} null values "
                f"({100 * null_count / len(df):.1f}% of rows)"
            )

    # Check value ranges if specified
    if column_ranges:
        for col, (min_val, max_val) in column_ranges.items():
            if col not in df.columns:
                continue  # Skip if column doesn't exist
            below_min = (df[col] < min_val).sum()
            above_max = (df[col] > max_val).sum()
            if below_min > 0 or above_max > 0:
                actual_min = df[col].min()
                actual_max = df[col].max()
                raise ValueError(
                    f"{name}: Column '{col}' has values outside valid range "
                    f"[{min_val}, {max_val}]. "
                    f"Actual range: [{actual_min:.2f}, {actual_max:.2f}]. "
                    f"Values below min: {below_min}, above max: {above_max}"
                )


def load_price_data(price_path: Path) -> pd.DataFrame:
    """
    Load and validate day-ahead price data from CSV.

    Args:
        price_path: Path to the price data CSV file.

    Returns:
        DataFrame with columns ['datetime', 'price_eur_mwh'].

    Raises:
        FileNotFoundError: If the price file does not exist.
        ValueError: If required columns are missing or data validation fails.
    """
    # Check file exists
    if not price_path.exists():
        raise FileNotFoundError(
            f"Price data file not found: {price_path}. "
            f"Please ensure the file exists at this location."
        )

    df = pd.read_csv(price_path, parse_dates=['datetime'])

    # Drop rows with null datetime (trailing empty rows in CSV)
    initial_len = len(df)
    df = df.dropna(subset=['datetime'])
    if len(df) < initial_len:
        dropped = initial_len - len(df)
        print(f"Note: Dropped {dropped} rows with null datetime from price data")

    # Validate required columns exist
    if 'price' not in df.columns:
        raise ValueError(
            f"Price data file missing required column 'price'. "
            f"Available columns: {list(df.columns)}"
        )
    if 'datetime' not in df.columns:
        raise ValueError(
            f"Price data file missing required column 'datetime'. "
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the wholesale price column, rename for clarity
    # Price is in EUR/kWh, convert to EUR/MWh for utility scale
    df['price_eur_mwh'] = df['price'] * 1000
    df = df[['datetime', 'price_eur_mwh']].copy()

    # Validate the processed data
    validate_dataframe(
        df,
        required_columns=['datetime', 'price_eur_mwh'],
        column_ranges={'price_eur_mwh': (PRICE_MIN_EUR_MWH, PRICE_MAX_EUR_MWH)},
        name="Price data"
    )

    return df


def load_weather_data(weather_path: Path) -> pd.DataFrame:
    """
    Load and validate weather/PV production data from CSV.

    Args:
        weather_path: Path to the weather/PV data CSV file.

    Returns:
        DataFrame with columns ['datetime', 'pv_actual_mwh', 'irradiance_direct',
        'irradiance_diffuse', 'temperature_c', 'wind_speed_ms'].

    Raises:
        FileNotFoundError: If the weather file does not exist.
        ValueError: If required columns are missing or data validation fails.
    """
    # Check file exists
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather data file not found: {weather_path}. "
            f"Please ensure the file exists at this location."
        )

    df = pd.read_csv(weather_path, parse_dates=['datetime'])

    # Drop rows with null datetime (trailing empty rows in CSV)
    initial_len = len(df)
    df = df.dropna(subset=['datetime'])
    if len(df) < initial_len:
        dropped = initial_len - len(df)
        print(f"Note: Dropped {dropped} rows with null datetime from weather data")

    # Validate required columns exist
    required_raw_cols = ['datetime', 'P', 'Gb(i)', 'Gd(i)', 'T2m', 'WS10m']
    missing_cols = [col for col in required_raw_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Weather data file missing required columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )

    # P is production in kWh for original ~5kW system
    # Scale to 20 MW plant (in MWh)
    df['pv_actual_mwh'] = df['P'] * SCALE_FACTOR / 1000  # Convert kWh to MWh

    # Keep relevant columns
    # Irradiance components useful for understanding production patterns
    df = df[['datetime', 'pv_actual_mwh', 'Gb(i)', 'Gd(i)', 'T2m', 'WS10m']].copy()
    df.columns = ['datetime', 'pv_actual_mwh', 'irradiance_direct', 'irradiance_diffuse',
                  'temperature_c', 'wind_speed_ms']

    # Validate the processed data
    validate_dataframe(
        df,
        required_columns=['datetime', 'pv_actual_mwh', 'temperature_c', 'wind_speed_ms'],
        column_ranges={
            # Allow 10% over capacity for real-world variations in original data
            'pv_actual_mwh': (0.0, PLANT_CAPACITY_MW * 1.1),
            'temperature_c': (TEMPERATURE_MIN_C, TEMPERATURE_MAX_C),
            'wind_speed_ms': (WIND_SPEED_MIN_MS, WIND_SPEED_MAX_MS),
        },
        name="Weather data"
    )

    return df


def generate_pv_forecast(actual: pd.Series, std_frac: float = FORECAST_ERROR_STD,
                         seed: int = 42) -> pd.Series:
    """
    Generate day-ahead PV forecast with realistic error.

    Day-ahead forecasts have ~15-25% RMSE error. Error is:
    - Proportional to actual production (no error at night)
    - Slightly biased toward overestimation (optimistic forecasts)
    - Temporally correlated (errors persist over hours)

    Args:
        actual: Actual PV production series
        std_frac: Standard deviation as fraction of production
        seed: Random seed for reproducibility

    Returns:
        Forecast series
    """
    np.random.seed(seed)
    n = len(actual)

    # Generate temporally correlated noise using AR(1) process
    # Correlation of ~0.8 between consecutive hours
    noise = np.zeros(n)
    noise[0] = np.random.normal(0, 1)
    for i in range(1, n):
        noise[i] = 0.8 * noise[i-1] + np.sqrt(1 - 0.8**2) * np.random.normal(0, 1)

    # Scale noise by production level (no error when production is 0)
    # Unbiased forecast - errors are symmetric around actual
    error = noise * std_frac * actual

    # Forecast = actual + error, but clipped to [0, capacity]
    forecast = actual + error
    forecast = forecast.clip(lower=0, upper=PLANT_CAPACITY_MW)

    return forecast


def derive_imbalance_prices(price_da: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Derive imbalance prices from day-ahead prices.

    In reality, imbalance prices depend on system conditions and can be
    very volatile. This is a simplified model:
    - Short (under-delivered): pay premium (1.5x DA price)
    - Long (over-delivered): receive discount (0.6x DA price)

    Returns:
        Tuple of (price_imbalance_short, price_imbalance_long)
    """
    price_short = price_da * IMBALANCE_SHORT_MULTIPLIER
    price_long = price_da * IMBALANCE_LONG_MULTIPLIER

    # Floor at 0 (even if DA price is negative, imbalance shouldn't pay you extra)
    price_short = price_short.clip(lower=0)
    price_long = price_long.clip(lower=0)

    return price_short, price_long


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based features for the model."""
    df = df.copy()

    # Extract time components
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month
    df['day_of_year'] = df['datetime'].dt.dayofyear

    # Cyclical encodings
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
    df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)

    return df


def main():
    # Paths
    base_path = Path(__file__).parent.parent.parent
    price_path = base_path / 'data' / 'prices' / 'France_clean.csv'
    weather_path = base_path / 'data' / 'weather' / 'PV_production_2015_2023.csv'
    output_path = base_path / 'data' / 'processed'

    print("Loading data...")
    prices = load_price_data(price_path)
    weather = load_weather_data(weather_path)

    print(f"Price data: {prices['datetime'].min()} to {prices['datetime'].max()}")
    print(f"Weather data: {weather['datetime'].min()} to {weather['datetime'].max()}")

    # Merge on datetime (inner join to keep only overlapping period)
    print("\nMerging datasets...")
    df = pd.merge(prices, weather, on='datetime', how='inner')
    print(f"Merged data: {len(df)} hours ({df['datetime'].min()} to {df['datetime'].max()})")

    # Generate PV forecast
    print("\nGenerating PV forecasts with realistic error...")
    df['pv_forecast_mwh'] = generate_pv_forecast(df['pv_actual_mwh'])

    # Calculate forecast error statistics
    mask = df['pv_actual_mwh'] > 0.1  # Only when there's meaningful production
    if mask.sum() > 0:
        mae = (df.loc[mask, 'pv_forecast_mwh'] - df.loc[mask, 'pv_actual_mwh']).abs().mean()
        rmse = np.sqrt(((df.loc[mask, 'pv_forecast_mwh'] - df.loc[mask, 'pv_actual_mwh'])**2).mean())
        print(f"Forecast MAE: {mae:.2f} MWh ({100*mae/PLANT_CAPACITY_MW:.1f}% of capacity)")
        print(f"Forecast RMSE: {rmse:.2f} MWh ({100*rmse/PLANT_CAPACITY_MW:.1f}% of capacity)")

    # Derive imbalance prices
    print("\nDeriving imbalance prices...")
    df['price_imbalance_short'], df['price_imbalance_long'] = derive_imbalance_prices(df['price_eur_mwh'])

    # Add time features
    print("Adding time features...")
    df = add_time_features(df)

    # Summary statistics
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Period: {df['datetime'].min().date()} to {df['datetime'].max().date()}")
    print(f"Total hours: {len(df):,}")
    print(f"Plant capacity: {PLANT_CAPACITY_MW} MW")
    print(f"\nPrice statistics (EUR/MWh):")
    print(f"  Mean: {df['price_eur_mwh'].mean():.1f}")
    print(f"  Std:  {df['price_eur_mwh'].std():.1f}")
    print(f"  Min:  {df['price_eur_mwh'].min():.1f}")
    print(f"  Max:  {df['price_eur_mwh'].max():.1f}")
    print(f"  Negative hours: {(df['price_eur_mwh'] < 0).sum()} ({100*(df['price_eur_mwh'] < 0).sum()/len(df):.1f}%)")
    print(f"\nPV production statistics (MWh):")
    print(f"  Max actual: {df['pv_actual_mwh'].max():.1f}")
    print(f"  Mean (daytime): {df.loc[df['pv_actual_mwh'] > 0, 'pv_actual_mwh'].mean():.1f}")
    print(f"  Annual production: ~{df['pv_actual_mwh'].sum() / (len(df)/8760):.0f} MWh/year")

    # Split into train/test
    # Train: 2015-2021, Test: 2022-2023
    train_end = '2021-12-31 23:00:00'

    train_df = df[df['datetime'] <= train_end].copy()
    test_df = df[df['datetime'] > train_end].copy()

    print(f"\nTrain set: {len(train_df):,} hours ({train_df['datetime'].min().date()} to {train_df['datetime'].max().date()})")
    print(f"Test set:  {len(test_df):,} hours ({test_df['datetime'].min().date()} to {test_df['datetime'].max().date()})")

    # Save processed data
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_path / 'train.csv', index=False)
    test_df.to_csv(output_path / 'test.csv', index=False)
    df.to_csv(output_path / 'full_dataset.csv', index=False)

    print(f"\nSaved to {output_path}:")
    print("  - train.csv")
    print("  - test.csv")
    print("  - full_dataset.csv")


if __name__ == '__main__':
    main()
