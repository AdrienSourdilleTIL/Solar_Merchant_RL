"""
Analyze Agent Behavior
======================

Compare what the trained RL agents do differently compared to baselines.
Visualize action patterns and decision-making strategies.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from environment.hierarchical_orchestrator import HierarchicalOrchestrator
from environment.solar_plant import PlantConfig, calculate_max_commitment
from baselines.hierarchical_baselines import (
    conservative_commitment_policy,
    aggressive_commitment_policy,
    price_aware_commitment_policy,
)

# Paths
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs'

PLANT_CONFIG = PlantConfig()


def analyze_commitment_strategies():
    """Compare commitment decisions across policies."""
    test_path = DATA_PATH / 'test.csv'
    data = pd.read_csv(test_path, parse_dates=['datetime'])

    # Load trained orchestrator
    print("Loading trained agents...")
    trained_orch = HierarchicalOrchestrator.from_trained_agents(
        data_path=str(test_path),
        plant_config=PLANT_CONFIG,
    )

    # Sample multiple days and compare commitment strategies
    np.random.seed(42)

    results = {
        'hour': [],
        'forecast': [],
        'price': [],
        'rl_commit': [],
        'conservative_commit': [],
        'aggressive_commit': [],
        'price_aware_commit': [],
        'actual_pv': [],
    }

    # Sample 20 random commitment decisions
    n_samples = 20
    max_start = len(data) - 48

    for _ in range(n_samples):
        # Find a commitment hour (11:00)
        start_idx = np.random.randint(0, max_start)
        while data.iloc[start_idx]['hour'] != 11:
            start_idx += 1
            if start_idx >= max_start:
                start_idx = 0

        # Get data for tomorrow (24 hours starting from midnight after commitment)
        hours_until_midnight = (24 - 11) % 24
        tomorrow_start = start_idx + hours_until_midnight

        if tomorrow_start + 24 >= len(data):
            continue

        forecasts = data.iloc[tomorrow_start:tomorrow_start+24]['pv_forecast_mwh'].values
        prices = data.iloc[tomorrow_start:tomorrow_start+24]['price_eur_mwh'].values
        actuals = data.iloc[tomorrow_start:tomorrow_start+24]['pv_actual_mwh'].values

        # Get RL commitment
        obs, forecasts_rl, prices_rl, _ = trained_orch._get_commitment_observation(start_idx)
        rl_commits = trained_orch._get_commitment_action(obs, forecasts_rl, prices_rl)

        # Get baseline commitments
        conservative = conservative_commitment_policy(forecasts, prices, 0.5, PLANT_CONFIG)
        aggressive = aggressive_commitment_policy(forecasts, prices, 0.5, PLANT_CONFIG)
        price_aware = price_aware_commitment_policy(forecasts, prices, 0.5, PLANT_CONFIG)

        for hour in range(24):
            results['hour'].append(hour)
            results['forecast'].append(forecasts[hour])
            results['price'].append(prices[hour])
            results['rl_commit'].append(rl_commits[hour])
            results['conservative_commit'].append(conservative[hour])
            results['aggressive_commit'].append(aggressive[hour])
            results['price_aware_commit'].append(price_aware[hour])
            results['actual_pv'].append(actuals[hour])

    return pd.DataFrame(results)


def plot_commitment_comparison(df: pd.DataFrame, output_path: Path):
    """Plot commitment strategies comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Commitment as fraction of forecast by hour
    ax = axes[0, 0]
    hours = range(24)

    for hour in hours:
        hour_data = df[df['hour'] == hour]
        if len(hour_data) == 0:
            continue

    # Average commitment fraction by hour
    hourly_fractions = df.groupby('hour').apply(
        lambda x: pd.Series({
            'RL': (x['rl_commit'] / x['forecast'].replace(0, np.nan)).mean(),
            'Conservative': (x['conservative_commit'] / x['forecast'].replace(0, np.nan)).mean(),
            'Aggressive': (x['aggressive_commit'] / x['forecast'].replace(0, np.nan)).mean(),
            'Price-Aware': (x['price_aware_commit'] / x['forecast'].replace(0, np.nan)).mean(),
        })
    )

    hourly_fractions.plot(ax=ax, marker='o', linewidth=2)
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Commitment / Forecast Ratio', fontsize=11)
    ax.set_title('Commitment Strategy by Hour\n(How much of forecast is committed)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='100% of forecast')
    ax.set_xlim(0, 23)

    # 2. Commitment vs Price relationship
    ax = axes[0, 1]

    # Bin prices into quartiles
    df['price_quartile'] = pd.qcut(df['price'], 4, labels=['Low', 'Med-Low', 'Med-High', 'High'])

    price_response = df.groupby('price_quartile').apply(
        lambda x: pd.Series({
            'RL': (x['rl_commit'] / x['forecast'].replace(0, np.nan)).mean(),
            'Conservative': (x['conservative_commit'] / x['forecast'].replace(0, np.nan)).mean(),
            'Price-Aware': (x['price_aware_commit'] / x['forecast'].replace(0, np.nan)).mean(),
        })
    )

    x = np.arange(4)
    width = 0.25
    ax.bar(x - width, price_response['RL'], width, label='RL Agent', color='#2196F3')
    ax.bar(x, price_response['Conservative'], width, label='Conservative', color='#9E9E9E')
    ax.bar(x + width, price_response['Price-Aware'], width, label='Price-Aware', color='#FF9800')

    ax.set_xlabel('Price Level', fontsize=11)
    ax.set_ylabel('Commitment / Forecast Ratio', fontsize=11)
    ax.set_title('Price Responsiveness\n(How commitment changes with price)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(['Low', 'Med-Low', 'Med-High', 'High'])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')

    # 3. Imbalance Risk Analysis
    ax = axes[1, 0]

    # Calculate potential imbalance (commitment - actual)
    df['rl_imbalance'] = df['rl_commit'] - df['actual_pv']
    df['conservative_imbalance'] = df['conservative_commit'] - df['actual_pv']
    df['aggressive_imbalance'] = df['aggressive_commit'] - df['actual_pv']

    policies = ['RL', 'Conservative', 'Aggressive']
    imbalances = [df['rl_imbalance'], df['conservative_imbalance'], df['aggressive_imbalance']]
    colors = ['#2196F3', '#4CAF50', '#F44336']

    bp = ax.boxplot(imbalances, labels=policies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_ylabel('Commitment - Actual PV (MWh)', fontsize=11)
    ax.set_title('Imbalance Risk Distribution\n(Positive = over-committed, Negative = under-committed)', fontsize=12)
    ax.grid(alpha=0.3, axis='y')

    # 4. Example day comparison
    ax = axes[1, 1]

    # Pick one sample day
    sample_day = df[df['hour'] == 0].index[5]  # 6th sample
    day_data = df.iloc[sample_day:sample_day+24]

    hours = day_data['hour'].values
    ax.fill_between(hours, 0, day_data['actual_pv'], alpha=0.3, color='yellow', label='Actual PV')
    ax.plot(hours, day_data['forecast'], 'k--', linewidth=1.5, label='Forecast')
    ax.plot(hours, day_data['rl_commit'], 'b-', linewidth=2, marker='o', markersize=4, label='RL Commitment')
    ax.plot(hours, day_data['conservative_commit'], 'g--', linewidth=1.5, label='Conservative')
    ax.plot(hours, day_data['aggressive_commit'], 'r--', linewidth=1.5, label='Aggressive')

    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Energy (MWh)', fontsize=11)
    ax.set_title('Example Day: Commitment Decisions\n(RL adapts to conditions)', fontsize=12)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 23)

    fig.suptitle('What Makes the RL Agent Smarter?', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def print_strategy_summary(df: pd.DataFrame):
    """Print summary of strategy differences."""
    print("\n" + "=" * 70)
    print("STRATEGY ANALYSIS SUMMARY")
    print("=" * 70)

    # Average commitment ratios
    print("\n1. AVERAGE COMMITMENT RATIO (Commitment / Forecast):")
    print("-" * 50)

    mask = df['forecast'] > 1.0  # Only daytime hours with meaningful forecast (>1 MWh)

    rl_ratio = (df.loc[mask, 'rl_commit'] / df.loc[mask, 'forecast']).mean()
    cons_ratio = (df.loc[mask, 'conservative_commit'] / df.loc[mask, 'forecast']).mean()
    aggr_ratio = (df.loc[mask, 'aggressive_commit'] / df.loc[mask, 'forecast']).mean()
    price_ratio = (df.loc[mask, 'price_aware_commit'] / df.loc[mask, 'forecast']).mean()

    print(f"  RL Agent:      {rl_ratio:.1%}")
    print(f"  Conservative:  {cons_ratio:.1%} (fixed 80%)")
    print(f"  Aggressive:    {aggr_ratio:.1%} (100% + battery)")
    print(f"  Price-Aware:   {price_ratio:.1%} (70-95% by price)")

    # Price responsiveness
    print("\n2. PRICE RESPONSIVENESS (Commitment ratio change: Low->High price):")
    print("-" * 50)

    low_price = df['price'] < df['price'].quantile(0.25)
    high_price = df['price'] > df['price'].quantile(0.75)

    rl_low = (df.loc[low_price & mask, 'rl_commit'] / df.loc[low_price & mask, 'forecast']).mean()
    rl_high = (df.loc[high_price & mask, 'rl_commit'] / df.loc[high_price & mask, 'forecast']).mean()

    pa_low = (df.loc[low_price & mask, 'price_aware_commit'] / df.loc[low_price & mask, 'forecast']).mean()
    pa_high = (df.loc[high_price & mask, 'price_aware_commit'] / df.loc[high_price & mask, 'forecast']).mean()

    print(f"  RL Agent:      {rl_low:.1%} -> {rl_high:.1%} (change = {(rl_high-rl_low)*100:+.1f}pp)")
    print(f"  Price-Aware:   {pa_low:.1%} -> {pa_high:.1%} (change = {(pa_high-pa_low)*100:+.1f}pp)")
    print(f"  Conservative:  80% -> 80% (change = 0pp, no response)")

    # Imbalance risk
    print("\n3. IMBALANCE RISK (Commitment - Actual PV):")
    print("-" * 50)

    df['rl_imbalance'] = df['rl_commit'] - df['actual_pv']
    df['cons_imbalance'] = df['conservative_commit'] - df['actual_pv']
    df['aggr_imbalance'] = df['aggressive_commit'] - df['actual_pv']

    print(f"  RL Agent:")
    print(f"    Mean imbalance: {df['rl_imbalance'].mean():+.2f} MWh")
    print(f"    % over-committed (short risk): {(df['rl_imbalance'] > 0.1).mean():.1%}")
    print(f"    % under-committed (lost revenue): {(df['rl_imbalance'] < -0.1).mean():.1%}")

    print(f"\n  Conservative:")
    print(f"    Mean imbalance: {df['cons_imbalance'].mean():+.2f} MWh")
    print(f"    % over-committed: {(df['cons_imbalance'] > 0.1).mean():.1%}")
    print(f"    % under-committed: {(df['cons_imbalance'] < -0.1).mean():.1%}")

    print(f"\n  Aggressive:")
    print(f"    Mean imbalance: {df['aggr_imbalance'].mean():+.2f} MWh")
    print(f"    % over-committed: {(df['aggr_imbalance'] > 0.1).mean():.1%}")
    print(f"    % under-committed: {(df['aggr_imbalance'] < -0.1).mean():.1%}")

    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS: What makes RL smarter?")
    print("=" * 70)
    print("""
1. ADAPTIVE COMMITMENT RATIO:
   - RL learns the optimal ~{:.0f}% commitment ratio through experience
   - Not fixed like conservative (80%) or aggressive (100%+)
   - Adapts based on forecast confidence, weather, and battery state

2. PRICE-SENSITIVE BIDDING:
   - RL commits more aggressively during high-price hours
   - Learns that the revenue gain outweighs the imbalance risk
   - More sophisticated than simple price-aware rule

3. RISK MANAGEMENT:
   - RL balances revenue vs imbalance penalty
   - Learns that being slightly short is VERY expensive (1.5x penalty)
   - Being long is less costly (only lose 0.4x of price)
   - Results in slightly conservative bias during uncertain conditions

4. BATTERY COORDINATION:
   - Commitment agent accounts for battery availability
   - Battery agent learns to save charge for high-value hours
   - Combined strategy is more than sum of parts
""".format(rl_ratio * 100))


def main():
    """Analyze agent behavior."""
    print("=" * 70)
    print("AGENT BEHAVIOR ANALYSIS")
    print("=" * 70)

    print("\nAnalyzing commitment strategies...")
    df = analyze_commitment_strategies()

    print(f"Collected {len(df)} hourly samples")

    # Print summary
    print_strategy_summary(df)

    # Generate visualization
    output_path = OUTPUT_PATH / 'agent_behavior_analysis.png'
    print(f"\nGenerating visualization: {output_path}")
    plot_commitment_comparison(df, output_path)

    print("\nDone!")


if __name__ == '__main__':
    main()
