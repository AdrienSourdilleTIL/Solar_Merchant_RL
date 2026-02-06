"""
Tests for Solar Plant Module
"""

import numpy as np
import pandas as pd
import pytest

from src.environment.solar_plant import (
    PlantConfig,
    Battery,
    Settlement,
    DataManager,
    calculate_max_commitment,
    heuristic_battery_policy,
)


class TestPlantConfig:
    """Tests for PlantConfig dataclass."""

    def test_default_values(self):
        config = PlantConfig()
        assert config.plant_capacity_mw == 20.0
        assert config.battery_capacity_mwh == 10.0
        assert config.battery_power_mw == 5.0
        assert config.battery_efficiency == 0.92
        assert config.battery_degradation_cost == 0.01
        assert config.commitment_hour == 11

    def test_custom_values(self):
        config = PlantConfig(
            plant_capacity_mw=30.0,
            battery_capacity_mwh=15.0,
            battery_power_mw=7.5,
        )
        assert config.plant_capacity_mw == 30.0
        assert config.battery_capacity_mwh == 15.0
        assert config.battery_power_mw == 7.5


class TestBattery:
    """Tests for Battery class."""

    @pytest.fixture
    def battery(self):
        return Battery(
            capacity_mwh=10.0,
            power_mw=5.0,
            efficiency=0.92,
            degradation_cost=0.01
        )

    def test_initialization(self, battery):
        assert battery.capacity_mwh == 10.0
        assert battery.power_mw == 5.0
        assert battery.efficiency == 0.92
        assert battery.soc == 5.0  # 50% default
        assert battery.one_way_efficiency == pytest.approx(np.sqrt(0.92))

    def test_reset_default(self, battery):
        battery.soc = 8.0
        battery.reset()
        assert battery.soc == 5.0  # Back to 50%

    def test_reset_custom(self, battery):
        battery.reset(initial_soc=3.0)
        assert battery.soc == 3.0

    def test_reset_clamps_values(self, battery):
        battery.reset(initial_soc=15.0)  # Over capacity
        assert battery.soc == 10.0

        battery.reset(initial_soc=-5.0)  # Negative
        assert battery.soc == 0.0

    def test_charge_basic(self, battery):
        battery.reset(initial_soc=5.0)
        energy_used, throughput = battery.charge(2.0, available_pv=10.0)

        assert energy_used == pytest.approx(2.0)
        assert throughput == pytest.approx(2.0)
        # SOC increases by charge * efficiency
        expected_soc = 5.0 + 2.0 * battery.one_way_efficiency
        assert battery.soc == pytest.approx(expected_soc)

    def test_charge_limited_by_pv(self, battery):
        battery.reset(initial_soc=5.0)
        energy_used, throughput = battery.charge(5.0, available_pv=2.0)

        assert energy_used == pytest.approx(2.0)  # Limited by PV
        assert throughput == pytest.approx(2.0)

    def test_charge_limited_by_power(self, battery):
        battery.reset(initial_soc=0.0)
        energy_used, throughput = battery.charge(10.0, available_pv=10.0)

        assert energy_used == pytest.approx(5.0)  # Limited by power_mw

    def test_charge_limited_by_capacity(self, battery):
        battery.reset(initial_soc=9.5)
        # Only ~0.5 MWh room left (accounting for efficiency)
        energy_used, throughput = battery.charge(5.0, available_pv=10.0)

        assert battery.soc == pytest.approx(10.0)  # Clamped to capacity

    def test_charge_negative_does_nothing(self, battery):
        battery.reset(initial_soc=5.0)
        energy_used, throughput = battery.charge(-2.0, available_pv=10.0)

        assert energy_used == 0.0
        assert throughput == 0.0
        assert battery.soc == 5.0

    def test_discharge_basic(self, battery):
        battery.reset(initial_soc=5.0)
        energy_delivered, throughput = battery.discharge(2.0)

        assert energy_delivered == pytest.approx(2.0)
        assert throughput == pytest.approx(2.0)
        # SOC decreases by discharge / efficiency
        expected_soc = 5.0 - 2.0 / battery.one_way_efficiency
        assert battery.soc == pytest.approx(expected_soc)

    def test_discharge_limited_by_power(self, battery):
        battery.reset(initial_soc=10.0)
        energy_delivered, throughput = battery.discharge(10.0)

        assert energy_delivered == pytest.approx(5.0)  # Limited by power_mw

    def test_discharge_limited_by_soc(self, battery):
        battery.reset(initial_soc=1.0)
        # Can only deliver soc * efficiency
        max_deliverable = 1.0 * battery.one_way_efficiency
        energy_delivered, throughput = battery.discharge(5.0)

        assert energy_delivered == pytest.approx(max_deliverable)
        assert battery.soc == pytest.approx(0.0)

    def test_discharge_negative_does_nothing(self, battery):
        battery.reset(initial_soc=5.0)
        energy_delivered, throughput = battery.discharge(-2.0)

        assert energy_delivered == 0.0
        assert throughput == 0.0
        assert battery.soc == 5.0

    def test_step_charge(self, battery):
        battery.reset(initial_soc=5.0)
        energy_delta, throughput, degradation = battery.step(1.0, available_pv=10.0)

        # action=1.0 -> full charge
        assert energy_delta < 0  # Charging consumes energy
        assert throughput > 0
        assert degradation == pytest.approx(throughput * 0.01)
        assert battery.soc > 5.0

    def test_step_discharge(self, battery):
        battery.reset(initial_soc=5.0)
        energy_delta, throughput, degradation = battery.step(0.0, available_pv=0.0)

        # action=0.0 -> full discharge
        assert energy_delta > 0  # Discharging adds energy
        assert throughput > 0
        assert degradation == pytest.approx(throughput * 0.01)
        assert battery.soc < 5.0

    def test_step_idle(self, battery):
        battery.reset(initial_soc=5.0)
        energy_delta, throughput, degradation = battery.step(0.5, available_pv=10.0)

        # action=0.5 -> idle
        assert energy_delta == 0.0
        assert throughput == 0.0
        assert degradation == 0.0
        assert battery.soc == 5.0

    def test_soc_normalized(self, battery):
        battery.reset(initial_soc=7.5)
        assert battery.soc_normalized == pytest.approx(0.75)

        battery.reset(initial_soc=0.0)
        assert battery.soc_normalized == 0.0

        battery.reset(initial_soc=10.0)
        assert battery.soc_normalized == 1.0


class TestSettlement:
    """Tests for Settlement class."""

    def test_balanced_position(self):
        revenue, imbalance_cost = Settlement.calculate(
            committed=10.0,
            delivered=10.0,
            price_da=50.0,
            price_short=75.0,
            price_long=30.0
        )

        assert revenue == pytest.approx(500.0)  # 10 * 50
        assert imbalance_cost == 0.0

    def test_long_position(self):
        revenue, imbalance_cost = Settlement.calculate(
            committed=10.0,
            delivered=12.0,  # Over-delivered by 2
            price_da=50.0,
            price_short=75.0,
            price_long=30.0
        )

        # Revenue = 10*50 + 2*30 = 560
        assert revenue == pytest.approx(560.0)
        assert imbalance_cost == 0.0

    def test_short_position(self):
        revenue, imbalance_cost = Settlement.calculate(
            committed=10.0,
            delivered=8.0,  # Under-delivered by 2
            price_da=50.0,
            price_short=75.0,
            price_long=30.0
        )

        # Revenue = 10*50 = 500 (for commitment)
        # Imbalance cost = 2 * 75 = 150
        assert revenue == pytest.approx(500.0)
        assert imbalance_cost == pytest.approx(150.0)

    def test_zero_commitment(self):
        revenue, imbalance_cost = Settlement.calculate(
            committed=0.0,
            delivered=5.0,
            price_da=50.0,
            price_short=75.0,
            price_long=30.0
        )

        # All delivered is "excess" at long price
        assert revenue == pytest.approx(150.0)  # 5 * 30
        assert imbalance_cost == 0.0


class TestDataManager:
    """Tests for DataManager class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing."""
        hours = 48
        return pd.DataFrame({
            'datetime': pd.date_range('2023-01-01', periods=hours, freq='h'),
            'hour': [h % 24 for h in range(hours)],
            'price_eur_mwh': [50.0 + 10 * np.sin(h * np.pi / 12) for h in range(hours)],
            'pv_actual_mwh': [max(0, 10 * np.sin((h - 6) * np.pi / 12)) for h in range(hours)],
            'pv_forecast_mwh': [max(0, 9 * np.sin((h - 6) * np.pi / 12)) for h in range(hours)],
            'price_imbalance_short': [75.0] * hours,
            'price_imbalance_long': [30.0] * hours,
            'temperature_c': [15.0 + 5 * np.sin(h * np.pi / 12) for h in range(hours)],
            'irradiance_direct': [max(0, 800 * np.sin((h - 6) * np.pi / 12)) for h in range(hours)],
            'hour_sin': [np.sin(h * 2 * np.pi / 24) for h in range(hours)],
            'hour_cos': [np.cos(h * 2 * np.pi / 24) for h in range(hours)],
            'day_sin': [0.0] * hours,
            'day_cos': [1.0] * hours,
            'month_sin': [0.0] * hours,
            'month_cos': [1.0] * hours,
        })

    @pytest.fixture
    def data_manager(self, sample_data):
        config = PlantConfig()
        return DataManager(sample_data, config)

    def test_initialization(self, data_manager):
        assert len(data_manager) == 48
        assert 'price' in data_manager.norm_factors
        assert 'pv' in data_manager.norm_factors

    def test_missing_columns_raises(self):
        bad_data = pd.DataFrame({'datetime': [], 'hour': []})
        config = PlantConfig()

        with pytest.raises(ValueError, match="Missing required columns"):
            DataManager(bad_data, config)

    def test_get_row(self, data_manager):
        row = data_manager.get_row(0)
        assert row['hour'] == 0

        row = data_manager.get_row(11)
        assert row['hour'] == 11

    def test_get_hour(self, data_manager):
        assert data_manager.get_hour(0) == 0
        assert data_manager.get_hour(11) == 11
        assert data_manager.get_hour(25) == 1  # 25 % 24 = 1

    def test_get_forecasts(self, data_manager):
        forecasts = data_manager.get_forecasts(start_idx=0, hours=24)
        assert len(forecasts) == 24
        assert all(f >= 0 for f in forecasts)

    def test_get_forecasts_near_end(self, data_manager):
        # Start near end of data
        forecasts = data_manager.get_forecasts(start_idx=40, hours=24)
        assert len(forecasts) == 24
        # Last few should be zero-padded
        assert forecasts[-1] == 0.0

    def test_get_prices(self, data_manager):
        prices = data_manager.get_prices(start_idx=0, hours=24)
        assert len(prices) == 24

    def test_get_actuals(self, data_manager):
        actuals = data_manager.get_actuals(start_idx=0, hours=24)
        assert len(actuals) == 24

    def test_get_imbalance_prices(self, data_manager):
        short, long = data_manager.get_imbalance_prices(0)
        assert short == 75.0
        assert long == 30.0

    def test_normalize_price(self, data_manager):
        normalized = data_manager.normalize_price(50.0)
        assert 0 < normalized <= 1  # Should be in reasonable range

    def test_normalize_pv(self, data_manager):
        normalized = data_manager.normalize_pv(10.0)
        assert normalized == pytest.approx(10.0 / 20.0)  # plant_capacity = 20


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_calculate_max_commitment(self):
        forecasts = np.array([5.0, 10.0, 15.0, 8.0])
        battery_power = 5.0

        max_commit = calculate_max_commitment(forecasts, battery_power)

        assert max_commit[0] == pytest.approx(10.0)  # 5 + 5
        assert max_commit[1] == pytest.approx(15.0)  # 10 + 5
        assert max_commit[2] == pytest.approx(20.0)  # 15 + 5
        assert max_commit[3] == pytest.approx(13.0)  # 8 + 5

    def test_heuristic_battery_policy_short(self):
        # Short position: PV < committed
        action = heuristic_battery_policy(
            soc=5.0,
            committed=10.0,
            pv_actual=5.0,  # Short by 5
            battery_power_mw=5.0,
            battery_capacity_mwh=10.0
        )

        # Should discharge (action < 0.5)
        assert action < 0.5
        assert action >= 0.0

    def test_heuristic_battery_policy_long(self):
        # Long position: PV > committed
        action = heuristic_battery_policy(
            soc=5.0,
            committed=5.0,
            pv_actual=10.0,  # Long by 5
            battery_power_mw=5.0,
            battery_capacity_mwh=10.0
        )

        # Should charge (action > 0.5)
        assert action > 0.5
        assert action <= 1.0

    def test_heuristic_battery_policy_balanced(self):
        # Balanced: PV == committed
        action = heuristic_battery_policy(
            soc=5.0,
            committed=10.0,
            pv_actual=10.0,
            battery_power_mw=5.0,
            battery_capacity_mwh=10.0
        )

        # Should be idle (action ~ 0.5)
        assert action == pytest.approx(0.5)

    def test_heuristic_battery_policy_full_battery(self):
        # Long but battery full - should idle
        action = heuristic_battery_policy(
            soc=10.0,  # Full
            committed=5.0,
            pv_actual=10.0,  # Long by 5
            battery_power_mw=5.0,
            battery_capacity_mwh=10.0
        )

        # Should idle since battery full
        assert action == pytest.approx(0.5)

    def test_heuristic_battery_policy_clamps(self):
        # Extreme case: should still be in [0, 1]
        action = heuristic_battery_policy(
            soc=0.0,
            committed=100.0,  # Very short
            pv_actual=0.0,
            battery_power_mw=5.0,
            battery_capacity_mwh=10.0
        )

        assert 0.0 <= action <= 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
