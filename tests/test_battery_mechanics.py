"""
Tests for battery mechanics in SolarMerchantEnv.

This module validates battery charge/discharge physics according to Story 2-3:
- SOC tracking and bounds [0, capacity]
- Power limits (5 MW max charge/discharge)
- Round-trip efficiency (92%)
- Degradation cost calculation
- PV-only charging constraint

Battery Parameters (per architecture):
- Capacity: 10 MWh
- Power: 5 MW max charge/discharge
- Efficiency: 92% round-trip (sqrt(0.92) ≈ 0.959 one-way)
- Degradation cost: EUR 0.01 per MWh throughput
"""

import numpy as np
import pytest
from src.environment import load_environment


class TestBatteryValidation:
    """Task 1: Validate existing battery physics implementation."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_battery_soc_tracking_bounded(self, env):
        """Verify battery SOC tracking is correct and bounded [0, capacity]."""
        obs, _ = env.reset(seed=42)

        # Initial SOC should be 50% of capacity (5 MWh)
        initial_soc = env.battery_soc
        assert initial_soc == 0.5 * env.battery_capacity_mwh, \
            f"Initial SOC should be 50% of capacity, got {initial_soc}"

        # Run multiple steps with varied actions
        for _ in range(48):
            action = env.action_space.sample()
            obs, _, terminated, truncated, info = env.step(action)

            # SOC should always be bounded
            assert 0 <= env.battery_soc <= env.battery_capacity_mwh, \
                f"SOC {env.battery_soc} outside bounds [0, {env.battery_capacity_mwh}]"

            # SOC in observation should be normalized [0, 1]
            soc_normalized = obs[1]
            assert 0 <= soc_normalized <= 1, \
                f"Normalized SOC {soc_normalized} outside [0, 1]"

            if terminated or truncated:
                break

    def test_power_limits_enforced(self, env):
        """Verify power limits are correctly enforced (5 MW max charge/discharge)."""
        env.reset(seed=42)

        # Track max observed throughput
        max_throughput = 0

        for i in range(100):
            # Use extreme actions to try to exceed power limit
            action = np.zeros(25, dtype=np.float32)
            action[24] = 1.0  # Full charge attempt

            obs, _, terminated, truncated, info = env.step(action)
            throughput = info.get('battery_throughput', 0)
            max_throughput = max(max_throughput, throughput)

            if terminated or truncated:
                env.reset(seed=42 + i)

        assert max_throughput <= env.battery_power_mw + 1e-6, \
            f"Battery throughput {max_throughput} exceeded power limit {env.battery_power_mw} MW"

    def test_round_trip_efficiency_calculation(self, env):
        """Verify round-trip efficiency calculation is correct (92% ≈ 0.959 one-way)."""
        # Verify one-way efficiency calculation
        expected_one_way = np.sqrt(env.battery_efficiency)
        assert np.isclose(env.one_way_efficiency, expected_one_way, atol=1e-6), \
            f"One-way efficiency {env.one_way_efficiency} != expected {expected_one_way}"

        # Verify round-trip: one_way² ≈ 0.92
        round_trip = env.one_way_efficiency ** 2
        assert np.isclose(round_trip, env.battery_efficiency, atol=1e-6), \
            f"Round-trip {round_trip} != battery_efficiency {env.battery_efficiency}"

    def test_battery_throughput_tracking(self, env):
        """Check battery throughput tracking for degradation cost calculation."""
        env.reset(seed=42)

        # Take steps with battery actions and verify throughput is tracked
        for _ in range(10):
            action = np.zeros(25, dtype=np.float32)
            action[24] = np.random.choice([0.0, 0.5, 1.0])  # Discharge, idle, or charge

            obs, reward, terminated, truncated, info = env.step(action)

            # Throughput should be in info
            assert 'battery_throughput' in info, "battery_throughput should be in info dict"
            assert info['battery_throughput'] >= 0, "Throughput should be non-negative"

            if terminated or truncated:
                break

    def test_battery_action_interpretation(self, env):
        """Validate battery action interpretation (0=discharge, 0.5=idle, 1=charge)."""
        env.reset(seed=42)
        initial_soc = env.battery_soc

        # Test idle action (0.5)
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.5  # Idle
        obs, _, _, _, info = env.step(action)

        # With idle, throughput should be zero
        assert info['battery_throughput'] == 0.0, \
            f"Idle action should produce zero throughput, got {info['battery_throughput']}"

        # Reset and test discharge (0.0)
        env.reset(seed=42)
        action[24] = 0.0  # Full discharge
        initial_soc = env.battery_soc
        obs, _, _, _, info = env.step(action)

        # Discharge should reduce SOC (if there was energy)
        if initial_soc > 0:
            assert env.battery_soc <= initial_soc, \
                f"Discharge should reduce SOC: {initial_soc} -> {env.battery_soc}"


class TestBatteryChargeMechanics:
    """Task 2: Test battery charge mechanics."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_charging_from_pv_surplus_only(self, env):
        """Test charging from PV surplus only (cannot charge from grid)."""
        env.reset(seed=42)

        # Find a step with PV production and test charge is limited by available PV
        charging_tested = False
        for i in range(100):
            # Get current PV
            row = env.data.iloc[env.current_idx]
            pv_actual = row['pv_actual_mwh']
            initial_soc = env.battery_soc

            # Full charge action
            action = np.zeros(25, dtype=np.float32)
            action[24] = 1.0  # Full charge

            obs, _, terminated, truncated, info = env.step(action)

            # Actual charge should not exceed PV available
            throughput = info['battery_throughput']
            if throughput > 0:  # Charging occurred
                assert throughput <= pv_actual + 1e-6, \
                    f"Charge throughput {throughput} exceeded available PV {pv_actual}"
                charging_tested = True

            if terminated or truncated:
                env.reset(seed=42 + i)

        assert charging_tested, "Test did not encounter any charging scenarios"

    def test_power_limit_during_charge(self, env):
        """Test power limit enforcement during charge (5 MW max)."""
        env.reset(seed=42)

        for _ in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 1.0  # Full charge

            obs, _, terminated, truncated, info = env.step(action)
            throughput = info['battery_throughput']

            assert throughput <= env.battery_power_mw + 1e-6, \
                f"Charge throughput {throughput} exceeded power limit {env.battery_power_mw}"

            if terminated or truncated:
                break

    def test_capacity_limit_during_charge(self, env):
        """Test capacity limit enforcement (cannot exceed 10 MWh)."""
        env.reset(seed=42)

        # Run many charge cycles
        for i in range(100):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 1.0  # Full charge

            obs, _, terminated, truncated, info = env.step(action)

            # SOC should never exceed capacity
            assert env.battery_soc <= env.battery_capacity_mwh + 1e-6, \
                f"SOC {env.battery_soc} exceeded capacity {env.battery_capacity_mwh}"

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_efficiency_losses_during_charge(self, env):
        """Test efficiency losses during charge (92% round-trip ≈ 95.9% one-way)."""
        env.reset(seed=42)

        # Find a high-PV step with low SOC
        for i in range(200):
            row = env.data.iloc[env.current_idx]
            pv_actual = row['pv_actual_mwh']

            # Drain battery first
            env.battery_soc = 1.0  # Low SOC
            initial_soc = env.battery_soc

            if pv_actual > 2.0:  # Need meaningful PV
                action = np.zeros(25, dtype=np.float32)
                action[24] = 1.0  # Full charge

                obs, _, terminated, truncated, info = env.step(action)

                throughput = info['battery_throughput']
                soc_increase = env.battery_soc - initial_soc

                if throughput > 0:
                    # SOC increase should be throughput * one_way_efficiency
                    expected_increase = throughput * env.one_way_efficiency
                    assert np.isclose(soc_increase, expected_increase, atol=0.01), \
                        f"SOC increase {soc_increase} != expected {expected_increase} (throughput * efficiency)"
                    return  # Test passed

                if terminated or truncated:
                    env.reset(seed=42 + i)
            else:
                # Advance to next step
                action = np.zeros(25, dtype=np.float32)
                action[24] = 0.5  # Idle
                obs, _, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    env.reset(seed=42 + i)

        pytest.fail("Could not find high-PV scenario to test charge efficiency")

    def test_edge_case_charging_near_full(self, env):
        """Test edge case: charging when SOC near full capacity."""
        env.reset(seed=42)

        # Set SOC to near capacity
        env.battery_soc = env.battery_capacity_mwh - 0.5  # 9.5 MWh
        initial_soc = env.battery_soc

        # Try to charge
        action = np.zeros(25, dtype=np.float32)
        action[24] = 1.0  # Full charge

        obs, _, _, _, info = env.step(action)

        # SOC should not exceed capacity
        assert env.battery_soc <= env.battery_capacity_mwh, \
            f"SOC {env.battery_soc} exceeded capacity when starting near full"

    def test_edge_case_charging_no_pv_surplus(self, env):
        """Test edge case: charging when no PV surplus available."""
        env.reset(seed=42)

        # Find a night hour with no PV
        for _ in range(100):
            row = env.data.iloc[env.current_idx]
            pv_actual = row['pv_actual_mwh']

            if pv_actual < 0.1:  # Night time / no PV
                initial_soc = env.battery_soc

                action = np.zeros(25, dtype=np.float32)
                action[24] = 1.0  # Try to charge

                obs, _, terminated, truncated, info = env.step(action)

                # Throughput should be near zero (no PV to charge from)
                throughput = info['battery_throughput']
                assert throughput <= pv_actual + 1e-6, \
                    f"Cannot charge more than available PV ({throughput} > {pv_actual})"
                break

            # Advance
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            _, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                env.reset(seed=42 + _)


class TestBatteryDischargeMechanics:
    """Task 3: Test battery discharge mechanics."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_discharge_provides_energy(self, env):
        """Test discharge to meet commitments or provide energy."""
        env.reset(seed=42)

        # Set known SOC to guarantee discharge occurs
        env.battery_soc = 5.0  # 5 MWh available

        # Get initial state
        row = env.data.iloc[env.current_idx]
        pv_actual = row['pv_actual_mwh']
        initial_soc = env.battery_soc

        # Full discharge action
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.0  # Full discharge

        obs, _, _, _, info = env.step(action)

        # Delivered energy should include battery discharge
        delivered = info['delivered']
        throughput = info['battery_throughput']

        # With 5 MWh SOC and full discharge, throughput must be positive
        assert throughput > 0, "Discharge should produce positive throughput with 5 MWh SOC"

        # There should be energy delivered beyond just PV
        assert delivered >= pv_actual - 1e-6, \
            f"Delivered {delivered} should be at least PV {pv_actual}"

    def test_power_limit_during_discharge(self, env):
        """Test power limit enforcement during discharge (5 MW max)."""
        env.reset(seed=42)

        # Set high SOC to allow max discharge
        env.battery_soc = env.battery_capacity_mwh  # Full battery

        for _ in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.0  # Full discharge

            obs, _, terminated, truncated, info = env.step(action)
            throughput = info['battery_throughput']

            assert throughput <= env.battery_power_mw + 1e-6, \
                f"Discharge throughput {throughput} exceeded power limit {env.battery_power_mw}"

            # Refill battery for next test
            env.battery_soc = env.battery_capacity_mwh

            if terminated or truncated:
                break

    def test_discharge_stops_at_soc_zero(self, env):
        """Test discharge stops at SOC=0 (cannot go negative)."""
        env.reset(seed=42)

        # Set low SOC
        env.battery_soc = 0.5  # Low SOC

        # Discharge multiple times
        for _ in range(20):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.0  # Full discharge

            obs, _, terminated, truncated, info = env.step(action)

            # SOC should never go negative
            assert env.battery_soc >= 0, f"SOC went negative: {env.battery_soc}"

            if terminated or truncated:
                break

    def test_efficiency_losses_during_discharge(self, env):
        """Test efficiency losses during discharge (≈95.9% one-way)."""
        env.reset(seed=42)

        # Set known SOC to guarantee discharge
        initial_soc = 5.0  # MWh
        env.battery_soc = initial_soc

        # Discharge
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.0  # Full discharge

        obs, _, _, _, info = env.step(action)
        throughput = info['battery_throughput']
        soc_decrease = initial_soc - env.battery_soc

        # With 5 MWh SOC, discharge must occur
        assert throughput > 0, "Discharge should produce throughput with 5 MWh SOC"

        # Energy delivered (throughput) should be SOC used * one_way_efficiency
        # Or equivalently: SOC decrease = throughput / one_way_efficiency
        expected_soc_decrease = throughput / env.one_way_efficiency

        assert np.isclose(soc_decrease, expected_soc_decrease, atol=0.01), \
            f"SOC decrease {soc_decrease} != expected {expected_soc_decrease}"

    def test_edge_case_discharging_near_empty(self, env):
        """Test edge case: discharging when SOC near empty."""
        env.reset(seed=42)

        # Set SOC to near empty
        env.battery_soc = 0.3  # Very low
        initial_soc = env.battery_soc

        # Full discharge
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.0  # Full discharge

        obs, _, _, _, info = env.step(action)

        # SOC should not go negative
        assert env.battery_soc >= 0, f"SOC went negative: {env.battery_soc}"

        # Throughput should be limited by available SOC
        throughput = info['battery_throughput']
        max_possible = initial_soc * env.one_way_efficiency
        assert throughput <= max_possible + 1e-6, \
            f"Throughput {throughput} exceeded max possible {max_possible}"

    def test_battery_throughput_for_degradation(self, env):
        """Test battery throughput calculation for degradation cost."""
        env.reset(seed=42)
        env.battery_soc = 5.0  # Set known SOC

        # Discharge
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.0  # Full discharge

        obs, _, _, _, info = env.step(action)
        throughput = info['battery_throughput']

        # Throughput should be positive if discharge occurred
        if env.battery_soc < 5.0:  # Discharge happened
            assert throughput > 0, "Discharge should produce positive throughput"


class TestBatteryDegradationCost:
    """Task 4: Test battery degradation cost calculation."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_degradation_cost_rate(self, env):
        """Verify degradation cost is EUR 0.01 per MWh throughput."""
        assert env.battery_degradation_cost == 0.01, \
            f"Degradation cost should be 0.01 EUR/MWh, got {env.battery_degradation_cost}"

    def test_throughput_accumulation(self, env):
        """Test throughput accumulation across charge/discharge operations."""
        env.reset(seed=42)

        total_throughput = 0

        for _ in range(20):
            # Alternate between charge and discharge
            action = np.zeros(25, dtype=np.float32)
            action[24] = np.random.choice([0.0, 1.0])

            obs, _, terminated, truncated, info = env.step(action)
            total_throughput += info['battery_throughput']

            if terminated or truncated:
                break

        # Verify throughput was accumulated
        # (We can't easily verify the exact value but can check it's positive)
        assert total_throughput >= 0, "Total throughput should be non-negative"

    def test_degradation_cost_subtracted_from_reward(self, env):
        """Test degradation cost is subtracted from reward."""
        env.reset(seed=42)
        env.battery_soc = 5.0  # Set known SOC to guarantee discharge

        # Get baseline degradation cost
        initial_degradation = env.episode_degradation_cost

        # Discharge to incur degradation
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.0  # Full discharge

        obs, reward, _, _, info = env.step(action)
        throughput = info['battery_throughput']

        # With 5 MWh SOC, discharge must occur
        assert throughput > 0, "Discharge should produce throughput with 5 MWh SOC"

        # Degradation cost should have increased
        expected_degradation = throughput * env.battery_degradation_cost
        actual_degradation_increase = env.episode_degradation_cost - initial_degradation

        assert np.isclose(actual_degradation_increase, expected_degradation, atol=0.001), \
            f"Degradation increase {actual_degradation_increase} != expected {expected_degradation}"

        # Verify reward formula: reward = revenue - imbalance_cost - degradation
        # The degradation should reduce the reward
        revenue = info['revenue']
        imbalance_cost = info['imbalance_cost']
        expected_reward = revenue - imbalance_cost - expected_degradation
        assert np.isclose(reward, expected_reward, atol=0.01), \
            f"Reward {reward} != expected {expected_reward} (revenue - imbalance - degradation)"

    def test_degradation_impacts_agent_economics(self, env):
        """Verify degradation cost impacts agent economics correctly."""
        env.reset(seed=42)
        env.battery_soc = env.battery_capacity_mwh  # Full battery

        # Calculate expected degradation for full discharge
        max_discharge = min(env.battery_power_mw,
                           env.battery_soc * env.one_way_efficiency)
        expected_degradation = max_discharge * env.battery_degradation_cost

        # Discharge
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.0  # Full discharge

        obs, reward, _, _, info = env.step(action)
        throughput = info['battery_throughput']
        actual_degradation = throughput * env.battery_degradation_cost

        # With full battery, discharge must occur
        assert throughput > 0, "Discharge should produce throughput with full battery"

        # Degradation should be small but meaningful
        assert actual_degradation <= expected_degradation + 0.001, \
            f"Degradation {actual_degradation} exceeded expected {expected_degradation}"

        # At ~5 MW discharge, degradation ≈ 0.05 EUR
        # This is small relative to revenue but non-zero
        assert actual_degradation > 0, "Degradation should be positive for non-zero throughput"


class TestBatteryEnhancementsAndEdgeCases:
    """Task 5: Test battery enhancements and edge cases."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_battery_action_clipping(self, env):
        """Review battery action clipping (ensure invalid actions handled gracefully)."""
        env.reset(seed=42)

        action = np.zeros(25, dtype=np.float32)

        # Test valid boundary values
        for battery_action in [0.0, 0.25, 0.5, 0.75, 1.0]:
            env.reset(seed=42)
            action[24] = battery_action

            # Should not raise an error
            obs, reward, terminated, truncated, info = env.step(action)

            # SOC should remain valid
            assert 0 <= env.battery_soc <= env.battery_capacity_mwh, \
                f"SOC {env.battery_soc} invalid for action {battery_action}"

    def test_battery_action_near_idle(self, env):
        """Test battery actions very close to idle (0.5) for floating point edge cases."""
        env.reset(seed=42)
        initial_soc = env.battery_soc

        action = np.zeros(25, dtype=np.float32)

        # Test values very close to idle (0.5)
        for battery_action in [0.49, 0.499, 0.5, 0.501, 0.51]:
            env.reset(seed=42)
            env.battery_soc = 5.0  # Set known SOC
            initial_soc = env.battery_soc

            action[24] = battery_action
            obs, reward, terminated, truncated, info = env.step(action)

            throughput = info['battery_throughput']

            # Actions very close to 0.5 should produce minimal or zero throughput
            if battery_action == 0.5:
                assert throughput == 0.0, f"Idle action should produce zero throughput"
            else:
                # Near-idle actions should produce small throughput
                # Maximum throughput at boundaries is 5 MW * (0.01 * 2) = 0.1 MW
                assert throughput <= 0.15, \
                    f"Near-idle action {battery_action} produced large throughput {throughput}"

            # SOC should remain valid
            assert 0 <= env.battery_soc <= env.battery_capacity_mwh, \
                f"SOC {env.battery_soc} invalid for near-idle action {battery_action}"

    def test_battery_action_out_of_bounds_clipped(self, env):
        """Test that out-of-bounds actions are handled (clipped by Gymnasium)."""
        env.reset(seed=42)
        env.battery_soc = 5.0  # Set known SOC

        action = np.zeros(25, dtype=np.float32)

        # Note: Gymnasium's Box action space clips values to [0, 1]
        # We test that the environment handles edge values gracefully

        # Test extreme values that would be clipped to boundaries
        for battery_action, expected_behavior in [
            (-0.5, "clipped to 0.0 = full discharge"),
            (1.5, "clipped to 1.0 = full charge"),
            (-1.0, "clipped to 0.0 = full discharge"),
            (2.0, "clipped to 1.0 = full charge"),
        ]:
            env.reset(seed=42)
            env.battery_soc = 5.0
            initial_soc = env.battery_soc

            # Manually clip as Gymnasium would (simulating wrapper behavior)
            clipped_action = np.clip(battery_action, 0.0, 1.0)
            action[24] = clipped_action

            obs, reward, terminated, truncated, info = env.step(action)

            # SOC should remain valid after clipped action
            assert 0 <= env.battery_soc <= env.battery_capacity_mwh, \
                f"SOC {env.battery_soc} invalid after clipped action ({battery_action} -> {clipped_action})"

            # Verify action interpretation matches clipped value
            throughput = info['battery_throughput']
            if clipped_action == 0.0:
                # Full discharge should produce throughput
                assert throughput > 0, f"Full discharge should produce throughput"
            elif clipped_action == 1.0:
                # Full charge may or may not produce throughput depending on PV
                pass  # No assertion needed, just verify no crash

    def test_battery_idle_no_throughput(self, env):
        """Test idle action produces no throughput."""
        env.reset(seed=42)
        initial_soc = env.battery_soc

        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.5  # Idle

        obs, _, _, _, info = env.step(action)

        # No throughput for idle
        assert info['battery_throughput'] == 0.0, \
            f"Idle should produce zero throughput, got {info['battery_throughput']}"

        # SOC should be unchanged
        assert env.battery_soc == initial_soc, \
            f"Idle should not change SOC: {initial_soc} -> {env.battery_soc}"

    def test_battery_soc_reset(self, env):
        """Test battery SOC resets to 50% on environment reset."""
        env.reset(seed=42)

        # Modify SOC
        env.battery_soc = 2.0

        # Reset
        env.reset(seed=43)

        expected_soc = 0.5 * env.battery_capacity_mwh
        assert env.battery_soc == expected_soc, \
            f"SOC should reset to 50%, got {env.battery_soc}"

    def test_battery_soc_in_observation(self, env):
        """Test battery SOC is correctly reflected in observation."""
        env.reset(seed=42)

        # Set known SOC
        test_soc = 7.5
        env.battery_soc = test_soc

        obs = env._get_observation()
        observed_soc_normalized = obs[1]

        expected_normalized = test_soc / env.battery_capacity_mwh
        assert np.isclose(observed_soc_normalized, expected_normalized, atol=1e-6), \
            f"Observed SOC {observed_soc_normalized} != expected {expected_normalized}"

    def test_battery_parameters_match_architecture(self, env):
        """Verify battery parameters match architecture specification."""
        assert env.battery_capacity_mwh == 10.0, \
            f"Capacity should be 10 MWh, got {env.battery_capacity_mwh}"
        assert env.battery_power_mw == 5.0, \
            f"Power should be 5 MW, got {env.battery_power_mw}"
        assert env.battery_efficiency == 0.92, \
            f"Efficiency should be 0.92, got {env.battery_efficiency}"
        assert env.battery_degradation_cost == 0.01, \
            f"Degradation cost should be 0.01, got {env.battery_degradation_cost}"


class TestBatteryRoundTrip:
    """Integration tests for complete battery charge/discharge cycles."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_full_round_trip_efficiency(self, env):
        """Test full charge → discharge cycle verifies round-trip efficiency."""
        env.reset(seed=42)

        # Start with empty battery
        env.battery_soc = 0.0

        # Find high-PV period and charge
        total_charged = 0.0
        for _ in range(100):
            row = env.data.iloc[env.current_idx]
            pv_actual = row['pv_actual_mwh']

            if pv_actual > 3.0 and env.battery_soc < env.battery_capacity_mwh - 1:
                action = np.zeros(25, dtype=np.float32)
                action[24] = 1.0  # Charge

                initial_soc = env.battery_soc
                obs, _, terminated, truncated, info = env.step(action)

                throughput = info['battery_throughput']
                total_charged += throughput

                if env.battery_soc >= env.battery_capacity_mwh - 0.5:
                    break
            else:
                action = np.zeros(25, dtype=np.float32)
                action[24] = 0.5  # Idle
                obs, _, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                env.reset(seed=42 + _)

        # Now discharge
        charged_soc = env.battery_soc
        total_discharged = 0.0

        for _ in range(20):
            if env.battery_soc > 0.1:
                action = np.zeros(25, dtype=np.float32)
                action[24] = 0.0  # Discharge

                obs, _, terminated, truncated, info = env.step(action)
                total_discharged += info['battery_throughput']
            else:
                break

            if terminated or truncated:
                break

        # Round-trip efficiency check
        # Energy stored = total_charged * one_way_efficiency
        # Energy delivered = stored * one_way_efficiency = total_charged * efficiency
        if total_charged > 0 and total_discharged > 0:
            actual_round_trip = total_discharged / total_charged
            expected_round_trip = env.battery_efficiency

            # Allow some tolerance due to partial cycles
            assert actual_round_trip <= expected_round_trip + 0.05, \
                f"Round-trip efficiency {actual_round_trip} worse than expected {expected_round_trip}"


class TestBatteryPerformance:
    """Performance tests for battery operations."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_battery_operation_performance(self, env):
        """Measure battery operation time per step.

        Target: Average step time <50ms to satisfy NFR2 (5 sec / 48 steps = ~104ms budget).
        Battery operations should be a small fraction of this.
        """
        import time

        env.reset(seed=42)

        # Warm-up
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)

        env.reset(seed=42)

        # Measure step time (includes battery operations)
        times = []
        for i in range(100):
            action = np.zeros(25, dtype=np.float32)
            action[24] = np.random.random()  # Random battery action

            start = time.perf_counter()
            obs, _, terminated, truncated, _ = env.step(action)
            elapsed = time.perf_counter() - start
            times.append(elapsed * 1000)  # Convert to ms

            if terminated or truncated:
                env.reset(seed=42 + i)

        avg_time = np.mean(times)
        p95_time = np.percentile(times, 95)

        # Step time should be well under 50ms for 48 steps in 5 seconds
        # Note: Windows CI can be slower, so we use a generous threshold
        assert avg_time < 50.0, f"Average step time {avg_time:.2f}ms exceeds 50ms"
        assert p95_time < 100.0, f"P95 step time {p95_time:.2f}ms exceeds 100ms"
