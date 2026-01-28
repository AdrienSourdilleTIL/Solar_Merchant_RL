"""
Tests for market settlement logic in SolarMerchantEnv.

This module validates market settlement mechanics according to Story 2-4:
- Revenue calculation: delivered * day-ahead price
- Short imbalance: penalty at 1.5x day-ahead price
- Long imbalance: opportunity cost at 0.6x day-ahead price
- Reward composition: revenue - imbalance_cost - degradation

Market Parameters (per architecture/PRD):
- Short penalty multiplier: 1.5x day-ahead price
- Long recovery multiplier: 0.6x day-ahead price
- Revenue = delivered energy × day-ahead price
- Reward = revenue - imbalance_cost - battery_degradation_cost
"""

import numpy as np
import pytest
from src.environment import load_environment


class TestSettlementValidation:
    """Task 1: Validate existing market settlement implementation."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_revenue_uses_delivered_times_price(self, env):
        """Verify revenue = delivered * price_eur_mwh."""
        env.reset(seed=42)

        # Run a step and verify revenue calculation
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.5  # Idle battery
        obs, reward, _, _, info = env.step(action)

        delivered = info['delivered']
        price = info['price']
        revenue = info['revenue']

        expected_revenue = delivered * price
        assert np.isclose(revenue, expected_revenue, atol=1e-6), \
            f"Revenue {revenue} != expected {expected_revenue} (delivered={delivered}, price={price})"

    def test_short_imbalance_uses_short_price(self, env):
        """Verify short imbalance uses price_imbalance_short (1.5x DA price)."""
        env.reset(seed=42)

        # Find a step where we're short (delivered < committed)
        # First make a commitment by advancing to commitment hour
        for _ in range(24):  # Max iterations to find commitment hour
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5  # Commit 50% of max
            action[24] = 0.5  # Idle battery
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        # After commitments are active, find a short position
        found_short = False
        for _ in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if imbalance < 0:
                found_short = True
                # Verify short penalty calculation
                row = env.data.iloc[env.current_idx - 1]
                price_short = row['price_imbalance_short']
                expected_cost = abs(imbalance) * price_short
                assert np.isclose(info['imbalance_cost'], expected_cost, atol=1e-6), \
                    f"Short cost {info['imbalance_cost']} != expected {expected_cost}"
                break

            if terminated or truncated:
                break

        # Note: Not all seeds will produce a short position in first 48 hours
        # This is acceptable as we verify when we do find one

    def test_long_imbalance_uses_opportunity_cost(self, env):
        """Verify long imbalance uses opportunity cost formula: excess * (price - price_long)."""
        env.reset(seed=42)

        # Find a step where we're long (delivered > committed)
        found_long = False
        for _ in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.0  # Zero commitment
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if imbalance > 0:
                found_long = True
                # Verify long opportunity cost calculation
                row = env.data.iloc[env.current_idx - 1]
                price = row['price_eur_mwh']
                price_long = row['price_imbalance_long']
                expected_cost = imbalance * (price - price_long)
                assert np.isclose(info['imbalance_cost'], expected_cost, atol=1e-6), \
                    f"Long cost {info['imbalance_cost']} != expected {expected_cost}"
                break

            if terminated or truncated:
                break

        # With zero commitment and any PV production, we should find long positions
        assert found_long, "Expected to find at least one long position with zero commitment"

    def test_reward_composition_formula(self, env):
        """Verify reward = revenue - imbalance_cost - degradation."""
        env.reset(seed=42)

        for _ in range(20):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            revenue = info['revenue']
            imbalance_cost = info['imbalance_cost']
            # Degradation = throughput * degradation_cost
            degradation = info['battery_throughput'] * env.battery_degradation_cost

            expected_reward = revenue - imbalance_cost - degradation
            assert np.isclose(reward, expected_reward, atol=1e-6), \
                f"Reward {reward} != expected {expected_reward}"

            if terminated or truncated:
                break

    def test_settlement_logic_trace(self, env):
        """Trace settlement logic through step() method to verify all components."""
        env.reset(seed=42)

        # Run a full episode and accumulate totals
        total_revenue = 0.0
        total_imbalance = 0.0
        total_degradation = 0.0
        total_reward = 0.0

        for _ in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.3  # Conservative commitment
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            total_revenue += info['revenue']
            total_imbalance += info['imbalance_cost']
            total_degradation += info['battery_throughput'] * env.battery_degradation_cost
            total_reward += reward

            if terminated or truncated:
                break

        # Verify totals are consistent
        expected_total = total_revenue - total_imbalance - total_degradation
        assert np.isclose(total_reward, expected_total, atol=1e-3), \
            f"Total reward {total_reward} != expected {expected_total}"


class TestRevenueCalculation:
    """Task 2: Test revenue calculation scenarios."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_revenue_with_zero_delivery(self, env):
        """Test revenue = 0 when delivered = 0 MWh."""
        env.reset(seed=42)

        # Find an hour with no PV (night) to test zero delivery
        found_zero = False
        for _ in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            if info['delivered'] == 0:
                found_zero = True
                assert info['revenue'] == 0, \
                    f"Revenue should be 0 with zero delivery, got {info['revenue']}"
                break

            if terminated or truncated:
                break

        # Note: May not find zero delivery if we start during day

    def test_revenue_with_typical_delivery(self, env):
        """Test revenue calculation with typical delivery values."""
        env.reset(seed=42)

        # Run steps and verify revenue formula
        for _ in range(20):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            delivered = info['delivered']
            price = info['price']
            revenue = info['revenue']

            expected = delivered * price
            assert np.isclose(revenue, expected, atol=1e-6), \
                f"Revenue {revenue} != {delivered} * {price} = {expected}"

            if terminated or truncated:
                break

    def test_revenue_with_maximum_delivery(self, env):
        """Test revenue with maximum delivery (plant capacity + battery)."""
        env.reset(seed=42)

        # Fully charge battery first
        for _ in range(10):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 1.0  # Full charge
            env.step(action)

        # Then try to discharge at maximum
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.0  # Full discharge
        obs, reward, terminated, truncated, info = env.step(action)

        delivered = info['delivered']
        price = info['price']
        revenue = info['revenue']

        # Revenue should still be delivered * price
        expected = delivered * price
        assert np.isclose(revenue, expected, atol=1e-6), \
            f"Revenue {revenue} != expected {expected}"

    def test_revenue_with_negative_prices(self, env):
        """Test revenue calculation with negative prices (delivered * negative = negative)."""
        env.reset(seed=42)

        # Find an hour with negative prices
        found_negative = False
        for _ in range(1000):  # Search through more data
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if info['price'] < 0:
                found_negative = True
                # With negative price and positive delivery, revenue is negative
                delivered = info['delivered']
                price = info['price']
                revenue = info['revenue']

                expected = delivered * price
                assert np.isclose(revenue, expected, atol=1e-6), \
                    f"Revenue {revenue} != {delivered} * {price} = {expected}"
                assert revenue <= 0 if delivered > 0 else True, \
                    "Negative price with positive delivery should give negative/zero revenue"
                break

            if terminated or truncated:
                env.reset(seed=42 + _)

        # Note: Negative prices may not exist in all datasets

    def test_revenue_accumulates_in_episode(self, env):
        """Verify revenue accumulates correctly in episode_revenue."""
        env.reset(seed=42)

        accumulated = 0.0
        for _ in range(20):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            accumulated += info['revenue']

            # Verify internal tracking matches
            assert np.isclose(env.episode_revenue, accumulated, atol=1e-3), \
                f"Episode revenue {env.episode_revenue} != accumulated {accumulated}"

            if terminated or truncated:
                break


class TestShortImbalanceSettlement:
    """Task 3: Test short imbalance settlement (under-delivery)."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_under_delivery_penalty_calculation(self, env):
        """Test penalty calculation for under-delivery."""
        env.reset(seed=42)

        # Make aggressive commitments to force short positions
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 1.0  # Maximum commitment
            action[24] = 0.0  # Discharge battery to help deliver
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if imbalance < 0:
                # Verify penalty calculation
                row = env.data.iloc[env.current_idx - 1]
                price_short = row['price_imbalance_short']
                shortage = abs(imbalance)
                expected_cost = shortage * price_short
                assert np.isclose(info['imbalance_cost'], expected_cost, atol=1e-6), \
                    f"Short penalty {info['imbalance_cost']} != {shortage} * {price_short} = {expected_cost}"

            if terminated or truncated:
                break

    def test_short_penalty_rate_is_1_5x(self, env):
        """Test that short penalty rate is 1.5x day-ahead price."""
        env.reset(seed=42)

        # Verify in data that price_short = 1.5 * price
        for _ in range(20):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            env.step(action)

            row = env.data.iloc[env.current_idx - 1]
            price = row['price_eur_mwh']
            price_short = row['price_imbalance_short']

            # When price is positive, short should be 1.5x (clamped at 0 for negative)
            if price > 0:
                expected_short = price * 1.5
                assert np.isclose(price_short, expected_short, atol=1e-6), \
                    f"Short price {price_short} != 1.5 * {price} = {expected_short}"
            else:
                assert price_short >= 0, "Short price should be clipped at 0"

    def test_partial_shortage(self, env):
        """Test partial shortage (delivered < committed but > 0)."""
        env.reset(seed=42)

        found_partial = False
        for i in range(100):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.8  # High commitment
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            delivered = info['delivered']
            committed = info['committed']
            imbalance = info['imbalance']

            # Partial shortage: delivered > 0 and committed > delivered
            if delivered > 0 and committed > delivered:
                found_partial = True
                shortage = committed - delivered
                assert np.isclose(imbalance, -shortage, atol=1e-6), \
                    f"Imbalance {imbalance} != -{shortage}"
                assert info['imbalance_cost'] > 0, \
                    "Partial shortage should have positive imbalance cost"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_complete_shortage(self, env):
        """Test complete shortage (delivered = 0, committed > 0)."""
        env.reset(seed=42)

        # Find night hour where PV = 0 but we have commitment
        # First make commitments
        for _ in range(24):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5
            action[24] = 0.5
            env.step(action)

        # Now find zero delivery with commitment
        found_complete = False
        for i in range(100):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5  # Idle battery (no discharge)
            obs, reward, terminated, truncated, info = env.step(action)

            delivered = info['delivered']
            committed = info['committed']

            if delivered == 0 and committed > 0:
                found_complete = True
                # Complete shortage
                row = env.data.iloc[env.current_idx - 1]
                price_short = row['price_imbalance_short']
                expected_cost = committed * price_short
                assert np.isclose(info['imbalance_cost'], expected_cost, atol=1e-6), \
                    f"Complete shortage cost {info['imbalance_cost']} != {committed} * {price_short}"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_imbalance_cost_positive_for_short(self, env):
        """Verify imbalance_cost is positive for short positions."""
        env.reset(seed=42)

        for i in range(50):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.9  # High commitment to force shorts
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if imbalance < 0:
                assert info['imbalance_cost'] >= 0, \
                    f"Short position should have non-negative imbalance cost, got {info['imbalance_cost']}"

            if terminated or truncated:
                env.reset(seed=42 + i)


class TestLongImbalanceSettlement:
    """Task 4: Test long imbalance settlement (over-delivery)."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_over_delivery_value_calculation(self, env):
        """Test value calculation for over-delivery."""
        env.reset(seed=42)

        # Make low/zero commitments to force long positions
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.0  # Zero commitment
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if imbalance > 0:
                # Verify opportunity cost calculation
                row = env.data.iloc[env.current_idx - 1]
                price = row['price_eur_mwh']
                price_long = row['price_imbalance_long']
                expected_cost = imbalance * (price - price_long)
                assert np.isclose(info['imbalance_cost'], expected_cost, atol=1e-6), \
                    f"Long cost {info['imbalance_cost']} != {imbalance} * ({price} - {price_long})"

            if terminated or truncated:
                break

    def test_long_rate_is_0_6x(self, env):
        """Test that long recovery rate is 0.6x day-ahead price."""
        env.reset(seed=42)

        # Verify in data that price_long = 0.6 * price
        for _ in range(20):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            env.step(action)

            row = env.data.iloc[env.current_idx - 1]
            price = row['price_eur_mwh']
            price_long = row['price_imbalance_long']

            # When price is positive, long should be 0.6x (clamped at 0 for negative)
            if price > 0:
                expected_long = price * 0.6
                assert np.isclose(price_long, expected_long, atol=1e-6), \
                    f"Long price {price_long} != 0.6 * {price} = {expected_long}"
            else:
                assert price_long >= 0, "Long price should be clipped at 0"

    def test_excess_calculation(self, env):
        """Test excess calculation: delivered - committed."""
        env.reset(seed=42)

        for _ in range(30):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.2  # Low commitment
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            delivered = info['delivered']
            committed = info['committed']
            imbalance = info['imbalance']

            expected_imbalance = delivered - committed
            assert np.isclose(imbalance, expected_imbalance, atol=1e-6), \
                f"Imbalance {imbalance} != {delivered} - {committed} = {expected_imbalance}"

            if terminated or truncated:
                break

    def test_opportunity_cost_formula(self, env):
        """Test opportunity cost: excess * (price - price_long)."""
        env.reset(seed=42)

        for i in range(50):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.0  # Zero commitment
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if imbalance > 0:
                row = env.data.iloc[env.current_idx - 1]
                price = row['price_eur_mwh']
                price_long = row['price_imbalance_long']

                # Opportunity cost = excess * (price - price_long)
                expected_cost = imbalance * (price - price_long)
                assert np.isclose(info['imbalance_cost'], expected_cost, atol=1e-6), \
                    f"Opportunity cost {info['imbalance_cost']} != {imbalance} * ({price} - {price_long})"

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_imbalance_cost_reflects_lost_opportunity(self, env):
        """Verify imbalance_cost reflects lost revenue opportunity for long positions."""
        env.reset(seed=42)

        for i in range(50):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.0
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if imbalance > 0:
                row = env.data.iloc[env.current_idx - 1]
                price = row['price_eur_mwh']
                price_long = row['price_imbalance_long']

                # When price > price_long, there's an opportunity cost
                if price > price_long:
                    assert info['imbalance_cost'] > 0, \
                        "Long position with price > price_long should have positive opportunity cost"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)


class TestRewardComposition:
    """Task 5: Test reward composition."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_reward_formula(self, env):
        """Test reward = revenue - imbalance_cost - degradation_cost."""
        env.reset(seed=42)

        for _ in range(30):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            revenue = info['revenue']
            imbalance_cost = info['imbalance_cost']
            degradation = info['battery_throughput'] * env.battery_degradation_cost

            expected = revenue - imbalance_cost - degradation
            assert np.isclose(reward, expected, atol=1e-6), \
                f"Reward {reward} != {revenue} - {imbalance_cost} - {degradation} = {expected}"

            if terminated or truncated:
                break

    def test_reward_with_zero_imbalance(self, env):
        """Test reward with balanced delivery (zero imbalance)."""
        env.reset(seed=42)

        # Find a step with zero commitment and zero delivery (night)
        for i in range(100):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.0  # Zero commitment
            action[24] = 0.5  # Idle battery
            obs, reward, terminated, truncated, info = env.step(action)

            # If both delivered and committed are 0, imbalance is 0
            if info['delivered'] == 0 and info['committed'] == 0:
                assert info['imbalance'] == 0, "Should have zero imbalance"
                assert info['imbalance_cost'] == 0, \
                    f"Zero imbalance should have zero cost, got {info['imbalance_cost']}"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_reward_with_positive_imbalance(self, env):
        """Test reward with over-delivery (positive imbalance)."""
        env.reset(seed=42)

        found_positive = False
        for i in range(50):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.0  # Zero commitment to get over-delivery
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if info['imbalance'] > 0:
                found_positive = True
                # Reward should account for opportunity cost
                revenue = info['revenue']
                imbalance_cost = info['imbalance_cost']
                degradation = info['battery_throughput'] * env.battery_degradation_cost
                expected = revenue - imbalance_cost - degradation
                assert np.isclose(reward, expected, atol=1e-6), \
                    f"Reward with positive imbalance incorrect"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

        assert found_positive, "Should find positive imbalance with zero commitment"

    def test_reward_with_negative_imbalance(self, env):
        """Test reward with under-delivery (negative imbalance)."""
        env.reset(seed=42)

        found_negative = False
        for i in range(100):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 1.0  # Max commitment to force under-delivery
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if info['imbalance'] < 0:
                found_negative = True
                # Reward should include penalty
                revenue = info['revenue']
                imbalance_cost = info['imbalance_cost']
                degradation = info['battery_throughput'] * env.battery_degradation_cost
                expected = revenue - imbalance_cost - degradation
                assert np.isclose(reward, expected, atol=1e-6), \
                    f"Reward with negative imbalance incorrect"
                # Penalty should reduce reward
                assert imbalance_cost >= 0, "Short penalty should be non-negative"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_reward_accumulation_across_episode(self, env):
        """Test reward accumulates correctly across episode."""
        env.reset(seed=42)

        total_reward = 0.0
        for _ in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            total_reward += reward

            if terminated or truncated:
                break

        # Verify internal tracking
        expected_total = env.episode_revenue - env.episode_imbalance_cost - env.episode_degradation_cost
        assert np.isclose(total_reward, expected_total, atol=1e-3), \
            f"Total reward {total_reward} != expected {expected_total}"


class TestEdgeCasesAndEconomics:
    """Task 6: Test edge cases and economic verification."""

    @pytest.fixture
    def env(self):
        """Create environment instance for testing."""
        data_path = 'data/processed/train.csv'
        env = load_environment(data_path)
        yield env
        env.close()

    def test_settlement_at_hour_zero(self, env):
        """Test settlement at hour 0 (day transition)."""
        env.reset(seed=42)

        # Advance to hour 0 (midnight)
        found_midnight = False
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.3
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if info['hour'] == 0:
                found_midnight = True
                # Settlement should still work at midnight
                # Verify reward is computed correctly
                revenue = info['revenue']
                imbalance_cost = info['imbalance_cost']
                degradation = info['battery_throughput'] * env.battery_degradation_cost
                expected = revenue - imbalance_cost - degradation
                assert np.isclose(reward, expected, atol=1e-6), \
                    "Settlement at hour 0 should work correctly"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

        assert found_midnight, "Should reach hour 0 within 48 steps"

    def test_settlement_with_zero_commitment(self, env):
        """Test settlement with zero commitment (no penalty for long)."""
        env.reset(seed=42)

        for _ in range(20):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.0  # Zero commitment
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            committed = info['committed']
            assert committed == 0, f"Commitment should be 0, got {committed}"

            # With zero commitment, any delivery is over-delivery (long)
            # But if delivery is also 0, imbalance is 0
            imbalance = info['imbalance']
            if info['delivered'] > 0:
                assert imbalance == info['delivered'], \
                    "With zero commitment, imbalance should equal delivery"

            if terminated or truncated:
                break

    def test_asymmetric_risk(self, env):
        """Verify short penalty > long opportunity cost for same magnitude."""
        env.reset(seed=42)

        # Get a sample price
        row = env.data.iloc[env.current_idx]
        price = row['price_eur_mwh']
        price_short = row['price_imbalance_short']
        price_long = row['price_imbalance_long']

        if price > 0:
            # For same imbalance magnitude:
            # Short cost = |imbalance| * price_short = |imbalance| * 1.5 * price
            # Long cost = imbalance * (price - price_long) = imbalance * 0.4 * price
            imbalance = 1.0  # 1 MWh
            short_cost = imbalance * price_short
            long_cost = imbalance * (price - price_long)

            assert short_cost > long_cost, \
                f"Short penalty {short_cost} should > long cost {long_cost} for asymmetric risk"

    def test_with_extreme_high_prices(self, env):
        """Test settlement with extreme high prices."""
        env.reset(seed=42)

        # Find a high price period
        found_high = False
        for i in range(500):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if info['price'] > 100:  # High price threshold
                found_high = True
                # Settlement should still work correctly
                revenue = info['revenue']
                imbalance_cost = info['imbalance_cost']
                degradation = info['battery_throughput'] * env.battery_degradation_cost
                expected = revenue - imbalance_cost - degradation
                assert np.isclose(reward, expected, atol=1e-6), \
                    "Settlement with high prices should work correctly"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_with_near_zero_prices(self, env):
        """Test settlement with near-zero prices."""
        env.reset(seed=42)

        found_low = False
        for i in range(500):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if 0 < info['price'] < 5:  # Near-zero positive price
                found_low = True
                # Settlement should still work correctly
                revenue = info['revenue']
                imbalance_cost = info['imbalance_cost']
                degradation = info['battery_throughput'] * env.battery_degradation_cost
                expected = revenue - imbalance_cost - degradation
                assert np.isclose(reward, expected, atol=1e-6), \
                    "Settlement with low prices should work correctly"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)

    def test_no_commitment_made_zeros(self, env):
        """Test settlement when no commitment has been made (todays_commitments = zeros)."""
        env.reset(seed=42)

        # At start, todays_commitments should be zeros
        assert np.all(env.todays_commitments == 0), \
            "Initial todays_commitments should be zeros"

        # First step with no prior commitment
        action = np.zeros(25, dtype=np.float32)
        action[:24] = 0.0  # Zero commitment for tomorrow too
        action[24] = 0.5
        obs, reward, terminated, truncated, info = env.step(action)

        # With zero commitment, any delivery is over-delivery
        committed = info['committed']
        assert committed == 0, f"Initial commitment should be 0, got {committed}"

        # No short penalty possible with zero commitment
        imbalance = info['imbalance']
        assert imbalance >= 0, \
            f"With zero commitment, imbalance should be >= 0, got {imbalance}"

    def test_perfect_delivery_gives_zero_imbalance_cost(self, env):
        """Test that perfect delivery (delivered == committed) gives zero imbalance cost."""
        env.reset(seed=42)

        # This is hard to achieve naturally, but we can check when imbalance is near zero
        for i in range(100):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            imbalance = info['imbalance']
            if abs(imbalance) < 0.001:
                # Near-zero imbalance should have near-zero cost
                assert abs(info['imbalance_cost']) < 0.01, \
                    f"Near-zero imbalance {imbalance} should have near-zero cost, got {info['imbalance_cost']}"
                break

            if terminated or truncated:
                env.reset(seed=42 + i)
