"""
Tests for episode flow and reset logic in SolarMerchantEnv.

This module validates episode mechanics according to Story 2-5:
- Episode reset to random starting points
- Initial state configuration (SOC, commitments, tracking)
- 48-hour episode structure for credit assignment
- Commitment hour processing
- Episode termination and boundary handling

Key Design Decisions (from CLAUDE.md):
- 48-hour episodes fix credit assignment problem
- Commitments made at hour 11 are for the next day
- Agent sees imbalance costs before episode ends
- Previous 24h design was fundamentally broken

Episode Flow:
- Hour 0-10: Agent operates with no commitments (or inherited)
- Hour 11: Agent makes commitments for NEXT day
- Hours 12-23: Rest of first day
- Hour 24 (midnight): Tomorrow's commitments become today's
- Hours 24-47: Agent operates under its own commitments
- Hour 48: Episode terminates AFTER consequences experienced
"""

import numpy as np
import pytest
import time
from src.environment import load_environment


class TestResetRandomization:
    """Task 1: Validate reset() randomization logic (AC: #1)."""

    # Fixture 'env' is provided by conftest.py

    def test_random_episode_start_within_valid_range(self, env):
        """Verify random episode start selection within valid data range."""
        env.reset(seed=42)

        # Episode start should be within valid range
        assert env.episode_start_idx >= 0, \
            f"Episode start {env.episode_start_idx} should be >= 0"

        max_valid_start = len(env.data) - 48 * 30  # Leave 30 days buffer
        assert env.episode_start_idx < max_valid_start, \
            f"Episode start {env.episode_start_idx} should be < {max_valid_start}"

    def test_episode_start_avoids_boundary(self, env):
        """Verify episode start avoids last 48*30 hours (boundary protection)."""
        # Test with multiple seeds
        for seed in range(100):
            env.reset(seed=seed)

            # Should never start in the last 48*30 hours
            max_start = len(env.data) - 48 * 30
            assert env.episode_start_idx < max_start, \
                f"Seed {seed}: Start {env.episode_start_idx} too close to end (max={max_start})"

    def test_commitment_hour_alignment_adjustment(self, env):
        """Verify commitment hour alignment adjustment."""
        env.reset(seed=42)

        # Check that episode will encounter commitment hour (11)
        episode_hours = []
        for i in range(48):
            if env.current_idx + i < len(env.data):
                episode_hours.append(int(env.data.iloc[env.current_idx + i]['hour']))

        assert env.commitment_hour in episode_hours, \
            f"Commitment hour {env.commitment_hour} should be in episode hours: {set(episode_hours)}"

    def test_different_seeds_produce_different_starts(self, env):
        """Verify different seeds produce different starting points."""
        starts = set()
        for seed in range(50):
            env.reset(seed=seed)
            starts.add(env.episode_start_idx)

        # Should have many different starting points
        assert len(starts) > 10, \
            f"Expected >10 different starts with 50 seeds, got {len(starts)}"

    def test_same_seed_produces_reproducible_starts(self, env):
        """Verify same seed produces reproducible starting points."""
        # Reset with same seed multiple times
        starts = []
        for _ in range(5):
            env.reset(seed=42)
            starts.append(env.episode_start_idx)

        # All starts should be identical
        assert all(s == starts[0] for s in starts), \
            f"Same seed should produce identical starts, got {starts}"


class TestInitialStateConfiguration:
    """Task 2: Test initial state configuration (AC: #1)."""

    # Fixture 'env' is provided by conftest.py

    def test_default_battery_soc(self, env):
        """Test default battery SOC is 0.5 * capacity (5 MWh)."""
        env.reset(seed=42)

        expected_soc = 0.5 * env.battery_capacity_mwh  # 5 MWh
        assert np.isclose(env.battery_soc, expected_soc, atol=1e-6), \
            f"Battery SOC {env.battery_soc} != expected {expected_soc}"

    def test_configurable_initial_soc(self, env):
        """Test initial SOC can be configured via options (AC: #1 - configurable SOC)."""
        # Test various initial SOC values
        for initial_soc in [0.0, 0.25, 0.5, 0.75, 1.0]:
            env.reset(seed=42, options={'initial_soc': initial_soc})

            expected_soc = initial_soc * env.battery_capacity_mwh
            assert np.isclose(env.battery_soc, expected_soc, atol=1e-6), \
                f"Initial SOC {initial_soc}: battery_soc {env.battery_soc} != expected {expected_soc}"

    def test_initial_soc_clamps_invalid_values(self, env):
        """Test that invalid initial_soc values are clamped to [0, 1]."""
        # Test value below 0
        env.reset(seed=42, options={'initial_soc': -0.5})
        assert env.battery_soc == 0.0, \
            f"Negative initial_soc should clamp to 0, got {env.battery_soc}"

        # Test value above 1
        env.reset(seed=42, options={'initial_soc': 1.5})
        expected_max = 1.0 * env.battery_capacity_mwh
        assert np.isclose(env.battery_soc, expected_max, atol=1e-6), \
            f"initial_soc > 1 should clamp to 1.0, got {env.battery_soc / env.battery_capacity_mwh}"

    def test_todays_commitments_initialized_zeros(self, env):
        """Test todays_commitments initialized to zeros."""
        env.reset(seed=42)

        assert np.all(env.todays_commitments == 0), \
            f"todays_commitments should be zeros: {env.todays_commitments}"
        assert env.todays_commitments.shape == (24,), \
            f"todays_commitments shape should be (24,): {env.todays_commitments.shape}"

    def test_tomorrows_commitments_initialized_zeros(self, env):
        """Test tomorrows_commitments initialized to zeros."""
        env.reset(seed=42)

        assert np.all(env.tomorrows_commitments == 0), \
            f"tomorrows_commitments should be zeros: {env.tomorrows_commitments}"
        assert env.tomorrows_commitments.shape == (24,), \
            f"tomorrows_commitments shape should be (24,): {env.tomorrows_commitments.shape}"

    def test_hourly_delivered_empty(self, env):
        """Test hourly_delivered dict is empty."""
        env.reset(seed=42)

        assert len(env.hourly_delivered) == 0, \
            f"hourly_delivered should be empty: {env.hourly_delivered}"
        assert isinstance(env.hourly_delivered, dict), \
            f"hourly_delivered should be dict: {type(env.hourly_delivered)}"

    def test_episode_tracking_counters_reset(self, env):
        """Test episode tracking counters reset (revenue, imbalance, degradation)."""
        # First run some steps
        env.reset(seed=42)
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)

        # Reset and verify counters are zeroed
        env.reset(seed=43)

        assert env.episode_revenue == 0.0, \
            f"episode_revenue should be 0: {env.episode_revenue}"
        assert env.episode_imbalance_cost == 0.0, \
            f"episode_imbalance_cost should be 0: {env.episode_imbalance_cost}"
        assert env.episode_degradation_cost == 0.0, \
            f"episode_degradation_cost should be 0: {env.episode_degradation_cost}"


class TestObservationValidityOnReset:
    """Task 3: Test observation validity on reset (AC: #1)."""

    # Fixture 'env' is provided by conftest.py

    def test_reset_returns_valid_84_dim_observation(self, env):
        """Verify reset() returns valid 84-dim observation."""
        obs, info = env.reset(seed=42)

        assert obs.shape == (84,), \
            f"Observation shape should be (84,): {obs.shape}"
        assert obs.dtype == np.float32, \
            f"Observation dtype should be float32: {obs.dtype}"

    def test_observation_no_nan_values(self, env):
        """Verify observation contains no NaN values."""
        obs, info = env.reset(seed=42)

        assert not np.any(np.isnan(obs)), \
            f"Observation contains NaN values at indices: {np.where(np.isnan(obs))[0]}"

    def test_observation_no_inf_values(self, env):
        """Verify observation contains no Inf values."""
        obs, info = env.reset(seed=42)

        assert not np.any(np.isinf(obs)), \
            f"Observation contains Inf values at indices: {np.where(np.isinf(obs))[0]}"

    def test_hour_normalization_range(self, env):
        """Verify hour normalization is in [0, 1) range."""
        for seed in range(20):
            obs, info = env.reset(seed=seed)
            hour_normalized = obs[0]  # First element is hour

            assert 0 <= hour_normalized < 1, \
                f"Hour normalization {hour_normalized} not in [0, 1)"

    def test_soc_normalization_range(self, env):
        """Verify SOC normalization is in [0, 1] range."""
        for seed in range(20):
            obs, info = env.reset(seed=seed)
            soc_normalized = obs[1]  # Second element is SOC

            assert 0 <= soc_normalized <= 1, \
                f"SOC normalization {soc_normalized} not in [0, 1]"

            # After reset, SOC should be 0.5
            assert np.isclose(soc_normalized, 0.5, atol=0.01), \
                f"Initial SOC should be ~0.5: {soc_normalized}"

    def test_info_dict_empty_on_reset(self, env):
        """Verify info dict is empty on reset."""
        obs, info = env.reset(seed=42)

        assert info == {}, \
            f"Info dict should be empty on reset: {info}"


class TestEpisodeStructure:
    """Task 4: Test 48-hour episode structure (AC: #2)."""

    # Fixture 'env' is provided by conftest.py

    def test_episode_terminates_at_48_steps(self, env):
        """Test episode terminates exactly at 48 steps."""
        env.reset(seed=42)

        steps = 0
        terminated = False
        for i in range(100):  # More than enough
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1

            if terminated:
                break

        assert terminated, "Episode should terminate"
        assert steps == 48, f"Episode should terminate at 48 steps, got {steps}"

    def test_terminated_flag_after_48_steps(self, env):
        """Test terminated flag is True after 48 steps."""
        env.reset(seed=42)

        # Run exactly 47 steps
        for _ in range(47):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)
            assert not terminated, "Should not terminate before step 48"

        # 48th step should terminate
        action = np.zeros(25, dtype=np.float32)
        action[24] = 0.5
        obs, reward, terminated, truncated, info = env.step(action)
        assert terminated, "Should terminate at step 48"

    def test_truncated_flag_handles_data_boundary(self, env):
        """Test truncated flag handles data boundary."""
        # Force start near end of data
        env.reset(seed=42)

        # Manually set current_idx near end (for testing only)
        original_idx = env.current_idx
        env.current_idx = len(env.data) - 10
        env.episode_start_idx = env.current_idx

        # Run until truncation
        truncated = False
        for _ in range(20):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)
            if truncated:
                break

        assert truncated, "Should truncate when reaching data boundary"

        # Reset back
        env.reset(seed=42)

    def test_commitment_at_hour_11_processed(self, env):
        """Test commitment at hour 11 is processed correctly."""
        env.reset(seed=42)

        # Advance until we hit commitment hour
        found_commitment = False
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.7  # Commit 70% of capacity
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if 'new_commitment' in info:
                found_commitment = True
                # Verify commitment was set
                assert not np.all(info['new_commitment'] == 0), \
                    "Commitment should be non-zero"
                assert info['new_commitment'].shape == (24,), \
                    "Commitment should have shape (24,)"
                break

            if terminated or truncated:
                break

        assert found_commitment, "Should find commitment hour within 48 steps"

    def test_midnight_transition_moves_commitments(self, env):
        """Test midnight transition moves commitments correctly."""
        env.reset(seed=42)

        # Advance until we hit commitment hour and make commitment
        tomorrows_commitment = None
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.7
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if 'new_commitment' in info:
                tomorrows_commitment = info['new_commitment'].copy()
                break

            if terminated or truncated:
                break

        if tomorrows_commitment is None:
            pytest.skip("Could not reach commitment hour")

        # Continue until midnight (hour 0)
        found_midnight = False
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if info['hour'] == 0:
                found_midnight = True
                # After midnight, todays_commitments should match tomorrows
                assert np.allclose(env.todays_commitments, tomorrows_commitment, atol=1e-6), \
                    "Midnight transition should move tomorrows_commitments to todays_commitments"
                # tomorrows should be reset to zeros
                assert np.all(env.tomorrows_commitments == 0), \
                    "tomorrows_commitments should be zeros after midnight"
                break

            if terminated or truncated:
                break


class TestCommitmentHourValidation:
    """Task 5: Test commitment hour validation (AC: #2)."""

    # Fixture 'env' is provided by conftest.py

    def test_episodes_contain_commitment_hour(self, env):
        """Verify episodes are adjusted to contain commitment hour."""
        for seed in range(20):
            env.reset(seed=seed)

            # Scan episode hours
            episode_hours = []
            for i in range(48):
                if env.current_idx + i < len(env.data):
                    episode_hours.append(int(env.data.iloc[env.current_idx + i]['hour']))

            assert env.commitment_hour in episode_hours, \
                f"Seed {seed}: Commitment hour {env.commitment_hour} not in episode hours"

    def test_commitment_processing_only_at_hour_11(self, env):
        """Test commitment processing only occurs at hour 11."""
        env.reset(seed=42)

        commitment_hours_found = []
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if 'new_commitment' in info:
                # Get the hour when commitment was made
                hour = env.data.iloc[env.current_idx - 1]['hour']
                commitment_hours_found.append(hour)

            if terminated or truncated:
                break

        # All commitment hours should be 11
        for h in commitment_hours_found:
            assert h == 11, \
                f"Commitment should only occur at hour 11, found at hour {h}"

    def test_tomorrows_commitments_populated_at_commitment_hour(self, env):
        """Test tomorrows_commitments populated at commitment hour."""
        env.reset(seed=42)

        # Before commitment hour, tomorrows should be zeros
        assert np.all(env.tomorrows_commitments == 0), \
            "tomorrows_commitments should start as zeros"

        # Advance to commitment hour
        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.6  # 60% commitment
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if 'new_commitment' in info:
                # After commitment, tomorrows should be populated
                assert not np.all(env.tomorrows_commitments == 0), \
                    "tomorrows_commitments should be populated after commitment hour"
                break

            if terminated or truncated:
                break

    def test_commitment_values_in_expected_range(self, env):
        """Verify commitment values in expected range."""
        env.reset(seed=42)

        for i in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[:24] = 0.5  # 50% commitment
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            if 'new_commitment' in info:
                commitment = info['new_commitment']

                # Commitments should be non-negative
                assert np.all(commitment >= 0), \
                    f"Commitments should be >= 0: {commitment}"

                # Commitments should not exceed max possible
                # max = forecast + battery_power = up to ~25 MW
                max_possible = env.plant_capacity_mw + env.battery_power_mw
                assert np.all(commitment <= max_possible * 1.1), \
                    f"Commitments too high: max {commitment.max()} vs limit {max_possible}"
                break

            if terminated or truncated:
                break


class TestEpisodePerformance:
    """Task 6: Test episode performance (AC: #2, NFR2)."""

    # Fixture 'env' is provided by conftest.py

    def test_single_episode_within_5_seconds(self, env):
        """Test single episode (48 steps) completes within 5 seconds."""
        env.reset(seed=42)

        start = time.time()
        for i in range(48):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        elapsed = time.time() - start

        assert elapsed < 5.0, \
            f"Single episode took {elapsed:.2f}s, should be < 5s (NFR2)"

    def test_10_episodes_within_50_seconds(self, env):
        """Test 10 episodes complete within 50 seconds."""
        start = time.time()

        for episode in range(10):
            env.reset(seed=episode)
            for step in range(48):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

        elapsed = time.time() - start

        assert elapsed < 50.0, \
            f"10 episodes took {elapsed:.2f}s, should be < 50s"

    def test_no_memory_leak_multiple_episodes(self, env):
        """Verify no memory leaks across multiple episodes."""
        import sys

        # Get initial reference count for data
        initial_ref = sys.getrefcount(env.data)

        # Run many episodes
        for episode in range(100):
            env.reset(seed=episode)
            for step in range(48):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break

        # Reference count should not grow significantly
        final_ref = sys.getrefcount(env.data)
        ref_growth = final_ref - initial_ref

        assert ref_growth < 10, \
            f"Reference count grew by {ref_growth}, possible memory leak"

    def test_reset_step_cycle_consistent(self, env):
        """Test reset/step cycle is consistent."""
        # Multiple reset-step cycles should produce consistent behavior
        results = []

        for _ in range(5):
            env.reset(seed=42)  # Same seed each time

            episode_data = {
                'start_idx': env.episode_start_idx,
                'initial_soc': env.battery_soc,
                'first_obs': None,
                'steps': 0
            }

            obs, info = env.reset(seed=42)
            episode_data['first_obs'] = obs.copy()

            for step in range(48):
                action = np.zeros(25, dtype=np.float32)
                action[24] = 0.5
                obs, reward, terminated, truncated, info = env.step(action)
                episode_data['steps'] += 1
                if terminated or truncated:
                    break

            results.append(episode_data)

        # All results should be identical
        for i in range(1, len(results)):
            assert results[i]['start_idx'] == results[0]['start_idx'], \
                f"Start idx mismatch: {results[i]['start_idx']} vs {results[0]['start_idx']}"
            assert results[i]['initial_soc'] == results[0]['initial_soc'], \
                f"Initial SOC mismatch"
            assert np.allclose(results[i]['first_obs'], results[0]['first_obs']), \
                f"First observation mismatch"
            assert results[i]['steps'] == results[0]['steps'], \
                f"Step count mismatch: {results[i]['steps']} vs {results[0]['steps']}"


class TestEdgeCases:
    """Additional edge case tests for episode flow."""

    # Fixture 'env' is provided by conftest.py

    def test_reset_near_data_start(self, env):
        """Test reset behavior when starting near data beginning."""
        env.reset(seed=0)

        # Should still work correctly
        obs, info = env.reset(seed=0)
        assert obs.shape == (84,), "Observation should be valid"
        assert env.episode_start_idx >= 0, "Start should be non-negative"

    def test_multiple_resets_in_sequence(self, env):
        """Test multiple resets in sequence work correctly."""
        for i in range(10):
            obs, info = env.reset(seed=i)

            # Verify each reset produces valid state
            assert obs.shape == (84,), f"Reset {i}: Invalid observation shape"
            assert not np.any(np.isnan(obs)), f"Reset {i}: NaN in observation"
            assert env.battery_soc == 0.5 * env.battery_capacity_mwh, \
                f"Reset {i}: SOC not reset correctly"
            assert np.all(env.todays_commitments == 0), \
                f"Reset {i}: Commitments not cleared"

    def test_episode_start_idx_matches_current_idx_on_reset(self, env):
        """Test that episode_start_idx matches current_idx after reset."""
        for seed in range(10):
            env.reset(seed=seed)

            assert env.episode_start_idx == env.current_idx, \
                f"Seed {seed}: episode_start_idx {env.episode_start_idx} != current_idx {env.current_idx}"

    def test_episode_hours_calculation(self, env):
        """Test episode hours calculation is correct."""
        env.reset(seed=42)

        for step in range(48):
            action = np.zeros(25, dtype=np.float32)
            action[24] = 0.5
            obs, reward, terminated, truncated, info = env.step(action)

            # Verify episode_hours calculation
            episode_hours = env.current_idx - env.episode_start_idx
            assert episode_hours == step + 1, \
                f"Step {step}: episode_hours {episode_hours} != expected {step + 1}"

            if terminated:
                assert episode_hours == 48, \
                    f"Terminated at {episode_hours} hours, should be 48"
                break
