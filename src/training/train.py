"""
Training Script for Solar Merchant RL Agent

Uses Soft Actor-Critic (SAC) algorithm to train an agent for
day-ahead bidding and battery management.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.solar_merchant_env import SolarMerchantEnv

# Configuration
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models'

# Training hyperparameters
TOTAL_TIMESTEPS = 500_000  # ~1.4 passes through 7 years of data
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 100_000
GAMMA = 0.999  # High discount for long-horizon planning
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1

# Plant configuration
PLANT_CONFIG = {
    'plant_capacity_mw': 20.0,
    'battery_capacity_mwh': 10.0,
    'battery_power_mw': 5.0,
    'battery_efficiency': 0.92,
    'battery_degradation_cost': 0.01,
}


def create_env(data_path: Path, **config) -> SolarMerchantEnv:
    """Create environment from data file."""
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    return SolarMerchantEnv(df, **config)


def main():
    # Ensure output directories exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    # Load training data
    train_path = DATA_PATH / 'train.csv'
    test_path = DATA_PATH / 'test.csv'

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        print("Run prepare_dataset.py first.")
        return

    print("Creating training environment...")
    train_env = Monitor(create_env(train_path, **PLANT_CONFIG))

    # Create eval environment if test data exists
    eval_env = None
    eval_callback = None
    if test_path.exists():
        print("Creating evaluation environment...")
        eval_env = Monitor(create_env(test_path, **PLANT_CONFIG))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_PATH / 'best'),
            log_path=str(OUTPUT_PATH / 'eval_logs'),
            eval_freq=10_000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=50_000,
        save_path=str(MODEL_PATH / 'checkpoints'),
        name_prefix='solar_merchant'
    )

    # Combine callbacks
    callbacks = [checkpoint_callback]
    if eval_callback:
        callbacks.append(eval_callback)

    # Create SAC agent
    print("\nInitializing SAC agent...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Buffer size: {BUFFER_SIZE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")

    model = SAC(
        'MlpPolicy',
        train_env,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        gamma=GAMMA,
        tau=TAU,
        train_freq=TRAIN_FREQ,
        gradient_steps=GRADIENT_STEPS,
        verbose=1,
        tensorboard_log=str(OUTPUT_PATH / 'tensorboard')
    )

    # Train
    print("\nStarting training...")
    print("="*60)

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    final_model_path = MODEL_PATH / 'solar_merchant_final.zip'
    model.save(str(final_model_path))
    print(f"\nModel saved to {final_model_path}")

    # Quick evaluation
    print("\nRunning quick evaluation on training data...")
    eval_episodes = 3
    total_rewards = []

    for ep in range(eval_episodes):
        obs, _ = train_env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 24 * 7  # 1 week

        while steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = train_env.step(action)
            episode_reward += reward
            steps += 1
            if terminated or truncated:
                break

        total_rewards.append(episode_reward)
        print(f"  Episode {ep+1}: reward = {episode_reward:.2f}")

    print(f"\nMean episode reward: {np.mean(total_rewards):.2f}")
    print(f"Std episode reward: {np.std(total_rewards):.2f}")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
