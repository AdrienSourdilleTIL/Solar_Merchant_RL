"""
Training Script for Battery Agent (Hierarchical RL)
====================================================

Trains the low-level battery management agent using SAC.
The agent learns to optimally charge/discharge the battery
to minimize imbalance costs given fixed commitment schedules.

Episode: 24 hours of hourly battery decisions
Observation: 21 dimensions (SOC, commitment, PV, lookahead, etc.)
Action: 1 dimension (charge/discharge)
Reward: Immediate (revenue - imbalance - degradation)
"""

import argparse
import random
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from environment.battery_env import BatteryEnv
from environment.solar_plant import PlantConfig

# Configuration
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs' / 'battery_agent'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'battery_agent'
TENSORBOARD_LOG_DIR = OUTPUT_PATH / 'tensorboard'

# Training hyperparameters
# Shorter episodes (24 steps) allow faster learning
TOTAL_TIMESTEPS = 200_000  # ~8,300 episodes
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 50_000  # Smaller buffer ok for shorter episodes
GAMMA = 0.99  # Lower gamma for shorter horizon (24h vs 48h)
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
SEED = 42

# Training loop configuration
CHECKPOINT_FREQ = 25_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10  # More episodes since they're shorter

# Network architecture (simpler for 1-dim action)
NET_ARCH = [128, 128]
ACTIVATION_FN = torch.nn.ReLU

# Plant configuration
PLANT_CONFIG = PlantConfig(
    plant_capacity_mw=20.0,
    battery_capacity_mwh=10.0,
    battery_power_mw=5.0,
    battery_efficiency=0.92,
    battery_degradation_cost=0.01,
)


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_env(data_path: Path, config: PlantConfig, commitment_policy: str = 'random') -> BatteryEnv:
    """Create a BatteryEnv from a CSV data file.

    Args:
        data_path: Path to the CSV data file
        config: Plant configuration
        commitment_policy: How to generate commitments ('random' or 'trained')
    """
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    return BatteryEnv(df, plant_config=config, commitment_policy=commitment_policy)


def evaluate_agent(model, env, n_episodes: int = 10) -> dict:
    """Evaluate agent performance over multiple episodes.

    Returns:
        Dict with mean/std of key metrics
    """
    episode_rewards = []
    episode_profits = []
    episode_imbalance_costs = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        episode_rewards.append(episode_reward)
        if 'episode_profit' in info:
            episode_profits.append(info['episode_profit'])
            episode_imbalance_costs.append(info['episode_imbalance_cost'])

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_profit': np.mean(episode_profits) if episode_profits else 0,
        'mean_imbalance_cost': np.mean(episode_imbalance_costs) if episode_imbalance_costs else 0,
    }


def main() -> None:
    """Train SAC agent for battery management."""
    parser = argparse.ArgumentParser(description='Train Battery Agent')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help='Total training timesteps')
    parser.add_argument('--commitment-policy', type=str, default='trained',
                        choices=['random', 'trained'],
                        help='Commitment policy: random (legacy) or trained (use commitment agent)')
    args = parser.parse_args()

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    set_all_seeds(SEED)

    # Ensure output directories exist
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    TENSORBOARD_LOG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.mkdir(parents=True, exist_ok=True)
    (MODEL_PATH / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (MODEL_PATH / 'best').mkdir(parents=True, exist_ok=True)

    # Load data
    train_path = DATA_PATH / 'train.csv'
    test_path = DATA_PATH / 'test.csv'

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        print("Run prepare_dataset.py first.")
        return

    print(f"Creating training environment (commitment_policy={args.commitment_policy})...")
    train_env = Monitor(create_env(train_path, PLANT_CONFIG, args.commitment_policy))

    # Create eval environment if test data exists
    eval_env = None
    eval_callback = None
    if test_path.exists():
        print("Creating evaluation environment...")
        eval_env = Monitor(create_env(test_path, PLANT_CONFIG, args.commitment_policy))
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(MODEL_PATH / 'best'),
            log_path=str(OUTPUT_PATH / 'eval_logs'),
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False
        )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(MODEL_PATH / 'checkpoints'),
        name_prefix='battery_agent'
    )

    callbacks = [checkpoint_callback]
    if eval_callback:
        callbacks.append(eval_callback)

    # Create or resume agent
    total_timesteps = args.timesteps
    is_resuming = False

    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        print(f"\nResuming training from: {checkpoint_path}")
        model = SAC.load(str(checkpoint_path), env=train_env,
                         tensorboard_log=str(TENSORBOARD_LOG_DIR))
        print(f"  Resuming from timestep: {model.num_timesteps:,}")
        is_resuming = True
    else:
        policy_kwargs = dict(
            net_arch=NET_ARCH,
            activation_fn=ACTIVATION_FN,
        )

        print("\n" + "=" * 60)
        print("BATTERY AGENT TRAINING")
        print("=" * 60)
        print(f"\nHyperparameters:")
        print(f"  Learning rate: {LEARNING_RATE}")
        print(f"  Batch size: {BATCH_SIZE}")
        print(f"  Buffer size: {BUFFER_SIZE:,}")
        print(f"  Gamma: {GAMMA}")
        print(f"  Net arch: {NET_ARCH}")
        print(f"  Total timesteps: {total_timesteps:,}")
        print(f"\nEnvironment:")
        print(f"  Observation space: {train_env.observation_space.shape}")
        print(f"  Action space: {train_env.action_space.shape}")
        print(f"  Episode length: 24 steps")
        print(f"  Commitment policy: {args.commitment_policy}")
        print(f"\nOutput:")
        print(f"  Models: {MODEL_PATH}")
        print(f"  TensorBoard: {TENSORBOARD_LOG_DIR}")

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
            seed=SEED,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(TENSORBOARD_LOG_DIR)
        )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not is_resuming,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\nTraining time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Timesteps completed: {model.num_timesteps:,}")

    # Save final model
    final_model_path = MODEL_PATH / 'battery_agent_final.zip'
    model.save(str(final_model_path))
    print(f"\nModel saved to {final_model_path}")

    # Final evaluation
    if eval_env:
        print("\nFinal evaluation on test set...")
        metrics = evaluate_agent(model, eval_env, n_episodes=20)
        print(f"  Mean episode reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
        print(f"  Mean episode profit: {metrics['mean_profit']:.2f} EUR")
        print(f"  Mean imbalance cost: {metrics['mean_imbalance_cost']:.2f} EUR")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
