"""
Training Script for Commitment Agent (Hierarchical RL)
======================================================

Trains the high-level commitment agent using SAC.
The agent learns to make optimal day-ahead commitment decisions.

Episode: Single commitment decision + simulated 24h execution
Observation: 56 dimensions (forecasts, prices, SOC, weather, time)
Action: 24 dimensions (commitment fractions for each hour)
Reward: End-of-day profit (revenue - imbalance - degradation)
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

from environment.commitment_env import CommitmentEnv
from environment.solar_plant import PlantConfig

# Configuration
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs' / 'commitment_agent'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'commitment_agent'
TENSORBOARD_LOG_DIR = OUTPUT_PATH / 'tensorboard'
BATTERY_MODEL_PATH = Path(__file__).parent.parent.parent / 'models' / 'battery_agent' / 'best' / 'best_model.zip'

# Training hyperparameters
# Single-step episodes = needs more episodes to learn
TOTAL_TIMESTEPS = 100_000  # 100k commitment decisions
LEARNING_RATE = 3e-4
BATCH_SIZE = 64  # Smaller batch for single-step episodes
BUFFER_SIZE = 50_000
GAMMA = 0.0  # No discounting - single step episodes
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
SEED = 42

# Training loop configuration
CHECKPOINT_FREQ = 10_000
EVAL_FREQ = 2_000
N_EVAL_EPISODES = 20  # More episodes for single-step env

# Network architecture
NET_ARCH = [256, 256]
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


def create_env(
    data_path: Path,
    config: PlantConfig,
    battery_policy: str = 'heuristic'
) -> CommitmentEnv:
    """Create a CommitmentEnv from a CSV data file."""
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    return CommitmentEnv(df, plant_config=config, battery_policy=battery_policy)


def evaluate_agent(model, env, n_episodes: int = 20) -> dict:
    """Evaluate agent performance over multiple episodes.

    Returns:
        Dict with mean/std of key metrics
    """
    episode_rewards = []
    episode_profits = []
    episode_imbalance_costs = []
    episode_revenues = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_rewards.append(reward)
        episode_profits.append(info.get('total_profit', reward))
        episode_imbalance_costs.append(info.get('total_imbalance_cost', 0))
        episode_revenues.append(info.get('total_revenue', 0))

    return {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_profit': np.mean(episode_profits),
        'mean_imbalance_cost': np.mean(episode_imbalance_costs),
        'mean_revenue': np.mean(episode_revenues),
    }


def main() -> None:
    """Train SAC agent for commitment decisions."""
    parser = argparse.ArgumentParser(description='Train Commitment Agent')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--timesteps', type=int, default=TOTAL_TIMESTEPS,
                        help='Total training timesteps')
    parser.add_argument('--battery-policy', type=str, default='heuristic',
                        choices=['heuristic', 'trained'],
                        help='Battery policy for simulation')
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

    # Check for trained battery agent
    battery_policy = args.battery_policy
    if battery_policy == 'trained' and not BATTERY_MODEL_PATH.exists():
        print(f"Trained battery agent not found at {BATTERY_MODEL_PATH}")
        print("Falling back to heuristic policy.")
        print("Run train_battery.py first to train the battery agent.")
        battery_policy = 'heuristic'

    print(f"Creating training environment (battery policy: {battery_policy})...")
    train_env = Monitor(create_env(train_path, PLANT_CONFIG, battery_policy))

    # Create eval environment if test data exists
    eval_env = None
    eval_callback = None
    if test_path.exists():
        print("Creating evaluation environment...")
        eval_env = Monitor(create_env(test_path, PLANT_CONFIG, battery_policy))
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
        name_prefix='commitment_agent'
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
        print("COMMITMENT AGENT TRAINING")
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
        print(f"  Battery policy: {battery_policy}")
        print(f"  Episode length: 1 step (single commitment)")
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
    final_model_path = MODEL_PATH / 'commitment_agent_final.zip'
    model.save(str(final_model_path))
    print(f"\nModel saved to {final_model_path}")

    # Final evaluation
    if eval_env:
        print("\nFinal evaluation on test set...")
        metrics = evaluate_agent(model, eval_env, n_episodes=50)
        print(f"  Mean reward: {metrics['mean_reward']:.2f} +/- {metrics['std_reward']:.2f}")
        print(f"  Mean profit: {metrics['mean_profit']:.2f} EUR/day")
        print(f"  Mean revenue: {metrics['mean_revenue']:.2f} EUR/day")
        print(f"  Mean imbalance cost: {metrics['mean_imbalance_cost']:.2f} EUR/day")

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
