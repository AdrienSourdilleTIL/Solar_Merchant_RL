"""
Training Script for Solar Merchant RL Agent

Uses Soft Actor-Critic (SAC) algorithm to train an agent for
day-ahead bidding and battery management.
"""

import random
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from environment.solar_merchant_env import SolarMerchantEnv

# Configuration
DATA_PATH = Path(__file__).parent.parent.parent / 'data' / 'processed'
OUTPUT_PATH = Path(__file__).parent.parent.parent / 'outputs'
MODEL_PATH = Path(__file__).parent.parent.parent / 'models'
TENSORBOARD_LOG_DIR = OUTPUT_PATH / 'tensorboard'

# Training hyperparameters
TOTAL_TIMESTEPS = 500_000  # ~1.4 passes through 7 years of data
LEARNING_RATE = 3e-4
BATCH_SIZE = 256
BUFFER_SIZE = 100_000
GAMMA = 0.999  # High discount for long-horizon planning
TAU = 0.005
TRAIN_FREQ = 1
GRADIENT_STEPS = 1
SEED = 42

# Training loop configuration
CHECKPOINT_FREQ = 50_000   # Save checkpoint every N steps
EVAL_FREQ = 10_000         # Evaluate every N steps
N_EVAL_EPISODES = 5        # Episodes per evaluation

# Network architecture
NET_ARCH = [256, 256]
ACTIVATION_FN = torch.nn.ReLU

# Plant configuration
PLANT_CONFIG = {
    'plant_capacity_mw': 20.0,
    'battery_capacity_mwh': 10.0,
    'battery_power_mw': 5.0,
    'battery_efficiency': 0.92,
    'battery_degradation_cost': 0.01,
}


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility across all libraries.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_env(data_path: Path, **config) -> SolarMerchantEnv:
    """Create a SolarMerchantEnv from a CSV data file.

    Args:
        data_path: Path to the processed CSV data file with datetime column.
        **config: Plant configuration keyword arguments passed to SolarMerchantEnv
            (e.g. plant_capacity_mw, battery_capacity_mwh).

    Returns:
        Configured SolarMerchantEnv instance ready for training or evaluation.
    """
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    return SolarMerchantEnv(df, **config)


def main() -> None:
    """Train a SAC agent on the solar merchant environment.

    Sets random seeds for reproducibility, creates training and evaluation
    environments, configures the SAC agent with hyperparameters and network
    architecture, then runs the training loop with checkpoint and eval callbacks.
    """
    # SB3 imports deferred to main() so module-level constants and set_all_seeds
    # are importable in test environments without stable_baselines3 installed.
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

    # Load training data
    train_path = DATA_PATH / 'train.csv'
    test_path = DATA_PATH / 'test.csv'

    if not train_path.exists():
        print(f"Training data not found at {train_path}")
        print("Run prepare_dataset.py first.")
        return

    # No VecNormalize — observations are normalized internally by SolarMerchantEnv
    # (see solar_merchant_env.py norm_factors). Adding VecNormalize would double-normalize.
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
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
            deterministic=True,
            render=False
        )

    # Checkpoint callback
    # No VecNormalize stats to save — observations normalized internally by
    # SolarMerchantEnv (see architecture.md, internal normalization decision)
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(MODEL_PATH / 'checkpoints'),
        name_prefix='solar_merchant'
    )

    # Combine callbacks
    callbacks = [checkpoint_callback]
    if eval_callback:
        callbacks.append(eval_callback)

    # Create SAC agent
    policy_kwargs = dict(
        net_arch=NET_ARCH,
        activation_fn=ACTIVATION_FN,
    )

    print("\nInitializing SAC agent...")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Buffer size: {BUFFER_SIZE}")
    print(f"  Gamma: {GAMMA}")
    print(f"  Seed: {SEED}")
    print(f"  Net arch: {NET_ARCH}")
    print(f"  Activation: {ACTIVATION_FN.__name__}")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Checkpoint freq: {CHECKPOINT_FREQ:,}")
    print(f"  Eval freq: {EVAL_FREQ:,}")
    print(f"  Eval episodes: {N_EVAL_EPISODES}")
    print(f"  TensorBoard log: {TENSORBOARD_LOG_DIR}")

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
        # SB3 automatically logs to TensorBoard: rollout/ep_rew_mean,
        # rollout/ep_len_mean, train/actor_loss, train/critic_loss,
        # train/ent_coef, train/ent_coef_loss, train/learning_rate
        tensorboard_log=str(TENSORBOARD_LOG_DIR)
    )

    # Train
    print("\nStarting training...")
    print(f"Launch TensorBoard: tensorboard --logdir {TENSORBOARD_LOG_DIR}")
    print("="*60)

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    elapsed = time.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Timesteps completed: {model.num_timesteps:,} / {TOTAL_TIMESTEPS:,}")

    # Save final model
    final_model_path = MODEL_PATH / 'solar_merchant_final.zip'
    model.save(str(final_model_path))
    print(f"\nModel saved to {final_model_path}")

    print("\nTraining complete!")
    print("Run evaluate_baselines.py or evaluate.py for proper evaluation.")


if __name__ == '__main__':
    main()
