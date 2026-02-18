"""
Alternating Training Script for Hierarchical RL Agents
======================================================

Trains commitment and battery agents in alternating iterations,
where each agent learns against the current version of its partner.

This addresses the distribution shift problem where agents trained
independently with mismatched partner policies perform poorly together.

Training Flow:
    Iteration 0 (Bootstrap):
        - Train Battery with random commitments
        - Save and copy to best/

    Iteration 1+:
        - Train Commitment using trained Battery
        - Train Battery using trained Commitment
        - Evaluate joint performance
        - Log metrics

Usage:
    python train_alternating.py --iterations 5
    python train_alternating.py --resume-iteration 2
    python train_alternating.py --iterations 3 --timestep-decay 0.7
"""

import argparse
import json
import random
import shutil
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from environment.battery_env import BatteryEnv
from environment.commitment_env import CommitmentEnv
from environment.solar_plant import PlantConfig

# Paths
BASE_PATH = Path(__file__).parent.parent.parent
DATA_PATH = BASE_PATH / 'data' / 'processed'
MODEL_PATH = BASE_PATH / 'models'
OUTPUT_PATH = BASE_PATH / 'outputs' / 'alternating'

# Default plant configuration
DEFAULT_PLANT_CONFIG = PlantConfig(
    plant_capacity_mw=20.0,
    battery_capacity_mwh=10.0,
    battery_power_mw=5.0,
    battery_efficiency=0.92,
    battery_degradation_cost=0.01,
)

# Network architectures (matching existing training scripts)
BATTERY_NET_ARCH = [128, 128]
COMMITMENT_NET_ARCH = [256, 256]


@dataclass
class AlternatingConfig:
    """Configuration for alternating training."""
    # Iteration settings
    num_iterations: int = 5

    # Timesteps per iteration
    battery_timesteps_bootstrap: int = 200_000  # Iteration 0
    battery_timesteps: int = 150_000            # Iteration 1+
    commitment_timesteps: int = 100_000

    # Timestep decay (later iterations need fewer steps)
    timestep_decay_rate: float = 0.8
    min_timesteps: int = 50_000

    # Warm-start settings
    warm_start: bool = True  # Resume from previous iteration's model

    # Training hyperparameters
    learning_rate: float = 3e-4
    batch_size_battery: int = 256
    batch_size_commitment: int = 64
    buffer_size: int = 50_000
    gamma_battery: float = 0.99
    gamma_commitment: float = 0.0  # Single-step episodes
    tau: float = 0.005

    # Evaluation settings
    eval_episodes: int = 20
    eval_days_per_episode: int = 7
    eval_freq_battery: int = 5_000
    eval_freq_commitment: int = 2_000

    # Resume support
    resume_from_iteration: Optional[int] = None

    # Seed
    base_seed: int = 42


def set_all_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_timesteps_for_iteration(
    base_timesteps: int,
    iteration: int,
    decay_rate: float,
    min_timesteps: int
) -> int:
    """Calculate timesteps for given iteration with decay.

    Later iterations need fewer steps as we're fine-tuning, not learning from scratch.
    """
    steps = int(base_timesteps * (decay_rate ** iteration))
    return max(steps, min_timesteps)


def copy_model_to_best(src_path: Path, agent_type: str) -> None:
    """Copy trained model to 'best' directory for environment loading.

    Environments load from models/{agent}_agent/best/best_model.zip
    """
    if agent_type not in ('battery', 'commitment'):
        raise ValueError(f"agent_type must be 'battery' or 'commitment', got {agent_type}")

    dest_dir = MODEL_PATH / f'{agent_type}_agent' / 'best'
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / 'best_model.zip'
    shutil.copy2(src_path, dest_path)
    print(f"  Copied model to {dest_path}")


def get_iteration_model_dir(agent_type: str, iteration: int) -> Path:
    """Get the model directory for a specific iteration."""
    return MODEL_PATH / f'{agent_type}_agent' / f'iter_{iteration}'


class AlternatingTrainer:
    """Orchestrates alternating training of hierarchical agents."""

    def __init__(self, config: AlternatingConfig):
        self.config = config
        self.metrics_history: List[Dict] = []
        self.plant_config = DEFAULT_PLANT_CONFIG

        # Ensure output directories exist
        self.output_path = OUTPUT_PATH
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.tensorboard_path = self.output_path / 'tensorboard'
        self.tensorboard_path.mkdir(parents=True, exist_ok=True)

        # Load data
        self.train_path = DATA_PATH / 'train.csv'
        self.test_path = DATA_PATH / 'test.csv'

        if not self.train_path.exists():
            raise FileNotFoundError(f"Training data not found at {self.train_path}")

        self._load_or_create_progress()

    def _load_or_create_progress(self) -> None:
        """Load existing progress or create new tracking file."""
        self.progress_path = self.output_path / 'training_progress.csv'
        self.checkpoint_path = self.output_path / 'checkpoint.json'

        if self.progress_path.exists():
            self.metrics_history = pd.read_csv(self.progress_path).to_dict('records')
        else:
            self.metrics_history = []

    def _save_progress(self, iteration: int, metrics: Dict) -> None:
        """Save training progress to CSV."""
        self.metrics_history.append(metrics)
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.progress_path, index=False)

        # Save checkpoint
        checkpoint = {
            'last_completed_iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def run(self) -> None:
        """Run full alternating training loop."""
        start_iter = self.config.resume_from_iteration or 0

        print("\n" + "=" * 70)
        print("ALTERNATING TRAINING FOR HIERARCHICAL RL AGENTS")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(f"  Iterations: {self.config.num_iterations}")
        print(f"  Battery timesteps (bootstrap): {self.config.battery_timesteps_bootstrap:,}")
        print(f"  Battery timesteps (iter 1+): {self.config.battery_timesteps:,}")
        print(f"  Commitment timesteps: {self.config.commitment_timesteps:,}")
        print(f"  Timestep decay rate: {self.config.timestep_decay_rate}")
        print(f"  Warm-start: {self.config.warm_start}")
        print(f"  Starting from iteration: {start_iter}")
        print(f"\nOutput: {self.output_path}")
        print(f"TensorBoard: tensorboard --logdir {self.tensorboard_path}")
        print("=" * 70 + "\n")

        total_start_time = time.time()

        for iteration in range(start_iter, self.config.num_iterations):
            iter_start_time = time.time()
            print(f"\n{'#' * 70}")
            print(f"# ITERATION {iteration}")
            print(f"{'#' * 70}\n")

            # Set seed for this iteration
            set_all_seeds(self.config.base_seed + iteration * 1000)

            try:
                self._run_iteration(iteration)
            except KeyboardInterrupt:
                print(f"\n\nTraining interrupted at iteration {iteration}")
                break
            except Exception as e:
                print(f"\nError in iteration {iteration}: {e}")
                raise

            iter_time = time.time() - iter_start_time
            print(f"\nIteration {iteration} completed in {iter_time/3600:.2f} hours")

        total_time = time.time() - total_start_time
        print(f"\n{'=' * 70}")
        print(f"TRAINING COMPLETE")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Progress saved to: {self.progress_path}")
        print(f"{'=' * 70}\n")

    def _run_iteration(self, iteration: int) -> None:
        """Run single training iteration."""
        if iteration == 0:
            # Bootstrap: train battery with random commitments
            self._train_battery_bootstrap()
        else:
            # Alternating: commitment then battery
            self._train_commitment(iteration)
            self._train_battery(iteration)

        # Evaluate joint performance
        metrics = self._evaluate_joint(iteration)

        # Log and save
        self._save_progress(iteration, metrics)
        self._print_iteration_summary(iteration, metrics)

    def _train_battery_bootstrap(self) -> None:
        """Train battery agent with random commitments (iteration 0)."""
        print("\n" + "-" * 60)
        print("Training Battery Agent (Bootstrap - Random Commitments)")
        print("-" * 60)

        timesteps = self.config.battery_timesteps_bootstrap
        iter_dir = get_iteration_model_dir('battery', 0)
        iter_dir.mkdir(parents=True, exist_ok=True)

        # Create environment with random commitments
        train_df = pd.read_csv(self.train_path, parse_dates=['datetime'])
        train_env = Monitor(BatteryEnv(
            train_df,
            plant_config=self.plant_config,
            commitment_policy='random'
        ))

        # Create eval environment
        eval_env = None
        eval_callback = None
        if self.test_path.exists():
            test_df = pd.read_csv(self.test_path, parse_dates=['datetime'])
            eval_env = Monitor(BatteryEnv(
                test_df,
                plant_config=self.plant_config,
                commitment_policy='random'
            ))
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(iter_dir),
                log_path=str(self.output_path / 'eval_logs' / 'battery_iter_0'),
                eval_freq=self.config.eval_freq_battery,
                n_eval_episodes=10,
                deterministic=True,
            )

        # Create model
        policy_kwargs = dict(
            net_arch=BATTERY_NET_ARCH,
            activation_fn=torch.nn.ReLU,
        )

        model = SAC(
            'MlpPolicy',
            train_env,
            learning_rate=self.config.learning_rate,
            batch_size=self.config.batch_size_battery,
            buffer_size=self.config.buffer_size,
            gamma=self.config.gamma_battery,
            tau=self.config.tau,
            seed=self.config.base_seed,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=str(self.tensorboard_path / 'battery'),
        )

        print(f"\nTraining for {timesteps:,} timesteps...")
        callbacks = [eval_callback] if eval_callback else []

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
        )

        # Save final model
        final_path = iter_dir / 'final_model.zip'
        model.save(str(final_path))
        print(f"  Saved final model to {final_path}")

        # Copy best model to best/ for environment loading
        best_path = iter_dir / 'best_model.zip'
        if best_path.exists():
            copy_model_to_best(best_path, 'battery')
        else:
            copy_model_to_best(final_path, 'battery')

        train_env.close()
        if eval_env:
            eval_env.close()

    def _train_commitment(self, iteration: int) -> None:
        """Train commitment agent with current battery agent."""
        print("\n" + "-" * 60)
        print(f"Training Commitment Agent (Iteration {iteration})")
        print("-" * 60)

        # Calculate timesteps with decay
        base_timesteps = self.config.commitment_timesteps
        timesteps = get_timesteps_for_iteration(
            base_timesteps, iteration - 1,  # iteration-1 because commitment doesn't train at iter 0
            self.config.timestep_decay_rate,
            self.config.min_timesteps
        )

        iter_dir = get_iteration_model_dir('commitment', iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Using trained battery agent from best/")
        print(f"  Timesteps: {timesteps:,}")

        # Create environment with trained battery
        train_df = pd.read_csv(self.train_path, parse_dates=['datetime'])
        train_env = Monitor(CommitmentEnv(
            train_df,
            plant_config=self.plant_config,
            battery_policy='trained'
        ))

        # Create eval environment
        eval_env = None
        eval_callback = None
        if self.test_path.exists():
            test_df = pd.read_csv(self.test_path, parse_dates=['datetime'])
            eval_env = Monitor(CommitmentEnv(
                test_df,
                plant_config=self.plant_config,
                battery_policy='trained'
            ))
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(iter_dir),
                log_path=str(self.output_path / 'eval_logs' / f'commitment_iter_{iteration}'),
                eval_freq=self.config.eval_freq_commitment,
                n_eval_episodes=20,
                deterministic=True,
            )

        # Create or load model
        policy_kwargs = dict(
            net_arch=COMMITMENT_NET_ARCH,
            activation_fn=torch.nn.ReLU,
        )

        # Warm-start from previous iteration if available
        prev_model_path = None
        if self.config.warm_start and iteration > 1:
            prev_iter_dir = get_iteration_model_dir('commitment', iteration - 1)
            prev_best = prev_iter_dir / 'best_model.zip'
            prev_final = prev_iter_dir / 'final_model.zip'
            if prev_best.exists():
                prev_model_path = prev_best
            elif prev_final.exists():
                prev_model_path = prev_final

        if prev_model_path:
            print(f"  Warm-starting from {prev_model_path}")
            model = SAC.load(str(prev_model_path), env=train_env,
                           tensorboard_log=str(self.tensorboard_path / 'commitment'))
        else:
            model = SAC(
                'MlpPolicy',
                train_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size_commitment,
                buffer_size=self.config.buffer_size,
                gamma=self.config.gamma_commitment,
                tau=self.config.tau,
                seed=self.config.base_seed + iteration,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(self.tensorboard_path / 'commitment'),
            )

        print(f"\nTraining for {timesteps:,} timesteps...")
        callbacks = [eval_callback] if eval_callback else []

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not (prev_model_path is not None),
        )

        # Save final model
        final_path = iter_dir / 'final_model.zip'
        model.save(str(final_path))
        print(f"  Saved final model to {final_path}")

        # Copy to best/
        best_path = iter_dir / 'best_model.zip'
        if best_path.exists():
            copy_model_to_best(best_path, 'commitment')
        else:
            copy_model_to_best(final_path, 'commitment')

        train_env.close()
        if eval_env:
            eval_env.close()

    def _train_battery(self, iteration: int) -> None:
        """Train battery agent with current commitment agent."""
        print("\n" + "-" * 60)
        print(f"Training Battery Agent (Iteration {iteration})")
        print("-" * 60)

        # Calculate timesteps with decay
        base_timesteps = self.config.battery_timesteps
        timesteps = get_timesteps_for_iteration(
            base_timesteps, iteration - 1,
            self.config.timestep_decay_rate,
            self.config.min_timesteps
        )

        iter_dir = get_iteration_model_dir('battery', iteration)
        iter_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Using trained commitment agent from best/")
        print(f"  Timesteps: {timesteps:,}")

        # Create environment with trained commitment
        train_df = pd.read_csv(self.train_path, parse_dates=['datetime'])
        train_env = Monitor(BatteryEnv(
            train_df,
            plant_config=self.plant_config,
            commitment_policy='trained'
        ))

        # Create eval environment
        eval_env = None
        eval_callback = None
        if self.test_path.exists():
            test_df = pd.read_csv(self.test_path, parse_dates=['datetime'])
            eval_env = Monitor(BatteryEnv(
                test_df,
                plant_config=self.plant_config,
                commitment_policy='trained'
            ))
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(iter_dir),
                log_path=str(self.output_path / 'eval_logs' / f'battery_iter_{iteration}'),
                eval_freq=self.config.eval_freq_battery,
                n_eval_episodes=10,
                deterministic=True,
            )

        # Create or load model
        policy_kwargs = dict(
            net_arch=BATTERY_NET_ARCH,
            activation_fn=torch.nn.ReLU,
        )

        # Warm-start from previous iteration
        prev_model_path = None
        if self.config.warm_start:
            prev_iter_dir = get_iteration_model_dir('battery', iteration - 1)
            prev_best = prev_iter_dir / 'best_model.zip'
            prev_final = prev_iter_dir / 'final_model.zip'
            if prev_best.exists():
                prev_model_path = prev_best
            elif prev_final.exists():
                prev_model_path = prev_final

        if prev_model_path:
            print(f"  Warm-starting from {prev_model_path}")
            model = SAC.load(str(prev_model_path), env=train_env,
                           tensorboard_log=str(self.tensorboard_path / 'battery'))
        else:
            model = SAC(
                'MlpPolicy',
                train_env,
                learning_rate=self.config.learning_rate,
                batch_size=self.config.batch_size_battery,
                buffer_size=self.config.buffer_size,
                gamma=self.config.gamma_battery,
                tau=self.config.tau,
                seed=self.config.base_seed + iteration * 100,
                policy_kwargs=policy_kwargs,
                verbose=1,
                tensorboard_log=str(self.tensorboard_path / 'battery'),
            )

        print(f"\nTraining for {timesteps:,} timesteps...")
        callbacks = [eval_callback] if eval_callback else []

        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not (prev_model_path is not None),
        )

        # Save final model
        final_path = iter_dir / 'final_model.zip'
        model.save(str(final_path))
        print(f"  Saved final model to {final_path}")

        # Copy to best/
        best_path = iter_dir / 'best_model.zip'
        if best_path.exists():
            copy_model_to_best(best_path, 'battery')
        else:
            copy_model_to_best(final_path, 'battery')

        train_env.close()
        if eval_env:
            eval_env.close()

    def _evaluate_joint(self, iteration: int) -> Dict:
        """Evaluate joint performance of both trained agents."""
        print("\n" + "-" * 60)
        print(f"Evaluating Joint Performance (Iteration {iteration})")
        print("-" * 60)

        try:
            from environment.hierarchical_orchestrator import HierarchicalOrchestrator

            battery_model_path = MODEL_PATH / 'battery_agent' / 'best' / 'best_model.zip'
            commitment_model_path = MODEL_PATH / 'commitment_agent' / 'best' / 'best_model.zip'

            # Check if both models exist (commitment may not exist at iter 0)
            if not commitment_model_path.exists():
                print("  Commitment model not found, skipping joint evaluation")
                return self._create_empty_metrics(iteration)

            if not battery_model_path.exists():
                print("  Battery model not found, skipping joint evaluation")
                return self._create_empty_metrics(iteration)

            orchestrator = HierarchicalOrchestrator.from_trained_agents(
                data_path=str(self.test_path),
                battery_model_path=str(battery_model_path),
                commitment_model_path=str(commitment_model_path),
            )

            print(f"  Running {self.config.eval_episodes} episodes "
                  f"({self.config.eval_days_per_episode} days each)...")

            metrics = orchestrator.evaluate(
                num_episodes=self.config.eval_episodes,
                days_per_episode=self.config.eval_days_per_episode,
                seed=self.config.base_seed + iteration,
            )

            return {
                'iteration': iteration,
                'timestamp': datetime.now().isoformat(),
                'mean_daily_profit': metrics.get('mean_daily_profit', 0),
                'mean_episode_profit': metrics.get('mean_episode_profit', 0),
                'std_episode_profit': metrics.get('std_episode_profit', 0),
                'mean_revenue': metrics.get('mean_revenue', 0),
                'mean_imbalance_cost': metrics.get('mean_imbalance_cost', 0),
            }

        except Exception as e:
            print(f"  Warning: Joint evaluation failed: {e}")
            return self._create_empty_metrics(iteration)

    def _create_empty_metrics(self, iteration: int) -> Dict:
        """Create empty metrics dict when evaluation isn't possible."""
        return {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),
            'mean_daily_profit': None,
            'mean_episode_profit': None,
            'std_episode_profit': None,
            'mean_revenue': None,
            'mean_imbalance_cost': None,
        }

    def _print_iteration_summary(self, iteration: int, metrics: Dict) -> None:
        """Print summary of iteration results."""
        print("\n" + "=" * 60)
        print(f"ITERATION {iteration} SUMMARY")
        print("=" * 60)

        if metrics.get('mean_daily_profit') is not None:
            print(f"  Mean daily profit: {metrics['mean_daily_profit']:.2f} EUR")
            print(f"  Mean episode profit: {metrics['mean_episode_profit']:.2f} EUR")
            print(f"  Std episode profit: {metrics['std_episode_profit']:.2f} EUR")
            print(f"  Mean imbalance cost: {metrics['mean_imbalance_cost']:.2f} EUR")
        else:
            print("  Joint evaluation not available (missing models)")

        # Show improvement over previous iteration
        if len(self.metrics_history) >= 2:
            prev = self.metrics_history[-2]
            curr = metrics
            if prev.get('mean_daily_profit') and curr.get('mean_daily_profit'):
                improvement = curr['mean_daily_profit'] - prev['mean_daily_profit']
                pct = (improvement / abs(prev['mean_daily_profit'])) * 100
                print(f"\n  Improvement: {improvement:+.2f} EUR/day ({pct:+.1f}%)")

        print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Alternating training for hierarchical RL agents',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Iteration settings
    parser.add_argument('--iterations', type=int, default=5,
                        help='Number of training iterations')
    parser.add_argument('--resume-iteration', type=int, default=None,
                        help='Resume from specific iteration')

    # Timestep settings
    parser.add_argument('--battery-timesteps-bootstrap', type=int, default=200_000,
                        help='Timesteps for battery bootstrap (iter 0)')
    parser.add_argument('--battery-timesteps', type=int, default=150_000,
                        help='Timesteps for battery training (iter 1+)')
    parser.add_argument('--commitment-timesteps', type=int, default=100_000,
                        help='Timesteps for commitment training')
    parser.add_argument('--timestep-decay', type=float, default=0.8,
                        help='Decay rate for timesteps per iteration')
    parser.add_argument('--min-timesteps', type=int, default=50_000,
                        help='Minimum timesteps per training phase')

    # Warm-start
    parser.add_argument('--no-warm-start', action='store_true',
                        help='Disable warm-starting from previous iteration')

    # Evaluation
    parser.add_argument('--eval-episodes', type=int, default=20,
                        help='Episodes for joint evaluation')
    parser.add_argument('--eval-days', type=int, default=7,
                        help='Days per evaluation episode')

    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed')

    args = parser.parse_args()

    # Build configuration
    config = AlternatingConfig(
        num_iterations=args.iterations,
        battery_timesteps_bootstrap=args.battery_timesteps_bootstrap,
        battery_timesteps=args.battery_timesteps,
        commitment_timesteps=args.commitment_timesteps,
        timestep_decay_rate=args.timestep_decay,
        min_timesteps=args.min_timesteps,
        warm_start=not args.no_warm_start,
        eval_episodes=args.eval_episodes,
        eval_days_per_episode=args.eval_days,
        resume_from_iteration=args.resume_iteration,
        base_seed=args.seed,
    )

    # Run training
    trainer = AlternatingTrainer(config)
    trainer.run()


if __name__ == '__main__':
    main()
