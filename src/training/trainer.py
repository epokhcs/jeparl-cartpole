"""Main trainer for JEPA-RL.

This is the central training loop that orchestrates:
- Experience collection
- Loss computation
- Model updates
- Monitoring and logging
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
from pathlib import Path

from ..models.vit_encoder import ViTEncoder
from ..models.momentum_encoder import MomentumEncoder
from ..models.predictor import Predictor
from ..models.actor_critic import ActorCritic
from ..losses.combined_loss import CombinedLoss
from ..environment.preprocessing import extract_patches
from .rollout_buffer import RolloutBuffer
from .embedding_monitor import EmbeddingMonitor
from ..utils.logger import Logger
from ..utils.checkpoint import CheckpointManager
from ..utils.metrics import MetricsCollector


class JEPATrainer:
    """Main trainer for JEPA-RL with PPO.

    Handles the complete training loop including:
    - Rollout collection from environment
    - JEPA and PPO loss computation
    - Model updates with configuration-dependent gradient flow
    - Momentum encoder updates
    - Collapse monitoring
    - Logging and checkpointing
    """

    def __init__(
        self,
        env,
        x_encoder: ViTEncoder,
        y_encoder_momentum: MomentumEncoder,
        predictor: Predictor,
        actor_critic: ActorCritic,
        combined_loss: CombinedLoss,
        optimizer: torch.optim.Optimizer,
        config: Dict,
        logger: Optional[Logger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None,
        device: str = "cpu"
    ):
        """Initialize trainer.

        Args:
            env: Training environment
            x_encoder: Context encoder (X-encoder)
            y_encoder_momentum: Target encoder with momentum (Y-encoder)
            predictor: Predictor network
            actor_critic: Actor-critic network
            combined_loss: Combined loss function
            optimizer: Optimizer
            config: Configuration dictionary
            logger: Logger for metrics
            checkpoint_manager: Checkpoint manager
            device: Device for training
        """
        self.env = env
        self.x_encoder = x_encoder
        self.y_encoder_momentum = y_encoder_momentum
        self.predictor = predictor
        self.actor_critic = actor_critic
        self.combined_loss = combined_loss
        self.optimizer = optimizer
        self.config = config
        self.logger = logger
        self.checkpoint_manager = checkpoint_manager
        self.device = device

        # Training config
        self.total_steps = config.get('training', {}).get('total_steps', 100000)
        self.rollout_steps = config.get('training', {}).get('rollout_steps', 2048)
        self.ppo_epochs = config.get('training', {}).get('ppo_epochs', 4)
        self.mini_batch_size = config.get('training', {}).get('mini_batch_size', 64)
        self.max_grad_norm = config.get('training', {}).get('max_grad_norm', 0.5)
        self.gamma = config.get('training', {}).get('gamma', 0.99)
        self.gae_lambda = config.get('training', {}).get('gae_lambda', 0.95)

        # Patch extraction config
        self.patch_size = config.get('model', {}).get('patch_size', 16)

        # Logging config
        self.log_interval = config.get('logging', {}).get('log_interval', 1000)
        self.save_interval = config.get('logging', {}).get('save_interval', 10000)

        # Collapse monitoring
        self.embedding_monitor = EmbeddingMonitor(
            d_emb=config.get('model', {}).get('d_emb', 64),
            collapse_threshold=config.get('collapse', {}).get('variance_threshold', 0.01)
        )

        # Metrics
        self.metrics = MetricsCollector()

        # Training state
        self.global_step = 0
        self.episode_count = 0

        # Rollout buffer
        obs_shape = env.observation_space.shape
        self.rollout_buffer = RolloutBuffer(
            buffer_size=self.rollout_steps,
            observation_shape=obs_shape,
            num_envs=1,
            device=device
        )

    def train(self):
        """Main training loop."""
        obs, _ = self.env.reset()
        obs = torch.from_numpy(obs).float().to(self.device)

        episode_reward = 0.0
        episode_length = 0

        print(f"Starting training for {self.total_steps} steps...")
        print(f"Configuration: {self.combined_loss.get_config_description()}")

        while self.global_step < self.total_steps:
            # ===== Collect Rollout =====
            for step in range(self.rollout_steps):
                with torch.no_grad():
                    # Extract patches and positions
                    patches, positions = extract_patches(
                        obs.unsqueeze(0),
                        patch_size=self.patch_size
                    )

                    # Encode observation
                    embeddings = self.x_encoder(patches, positions)  # (1, d_emb)

                    # Get action and value
                    action, log_prob, entropy, value = self.actor_critic.get_action_and_value(
                        embeddings
                    )

                # Take action in environment
                next_obs, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated

                # Store transition
                self.rollout_buffer.add(
                    observation=obs.unsqueeze(0),
                    action=action,
                    reward=torch.tensor([reward], device=self.device),
                    done=torch.tensor([float(done)], device=self.device),
                    value=value.squeeze(),
                    log_prob=log_prob
                )

                # Update state
                obs = torch.from_numpy(next_obs).float().to(self.device)
                episode_reward += reward
                episode_length += 1
                self.global_step += 1

                # Handle episode end
                if done:
                    self.metrics.add_episode(episode_reward, episode_length)
                    self.episode_count += 1

                    if self.logger:
                        self.logger.log_scalar('train/episode_reward', episode_reward, self.global_step)
                        self.logger.log_scalar('train/episode_length', episode_length, self.global_step)

                    episode_reward = 0.0
                    episode_length = 0
                    obs, _ = self.env.reset()
                    obs = torch.from_numpy(obs).float().to(self.device)

            # ===== Compute Returns and Advantages =====
            with torch.no_grad():
                patches, positions = extract_patches(
                    obs.unsqueeze(0),
                    patch_size=self.patch_size
                )
                embeddings = self.x_encoder(patches, positions)
                _, _, _, last_value = self.actor_critic.get_action_and_value(embeddings)

            self.rollout_buffer.compute_returns_and_advantages(
                last_values=last_value.squeeze(),
                gamma=self.gamma,
                gae_lambda=self.gae_lambda
            )

            # ===== Update Policy =====
            update_stats = self._update_policy()

            # ===== Logging =====
            if self.global_step % self.log_interval == 0:
                self._log_metrics(update_stats)

            # ===== Checkpointing =====
            if self.checkpoint_manager and self.global_step % self.save_interval == 0:
                self._save_checkpoint()

            # Reset buffer
            self.rollout_buffer.reset()

        # Final checkpoint
        if self.checkpoint_manager:
            self._save_checkpoint()

        print("Training completed!")

    def _update_policy(self) -> Dict:
        """Update policy using collected rollout.

        Returns:
            Dictionary with update statistics
        """
        # Get all data
        data = self.rollout_buffer.get()

        total_stats = {
            'total_loss': 0.0,
            'jepa_loss': 0.0,
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'reg_loss': 0.0,
            'avg_variance': 0.0
        }
        num_updates = 0

        # Multiple PPO epochs
        for epoch in range(self.ppo_epochs):
            # Iterate over mini-batches
            for batch in self.rollout_buffer.get_batches(self.mini_batch_size):
                # Extract patches for JEPA
                # Context and target frames need to be derived from observations
                obs = batch['observations']  # (B, num_frames, C, H, W)

                # Extract patches
                patches_context, positions_context = extract_patches(
                    obs,
                    patch_size=self.patch_size
                )

                # Encode with X-encoder (context)
                context_emb = self.x_encoder(patches_context, positions_context)  # (B, d_emb)

                # Encode with Y-encoder (target) - for JEPA loss
                # NOTE: In actual implementation, target should use future frames
                # For simplicity here, we use same observations (this should be improved)
                with torch.no_grad():
                    target_emb = self.y_encoder_momentum(patches_context, positions_context)

                # Predict target embeddings
                predicted_emb = self.predictor(
                    context_emb,
                    batch['actions']
                )  # (B, num_target_frames, d_emb)

                # Evaluate actions for PPO
                log_probs, entropy, values = self.actor_critic.evaluate_actions(
                    context_emb,
                    batch['actions']
                )

                # Compute combined loss
                loss, loss_dict = self.combined_loss(
                    predicted_embeddings=predicted_emb,
                    target_embeddings=target_emb.unsqueeze(1),  # Add frame dimension
                    context_embeddings=context_emb,
                    log_probs=log_probs,
                    old_log_probs=batch['log_probs'],
                    advantages=batch['advantages'],
                    entropy=entropy,
                    values=values.squeeze(-1),
                    returns=batch['returns']
                )

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.x_encoder.parameters()) +
                    list(self.predictor.parameters()) +
                    list(self.actor_critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Update momentum encoder
                self.y_encoder_momentum.update()

                # Monitor embeddings
                collapse_stats = self.embedding_monitor.update(
                    context_emb,
                    self.global_step
                )

                # Accumulate stats
                for key in total_stats:
                    if key in loss_dict:
                        total_stats[key] += loss_dict[key]
                    elif key in collapse_stats:
                        total_stats[key] += collapse_stats[key]

                num_updates += 1

        # Average stats
        for key in total_stats:
            total_stats[key] /= max(num_updates, 1)

        return total_stats

    def _log_metrics(self, update_stats: Dict):
        """Log metrics.

        Args:
            update_stats: Update statistics
        """
        if not self.logger:
            return

        # Log losses
        for key, value in update_stats.items():
            self.logger.log_scalar(f'train/{key}', value, self.global_step)

        # Log episode stats
        episode_stats = self.metrics.get_episode_stats(last_n=100)
        for key, value in episode_stats.items():
            self.logger.log_scalar(f'train/{key}', value, self.global_step)

        # Print progress
        self.logger.print(
            f"Step {self.global_step}/{self.total_steps} | "
            f"Episodes: {self.episode_count} | "
            f"Reward: {episode_stats['mean_return']:.2f} ± {episode_stats['std_return']:.2f} | "
            f"Variance: {update_stats['avg_variance']:.4f}"
        )

    def _save_checkpoint(self):
        """Save checkpoint."""
        models = {
            'x_encoder': self.x_encoder,
            'y_encoder_momentum': self.y_encoder_momentum.target_encoder,
            'predictor': self.predictor,
            'actor_critic': self.actor_critic
        }

        metrics = {
            'episode_count': self.episode_count,
            **self.metrics.get_episode_stats()
        }

        path = self.checkpoint_manager.save_checkpoint(
            step=self.global_step,
            models=models,
            optimizer=self.optimizer,
            metrics=metrics
        )

        if self.logger:
            self.logger.print(f"Checkpoint saved: {path}")
