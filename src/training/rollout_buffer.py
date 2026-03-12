"""Rollout buffer for collecting and storing experience."""

import torch
import numpy as np
from typing import Optional, Tuple, Dict


class RolloutBuffer:
    """Buffer for storing rollout data with frame sequences for JEPA.

    Stores transitions along with context and target frame sequences needed
    for JEPA training.
    """

    def __init__(
        self,
        buffer_size: int,
        observation_shape: Tuple[int, ...],
        num_envs: int = 1,
        device: str = "cpu"
    ):
        """Initialize rollout buffer.

        Args:
            buffer_size: Number of steps to store
            observation_shape: Shape of observations (e.g., (3, 3, 400, 608))
            num_envs: Number of parallel environments
            device: Device to store tensors on
        """
        self.buffer_size = buffer_size
        self.observation_shape = observation_shape
        self.num_envs = num_envs
        self.device = device
        self.pos = 0
        self.full = False

        # Storage tensors
        self.observations = torch.zeros(
            (buffer_size, num_envs, *observation_shape),
            dtype=torch.float32,
            device=device
        )
        self.actions = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.long,
            device=device
        )
        self.rewards = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.dones = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.values = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )
        self.log_probs = torch.zeros(
            (buffer_size, num_envs),
            dtype=torch.float32,
            device=device
        )

        # GAE computation results
        self.advantages = None
        self.returns = None

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """Add a transition to the buffer.

        Args:
            observation: Observation (num_envs, *obs_shape)
            action: Action (num_envs,)
            reward: Reward (num_envs,)
            done: Done flag (num_envs,)
            value: Value estimate (num_envs,)
            log_prob: Action log probability (num_envs,)
        """
        self.observations[self.pos] = observation
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.log_probs[self.pos] = log_prob

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """Compute returns and advantages using GAE.

        Args:
            last_values: Value estimates for last states (num_envs,)
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        # Get actual buffer size (might not be full)
        buffer_size = self.buffer_size if self.full else self.pos

        advantages = torch.zeros_like(self.rewards[:buffer_size])
        last_gae_lam = torch.zeros(self.num_envs, device=self.device)

        # Compute GAE in reverse
        for step in reversed(range(buffer_size)):
            if step == buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_values = self.values[step + 1]

            # TD error
            delta = (
                self.rewards[step]
                + gamma * next_values * next_non_terminal
                - self.values[step]
            )

            # GAE
            advantages[step] = last_gae_lam = (
                delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
            )

        # Returns are advantages + values
        self.advantages = advantages
        self.returns = advantages + self.values[:buffer_size]

    def get(
        self,
        batch_size: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get all data from the buffer.

        Args:
            batch_size: Optional batch size for splitting data

        Returns:
            Dictionary with all buffer data
        """
        buffer_size = self.buffer_size if self.full else self.pos

        # Flatten batch and num_envs dimensions
        observations = self.observations[:buffer_size].reshape(
            -1, *self.observation_shape
        )
        actions = self.actions[:buffer_size].reshape(-1)
        values = self.values[:buffer_size].reshape(-1)
        log_probs = self.log_probs[:buffer_size].reshape(-1)
        advantages = self.advantages.reshape(-1)
        returns = self.returns.reshape(-1)

        data = {
            'observations': observations,
            'actions': actions,
            'values': values,
            'log_probs': log_probs,
            'advantages': advantages,
            'returns': returns
        }

        return data

    def get_batches(
        self,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Get data in batches (generator).

        Args:
            batch_size: Size of each batch

        Yields:
            Dictionary with batch data
        """
        data = self.get()
        total_size = data['observations'].shape[0]

        # Generate random indices
        indices = torch.randperm(total_size, device=self.device)

        # Yield batches
        for start_idx in range(0, total_size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            yield {
                key: value[batch_indices] for key, value in data.items()
            }

    def reset(self):
        """Reset the buffer."""
        self.pos = 0
        self.full = False
        self.advantages = None
        self.returns = None

    def size(self) -> int:
        """Get current buffer size.

        Returns:
            Current number of samples in buffer
        """
        return self.buffer_size if self.full else self.pos


class FrameSequenceBuffer:
    """Buffer that maintains frame sequences for JEPA training.

    This is an extension of RolloutBuffer that properly handles
    the frame sequences needed for context and target encoders.
    """

    def __init__(
        self,
        buffer_size: int,
        frame_shape: Tuple[int, int, int],  # (C, H, W)
        num_frames: int = 3,
        num_envs: int = 1,
        device: str = "cpu"
    ):
        """Initialize frame sequence buffer.

        Args:
            buffer_size: Number of steps to store
            frame_shape: Shape of single frame (C, H, W)
            num_frames: Number of frames in stack (default: 3)
            num_envs: Number of parallel environments
            device: Device for tensors
        """
        self.buffer_size = buffer_size
        self.frame_shape = frame_shape
        self.num_frames = num_frames
        self.num_envs = num_envs
        self.device = device

        # We need to store num_frames + 1 extra frames to get target frames
        # For context: frames[t-2:t+1] = {t-2, t-1, t}
        # For target: frames[t-1:t+2] = {t-1, t, t+1}
        self.frames = torch.zeros(
            (buffer_size + 1, num_envs, *frame_shape),
            dtype=torch.float32,
            device=device
        )

        # Use standard rollout buffer for other data
        observation_shape = (num_frames, *frame_shape)
        self.rollout_buffer = RolloutBuffer(
            buffer_size=buffer_size,
            observation_shape=observation_shape,
            num_envs=num_envs,
            device=device
        )

    def add(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        value: torch.Tensor,
        log_prob: torch.Tensor
    ):
        """Add transition to buffer.

        Args:
            observation: Stacked frames observation (num_envs, num_frames, C, H, W)
            action: Action (num_envs,)
            reward: Reward (num_envs,)
            done: Done flag (num_envs,)
            value: Value estimate (num_envs,)
            log_prob: Log probability (num_envs,)
        """
        # Store the latest frame from the observation
        # observation shape: (num_envs, num_frames, C, H, W)
        # We want the most recent frame: observation[:, -1]
        current_pos = self.rollout_buffer.pos
        self.frames[current_pos] = observation[:, -1]

        # Add to rollout buffer
        self.rollout_buffer.add(
            observation, action, reward, done, value, log_prob
        )

        # If this is the last position, store the next frame for target
        if current_pos == self.buffer_size - 1:
            # This will be used as t+1 for the last transition
            # In practice, this gets set when we call add() with the next observation
            pass

    def compute_returns_and_advantages(
        self,
        last_values: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        """Compute returns and advantages."""
        self.rollout_buffer.compute_returns_and_advantages(
            last_values, gamma, gae_lambda
        )

    def get_context_and_target_frames(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get context and target frame sequences.

        Returns:
            context_frames: Frames for context encoder (N, num_frames, C, H, W)
            target_frames: Frames for target encoder (N, num_frames, C, H, W)
        """
        buffer_size = self.rollout_buffer.size()

        # For each transition i, we need:
        # - Context: frames[i:i+num_frames] (the observation already stored)
        # - Target: frames[i+1:i+num_frames+1]

        # Context frames are just the stored observations
        context_frames = self.rollout_buffer.observations[:buffer_size].reshape(
            -1, self.num_frames, *self.frame_shape
        )

        # For target frames, we need to shift by one
        # This is tricky - we need the future frame for each transition
        # Use the individual frames we stored
        target_frame_list = []
        for step in range(buffer_size):
            # Target frames for step i: {frame[step+1], frame[step+2], frame[step+3]}
            # Which corresponds to temporal indices {t-1, t, t+1} when context is {t-2, t-1, t}
            target = self.frames[step + 1:step + 1 + self.num_frames]
            target_frame_list.append(target)

        target_frames = torch.stack(target_frame_list, dim=0).reshape(
            -1, self.num_frames, *self.frame_shape
        )

        return context_frames, target_frames

    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data including frame sequences."""
        data = self.rollout_buffer.get()

        # Add context and target frames
        context_frames, target_frames = self.get_context_and_target_frames()
        data['context_frames'] = context_frames
        data['target_frames'] = target_frames

        return data

    def get_batches(self, batch_size: int):
        """Get data in batches."""
        data = self.get()
        total_size = data['observations'].shape[0]

        indices = torch.randperm(total_size, device=self.device)

        for start_idx in range(0, total_size, batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]

            yield {
                key: value[batch_indices] for key, value in data.items()
            }

    def reset(self):
        """Reset buffer."""
        self.rollout_buffer.reset()
        self.frames.zero_()

    def size(self) -> int:
        """Get current buffer size."""
        return self.rollout_buffer.size()
