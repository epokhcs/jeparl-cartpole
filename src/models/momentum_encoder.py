"""Momentum encoder for JEPA target encoder."""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Optional


class MomentumEncoder:
    """Wrapper for target encoder with momentum updates.

    The target encoder (Y-encoder) is updated as an exponential moving average
    of the context encoder (X-encoder). No gradients flow through the target encoder.

    Update rule: θ_target = momentum * θ_target + (1 - momentum) * θ_context
    """

    def __init__(
        self,
        encoder: nn.Module,
        momentum: float = 0.99
    ):
        """Initialize momentum encoder.

        Args:
            encoder: Context encoder (X-encoder) to track
            momentum: Momentum coefficient (default: 0.99 as per paper)
        """
        self.encoder = encoder
        self.momentum = momentum

        # Create target encoder as a deep copy
        self.target_encoder = deepcopy(encoder)

        # Disable gradients for target encoder
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        # Set target encoder to eval mode
        self.target_encoder.eval()

    @torch.no_grad()
    def update(self):
        """Update target encoder parameters using momentum.

        Called after each optimization step on the context encoder.
        Uses exponential moving average: θ_bar = m * θ_bar + (1 - m) * θ
        """
        for param_encoder, param_target in zip(
            self.encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_target.data.mul_(self.momentum).add_(
                param_encoder.data, alpha=1 - self.momentum
            )

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Forward pass through target encoder (no gradients).

        Args:
            *args: Arguments to pass to encoder
            **kwargs: Keyword arguments to pass to encoder

        Returns:
            Target encoder output
        """
        with torch.no_grad():
            return self.target_encoder(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        """Make the object callable."""
        return self.forward(*args, **kwargs)

    def state_dict(self) -> dict:
        """Get state dict of target encoder.

        Returns:
            State dictionary
        """
        return {
            'target_encoder': self.target_encoder.state_dict(),
            'momentum': self.momentum
        }

    def load_state_dict(self, state_dict: dict):
        """Load state dict into target encoder.

        Args:
            state_dict: State dictionary to load
        """
        self.target_encoder.load_state_dict(state_dict['target_encoder'])
        self.momentum = state_dict.get('momentum', 0.99)

    def reset_target_encoder(self):
        """Reset target encoder to match context encoder.

        Useful for initialization or debugging.
        """
        self.target_encoder = deepcopy(self.encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        self.target_encoder.eval()

    def get_momentum(self) -> float:
        """Get current momentum value.

        Returns:
            Momentum coefficient
        """
        return self.momentum

    def set_momentum(self, momentum: float):
        """Set momentum value.

        Args:
            momentum: New momentum coefficient (should be in [0, 1])
        """
        assert 0 <= momentum <= 1, "Momentum must be in [0, 1]"
        self.momentum = momentum

    def to(self, device: torch.device):
        """Move target encoder to device.

        Args:
            device: Target device
        """
        self.target_encoder = self.target_encoder.to(device)
        return self

    def train(self, mode: bool = True):
        """Set training mode (target encoder always stays in eval mode).

        Args:
            mode: Training mode flag (ignored for target encoder)
        """
        # Target encoder always stays in eval mode
        self.target_encoder.eval()

    def eval(self):
        """Set evaluation mode (target encoder always in eval mode)."""
        self.target_encoder.eval()


class MomentumScheduler:
    """Optional scheduler for momentum coefficient.

    Can be used to gradually increase momentum during training.
    """

    def __init__(
        self,
        momentum_encoder: MomentumEncoder,
        initial_momentum: float = 0.99,
        final_momentum: float = 1.0,
        total_steps: int = 100000
    ):
        """Initialize momentum scheduler.

        Args:
            momentum_encoder: MomentumEncoder instance
            initial_momentum: Starting momentum value
            final_momentum: Final momentum value
            total_steps: Total training steps
        """
        self.momentum_encoder = momentum_encoder
        self.initial_momentum = initial_momentum
        self.final_momentum = final_momentum
        self.total_steps = total_steps
        self.current_step = 0

        # Set initial momentum
        self.momentum_encoder.set_momentum(initial_momentum)

    def step(self):
        """Update momentum for current step."""
        self.current_step += 1

        if self.current_step >= self.total_steps:
            momentum = self.final_momentum
        else:
            # Linear interpolation
            progress = self.current_step / self.total_steps
            momentum = self.initial_momentum + (
                self.final_momentum - self.initial_momentum
            ) * progress

        self.momentum_encoder.set_momentum(momentum)

    def get_momentum(self) -> float:
        """Get current momentum value.

        Returns:
            Current momentum
        """
        return self.momentum_encoder.get_momentum()
