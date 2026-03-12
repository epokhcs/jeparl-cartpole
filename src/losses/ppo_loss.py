"""PPO (Proximal Policy Optimization) loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PPOActorLoss(nn.Module):
    """PPO actor loss with clipped surrogate objective."""

    def __init__(
        self,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01
    ):
        """Initialize PPO actor loss.

        Args:
            clip_epsilon: PPO clipping parameter (typically 0.2)
            entropy_coef: Entropy bonus coefficient
        """
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        entropy: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """Compute PPO actor loss.

        Args:
            log_probs: Current log probabilities (B,)
            old_log_probs: Old log probabilities (B,)
            advantages: Advantage estimates (B,)
            entropy: Policy entropy (B,) - optional

        Returns:
            loss: Actor loss (scalar)
            info: Dictionary with loss components
        """
        # Normalize advantages (common practice)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages

        # Take minimum (pessimistic bound)
        policy_loss = -torch.min(surr1, surr2).mean()

        # Add entropy bonus (encourage exploration)
        if entropy is not None:
            entropy_loss = -self.entropy_coef * entropy.mean()
        else:
            entropy_loss = torch.tensor(0.0, device=log_probs.device)

        # Total actor loss
        total_loss = policy_loss + entropy_loss

        # Compute approximate KL divergence for monitoring
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean()
            clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()

        info = {
            'policy_loss': policy_loss.item(),
            'entropy_loss': entropy_loss.item() if isinstance(entropy_loss, torch.Tensor) else entropy_loss,
            'entropy': entropy.mean().item() if entropy is not None else 0.0,
            'approx_kl': approx_kl.item(),
            'clip_fraction': clip_fraction.item(),
            'ratio_mean': ratio.mean().item(),
            'ratio_std': ratio.std().item()
        }

        return total_loss, info


class PPOCriticLoss(nn.Module):
    """PPO critic (value function) loss."""

    def __init__(
        self,
        clip_value: bool = False,
        clip_epsilon: float = 0.2
    ):
        """Initialize PPO critic loss.

        Args:
            clip_value: Whether to clip value function updates
            clip_epsilon: Clipping parameter if clip_value is True
        """
        super().__init__()
        self.clip_value = clip_value
        self.clip_epsilon = clip_epsilon

    def forward(
        self,
        values: torch.Tensor,
        old_values: Optional[torch.Tensor],
        returns: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """Compute PPO critic loss.

        Args:
            values: Current value estimates (B, 1) or (B,)
            old_values: Old value estimates (B, 1) or (B,) - optional for clipping
            returns: Computed returns (B,)

        Returns:
            loss: Critic loss (scalar)
            info: Dictionary with loss components
        """
        # Flatten if needed
        if values.dim() > 1:
            values = values.squeeze(-1)
        if old_values is not None and old_values.dim() > 1:
            old_values = old_values.squeeze(-1)

        if self.clip_value and old_values is not None:
            # Clipped value loss (helps with stability)
            value_pred_clipped = old_values + torch.clamp(
                values - old_values,
                -self.clip_epsilon,
                self.clip_epsilon
            )
            value_loss1 = F.mse_loss(values, returns, reduction='none')
            value_loss2 = F.mse_loss(value_pred_clipped, returns, reduction='none')
            value_loss = torch.max(value_loss1, value_loss2).mean()
        else:
            # Standard MSE loss
            value_loss = F.mse_loss(values, returns)

        # Compute explained variance for monitoring
        with torch.no_grad():
            y_pred = values
            y_true = returns
            var_y = torch.var(y_true)
            explained_var = 1 - torch.var(y_true - y_pred) / (var_y + 1e-8)

        info = {
            'value_loss': value_loss.item(),
            'explained_variance': explained_var.item(),
            'value_mean': values.mean().item(),
            'value_std': values.std().item(),
            'return_mean': returns.mean().item(),
            'return_std': returns.std().item()
        }

        return value_loss, info


def ppo_actor_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float = 0.2
) -> torch.Tensor:
    """Functional interface for PPO actor loss (without entropy).

    Args:
        log_probs: Current log probabilities (B,)
        old_log_probs: Old log probabilities (B,)
        advantages: Advantage estimates (B,)
        clip_epsilon: Clipping parameter

    Returns:
        Actor loss (scalar)
    """
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Probability ratio
    ratio = torch.exp(log_probs - old_log_probs)

    # Clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

    loss = -torch.min(surr1, surr2).mean()

    return loss


def ppo_critic_loss(
    values: torch.Tensor,
    returns: torch.Tensor
) -> torch.Tensor:
    """Functional interface for PPO critic loss.

    Args:
        values: Value estimates (B, 1) or (B,)
        returns: Computed returns (B,)

    Returns:
        Critic loss (scalar)
    """
    if values.dim() > 1:
        values = values.squeeze(-1)

    loss = F.mse_loss(values, returns)

    return loss


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: Rewards (T,)
        values: Value estimates (T,)
        dones: Done flags (T,)
        next_value: Value of next state (scalar)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter

    Returns:
        advantages: Advantage estimates (T,)
        returns: Computed returns (T,)
    """
    T = len(rewards)
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0

    # Append next value to values
    values_extended = torch.cat([values, next_value.unsqueeze(0)])

    # Compute GAE in reverse
    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_value_t = next_value
        else:
            next_non_terminal = 1.0 - dones[t]
            next_value_t = values_extended[t + 1]

        # TD error
        delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]

        # GAE
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    # Returns are advantages + values
    returns = advantages + values

    return advantages, returns
