"""Combined loss function with configuration-dependent composition.

This is the critical component that implements the 4 experimental configurations
from the JEPA for RL paper.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .jepa_loss import jepa_loss
from .ppo_loss import ppo_actor_loss, ppo_critic_loss
from .regularization_loss import variance_regularization_loss


class CombinedLoss(nn.Module):
    """Combined loss for JEPA-RL with configuration-dependent composition.

    Four configurations (as per paper):
    1. (J_hat, ∇, R_hat): No JEPA, only RL gradients (baseline)
    2. (J, ∇, R_hat): JEPA + RL gradients (best performing)
    3. (J, ∇_hat, R_hat): JEPA without RL gradients (demonstrates collapse)
    4. (J, ∇_hat, R): JEPA + regularization, no RL gradients
    """

    def __init__(
        self,
        config_type: int = 2,
        jepa_weight: float = 1.0,
        actor_weight: float = 1.0,
        critic_weight: float = 0.5,
        reg_weight: float = 0.1,
        entropy_coef: float = 0.01,
        clip_epsilon: float = 0.2,
        d_emb: int = 64
    ):
        """Initialize combined loss.

        Args:
            config_type: Configuration type (1, 2, 3, or 4)
            jepa_weight: Weight for JEPA loss
            actor_weight: Weight for actor loss
            critic_weight: Weight for critic loss
            reg_weight: Weight for regularization loss
            entropy_coef: Entropy coefficient
            clip_epsilon: PPO clipping parameter
            d_emb: Embedding dimension (for regularization)
        """
        super().__init__()

        assert config_type in [1, 2, 3, 4], f"Invalid config_type: {config_type}"

        self.config_type = config_type
        self.jepa_weight = jepa_weight
        self.actor_weight = actor_weight
        self.critic_weight = critic_weight
        self.reg_weight = reg_weight
        self.entropy_coef = entropy_coef
        self.clip_epsilon = clip_epsilon
        self.d_emb = d_emb

        # Configuration flags
        self.use_jepa = config_type in [2, 3, 4]
        self.use_rl_gradients = config_type in [1, 2]
        self.use_regularization = config_type == 4

    def forward(
        self,
        # JEPA components
        predicted_embeddings: Optional[torch.Tensor] = None,
        target_embeddings: Optional[torch.Tensor] = None,
        context_embeddings: Optional[torch.Tensor] = None,
        # PPO components
        log_probs: Optional[torch.Tensor] = None,
        old_log_probs: Optional[torch.Tensor] = None,
        advantages: Optional[torch.Tensor] = None,
        entropy: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        returns: Optional[torch.Tensor] = None,
        # For gradient control
        detach_embeddings_for_rl: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined loss based on configuration.

        Args:
            predicted_embeddings: Predicted embeddings from predictor (B, num_frames, d_emb)
            target_embeddings: Target embeddings from Y-encoder (B, num_frames, d_emb)
            context_embeddings: Context embeddings from X-encoder (B, d_emb)
            log_probs: Current action log probabilities (B,)
            old_log_probs: Old action log probabilities (B,)
            advantages: Advantage estimates (B,)
            entropy: Policy entropy (B,)
            values: Value estimates (B,) or (B, 1)
            returns: Computed returns (B,)
            detach_embeddings_for_rl: Whether to detach embeddings before RL losses

        Returns:
            total_loss: Combined loss (scalar)
            loss_dict: Dictionary with individual loss components
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self._get_device())

        # ===== JEPA Loss =====
        if self.use_jepa and predicted_embeddings is not None and target_embeddings is not None:
            jepa_loss_value = jepa_loss(predicted_embeddings, target_embeddings)
            total_loss = total_loss + self.jepa_weight * jepa_loss_value
            loss_dict['jepa_loss'] = jepa_loss_value.item()
        else:
            loss_dict['jepa_loss'] = 0.0

        # ===== RL Losses =====
        # Configuration 3 and 4: Stop gradients before RL losses
        if not self.use_rl_gradients and detach_embeddings_for_rl:
            # Note: This flag should be set based on configuration
            # The caller should handle detaching context_embeddings if needed
            pass

        # Actor Loss
        if log_probs is not None and old_log_probs is not None and advantages is not None:
            # Normalize advantages
            advantages_normalized = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Compute actor loss
            actor_loss_value = ppo_actor_loss(
                log_probs,
                old_log_probs,
                advantages_normalized,
                clip_epsilon=self.clip_epsilon
            )

            # Add entropy bonus
            if entropy is not None:
                entropy_loss = -self.entropy_coef * entropy.mean()
                actor_loss_value = actor_loss_value + entropy_loss
                loss_dict['entropy'] = entropy.mean().item()
                loss_dict['entropy_loss'] = entropy_loss.item()

            # Apply weight
            weighted_actor_loss = self.actor_weight * actor_loss_value

            # For Config 3 & 4: detach actor loss so no gradients to encoder
            if not self.use_rl_gradients:
                weighted_actor_loss = weighted_actor_loss.detach()

            total_loss = total_loss + weighted_actor_loss
            loss_dict['actor_loss'] = actor_loss_value.item()
        else:
            loss_dict['actor_loss'] = 0.0

        # Critic Loss
        if values is not None and returns is not None:
            critic_loss_value = ppo_critic_loss(values, returns)

            # Apply weight
            weighted_critic_loss = self.critic_weight * critic_loss_value

            # For Config 3 & 4: detach critic loss so no gradients to encoder
            if not self.use_rl_gradients:
                weighted_critic_loss = weighted_critic_loss.detach()

            total_loss = total_loss + weighted_critic_loss
            loss_dict['critic_loss'] = critic_loss_value.item()
        else:
            loss_dict['critic_loss'] = 0.0

        # ===== Regularization Loss =====
        if self.use_regularization and context_embeddings is not None:
            reg_loss_value = variance_regularization_loss(
                context_embeddings,
                d_emb=self.d_emb
            )
            total_loss = total_loss + self.reg_weight * reg_loss_value
            loss_dict['reg_loss'] = reg_loss_value.item()

            # Compute variance for monitoring
            with torch.no_grad():
                var_per_dim = torch.var(context_embeddings, dim=0, unbiased=False)
                loss_dict['avg_variance'] = var_per_dim.mean().item()
        else:
            loss_dict['reg_loss'] = 0.0

            # Still monitor variance even if not using regularization
            if context_embeddings is not None:
                with torch.no_grad():
                    var_per_dim = torch.var(context_embeddings, dim=0, unbiased=False)
                    loss_dict['avg_variance'] = var_per_dim.mean().item()

        # Add total loss to dict
        loss_dict['total_loss'] = total_loss.item()
        loss_dict['config_type'] = self.config_type

        return total_loss, loss_dict

    def _get_device(self) -> torch.device:
        """Get device from parameters (if any)."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')

    def get_config_description(self) -> str:
        """Get human-readable description of configuration.

        Returns:
            Configuration description string
        """
        descriptions = {
            1: "Config 1: Baseline (No JEPA, only RL gradients)",
            2: "Config 2: JEPA + RL gradients (best performing)",
            3: "Config 3: JEPA without RL gradients (demonstrates collapse)",
            4: "Config 4: JEPA + regularization, no RL gradients"
        }
        return descriptions[self.config_type]


def compute_combined_loss(
    config_type: int,
    predicted_embeddings: Optional[torch.Tensor] = None,
    target_embeddings: Optional[torch.Tensor] = None,
    context_embeddings: Optional[torch.Tensor] = None,
    log_probs: Optional[torch.Tensor] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    advantages: Optional[torch.Tensor] = None,
    entropy: Optional[torch.Tensor] = None,
    values: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    weights: Optional[Dict[str, float]] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Functional interface for combined loss.

    Args:
        config_type: Configuration type (1-4)
        predicted_embeddings: Predicted embeddings
        target_embeddings: Target embeddings
        context_embeddings: Context embeddings
        log_probs: Current log probs
        old_log_probs: Old log probs
        advantages: Advantages
        entropy: Entropy
        values: Values
        returns: Returns
        weights: Optional weight overrides

    Returns:
        total_loss: Combined loss
        loss_dict: Loss components
    """
    if weights is None:
        weights = {}

    loss_fn = CombinedLoss(
        config_type=config_type,
        jepa_weight=weights.get('jepa_weight', 1.0),
        actor_weight=weights.get('actor_weight', 1.0),
        critic_weight=weights.get('critic_weight', 0.5),
        reg_weight=weights.get('reg_weight', 0.1),
        entropy_coef=weights.get('entropy_coef', 0.01),
        clip_epsilon=weights.get('clip_epsilon', 0.2),
        d_emb=weights.get('d_emb', 64)
    )

    return loss_fn(
        predicted_embeddings=predicted_embeddings,
        target_embeddings=target_embeddings,
        context_embeddings=context_embeddings,
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        entropy=entropy,
        values=values,
        returns=returns
    )
