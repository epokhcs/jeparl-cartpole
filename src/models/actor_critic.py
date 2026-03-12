"""Actor-Critic networks for PPO-style reinforcement learning."""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import torch.distributions as distributions


class ActorCritic(nn.Module):
    """Actor-Critic network built on top of encoder embeddings.

    The actor outputs action logits (policy), and the critic outputs value estimates.
    """

    def __init__(
        self,
        d_emb: int = 64,
        num_actions: int = 2,
        hidden_dim: Optional[int] = None
    ):
        """Initialize actor-critic network.

        Args:
            d_emb: Embedding dimension from encoder
            num_actions: Number of discrete actions (2 for CartPole)
            hidden_dim: Optional hidden layer dimension (if None, uses direct projection)
        """
        super().__init__()

        self.d_emb = d_emb
        self.num_actions = num_actions

        if hidden_dim is not None:
            # Actor with hidden layer
            self.actor = nn.Sequential(
                nn.Linear(d_emb, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_actions)
            )

            # Critic with hidden layer
            self.critic = nn.Sequential(
                nn.Linear(d_emb, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            # Direct projection (simpler, as per some PPO implementations)
            self.actor = nn.Linear(d_emb, num_actions)
            self.critic = nn.Linear(d_emb, 1)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Orthogonal initialization for better gradient flow
                nn.init.orthogonal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through actor-critic.

        Args:
            embeddings: Encoder embeddings (B, d_emb)

        Returns:
            action_logits: Action logits (B, num_actions)
            values: Value estimates (B, 1)
        """
        action_logits = self.actor(embeddings)
        values = self.critic(embeddings)
        return action_logits, values

    def get_action_and_value(
        self,
        embeddings: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value with log probability and entropy.

        Args:
            embeddings: Encoder embeddings (B, d_emb)
            action: Optional specific action to evaluate (B,)
            deterministic: If True, select argmax action (for evaluation)

        Returns:
            action: Selected action (B,)
            log_prob: Log probability of action (B,)
            entropy: Policy entropy (B,)
            value: Value estimate (B, 1)
        """
        action_logits, value = self.forward(embeddings)

        # Create categorical distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = distributions.Categorical(probs=probs)

        # Sample or use provided action
        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate given actions (used during training).

        Args:
            embeddings: Encoder embeddings (B, d_emb)
            actions: Actions to evaluate (B,)

        Returns:
            log_probs: Log probabilities of actions (B,)
            entropy: Policy entropy (B,)
            values: Value estimates (B, 1)
        """
        action_logits, values = self.forward(embeddings)

        # Create distribution
        probs = torch.softmax(action_logits, dim=-1)
        dist = distributions.Categorical(probs=probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values

    def get_value(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get only value estimate (critic).

        Args:
            embeddings: Encoder embeddings (B, d_emb)

        Returns:
            Value estimates (B, 1)
        """
        return self.critic(embeddings)

    def get_action_probs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get action probabilities (softmax of logits).

        Args:
            embeddings: Encoder embeddings (B, d_emb)

        Returns:
            Action probabilities (B, num_actions)
        """
        action_logits = self.actor(embeddings)
        return torch.softmax(action_logits, dim=-1)


class SeparateActorCritic(nn.Module):
    """Alternative implementation with completely separate actor and critic networks.

    This can be useful if we want different architectures or learning rates.
    """

    def __init__(
        self,
        d_emb: int = 64,
        num_actions: int = 2,
        actor_hidden: int = 64,
        critic_hidden: int = 64
    ):
        """Initialize separate actor-critic.

        Args:
            d_emb: Embedding dimension
            num_actions: Number of actions
            actor_hidden: Actor hidden dimension
            critic_hidden: Critic hidden dimension
        """
        super().__init__()

        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(d_emb, actor_hidden),
            nn.Tanh(),
            nn.Linear(actor_hidden, actor_hidden),
            nn.Tanh(),
            nn.Linear(actor_hidden, num_actions)
        )

        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(d_emb, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, critic_hidden),
            nn.Tanh(),
            nn.Linear(critic_hidden, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            embeddings: Encoder embeddings (B, d_emb)

        Returns:
            action_logits: Action logits (B, num_actions)
            values: Value estimates (B, 1)
        """
        action_logits = self.actor(embeddings)
        values = self.critic(embeddings)
        return action_logits, values

    def get_action_and_value(
        self,
        embeddings: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action and value (same interface as ActorCritic)."""
        action_logits, value = self.forward(embeddings)

        probs = torch.softmax(action_logits, dim=-1)
        dist = distributions.Categorical(probs=probs)

        if action is None:
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value

    def evaluate_actions(
        self,
        embeddings: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions (same interface as ActorCritic)."""
        action_logits, values = self.forward(embeddings)

        probs = torch.softmax(action_logits, dim=-1)
        dist = distributions.Categorical(probs=probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, entropy, values
