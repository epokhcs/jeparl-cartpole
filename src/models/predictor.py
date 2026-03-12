"""Predictor network for JEPA."""

import torch
import torch.nn as nn
from typing import Optional


class Predictor(nn.Module):
    """Shallow predictor network for JEPA.

    Predicts target embeddings from context embeddings and action.
    As per the paper, this is a shallow 2-layer MLP.
    """

    def __init__(
        self,
        d_emb: int = 64,
        hidden_dim: int = 128,
        num_actions: int = 2,
        num_context_frames: int = 3,
        num_target_frames: int = 3,
        dropout: float = 0.1
    ):
        """Initialize predictor network.

        Args:
            d_emb: Embedding dimension (from encoder)
            hidden_dim: Hidden layer dimension
            num_actions: Number of discrete actions
            num_context_frames: Number of context frames (default: 3)
            num_target_frames: Number of target frames to predict (default: 3)
            dropout: Dropout rate
        """
        super().__init__()

        self.d_emb = d_emb
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames

        # Context embeddings are from multiple frames
        # We'll flatten them: (num_context_frames * d_emb)
        context_dim = num_context_frames * d_emb

        # Action embedding: project action one-hot to hidden dimension
        self.action_embed = nn.Linear(num_actions, hidden_dim)

        # 2-layer MLP as per paper
        # Layer 1: context + action projection
        self.layer1 = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Layer 2: predict target embeddings
        # Output: num_target_frames * d_emb
        target_dim = num_target_frames * d_emb
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, target_dim),
            nn.Dropout(dropout)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through predictor.

        Args:
            context_embeddings: Context embeddings from X-encoder
                                Shape: (B, num_context_frames, d_emb) or (B, d_emb)
            actions: Action indices or one-hot vectors
                     Shape: (B,) for indices or (B, num_actions) for one-hot

        Returns:
            Predicted target embeddings: (B, num_target_frames, d_emb)
        """
        B = context_embeddings.shape[0]

        # Handle different context embedding shapes
        if context_embeddings.dim() == 2:
            # Single embedding per sample: (B, d_emb)
            # Assume this is already aggregated, replicate for context frames
            context_embeddings = context_embeddings.unsqueeze(1).repeat(1, self.num_context_frames, 1)
        elif context_embeddings.dim() == 3:
            # Multiple embeddings: (B, num_frames, d_emb)
            pass
        else:
            raise ValueError(f"Unexpected context_embeddings shape: {context_embeddings.shape}")

        # Flatten context embeddings
        context_flat = context_embeddings.reshape(B, -1)  # (B, num_context_frames * d_emb)

        # Process actions
        if actions.dim() == 1:
            # Action indices: convert to one-hot
            actions_onehot = torch.nn.functional.one_hot(
                actions.long(),
                num_classes=self.num_actions
            ).float()
        else:
            # Already one-hot
            actions_onehot = actions.float()

        # Embed action
        action_emb = self.action_embed(actions_onehot)  # (B, hidden_dim)

        # First layer: process context
        h = self.layer1(context_flat)  # (B, hidden_dim)

        # Add action embedding after first layer (as per paper description)
        h = h + action_emb

        # Second layer: predict target embeddings
        target_flat = self.layer2(h)  # (B, num_target_frames * d_emb)

        # Reshape to separate frames
        target_embeddings = target_flat.reshape(B, self.num_target_frames, self.d_emb)

        return target_embeddings


class PredictorV2(nn.Module):
    """Alternative predictor that processes each frame independently with attention.

    This version uses attention mechanism to aggregate context frames.
    Can be experimented with if the simple version doesn't work well.
    """

    def __init__(
        self,
        d_emb: int = 64,
        hidden_dim: int = 128,
        num_actions: int = 2,
        num_context_frames: int = 3,
        num_target_frames: int = 3,
        num_heads: int = 2,
        dropout: float = 0.1
    ):
        """Initialize attention-based predictor.

        Args:
            d_emb: Embedding dimension
            hidden_dim: Hidden layer dimension
            num_actions: Number of actions
            num_context_frames: Number of context frames
            num_target_frames: Number of target frames
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()

        self.d_emb = d_emb
        self.num_target_frames = num_target_frames

        # Action embedding
        self.action_embed = nn.Linear(num_actions, d_emb)

        # Cross-attention to aggregate context frames with action
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_emb,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # MLP to predict target embeddings
        self.mlp = nn.Sequential(
            nn.Linear(d_emb, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_emb * num_target_frames)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        context_embeddings: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            context_embeddings: (B, num_context_frames, d_emb)
            actions: (B,) or (B, num_actions)

        Returns:
            Predicted embeddings: (B, num_target_frames, d_emb)
        """
        B = context_embeddings.shape[0]

        # Process action
        if actions.dim() == 1:
            actions_onehot = torch.nn.functional.one_hot(
                actions.long(), num_classes=2
            ).float()
        else:
            actions_onehot = actions.float()

        action_emb = self.action_embed(actions_onehot).unsqueeze(1)  # (B, 1, d_emb)

        # Use action as query to attend over context frames
        aggregated, _ = self.cross_attention(
            action_emb, context_embeddings, context_embeddings
        )  # (B, 1, d_emb)

        aggregated = aggregated.squeeze(1)  # (B, d_emb)

        # Predict target embeddings
        target_flat = self.mlp(aggregated)  # (B, num_target_frames * d_emb)
        target_embeddings = target_flat.reshape(B, self.num_target_frames, self.d_emb)

        return target_embeddings
