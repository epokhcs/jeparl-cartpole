"""JEPA loss function."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class JEPALoss(nn.Module):
    """Joint-Embedding Predictive Architecture loss.

    Computes L2 distance between predicted and target embeddings.
    Target embeddings must be detached (no gradients).
    """

    def __init__(self, reduction: str = 'mean'):
        """Initialize JEPA loss.

        Args:
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        predicted_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        detach_target: bool = True
    ) -> torch.Tensor:
        """Compute JEPA loss.

        Args:
            predicted_embeddings: Predicted embeddings from predictor
                                  Shape: (B, num_frames, d_emb) or (B, d_emb)
            target_embeddings: Target embeddings from Y-encoder
                              Shape: (B, num_frames, d_emb) or (B, d_emb)
            detach_target: Whether to detach target (should always be True)

        Returns:
            JEPA loss (scalar or per-sample depending on reduction)
        """
        # Ensure target has no gradients
        if detach_target:
            target_embeddings = target_embeddings.detach()

        # Compute MSE loss (L2 distance)
        loss = F.mse_loss(
            predicted_embeddings,
            target_embeddings,
            reduction=self.reduction
        )

        return loss


def jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Functional interface for JEPA loss.

    Args:
        predicted: Predicted embeddings (B, num_frames, d_emb) or (B, d_emb)
        target: Target embeddings (B, num_frames, d_emb) or (B, d_emb)
        reduction: Reduction method

    Returns:
        JEPA loss
    """
    # Always detach target
    target = target.detach()

    # MSE loss
    loss = F.mse_loss(predicted, target, reduction=reduction)

    return loss


class ContrastiveJEPALoss(nn.Module):
    """Alternative JEPA loss with contrastive learning.

    This can be used as an alternative to the simple MSE loss.
    Not used in the paper but provided for experimentation.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        reduction: str = 'mean'
    ):
        """Initialize contrastive JEPA loss.

        Args:
            temperature: Temperature for softmax
            reduction: Reduction method
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        predicted_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive JEPA loss.

        Args:
            predicted_embeddings: (B, d_emb)
            target_embeddings: (B, d_emb)

        Returns:
            Contrastive loss
        """
        # Detach target
        target_embeddings = target_embeddings.detach()

        # Normalize embeddings
        predicted_norm = F.normalize(predicted_embeddings, dim=-1)
        target_norm = F.normalize(target_embeddings, dim=-1)

        # Compute similarity matrix
        logits = torch.matmul(predicted_norm, target_norm.T) / self.temperature

        # Diagonal should be positive pairs
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        return loss


class AdaptiveJEPALoss(nn.Module):
    """Adaptive JEPA loss with learned weighting.

    Learns to weight different dimensions or frames differently.
    """

    def __init__(
        self,
        d_emb: int = 64,
        num_frames: int = 3,
        learnable_weights: bool = True
    ):
        """Initialize adaptive JEPA loss.

        Args:
            d_emb: Embedding dimension
            num_frames: Number of frames
            learnable_weights: Whether to learn dimension weights
        """
        super().__init__()

        self.d_emb = d_emb
        self.num_frames = num_frames

        if learnable_weights:
            # Learnable per-dimension weights
            self.dim_weights = nn.Parameter(torch.ones(d_emb))
            # Learnable per-frame weights
            self.frame_weights = nn.Parameter(torch.ones(num_frames))
        else:
            self.register_buffer('dim_weights', torch.ones(d_emb))
            self.register_buffer('frame_weights', torch.ones(num_frames))

    def forward(
        self,
        predicted_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted JEPA loss.

        Args:
            predicted_embeddings: (B, num_frames, d_emb)
            target_embeddings: (B, num_frames, d_emb)

        Returns:
            Weighted JEPA loss
        """
        # Detach target
        target_embeddings = target_embeddings.detach()

        # Compute squared differences
        diff_squared = (predicted_embeddings - target_embeddings) ** 2

        # Apply dimension weights
        weighted_diff = diff_squared * self.dim_weights.view(1, 1, -1)

        # Apply frame weights
        weighted_diff = weighted_diff * self.frame_weights.view(1, -1, 1)

        # Mean over all dimensions
        loss = weighted_diff.mean()

        return loss
