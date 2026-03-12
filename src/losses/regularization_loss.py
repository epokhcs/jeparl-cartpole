"""Regularization loss for preventing embedding collapse."""

import torch
import torch.nn as nn
from typing import Optional, Dict


class VarianceRegularizationLoss(nn.Module):
    """Variance-based regularization to prevent embedding collapse.

    Encourages embeddings to use full dimensionality by penalizing low variance.
    As per paper: L_reg = -min(1, (1/d_emb) * Σ Var(s_x)_i)
    """

    def __init__(
        self,
        d_emb: int = 64,
        min_variance: float = 0.01,
        normalize: bool = True
    ):
        """Initialize variance regularization loss.

        Args:
            d_emb: Embedding dimension
            min_variance: Minimum acceptable variance (for monitoring)
            normalize: Whether to normalize variance by d_emb
        """
        super().__init__()
        self.d_emb = d_emb
        self.min_variance = min_variance
        self.normalize = normalize

    def forward(
        self,
        embeddings: torch.Tensor,
        return_stats: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, float]]:
        """Compute variance regularization loss.

        Args:
            embeddings: Embeddings to regularize (B, d_emb)
            return_stats: Whether to return variance statistics

        Returns:
            loss: Regularization loss (negative to encourage high variance)
            stats: Optional dictionary with variance statistics
        """
        # Compute variance per dimension across batch
        var_per_dim = torch.var(embeddings, dim=0, unbiased=False)  # (d_emb,)

        # Average variance across dimensions
        avg_variance = var_per_dim.mean()

        if self.normalize:
            # Normalize by embedding dimension
            normalized_variance = avg_variance / self.d_emb
        else:
            normalized_variance = avg_variance

        # Loss: -min(1, normalized_variance)
        # We want to maximize variance, so negative loss
        # Cap at 1.0 to prevent unbounded growth
        loss = -torch.min(torch.tensor(1.0, device=embeddings.device), normalized_variance)

        if return_stats:
            with torch.no_grad():
                # Check for collapse
                collapsed = avg_variance < self.min_variance

                stats = {
                    'avg_variance': avg_variance.item(),
                    'normalized_variance': normalized_variance.item(),
                    'min_variance': var_per_dim.min().item(),
                    'max_variance': var_per_dim.max().item(),
                    'std_of_variances': var_per_dim.std().item(),
                    'collapsed': collapsed.item() if isinstance(collapsed, torch.Tensor) else collapsed,
                    'num_low_var_dims': (var_per_dim < self.min_variance).sum().item()
                }

            return loss, stats

        return loss


class StdRegularizationLoss(nn.Module):
    """Alternative regularization using standard deviation.

    Encourages high standard deviation across batch for each dimension.
    """

    def __init__(
        self,
        d_emb: int = 64,
        target_std: float = 1.0
    ):
        """Initialize std regularization loss.

        Args:
            d_emb: Embedding dimension
            target_std: Target standard deviation
        """
        super().__init__()
        self.d_emb = d_emb
        self.target_std = target_std

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute std regularization loss.

        Args:
            embeddings: Embeddings (B, d_emb)

        Returns:
            Loss penalizing deviation from target std
        """
        # Std per dimension
        std_per_dim = torch.std(embeddings, dim=0, unbiased=False)

        # Penalize deviation from target std
        loss = F.mse_loss(std_per_dim, torch.full_like(std_per_dim, self.target_std))

        return loss


class VICRegLoss(nn.Module):
    """VICReg-style regularization (Variance-Invariance-Covariance).

    Prevents collapse through three objectives:
    1. Variance: encourage high variance per dimension
    2. Invariance: not used here (would be similarity loss)
    3. Covariance: decorrelate dimensions
    """

    def __init__(
        self,
        d_emb: int = 64,
        variance_weight: float = 1.0,
        covariance_weight: float = 0.04,
        variance_epsilon: float = 1e-4
    ):
        """Initialize VICReg loss.

        Args:
            d_emb: Embedding dimension
            variance_weight: Weight for variance loss
            covariance_weight: Weight for covariance loss
            variance_epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.d_emb = d_emb
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_epsilon = variance_epsilon

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute VICReg regularization.

        Args:
            embeddings: Embeddings (B, d_emb)

        Returns:
            Combined variance and covariance loss
        """
        B = embeddings.shape[0]

        # Variance loss: encourage std close to 1
        std_per_dim = torch.sqrt(embeddings.var(dim=0) + self.variance_epsilon)
        variance_loss = torch.mean(F.relu(1.0 - std_per_dim))

        # Covariance loss: decorrelate dimensions
        # Center embeddings
        embeddings_centered = embeddings - embeddings.mean(dim=0)

        # Covariance matrix (d_emb, d_emb)
        cov = (embeddings_centered.T @ embeddings_centered) / (B - 1)

        # Off-diagonal elements should be zero
        off_diagonal = cov.flatten()[:-1].view(self.d_emb - 1, self.d_emb + 1)[:, 1:].flatten()
        covariance_loss = off_diagonal.pow(2).sum() / self.d_emb

        # Total loss
        total_loss = self.variance_weight * variance_loss + self.covariance_weight * covariance_loss

        return total_loss


def variance_regularization_loss(
    embeddings: torch.Tensor,
    d_emb: int = 64
) -> torch.Tensor:
    """Functional interface for variance regularization.

    Args:
        embeddings: Embeddings (B, d_emb)
        d_emb: Embedding dimension

    Returns:
        Regularization loss
    """
    # Variance per dimension
    var_per_dim = torch.var(embeddings, dim=0, unbiased=False)

    # Average variance
    avg_variance = var_per_dim.mean()

    # Normalized variance
    normalized_variance = avg_variance / d_emb

    # Loss: -min(1, normalized_variance)
    loss = -torch.min(torch.tensor(1.0, device=embeddings.device), normalized_variance)

    return loss


# Fix import
import torch.nn.functional as F
