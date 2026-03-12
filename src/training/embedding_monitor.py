"""Embedding monitor for tracking variance and detecting collapse."""

import torch
import numpy as np
from collections import deque
from typing import Dict, List, Optional
import matplotlib.pyplot as plt


class EmbeddingMonitor:
    """Monitor embedding statistics to detect collapse.

    Tracks variance, dimensionality usage, and provides visualization tools.
    """

    def __init__(
        self,
        d_emb: int = 64,
        collapse_threshold: float = 0.01,
        history_size: int = 1000
    ):
        """Initialize embedding monitor.

        Args:
            d_emb: Embedding dimension
            collapse_threshold: Variance threshold below which collapse is detected
            history_size: Number of steps to keep in history
        """
        self.d_emb = d_emb
        self.collapse_threshold = collapse_threshold
        self.history_size = history_size

        # History tracking
        self.variance_history = deque(maxlen=history_size)
        self.per_dim_variance_history = []
        self.step_history = deque(maxlen=history_size)

        # Collapse tracking
        self.collapse_detected = False
        self.collapse_step = None

    def update(
        self,
        embeddings: torch.Tensor,
        step: int
    ) -> Dict[str, float]:
        """Update monitor with new embeddings.

        Args:
            embeddings: Batch of embeddings (B, d_emb)
            step: Current training step

        Returns:
            Dictionary with statistics
        """
        with torch.no_grad():
            # Compute variance per dimension
            var_per_dim = torch.var(embeddings, dim=0, unbiased=False)  # (d_emb,)

            # Average variance
            avg_variance = var_per_dim.mean().item()

            # Store in history
            self.variance_history.append(avg_variance)
            self.step_history.append(step)

            # Store per-dimension variance (less frequently to save memory)
            if step % 100 == 0 or len(self.per_dim_variance_history) == 0:
                self.per_dim_variance_history.append({
                    'step': step,
                    'variance': var_per_dim.cpu().numpy()
                })

            # Check for collapse
            if not self.collapse_detected and avg_variance < self.collapse_threshold:
                self.collapse_detected = True
                self.collapse_step = step

            # Compute statistics
            stats = {
                'avg_variance': avg_variance,
                'min_variance': var_per_dim.min().item(),
                'max_variance': var_per_dim.max().item(),
                'std_of_variances': var_per_dim.std().item(),
                'collapsed': self.collapse_detected,
                'num_low_var_dims': (var_per_dim < self.collapse_threshold).sum().item(),
                'effective_dimensionality': self._compute_effective_dimensionality(var_per_dim)
            }

            return stats

    def _compute_effective_dimensionality(
        self,
        var_per_dim: torch.Tensor
    ) -> float:
        """Compute effective dimensionality using entropy.

        Args:
            var_per_dim: Variance per dimension (d_emb,)

        Returns:
            Effective dimensionality (higher is better)
        """
        # Normalize variances to probabilities
        var_sum = var_per_dim.sum()
        if var_sum < 1e-10:
            return 0.0

        probs = var_per_dim / var_sum

        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-10)).sum()

        # Effective dimensionality is exp(entropy)
        effective_dim = torch.exp(entropy).item()

        return effective_dim

    def get_variance_trend(self, window_size: int = 100) -> float:
        """Get trend of variance (increasing/decreasing).

        Args:
            window_size: Window size for trend computation

        Returns:
            Trend value (positive = increasing, negative = decreasing)
        """
        if len(self.variance_history) < window_size:
            return 0.0

        recent = list(self.variance_history)[-window_size:]

        # Simple linear regression slope
        x = np.arange(len(recent))
        y = np.array(recent)

        if len(x) < 2:
            return 0.0

        slope = np.polyfit(x, y, 1)[0]

        return float(slope)

    def plot_variance_history(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot variance history over time.

        Args:
            save_path: Optional path to save figure
            show: Whether to show plot
        """
        if len(self.variance_history) == 0:
            print("No variance history to plot")
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        steps = list(self.step_history)
        variances = list(self.variance_history)

        ax.plot(steps, variances, linewidth=2, label='Average Variance')
        ax.axhline(
            y=self.collapse_threshold,
            color='r',
            linestyle='--',
            label=f'Collapse Threshold ({self.collapse_threshold})'
        )

        if self.collapse_detected:
            ax.axvline(
                x=self.collapse_step,
                color='r',
                linestyle='-',
                alpha=0.3,
                label=f'Collapse Detected (Step {self.collapse_step})'
            )

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Average Variance', fontsize=12)
        ax.set_title('Embedding Variance Over Time', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_per_dimension_variance(
        self,
        save_path: Optional[str] = None,
        show: bool = True,
        max_steps: int = 10
    ):
        """Plot variance per dimension over time (heatmap).

        Args:
            save_path: Optional path to save figure
            show: Whether to show plot
            max_steps: Maximum number of time steps to show
        """
        if len(self.per_dim_variance_history) == 0:
            print("No per-dimension variance history to plot")
            return

        # Get last max_steps entries
        history = self.per_dim_variance_history[-max_steps:]

        # Create matrix (time_steps, dimensions)
        variance_matrix = np.array([h['variance'] for h in history])
        steps = [h['step'] for h in history]

        fig, ax = plt.subplots(figsize=(14, 6))

        im = ax.imshow(
            variance_matrix.T,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest'
        )

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Dimension', fontsize=12)
        ax.set_title('Per-Dimension Variance Heatmap', fontsize=14)

        # Set x-axis labels
        ax.set_xticks(range(len(steps)))
        ax.set_xticklabels([str(s) for s in steps], rotation=45)

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Variance', fontsize=10)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def get_summary(self) -> Dict[str, any]:
        """Get summary statistics.

        Returns:
            Dictionary with summary information
        """
        if len(self.variance_history) == 0:
            return {
                'no_data': True
            }

        return {
            'current_variance': self.variance_history[-1],
            'min_variance_seen': min(self.variance_history),
            'max_variance_seen': max(self.variance_history),
            'mean_variance': np.mean(list(self.variance_history)),
            'collapsed': self.collapse_detected,
            'collapse_step': self.collapse_step,
            'variance_trend': self.get_variance_trend(),
            'num_samples': len(self.variance_history)
        }

    def reset(self):
        """Reset monitor."""
        self.variance_history.clear()
        self.per_dim_variance_history.clear()
        self.step_history.clear()
        self.collapse_detected = False
        self.collapse_step = None


class MultiEmbeddingMonitor:
    """Monitor multiple embedding sources (e.g., context and target encoders).

    This is useful for comparing context encoder embeddings with target encoder
    embeddings to ensure they're not diverging.
    """

    def __init__(
        self,
        d_emb: int = 64,
        collapse_threshold: float = 0.01,
        history_size: int = 1000
    ):
        """Initialize multi-embedding monitor.

        Args:
            d_emb: Embedding dimension
            collapse_threshold: Collapse threshold
            history_size: History size
        """
        self.monitors = {}
        self.d_emb = d_emb
        self.collapse_threshold = collapse_threshold
        self.history_size = history_size

    def add_monitor(self, name: str):
        """Add a monitor for a specific embedding source.

        Args:
            name: Name of the embedding source (e.g., 'context', 'target')
        """
        self.monitors[name] = EmbeddingMonitor(
            d_emb=self.d_emb,
            collapse_threshold=self.collapse_threshold,
            history_size=self.history_size
        )

    def update(
        self,
        embeddings_dict: Dict[str, torch.Tensor],
        step: int
    ) -> Dict[str, Dict[str, float]]:
        """Update all monitors.

        Args:
            embeddings_dict: Dictionary mapping names to embeddings
            step: Training step

        Returns:
            Dictionary mapping names to statistics
        """
        all_stats = {}

        for name, embeddings in embeddings_dict.items():
            if name not in self.monitors:
                self.add_monitor(name)

            stats = self.monitors[name].update(embeddings, step)
            all_stats[name] = stats

        return all_stats

    def plot_comparison(
        self,
        save_path: Optional[str] = None,
        show: bool = True
    ):
        """Plot variance comparison across all monitors.

        Args:
            save_path: Optional save path
            show: Whether to show plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        for name, monitor in self.monitors.items():
            if len(monitor.variance_history) > 0:
                steps = list(monitor.step_history)
                variances = list(monitor.variance_history)
                ax.plot(steps, variances, linewidth=2, label=name.capitalize())

        ax.axhline(
            y=self.collapse_threshold,
            color='r',
            linestyle='--',
            label='Collapse Threshold'
        )

        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Average Variance', fontsize=12)
        ax.set_title('Embedding Variance Comparison', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close()

    def get_all_summaries(self) -> Dict[str, Dict]:
        """Get summaries for all monitors.

        Returns:
            Dictionary mapping names to summaries
        """
        return {
            name: monitor.get_summary()
            for name, monitor in self.monitors.items()
        }
