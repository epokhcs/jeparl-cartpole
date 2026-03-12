"""Metrics collection and tracking."""

from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict


class MetricsCollector:
    """Collect and aggregate metrics during training."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics_history: Dict[str, List[float]] = defaultdict(list)
        self.episode_returns: List[float] = []
        self.episode_lengths: List[float] = []

    def add_scalar(self, name: str, value: float) -> None:
        """Add a scalar metric.

        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics_history[name].append(value)

    def add_episode(self, episode_return: float, episode_length: int) -> None:
        """Add completed episode.

        Args:
            episode_return: Total episode return
            episode_length: Episode length in steps
        """
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)

    def get_mean(self, name: str, last_n: Optional[int] = None) -> float:
        """Get mean of a metric.

        Args:
            name: Metric name
            last_n: If specified, only average over last N values

        Returns:
            Mean value or 0.0 if no values
        """
        if name not in self.metrics_history or len(self.metrics_history[name]) == 0:
            return 0.0

        values = self.metrics_history[name]
        if last_n is not None:
            values = values[-last_n:]

        return float(np.mean(values))

    def get_std(self, name: str, last_n: Optional[int] = None) -> float:
        """Get standard deviation of a metric.

        Args:
            name: Metric name
            last_n: If specified, only compute over last N values

        Returns:
            Standard deviation or 0.0 if no values
        """
        if name not in self.metrics_history or len(self.metrics_history[name]) == 0:
            return 0.0

        values = self.metrics_history[name]
        if last_n is not None:
            values = values[-last_n:]

        return float(np.std(values))

    def get_episode_stats(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """Get episode statistics.

        Args:
            last_n: If specified, only compute over last N episodes

        Returns:
            Dictionary with mean/std return and length
        """
        if not self.episode_returns:
            return {
                'mean_return': 0.0,
                'std_return': 0.0,
                'mean_length': 0.0,
                'std_length': 0.0
            }

        returns = self.episode_returns if last_n is None else self.episode_returns[-last_n:]
        lengths = self.episode_lengths if last_n is None else self.episode_lengths[-last_n:]

        return {
            'mean_return': float(np.mean(returns)),
            'std_return': float(np.std(returns)),
            'mean_length': float(np.mean(lengths)),
            'std_length': float(np.std(lengths))
        }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics as dictionary.

        Returns:
            Dictionary of all metrics
        """
        metrics = dict(self.metrics_history)
        metrics['episode_returns'] = self.episode_returns
        metrics['episode_lengths'] = self.episode_lengths
        return metrics

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics_history.clear()
        self.episode_returns.clear()
        self.episode_lengths.clear()

