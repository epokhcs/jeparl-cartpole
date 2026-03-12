"""Checkpoint management utilities."""

from typing import Dict, Any, Optional
from pathlib import Path
import torch


class CheckpointManager:
    """Manage model checkpoints during training."""

    def __init__(self, checkpoint_dir: str, keep_best: bool = True):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_best: Whether to track and keep best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.best_metric = None
        self.best_checkpoint_path = None

    def save_checkpoint(
        self,
        step: int,
        models: Dict[str, torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> str:
        """Save checkpoint.

        Args:
            step: Training step
            models: Dictionary of model name -> model
            optimizer: Optimizer state
            metrics: Optional metrics dictionary
            is_best: Whether this is the best checkpoint so far

        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'step': step,
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics or {}
        }

        # Save model state dicts
        for name, model in models.items():
            checkpoint[f'{name}_state_dict'] = model.state_dict()

        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Best checkpoint
        if is_best and self.keep_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path

        # Latest checkpoint (for easy resuming)
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)

        return str(checkpoint_path)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        models: Dict[str, torch.nn.Module],
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = "cpu"
    ) -> Dict[str, Any]:
        """Load checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            models: Dictionary of model name -> model to load into
            optimizer: Optional optimizer to load state into
            device: Device to load checkpoint to

        Returns:
            Dictionary containing checkpoint metadata (step, metrics, etc.)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model state dicts
        for name, model in models.items():
            key = f'{name}_state_dict'
            if key in checkpoint:
                model.load_state_dict(checkpoint[key])
            else:
                print(f"Warning: No state dict found for model '{name}' in checkpoint")

        # Load optimizer state
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'step': checkpoint.get('step', 0),
            'metrics': checkpoint.get('metrics', {})
        }

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get path to latest checkpoint if it exists.

        Returns:
            Path to latest checkpoint or None
        """
        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        return str(latest_path) if latest_path.exists() else None

    def get_best_checkpoint(self) -> Optional[str]:
        """Get path to best checkpoint if it exists.

        Returns:
            Path to best checkpoint or None
        """
        best_path = self.checkpoint_dir / "checkpoint_best.pt"
        return str(best_path) if best_path.exists() else None
