"""Logging utilities for training."""

import sys
from typing import Dict, Any, Optional
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Unified logger for console, TensorBoard, and WandB."""

    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize logger.

        Args:
            log_dir: Directory for log files
            use_tensorboard: Whether to log to TensorBoard
            use_wandb: Whether to log to Weights & Biases
            wandb_project: WandB project name
            wandb_entity: WandB entity/username
            config: Configuration dictionary to log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))

        # WandB
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    dir=str(self.log_dir),
                    config=config
                )
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Disabling WandB logging.")
                self.use_wandb = False

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar value.

        Args:
            tag: Name of the scalar
            value: Scalar value
            step: Training step
        """
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)

        if self.use_wandb:
            self.wandb.log({tag: value, "step": step})

    def log_scalars(self, scalar_dict: Dict[str, float], step: int) -> None:
        """Log multiple scalars at once.

        Args:
            scalar_dict: Dictionary of tag -> value
            step: Training step
        """
        for tag, value in scalar_dict.items():
            self.log_scalar(tag, value, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Log a histogram of values.

        Args:
            tag: Name of the histogram
            values: Tensor of values
            step: Training step
        """
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)

        if self.use_wandb:
            self.wandb.log({f"{tag}_hist": self.wandb.Histogram(values.cpu().numpy()), "step": step})

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log text.

        Args:
            tag: Name of the text
            text: Text content
            step: Training step
        """
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.add_text(tag, text, step)

        if self.use_wandb:
            self.wandb.log({tag: text, "step": step})

    def print(self, message: str) -> None:
        """Print message to console and flush.

        Args:
            message: Message to print
        """
        print(message, flush=True)
        sys.stdout.flush()

    def close(self) -> None:
        """Close logger and clean up resources."""
        if self.tb_writer:
            self.tb_writer.close()

        if self.use_wandb:
            self.wandb.finish()
