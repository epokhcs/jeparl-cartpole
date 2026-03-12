"""Frame buffer for stacking frames."""

from collections import deque
from typing import Optional
import numpy as np
import torch


class FrameBuffer:
    """Buffer for stacking multiple frames."""

    def __init__(self, num_frames: int = 3, frame_shape: tuple = (3, 400, 608)):
        """Initialize frame buffer.

        Args:
            num_frames: Number of frames to stack
            frame_shape: Shape of individual frames (C, H, W)
        """
        self.num_frames = num_frames
        self.frame_shape = frame_shape
        self.buffer = deque(maxlen=num_frames)

    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """Reset buffer with initial frame.

        Duplicates the initial frame to fill the buffer.

        Args:
            initial_frame: Initial frame (C, H, W)

        Returns:
            Stacked frames (num_frames, C, H, W)
        """
        self.buffer.clear()
        for _ in range(self.num_frames):
            self.buffer.append(initial_frame.copy())
        return self.get_stacked_frames()

    def append(self, frame: np.ndarray) -> np.ndarray:
        """Append new frame and return stacked frames.

        Args:
            frame: New frame to append (C, H, W)

        Returns:
            Stacked frames (num_frames, C, H, W)
        """
        self.buffer.append(frame.copy())
        return self.get_stacked_frames()

    def get_stacked_frames(self) -> np.ndarray:
        """Get current stacked frames.

        Returns:
            Stacked frames (num_frames, C, H, W)
        """
        return np.stack(list(self.buffer), axis=0)

    def get_context_and_target_frames(self) -> tuple[np.ndarray, np.ndarray]:
        """Get context and target frames for JEPA.

        Context frames: {t-2, t-1, t}
        Target frames: {t-1, t, t+1}

        Note: This assumes buffer is full and will need the next frame for target.
        The trainer needs to handle getting t+1 frame.

        Returns:
            context_frames: Shape (3, C, H, W) for frames t-2, t-1, t
            target_frames: Shape (3, C, H, W) for frames t-1, t, t+1
        """
        if len(self.buffer) < self.num_frames:
            raise ValueError("Buffer not full enough for context/target split")

        frames = list(self.buffer)

        # For a buffer of last 3 frames: [t-2, t-1, t]
        # Context: [t-2, t-1, t]
        # Target: [t-1, t, t+1] - but we don't have t+1 yet
        # The trainer will need to manage this by storing frames appropriately

        context_frames = np.stack(frames, axis=0)  # All current frames

        # For target, we use [t-1, t] from current buffer
        # The caller must provide t+1
        return context_frames, None  # Target construction handled in trainer

    def __len__(self) -> int:
        """Get current buffer length."""
        return len(self.buffer)
