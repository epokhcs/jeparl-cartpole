"""CartPole environment wrapper with pixel observations."""

import gymnasium as gym
import numpy as np
import torch
from typing import Optional, Tuple
from .frame_buffer import FrameBuffer


class CartPolePixels(gym.Wrapper):
    """Wrapper for CartPole that provides pixel observations instead of state vectors.

    Converts CartPole-v1 to return 400x600x3 (or padded 400x608x3) RGB images
    instead of the 4-dimensional state vector.
    """

    def __init__(
        self,
        env: Optional[gym.Env] = None,
        render_size: Tuple[int, int] = (600, 400),  # width, height
        pad_to: Optional[Tuple[int, int]] = (608, 400),  # width, height for clean patch division
        frame_stack: int = 3,
        normalize: bool = True,
    ):
        """Initialize CartPole pixel wrapper.

        Args:
            env: Base environment (if None, creates CartPole-v1)
            render_size: (width, height) for rendering
            pad_to: (width, height) to pad to (None for no padding)
            frame_stack: Number of frames to stack
            normalize: Whether to normalize pixels to [0, 1]
        """
        if env is None:
            env = gym.make("CartPole-v1", render_mode="rgb_array")

        super().__init__(env)

        self.render_size = render_size  # (width, height)
        self.pad_to = pad_to
        self.frame_stack_size = frame_stack
        self.normalize = normalize

        # Calculate final size
        if pad_to is not None:
            final_width, final_height = pad_to
        else:
            final_width, final_height = render_size

        # Frame buffer for stacking
        self.frame_buffer = FrameBuffer(
            num_frames=frame_stack,
            frame_shape=(3, final_height, final_width)  # C, H, W
        )

        # Update observation space
        # Stacked frames: (num_frames, C, H, W)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255 if not normalize else 1.0,
            shape=(frame_stack, 3, final_height, final_width),
            dtype=np.float32 if normalize else np.uint8
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and return initial stacked frames.

        Returns:
            observation: Stacked frames (num_frames, C, H, W)
            info: Info dictionary
        """
        state, info = self.env.reset(**kwargs)

        # Get initial pixel observation
        frame = self._get_pixel_observation()

        # Reset frame buffer with initial frame
        stacked_frames = self.frame_buffer.reset(frame)

        return stacked_frames, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: Action to take

        Returns:
            observation: Stacked frames (num_frames, C, H, W)
            reward: Reward received
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dictionary
        """
        state, reward, terminated, truncated, info = self.env.step(action)

        # Get pixel observation
        frame = self._get_pixel_observation()

        # Update frame buffer
        stacked_frames = self.frame_buffer.append(frame)

        return stacked_frames, reward, terminated, truncated, info

    def _get_pixel_observation(self) -> np.ndarray:
        """Get current pixel observation from environment.

        Returns:
            Pixel observation (C, H, W) in range [0, 1] or [0, 255]
        """
        # Render environment
        # gym.make with render_mode="rgb_array" already configured
        rgb_array = self.env.render()

        # Resize if needed
        if rgb_array.shape[:2] != (self.render_size[1], self.render_size[0]):
            # rgb_array is (H, W, C), render_size is (W, H)
            rgb_array = self._resize_image(rgb_array, self.render_size)

        # Convert to CHW format
        # From (H, W, C) to (C, H, W)
        frame = np.transpose(rgb_array, (2, 0, 1))

        # Pad if needed
        if self.pad_to is not None:
            frame = self._pad_frame(frame)

        # Normalize to [0, 1] if requested
        if self.normalize:
            frame = frame.astype(np.float32) / 255.0
        else:
            frame = frame.astype(np.uint8)

        return frame

    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to target size.

        Args:
            image: Image array (H, W, C)
            size: Target size (width, height)

        Returns:
            Resized image
        """
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            resized = pil_image.resize(size, Image.BILINEAR)
            return np.array(resized)
        except ImportError:
            # Fallback to basic interpolation if PIL not available
            print("Warning: PIL not available, using basic resizing")
            # Simple nearest neighbor (not ideal but works)
            import cv2
            return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    def _pad_frame(self, frame: np.ndarray) -> np.ndarray:
        """Pad frame to target size.

        Args:
            frame: Frame array (C, H, W)

        Returns:
            Padded frame
        """
        c, h, w = frame.shape
        target_w, target_h = self.pad_to

        if w == target_w and h == target_h:
            return frame

        # Calculate padding
        pad_w = target_w - w
        pad_h = target_h - h

        # Pad symmetrically (or right/bottom if odd)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Pad with zeros (black)
        padded = np.pad(
            frame,
            ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=0
        )

        return padded

    def get_current_stacked_frames(self) -> np.ndarray:
        """Get current stacked frames without taking a step.

        Returns:
            Stacked frames (num_frames, C, H, W)
        """
        return self.frame_buffer.get_stacked_frames()

    def render(self, mode: str = 'rgb_array'):
        """Render the environment."""
        return self.env.render()
