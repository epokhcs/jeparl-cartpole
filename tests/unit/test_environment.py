"""Unit tests for environment components."""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.environment.frame_buffer import FrameBuffer
from src.environment.cartpole_pixels import CartPolePixels
from src.environment.preprocessing import (
    extract_patches,
    create_positional_encodings,
    reconstruct_from_patches,
    normalize_positions
)


class TestFrameBuffer:
    """Tests for FrameBuffer class."""

    def test_initialization(self):
        """Test frame buffer initialization."""
        buffer = FrameBuffer(num_frames=3, frame_shape=(3, 400, 608))
        assert len(buffer) == 0
        assert buffer.num_frames == 3

    def test_reset(self):
        """Test buffer reset."""
        buffer = FrameBuffer(num_frames=3, frame_shape=(3, 400, 608))
        initial_frame = np.random.rand(3, 400, 608).astype(np.float32)

        stacked = buffer.reset(initial_frame)

        assert stacked.shape == (3, 3, 400, 608)
        assert len(buffer) == 3
        # All frames should be identical after reset
        assert np.allclose(stacked[0], stacked[1])
        assert np.allclose(stacked[1], stacked[2])

    def test_append(self):
        """Test appending frames."""
        buffer = FrameBuffer(num_frames=3, frame_shape=(3, 400, 608))

        # Reset with initial frame
        frame1 = np.ones((3, 400, 608), dtype=np.float32)
        buffer.reset(frame1)

        # Append new frame
        frame2 = np.ones((3, 400, 608), dtype=np.float32) * 2
        stacked = buffer.append(frame2)

        assert stacked.shape == (3, 3, 400, 608)
        # Latest frame should be at the end
        assert np.allclose(stacked[-1], frame2)

    def test_buffer_overflow(self):
        """Test that buffer maintains max length."""
        buffer = FrameBuffer(num_frames=3, frame_shape=(3, 400, 608))

        # Add more frames than buffer size
        for i in range(5):
            frame = np.ones((3, 400, 608), dtype=np.float32) * i
            if i == 0:
                buffer.reset(frame)
            else:
                buffer.append(frame)

        # Should only keep last 3 frames
        assert len(buffer) == 3
        stacked = buffer.get_stacked_frames()
        # Should have frames 2, 3, 4
        assert np.allclose(stacked[0], np.ones((3, 400, 608)) * 2)
        assert np.allclose(stacked[1], np.ones((3, 400, 608)) * 3)
        assert np.allclose(stacked[2], np.ones((3, 400, 608)) * 4)


class TestCartPolePixels:
    """Tests for CartPolePixels wrapper."""

    def test_initialization(self):
        """Test environment initialization."""
        env = CartPolePixels(
            render_size=(600, 400),
            pad_to=(608, 400),
            frame_stack=3,
            normalize=True
        )

        assert env.observation_space.shape == (3, 3, 400, 608)
        assert env.observation_space.dtype == np.float32

    def test_reset(self):
        """Test environment reset."""
        env = CartPolePixels(
            render_size=(600, 400),
            pad_to=(608, 400),
            frame_stack=3,
            normalize=True
        )

        obs, info = env.reset(seed=42)

        assert obs.shape == (3, 3, 400, 608)
        assert obs.dtype == np.float32
        assert 0 <= obs.min() <= obs.max() <= 1.0  # Normalized

    def test_step(self):
        """Test environment step."""
        env = CartPolePixels(
            render_size=(600, 400),
            pad_to=(608, 400),
            frame_stack=3,
            normalize=True
        )

        obs, info = env.reset(seed=42)
        obs, reward, terminated, truncated, info = env.step(0)

        assert obs.shape == (3, 3, 400, 608)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_multiple_steps(self):
        """Test multiple environment steps."""
        env = CartPolePixels(
            render_size=(600, 400),
            pad_to=(608, 400),
            frame_stack=3,
            normalize=True
        )

        obs, info = env.reset(seed=42)

        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == (3, 3, 400, 608)

            if terminated or truncated:
                obs, info = env.reset()


class TestPreprocessing:
    """Tests for preprocessing utilities."""

    def test_extract_patches(self):
        """Test patch extraction."""
        # Create dummy frames (3 frames, 3 channels, 400x608)
        frames = torch.rand(3, 3, 400, 608)

        patches, positions = extract_patches(frames, patch_size=16)

        # Calculate expected dimensions
        num_frames = 3
        num_patches_h = 400 // 16  # 25
        num_patches_w = 608 // 16  # 38
        num_patches = num_frames * num_patches_h * num_patches_w  # 2850
        patch_dim = 3 * 16 * 16  # 768

        assert patches.shape == (num_patches, patch_dim)
        assert positions.shape == (num_patches, 3)

    def test_extract_patches_batched(self):
        """Test patch extraction with batch dimension."""
        # Create dummy batch (batch_size=2, 3 frames, 3 channels, 400x608)
        frames = torch.rand(2, 3, 3, 400, 608)

        patches, positions = extract_patches(frames, patch_size=16)

        num_frames = 3
        num_patches_h = 400 // 16
        num_patches_w = 608 // 16
        num_patches = num_frames * num_patches_h * num_patches_w
        patch_dim = 3 * 16 * 16

        assert patches.shape == (2, num_patches, patch_dim)
        assert positions.shape == (2, num_patches, 3)

    def test_positional_encodings(self):
        """Test positional encoding creation."""
        positions = create_positional_encodings(
            num_frames=3,
            num_patches_h=25,
            num_patches_w=38,
            batch_size=1
        )

        expected_patches = 3 * 25 * 38
        assert positions.shape == (1, expected_patches, 3)

        # Check temporal indices (should be -2, -1, 0 for 3 frames)
        temporal_values = positions[0, :, 2].unique()
        assert len(temporal_values) == 3
        assert -2 in temporal_values
        assert -1 in temporal_values
        assert 0 in temporal_values

    def test_reconstruct_from_patches(self):
        """Test patch reconstruction."""
        # Create dummy frames
        original_frames = torch.rand(3, 3, 400, 608)

        # Extract patches
        patches, positions = extract_patches(original_frames, patch_size=16)

        # Reconstruct
        reconstructed = reconstruct_from_patches(
            patches,
            num_frames=3,
            img_size=(400, 608),
            patch_size=16,
            num_channels=3
        )

        assert reconstructed.shape == original_frames.shape
        # Should be very close (perfect reconstruction)
        assert torch.allclose(reconstructed, original_frames, atol=1e-6)

    def test_normalize_positions(self):
        """Test position normalization."""
        positions = create_positional_encodings(
            num_frames=3,
            num_patches_h=25,
            num_patches_w=38,
            batch_size=1
        )

        normalized = normalize_positions(positions, max_h=25, max_w=38, max_t=2)

        # Spatial positions should be in [0, 1]
        assert normalized[..., 0].min() >= 0
        assert normalized[..., 0].max() <= 1
        assert normalized[..., 1].min() >= 0
        assert normalized[..., 1].max() <= 1

        # Temporal positions should be normalized
        assert normalized[..., 2].min() >= -1
        assert normalized[..., 2].max() <= 1


def test_integration():
    """Integration test: environment + preprocessing."""
    # Create environment
    env = CartPolePixels(
        render_size=(600, 400),
        pad_to=(608, 400),
        frame_stack=3,
        normalize=True
    )

    # Reset and get observation
    obs, info = env.reset(seed=42)

    # Convert to torch and extract patches
    obs_tensor = torch.from_numpy(obs)
    patches, positions = extract_patches(obs_tensor, patch_size=16)

    # Verify shapes
    assert patches.shape[0] == 2850  # 3 * 25 * 38
    assert patches.shape[1] == 768   # 3 * 16 * 16
    assert positions.shape == (2850, 3)

    # Reconstruct and verify
    reconstructed = reconstruct_from_patches(
        patches,
        num_frames=3,
        img_size=(400, 608),
        patch_size=16,
        num_channels=3
    )

    assert torch.allclose(reconstructed, obs_tensor, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
