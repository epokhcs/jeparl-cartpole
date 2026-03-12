"""Preprocessing utilities for Vision Transformer input."""

import torch
import numpy as np
from typing import Tuple, Optional


def extract_patches(
    frames: torch.Tensor,
    patch_size: int = 16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract patches from stacked frames for Vision Transformer.

    Args:
        frames: Tensor of shape (B, num_frames, C, H, W) or (num_frames, C, H, W)
        patch_size: Size of square patches

    Returns:
        patches: Tensor of shape (B, num_patches, patch_dim) where
                 num_patches = num_frames * (H // patch_size) * (W // patch_size)
                 patch_dim = patch_size * patch_size * C
        positions: Tensor of shape (B, num_patches, 3) containing (i, j, t)
                   where i, j are spatial positions and t is temporal position
    """
    # Handle both batched and unbatched input
    if frames.dim() == 4:
        # Add batch dimension (num_frames, C, H, W) -> (1, num_frames, C, H, W)
        frames = frames.unsqueeze(0)
        unbatched = True
    else:
        unbatched = False

    B, num_frames, C, H, W = frames.shape

    # Check if image dimensions are divisible by patch_size
    assert H % patch_size == 0, f"Height {H} not divisible by patch_size {patch_size}"
    assert W % patch_size == 0, f"Width {W} not divisible by patch_size {patch_size}"

    # Calculate number of patches per dimension
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches_per_frame = num_patches_h * num_patches_w
    total_patches = num_frames * num_patches_per_frame

    # Reshape frames to extract patches
    # (B, num_frames, C, H, W) -> (B, num_frames, C, num_patches_h, patch_size, num_patches_w, patch_size)
    frames_reshaped = frames.reshape(
        B, num_frames, C,
        num_patches_h, patch_size,
        num_patches_w, patch_size
    )

    # Permute to group patch dimensions together
    # -> (B, num_frames, num_patches_h, num_patches_w, C, patch_size, patch_size)
    frames_reshaped = frames_reshaped.permute(0, 1, 3, 5, 2, 4, 6)

    # Reshape to final patch format
    # -> (B, num_frames * num_patches_h * num_patches_w, C * patch_size * patch_size)
    patch_dim = C * patch_size * patch_size
    patches = frames_reshaped.reshape(B, total_patches, patch_dim)

    # Create positional encodings (i, j, t)
    positions = create_positional_encodings(
        num_frames=num_frames,
        num_patches_h=num_patches_h,
        num_patches_w=num_patches_w,
        batch_size=B,
        device=frames.device
    )

    if unbatched:
        # Remove batch dimension
        patches = patches.squeeze(0)
        positions = positions.squeeze(0)

    return patches, positions


def create_positional_encodings(
    num_frames: int,
    num_patches_h: int,
    num_patches_w: int,
    batch_size: int = 1,
    device: Optional[torch.device] = None
) -> torch.Tensor:
    """Create positional encodings (i, j, t) for patches.

    Args:
        num_frames: Number of frames
        num_patches_h: Number of patches in height dimension
        num_patches_w: Number of patches in width dimension
        batch_size: Batch size
        device: Device to create tensor on

    Returns:
        Positions: Tensor of shape (B, num_patches, 3) containing (i, j, t)
                   i: patch row index (0 to num_patches_h - 1)
                   j: patch column index (0 to num_patches_w - 1)
                   t: temporal index (for 3 frames: -2, -1, 0 for context)
    """
    positions = []

    # Temporal encoding: for 3 frames, use t = -2, -1, 0
    # This represents the relative temporal position
    temporal_indices = torch.arange(num_frames, device=device) - (num_frames - 1)

    for t in temporal_indices:
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                positions.append([i, j, t.item()])

    # Convert to tensor (num_patches, 3)
    positions = torch.tensor(positions, dtype=torch.float32, device=device)

    # Expand for batch (B, num_patches, 3)
    positions = positions.unsqueeze(0).expand(batch_size, -1, -1)

    return positions


def normalize_positions(positions: torch.Tensor, max_h: int, max_w: int, max_t: int) -> torch.Tensor:
    """Normalize positional encodings to [-1, 1] range.

    Args:
        positions: Tensor of shape (B, num_patches, 3) or (num_patches, 3)
        max_h: Maximum height (number of patches in height)
        max_w: Maximum width (number of patches in width)
        max_t: Maximum temporal index

    Returns:
        Normalized positions in [-1, 1] range
    """
    positions_norm = positions.clone()

    # Normalize spatial positions to [0, 1]
    positions_norm[..., 0] = positions[..., 0] / (max_h - 1) if max_h > 1 else 0.5
    positions_norm[..., 1] = positions[..., 1] / (max_w - 1) if max_w > 1 else 0.5

    # Temporal positions are already in a reasonable range (-2, -1, 0)
    # Normalize to [-1, 1]: divide by max absolute value
    if max_t > 0:
        positions_norm[..., 2] = positions[..., 2] / max_t

    return positions_norm


def reconstruct_from_patches(
    patches: torch.Tensor,
    num_frames: int,
    img_size: Tuple[int, int],
    patch_size: int,
    num_channels: int = 3
) -> torch.Tensor:
    """Reconstruct frames from patches (inverse of extract_patches).

    Useful for visualization and debugging.

    Args:
        patches: Tensor of shape (B, num_patches, patch_dim) or (num_patches, patch_dim)
        num_frames: Number of frames
        img_size: (H, W) of original image
        patch_size: Size of square patches
        num_channels: Number of channels

    Returns:
        Reconstructed frames of shape (B, num_frames, C, H, W) or (num_frames, C, H, W)
    """
    if patches.dim() == 2:
        patches = patches.unsqueeze(0)
        unbatched = True
    else:
        unbatched = False

    B, num_patches, patch_dim = patches.shape
    H, W = img_size

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    # Reshape patches
    # (B, num_patches, patch_dim) -> (B, num_frames, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches_reshaped = patches.reshape(
        B, num_frames, num_patches_h, num_patches_w,
        num_channels, patch_size, patch_size
    )

    # Permute back to image format
    # -> (B, num_frames, C, num_patches_h, patch_size, num_patches_w, patch_size)
    patches_reshaped = patches_reshaped.permute(0, 1, 4, 2, 5, 3, 6)

    # Reshape to final image
    # -> (B, num_frames, C, H, W)
    frames = patches_reshaped.reshape(B, num_frames, num_channels, H, W)

    if unbatched:
        frames = frames.squeeze(0)

    return frames


def visualize_patches(
    frames: torch.Tensor,
    patch_size: int = 16,
    save_path: Optional[str] = None
) -> None:
    """Visualize patch extraction for debugging.

    Args:
        frames: Tensor of shape (num_frames, C, H, W)
        patch_size: Size of square patches
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt

    # Extract patches
    patches, positions = extract_patches(frames, patch_size)

    # Reconstruct
    reconstructed = reconstruct_from_patches(
        patches,
        num_frames=frames.shape[0],
        img_size=(frames.shape[2], frames.shape[3]),
        patch_size=patch_size,
        num_channels=frames.shape[1]
    )

    # Plot original vs reconstructed
    num_frames = frames.shape[0]
    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 4, 8))

    for i in range(num_frames):
        # Original
        img = frames[i].permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Original Frame {i}")
        axes[0, i].axis('off')

        # Reconstructed
        img_recon = reconstructed[i].permute(1, 2, 0).cpu().numpy()
        axes[1, i].imshow(img_recon)
        axes[1, i].set_title(f"Reconstructed Frame {i}")
        axes[1, i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()
