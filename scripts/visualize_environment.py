"""Visualize CartPole pixel observations and verify environment setup."""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment.cartpole_pixels import CartPolePixels
from src.environment.preprocessing import extract_patches, visualize_patches


def visualize_cartpole_frames(num_steps: int = 10, save_dir: str = "results/visualization"):
    """Visualize CartPole pixel observations.

    Args:
        num_steps: Number of steps to visualize
        save_dir: Directory to save visualizations
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Creating CartPole environment with pixel observations...")
    env = CartPolePixels(
        render_size=(600, 400),
        pad_to=(608, 400),
        frame_stack=3,
        normalize=True
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")

    # Reset environment
    obs, info = env.reset(seed=42)
    print(f"Initial observation shape: {obs.shape}")
    print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

    # Collect some frames
    frames_to_viz = [obs.copy()]
    actions = []

    for step in range(num_steps):
        action = env.action_space.sample()
        actions.append(action)
        obs, reward, terminated, truncated, info = env.step(action)
        frames_to_viz.append(obs.copy())

        if terminated or truncated:
            print(f"Episode ended at step {step}")
            break

    # Visualize initial frames
    print("\nVisualizing initial stacked frames...")
    initial_frames = frames_to_viz[0]  # Shape: (3, 3, 400, 608)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        # Convert CHW to HWC for display
        img = initial_frames[i].transpose(1, 2, 0)  # (3, 400, 608) -> (400, 608, 3)
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {i} (t={-2+i})")
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / "initial_stacked_frames.png", dpi=150)
    print(f"Saved: {save_dir / 'initial_stacked_frames.png'}")
    plt.close()

    # Visualize sequence
    print("\nVisualizing frame sequence...")
    num_viz = min(6, len(frames_to_viz))
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx in range(num_viz):
        if idx < len(frames_to_viz):
            # Show the most recent frame from the stack
            img = frames_to_viz[idx][-1].transpose(1, 2, 0)  # Last frame, CHW -> HWC
            axes[idx].imshow(img)
            action_text = f"Action: {actions[idx-1]}" if idx > 0 else "Initial"
            axes[idx].set_title(f"Step {idx} - {action_text}")
            axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(save_dir / "frame_sequence.png", dpi=150)
    print(f"Saved: {save_dir / 'frame_sequence.png'}")
    plt.close()

    # Test patch extraction
    print("\nTesting patch extraction...")
    obs_tensor = torch.from_numpy(initial_frames)
    patches, positions = extract_patches(obs_tensor, patch_size=16)

    print(f"Patches shape: {patches.shape}")
    print(f"Positions shape: {positions.shape}")
    print(f"Number of patches per frame: {patches.shape[0] // 3}")
    print(f"Patch dimension: {patches.shape[1]}")

    # Visualize patches
    print("\nVisualizing patch extraction and reconstruction...")
    visualize_patches(
        obs_tensor,
        patch_size=16,
        save_path=save_dir / "patch_reconstruction.png"
    )
    print(f"Saved: {save_dir / 'patch_reconstruction.png'}")

    # Analyze positional encodings
    print("\nAnalyzing positional encodings...")
    print(f"Spatial position range (i): [{positions[:, 0].min():.0f}, {positions[:, 0].max():.0f}]")
    print(f"Spatial position range (j): [{positions[:, 1].min():.0f}, {positions[:, 1].max():.0f}]")
    print(f"Temporal position values: {positions[:, 2].unique().tolist()}")

    # Visualize positional encoding distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(positions[:, 0].numpy(), bins=30, edgecolor='black')
    axes[0].set_title("Spatial Position (i) - Row")
    axes[0].set_xlabel("Row Index")
    axes[0].set_ylabel("Count")

    axes[1].hist(positions[:, 1].numpy(), bins=30, edgecolor='black')
    axes[1].set_title("Spatial Position (j) - Column")
    axes[1].set_xlabel("Column Index")
    axes[1].set_ylabel("Count")

    axes[2].hist(positions[:, 2].numpy(), bins=10, edgecolor='black')
    axes[2].set_title("Temporal Position (t)")
    axes[2].set_xlabel("Time Index")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(save_dir / "positional_encodings.png", dpi=150)
    print(f"Saved: {save_dir / 'positional_encodings.png'}")
    plt.close()

    print("\n" + "="*60)
    print("Environment visualization complete!")
    print(f"All visualizations saved to: {save_dir}")
    print("="*60)

    env.close()


def test_environment_stability(num_episodes: int = 5, max_steps: int = 500):
    """Test environment stability over multiple episodes.

    Args:
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
    """
    print(f"\nTesting environment stability over {num_episodes} episodes...")

    env = CartPolePixels(
        render_size=(600, 400),
        pad_to=(608, 400),
        frame_stack=3,
        normalize=True
    )

    episode_lengths = []
    episode_returns = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_return = 0
        step = 0

        while step < max_steps:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            episode_return += reward
            step += 1

            # Verify observation shape
            assert obs.shape == (3, 3, 400, 608), f"Unexpected shape: {obs.shape}"

            if terminated or truncated:
                break

        episode_lengths.append(step)
        episode_returns.append(episode_return)

        print(f"Episode {episode + 1}: Length={step}, Return={episode_return:.1f}")

    print(f"\nAverage episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Average episode return: {np.mean(episode_returns):.1f} ± {np.std(episode_returns):.1f}")

    env.close()
    print("\nEnvironment stability test passed! ✓")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize CartPole environment")
    parser.add_argument("--steps", type=int, default=10, help="Number of steps to visualize")
    parser.add_argument("--save-dir", type=str, default="results/visualization",
                        help="Directory to save visualizations")
    parser.add_argument("--test-stability", action="store_true",
                        help="Run stability test")

    args = parser.parse_args()

    # Run visualization
    visualize_cartpole_frames(num_steps=args.steps, save_dir=args.save_dir)

    # Run stability test if requested
    if args.test_stability:
        test_environment_stability()
