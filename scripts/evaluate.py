"""Evaluation script for trained JEPA-RL models.

Loads a trained model and evaluates it on the environment.
"""

import sys
import argparse
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.environment.cartpole_pixels import CartPolePixels
from src.models.vit_encoder import ViTEncoder
from src.models.actor_critic import ActorCritic
from src.environment.preprocessing import extract_patches


def evaluate_model(
    env,
    encoder: ViTEncoder,
    actor_critic: ActorCritic,
    num_episodes: int = 100,
    patch_size: int = 16,
    device: str = "cpu",
    render: bool = False,
    deterministic: bool = True
):
    """Evaluate a trained model.

    Args:
        env: Environment
        encoder: Trained encoder
        actor_critic: Trained actor-critic
        num_episodes: Number of evaluation episodes
        patch_size: Patch size for ViT
        device: Device
        render: Whether to render episodes
        deterministic: Whether to use deterministic actions

    Returns:
        Dictionary with evaluation statistics
    """
    encoder.eval()
    actor_critic.eval()

    episode_returns = []
    episode_lengths = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.from_numpy(obs).float().to(device)

        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            with torch.no_grad():
                # Extract patches
                patches, positions = extract_patches(
                    obs.unsqueeze(0),
                    patch_size=patch_size
                )

                # Encode
                embeddings = encoder(patches, positions)

                # Get action
                action, _, _, _ = actor_critic.get_action_and_value(
                    embeddings,
                    deterministic=deterministic
                )

            # Step
            obs, reward, terminated, truncated, info = env.step(action.item())
            obs = torch.from_numpy(obs).float().to(device)

            episode_return += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                env.render()

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: "
                  f"Return = {episode_return:.1f}, Length = {episode_length}")

    # Statistics
    stats = {
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths
    }

    return stats


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained JEPA-RL model')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cuda/cpu/mps)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes'
    )
    parser.add_argument(
        '--stochastic',
        action='store_true',
        help='Use stochastic actions instead of deterministic'
    )
    parser.add_argument(
        '--save-video',
        type=str,
        default=None,
        help='Path to save video (if rendering)'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create environment
    print("Creating environment...")
    env = CartPolePixels(
        render_size=tuple(config.environment.render_size),
        pad_to=tuple(config.environment.pad_to),
        frame_stack=config.environment.frame_stack,
        normalize=config.environment.normalize_pixels
    )

    # Create models
    print("Creating models...")
    d_emb = config.model.d_emb
    patch_size = config.model.patch_size
    num_layers = config.model.num_layers
    num_heads = config.model.num_heads
    dropout = config.model.dropout

    patch_dim = 3 * patch_size * patch_size

    encoder = ViTEncoder(
        patch_dim=patch_dim,
        d_model=d_emb,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)

    actor_critic = ActorCritic(
        d_emb=d_emb,
        num_actions=2
    ).to(device)

    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if 'x_encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['x_encoder_state_dict'])
    if 'actor_critic_state_dict' in checkpoint:
        actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])

    print(f"Checkpoint loaded (step {checkpoint.get('step', 'unknown')})")

    # Evaluate
    print(f"\nEvaluating for {args.num_episodes} episodes...")
    print("="*60)

    stats = evaluate_model(
        env=env,
        encoder=encoder,
        actor_critic=actor_critic,
        num_episodes=args.num_episodes,
        patch_size=patch_size,
        device=device,
        render=args.render,
        deterministic=not args.stochastic
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Episodes: {args.num_episodes}")
    print(f"Mean Return: {stats['mean_return']:.2f} ± {stats['std_return']:.2f}")
    print(f"Min/Max Return: {stats['min_return']:.1f} / {stats['max_return']:.1f}")
    print(f"Mean Length: {stats['mean_length']:.1f} ± {stats['std_length']:.1f}")
    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
