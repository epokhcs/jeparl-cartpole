"""Visualize embeddings from trained models.

This script loads trained models and visualizes:
- Embedding space structure (PCA, t-SNE)
- Per-dimension variance over time
- Collapse signatures
- Attention patterns

Usage:
    python scripts/visualize_embeddings.py --checkpoint results/config2/seed_42/checkpoints/step_100000.pt
"""

import sys
from pathlib import Path
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.checkpoint import CheckpointManager
from src.environment.cartpole_pixels import CartPolePixels
from src.environment.preprocessing import extract_patches
from src.models.vit_encoder import ViTEncoder


def collect_embeddings(env, encoder, device, num_episodes=10):
    """Collect embeddings from random rollouts.

    Args:
        env: Environment
        encoder: Trained encoder
        device: Device
        num_episodes: Number of episodes to collect

    Returns:
        Dictionary with embeddings and metadata
    """
    encoder.eval()

    embeddings_list = []
    rewards_list = []
    episode_ids = []

    with torch.no_grad():
        for episode_idx in range(num_episodes):
            obs, _ = env.reset()
            obs = torch.from_numpy(obs).float().to(device)

            episode_reward = 0.0
            done = False

            while not done:
                # Extract patches
                patches, positions = extract_patches(
                    obs.unsqueeze(0),
                    patch_size=16
                )

                # Get embedding
                emb = encoder(patches, positions)  # (1, d_emb)

                # Store
                embeddings_list.append(emb.squeeze(0).cpu().numpy())
                rewards_list.append(episode_reward)
                episode_ids.append(episode_idx)

                # Random action
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                obs = torch.from_numpy(obs).float().to(device)
                episode_reward += reward

    return {
        'embeddings': np.array(embeddings_list),
        'rewards': np.array(rewards_list),
        'episode_ids': np.array(episode_ids)
    }


def plot_embedding_space_pca(embeddings, episode_ids, save_path=None):
    """Plot 2D PCA projection of embeddings.

    Args:
        embeddings: (N, d_emb) array
        episode_ids: Episode labels for coloring
        save_path: Optional save path
    """
    # Compute PCA
    pca = PCA(n_components=2)
    emb_2d = pca.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=episode_ids,
        cmap='tab10',
        alpha=0.6,
        s=20
    )

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontsize=12)
    ax.set_title('Embedding Space (PCA)', fontsize=14)

    plt.colorbar(scatter, label='Episode ID')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")


def plot_embedding_space_tsne(embeddings, episode_ids, save_path=None):
    """Plot 2D t-SNE projection of embeddings.

    Args:
        embeddings: (N, d_emb) array
        episode_ids: Episode labels for coloring
        save_path: Optional save path
    """
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    scatter = ax.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=episode_ids,
        cmap='tab10',
        alpha=0.6,
        s=20
    )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('Embedding Space (t-SNE)', fontsize=14)

    plt.colorbar(scatter, label='Episode ID')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_variance_per_dimension(embeddings, save_path=None):
    """Plot variance for each embedding dimension.

    Args:
        embeddings: (N, d_emb) array
        save_path: Optional save path
    """
    # Compute variance per dimension
    var_per_dim = np.var(embeddings, axis=0)
    d_emb = len(var_per_dim)

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Bar plot
    ax1.bar(range(d_emb), var_per_dim, alpha=0.7, color='steelblue')
    ax1.set_xlabel('Dimension', fontsize=12)
    ax1.set_ylabel('Variance', fontsize=12)
    ax1.set_title('Variance per Embedding Dimension', fontsize=14)
    ax1.axhline(y=0.01, color='red', linestyle='--', label='Collapse threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Heatmap
    var_2d = var_per_dim.reshape(1, -1)
    sns.heatmap(var_2d, ax=ax2, cmap='viridis', cbar_kws={'label': 'Variance'})
    ax2.set_xlabel('Dimension', fontsize=12)
    ax2.set_yticks([])
    ax2.set_title('Variance Heatmap', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

    # Statistics
    print(f"\nVariance Statistics:")
    print(f"  Mean: {var_per_dim.mean():.6f}")
    print(f"  Std:  {var_per_dim.std():.6f}")
    print(f"  Min:  {var_per_dim.min():.6f}")
    print(f"  Max:  {var_per_dim.max():.6f}")
    print(f"  Collapsed dimensions (< 0.01): {(var_per_dim < 0.01).sum()}/{d_emb}")


def plot_embedding_distribution(embeddings, save_path=None):
    """Plot distribution of embedding values.

    Args:
        embeddings: (N, d_emb) array
        save_path: Optional save path
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Overall histogram
    axes[0, 0].hist(embeddings.flatten(), bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].set_xlabel('Embedding Value', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Overall Distribution', fontsize=14)
    axes[0, 0].grid(True, alpha=0.3)

    # Per-dimension mean
    mean_per_dim = embeddings.mean(axis=0)
    axes[0, 1].plot(mean_per_dim, marker='o', markersize=3, alpha=0.7)
    axes[0, 1].set_xlabel('Dimension', fontsize=12)
    axes[0, 1].set_ylabel('Mean Value', fontsize=12)
    axes[0, 1].set_title('Mean per Dimension', fontsize=14)
    axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(True, alpha=0.3)

    # Per-dimension std
    std_per_dim = embeddings.std(axis=0)
    axes[1, 0].plot(std_per_dim, marker='o', markersize=3, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('Dimension', fontsize=12)
    axes[1, 0].set_ylabel('Std Dev', fontsize=12)
    axes[1, 0].set_title('Standard Deviation per Dimension', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)

    # Correlation matrix (sample if too large)
    if embeddings.shape[1] <= 64:
        corr = np.corrcoef(embeddings.T)
        im = axes[1, 1].imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        axes[1, 1].set_xlabel('Dimension', fontsize=12)
        axes[1, 1].set_ylabel('Dimension', fontsize=12)
        axes[1, 1].set_title('Dimension Correlation', fontsize=14)
        plt.colorbar(im, ax=axes[1, 1])
    else:
        axes[1, 1].text(0.5, 0.5, 'Too many dimensions\nfor correlation plot',
                        ha='center', va='center', fontsize=12)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Visualize embeddings from trained models')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    parser.add_argument(
        '--num-episodes',
        type=int,
        default=10,
        help='Number of episodes to collect embeddings from'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/visualizations',
        help='Output directory for plots'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cuda/cpu/mps)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("EMBEDDING VISUALIZATION")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Output: {output_dir}")

    # Load checkpoint
    print("\nLoading checkpoint...")
    checkpoint_manager = CheckpointManager(checkpoint_dir=str(Path(args.checkpoint).parent))
    checkpoint_data = torch.load(args.checkpoint, map_location='cpu')

    # Create encoder
    device = torch.device(args.device)
    patch_dim = 3 * 16 * 16  # Assuming 16x16 patches
    d_emb = 64  # From checkpoint or config

    encoder = ViTEncoder(
        patch_dim=patch_dim,
        d_model=d_emb,
        num_layers=4,
        num_heads=4,
        dropout=0.1
    ).to(device)

    # Load weights
    encoder.load_state_dict(checkpoint_data['models']['x_encoder'])
    print(f"Loaded encoder with {sum(p.numel() for p in encoder.parameters())} parameters")

    # Create environment
    print("\nCreating environment...")
    env = CartPolePixels(
        render_size=(600, 400),
        pad_to=(608, 400),
        frame_stack=3,
        normalize=True
    )

    # Collect embeddings
    print(f"\nCollecting embeddings from {args.num_episodes} episodes...")
    data = collect_embeddings(env, encoder, device, num_episodes=args.num_episodes)

    embeddings = data['embeddings']
    episode_ids = data['episode_ids']

    print(f"Collected {len(embeddings)} embeddings")
    print(f"Embedding shape: {embeddings.shape}")

    # Create visualizations
    print("\nGenerating visualizations...")

    print("  1. PCA projection...")
    plot_embedding_space_pca(
        embeddings,
        episode_ids,
        save_path=output_dir / 'embedding_pca.png'
    )

    print("  2. t-SNE projection...")
    plot_embedding_space_tsne(
        embeddings,
        episode_ids,
        save_path=output_dir / 'embedding_tsne.png'
    )

    print("  3. Per-dimension variance...")
    plot_variance_per_dimension(
        embeddings,
        save_path=output_dir / 'variance_per_dimension.png'
    )

    print("  4. Distribution analysis...")
    plot_embedding_distribution(
        embeddings,
        save_path=output_dir / 'embedding_distribution.png'
    )

    env.close()

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
