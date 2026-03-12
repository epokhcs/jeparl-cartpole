"""Training script for JEPA-RL.

Run with:
    python scripts/train.py --config configs/experiments/config2.yaml
"""

import sys
import argparse
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import Logger
from src.utils.checkpoint import CheckpointManager
from src.environment.cartpole_pixels import CartPolePixels
from src.models.vit_encoder import ViTEncoder
from src.models.momentum_encoder import MomentumEncoder
from src.models.predictor import Predictor
from src.models.actor_critic import ActorCritic
from src.losses.combined_loss import CombinedLoss
from src.training.trainer import JEPATrainer


def create_models(config, device):
    """Create all models.

    Args:
        config: Configuration object
        device: Device for models

    Returns:
        Dictionary with all models
    """
    # Get model config
    model_config = config.model
    d_emb = model_config.d_emb
    patch_size = model_config.patch_size
    num_layers = model_config.num_layers
    num_heads = model_config.num_heads
    dropout = model_config.dropout
    momentum = model_config.momentum

    # Predictor config
    predictor_config = config.predictor
    hidden_dim = predictor_config.hidden_dim

    # Calculate patch dimension (3 channels * patch_size^2)
    patch_dim = 3 * patch_size * patch_size

    # Create X-encoder (context encoder)
    x_encoder = ViTEncoder(
        patch_dim=patch_dim,
        d_model=d_emb,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout
    ).to(device)

    # Create Y-encoder with momentum (target encoder)
    y_encoder_momentum = MomentumEncoder(
        encoder=x_encoder,
        momentum=momentum
    )
    y_encoder_momentum.to(device)

    # Create predictor
    predictor = Predictor(
        d_emb=d_emb,
        hidden_dim=hidden_dim,
        num_actions=2,  # CartPole has 2 actions
        num_context_frames=1,  # Using single [CLS] token
        num_target_frames=1
    ).to(device)

    # Create actor-critic
    actor_critic = ActorCritic(
        d_emb=d_emb,
        num_actions=2
    ).to(device)

    return {
        'x_encoder': x_encoder,
        'y_encoder_momentum': y_encoder_momentum,
        'predictor': predictor,
        'actor_critic': actor_critic
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train JEPA-RL on CartPole')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device (cuda/cpu/mps, overrides config)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for logs and checkpoints'
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Override config with command-line arguments
    if args.seed is not None:
        config.experiment.seed = args.seed

    if args.device is not None:
        config.experiment.device = args.device

    # Set random seed
    seed = config.experiment.seed
    print(f"Setting random seed: {seed}")
    set_seed(seed)

    # Determine device
    if hasattr(config.experiment, 'device'):
        device = config.experiment.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)
    print(f"Using device: {device}")

    # Create output directory
    output_dir = Path(args.output_dir) / config.experiment.name / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

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
    models = create_models(config, device)

    # Create combined loss
    print(f"Creating loss (Configuration {config.configuration.type})...")
    combined_loss = CombinedLoss(
        config_type=config.configuration.type,
        jepa_weight=config.losses.jepa_weight,
        actor_weight=config.losses.actor_weight,
        critic_weight=config.losses.critic_weight,
        reg_weight=config.losses.reg_weight,
        entropy_coef=config.training.entropy_coef,
        clip_epsilon=config.training.clip_epsilon,
        d_emb=config.model.d_emb
    )
    print(f"  {combined_loss.get_config_description()}")

    # Create optimizer
    params = (
        list(models['x_encoder'].parameters()) +
        list(models['predictor'].parameters()) +
        list(models['actor_critic'].parameters())
    )
    optimizer = torch.optim.Adam(params, lr=config.training.learning_rate)

    # Create logger
    print("Setting up logging...")
    logger = Logger(
        log_dir=str(output_dir / "logs"),
        use_tensorboard=config.logging.use_tensorboard,
        use_wandb=config.logging.use_wandb,
        wandb_project=config.logging.wandb_project if config.logging.use_wandb else None,
        wandb_entity=config.logging.wandb_entity if config.logging.use_wandb else None,
        config=config.to_dict() if hasattr(config, 'to_dict') else None
    )

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=str(output_dir / "checkpoints")
    )

    # Create trainer
    print("Creating trainer...")
    trainer = JEPATrainer(
        env=env,
        x_encoder=models['x_encoder'],
        y_encoder_momentum=models['y_encoder_momentum'],
        predictor=models['predictor'],
        actor_critic=models['actor_critic'],
        combined_loss=combined_loss,
        optimizer=optimizer,
        config=config.to_dict() if hasattr(config, 'to_dict') else config._config,
        logger=logger,
        checkpoint_manager=checkpoint_manager,
        device=device
    )

    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    finally:
        # Cleanup
        logger.close()
        env.close()

    print("\nTraining finished!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
