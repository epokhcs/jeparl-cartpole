"""Smoke test for JEPA-RL training system.

Quick test to verify all components work together.
Runs a very short training session to catch any bugs.
"""

import sys
from pathlib import Path
import torch
import tempfile

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.environment.cartpole_pixels import CartPolePixels
from src.models.vit_encoder import ViTEncoder
from src.models.momentum_encoder import MomentumEncoder
from src.models.predictor import Predictor
from src.models.actor_critic import ActorCritic
from src.losses.combined_loss import CombinedLoss
from src.training.trainer import JEPATrainer


def smoke_test():
    """Run smoke test."""
    print("="*60)
    print("JEPA-RL SMOKE TEST")
    print("="*60)

    # Set seed
    set_seed(42)
    device = torch.device('cpu')
    print(f"Device: {device}")

    # Load minimal config
    print("\n1. Loading configuration...")
    config = load_config("configs/base_config.yaml")

    # Override for quick test
    config.training.total_steps = 100  # Very short
    config.training.rollout_steps = 32  # Small rollout
    config.training.mini_batch_size = 16
    config.logging.log_interval = 50

    print(f"   Training steps: {config.training.total_steps}")
    print(f"   Rollout steps: {config.training.rollout_steps}")

    # Create environment
    print("\n2. Creating environment...")
    env = CartPolePixels(
        render_size=(600, 400),
        pad_to=(608, 400),
        frame_stack=3,
        normalize=True
    )
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space}")

    # Create models
    print("\n3. Creating models...")
    d_emb = 64
    patch_size = 16
    patch_dim = 3 * patch_size * patch_size

    x_encoder = ViTEncoder(
        patch_dim=patch_dim,
        d_model=d_emb,
        num_layers=2,  # Smaller for speed
        num_heads=2,
        dropout=0.1
    ).to(device)
    print(f"   X-encoder: {sum(p.numel() for p in x_encoder.parameters())} params")

    y_encoder_momentum = MomentumEncoder(x_encoder, momentum=0.99)
    y_encoder_momentum.to(device)
    print(f"   Y-encoder: Momentum wrapper")

    predictor = Predictor(
        d_emb=d_emb,
        hidden_dim=128,
        num_actions=2,
        num_context_frames=1,
        num_target_frames=1
    ).to(device)
    print(f"   Predictor: {sum(p.numel() for p in predictor.parameters())} params")

    actor_critic = ActorCritic(d_emb=d_emb, num_actions=2).to(device)
    print(f"   Actor-Critic: {sum(p.numel() for p in actor_critic.parameters())} params")

    # Test all 4 configurations
    print("\n4. Testing all configurations...")

    for config_type in [1, 2, 3, 4]:
        print(f"\n   Testing Config {config_type}...")

        combined_loss = CombinedLoss(
            config_type=config_type,
            jepa_weight=1.0,
            actor_weight=1.0,
            critic_weight=0.5,
            reg_weight=0.1,
            d_emb=d_emb
        )
        print(f"   - {combined_loss.get_config_description()}")

        # Create optimizer
        params = (
            list(x_encoder.parameters()) +
            list(predictor.parameters()) +
            list(actor_critic.parameters())
        )
        optimizer = torch.optim.Adam(params, lr=3e-4)

        # Create trainer with temporary output
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = JEPATrainer(
                env=env,
                x_encoder=x_encoder,
                y_encoder_momentum=y_encoder_momentum,
                predictor=predictor,
                actor_critic=actor_critic,
                combined_loss=combined_loss,
                optimizer=optimizer,
                config=config._config,
                logger=None,  # No logging for smoke test
                checkpoint_manager=None,
                device=device
            )

            # Override total steps for quick test
            trainer.total_steps = 100
            trainer.log_interval = 50

            try:
                print(f"   - Training for {trainer.total_steps} steps...")
                trainer.train()
                print(f"   - ✓ Config {config_type} SUCCESS")
            except Exception as e:
                print(f"   - ✗ Config {config_type} FAILED: {e}")
                raise

    env.close()

    print("\n" + "="*60)
    print("SMOKE TEST PASSED! ✓")
    print("="*60)
    print("\nAll configurations work correctly.")
    print("The training system is ready for full experiments.")
    print("\nNext steps:")
    print("1. Run single config: python scripts/train.py --config configs/experiments/config2.yaml")
    print("2. Run all experiments: python scripts/run_experiments.py --num-seeds 5")


if __name__ == "__main__":
    try:
        smoke_test()
    except Exception as e:
        print(f"\n✗ SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
