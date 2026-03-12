"""Unit tests for neural network models."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.vit_encoder import (
    ViTEncoder,
    SpatioTemporalPositionalEncoding,
    TransformerBlock
)
from src.models.momentum_encoder import MomentumEncoder, MomentumScheduler
from src.models.predictor import Predictor, PredictorV2
from src.models.actor_critic import ActorCritic, SeparateActorCritic


class TestViTEncoder:
    """Tests for Vision Transformer encoder."""

    def test_initialization(self):
        """Test ViT encoder initialization."""
        encoder = ViTEncoder(
            patch_dim=768,
            d_model=64,
            num_layers=4,
            num_heads=4
        )
        assert encoder.d_model == 64
        assert encoder.num_layers == 4

    def test_forward_pass(self):
        """Test forward pass with correct shapes."""
        encoder = ViTEncoder(
            patch_dim=768,
            d_model=64,
            num_layers=4,
            num_heads=4
        )

        # Create dummy input
        B, N = 2, 2850  # Batch size, number of patches
        patches = torch.randn(B, N, 768)
        positions = torch.zeros(B, N, 3)
        positions[..., 0] = torch.arange(N) % 25  # i
        positions[..., 1] = torch.arange(N) % 38  # j
        positions[..., 2] = -2  # t

        # Forward pass
        output = encoder(patches, positions)

        assert output.shape == (B, 64)

    def test_return_all_tokens(self):
        """Test returning all tokens."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)

        B, N = 2, 100
        patches = torch.randn(B, N, 768)
        positions = torch.zeros(B, N, 3)

        output = encoder(patches, positions, return_all_tokens=True)

        # Should return [CLS] + all patches
        assert output.shape == (B, N + 1, 64)

    def test_positional_encoding(self):
        """Test spatio-temporal positional encoding."""
        pos_enc = SpatioTemporalPositionalEncoding(
            d_model=64,
            max_spatial_h=25,
            max_spatial_w=38,
            max_temporal=3
        )

        B, N = 2, 100
        positions = torch.zeros(B, N, 3)
        positions[..., 0] = torch.randint(0, 25, (B, N))
        positions[..., 1] = torch.randint(0, 38, (B, N))
        positions[..., 2] = torch.randint(-2, 1, (B, N))

        encoding = pos_enc(positions)

        assert encoding.shape == (B, N, 64)

    def test_transformer_block(self):
        """Test transformer block."""
        block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)

        B, N, D = 2, 100, 64
        x = torch.randn(B, N, D)

        output = block(x)

        assert output.shape == (B, N, D)


class TestMomentumEncoder:
    """Tests for momentum encoder."""

    def test_initialization(self):
        """Test momentum encoder initialization."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)
        momentum_enc = MomentumEncoder(encoder, momentum=0.99)

        assert momentum_enc.momentum == 0.99
        assert momentum_enc.target_encoder is not None

        # Check that target encoder parameters don't require gradients
        for param in momentum_enc.target_encoder.parameters():
            assert not param.requires_grad

    def test_momentum_update(self):
        """Test momentum update mechanism."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)
        momentum_enc = MomentumEncoder(encoder, momentum=0.99)

        # Get initial parameters
        initial_params = [
            p.clone() for p in momentum_enc.target_encoder.parameters()
        ]

        # Modify encoder parameters
        with torch.no_grad():
            for param in encoder.parameters():
                param.add_(torch.randn_like(param) * 0.1)

        # Update momentum encoder
        momentum_enc.update()

        # Check that target parameters changed
        updated_params = list(momentum_enc.target_encoder.parameters())
        for init_p, updated_p in zip(initial_params, updated_params):
            assert not torch.allclose(init_p, updated_p)

    def test_no_gradients_through_target(self):
        """Test that no gradients flow through target encoder."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)
        momentum_enc = MomentumEncoder(encoder, momentum=0.99)

        # Create dummy input
        B, N = 2, 100
        patches = torch.randn(B, N, 768, requires_grad=True)
        positions = torch.zeros(B, N, 3)

        # Forward through target encoder (no_grad context is inside forward)
        output = momentum_enc(patches, positions)

        # Output should not require gradients
        assert not output.requires_grad

        # Target encoder parameters should not have gradients
        for param in momentum_enc.target_encoder.parameters():
            assert param.grad is None

    def test_forward_pass(self):
        """Test forward pass through momentum encoder."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)
        momentum_enc = MomentumEncoder(encoder, momentum=0.99)

        B, N = 2, 100
        patches = torch.randn(B, N, 768)
        positions = torch.zeros(B, N, 3)

        output = momentum_enc(patches, positions)

        assert output.shape == (B, 64)

    def test_momentum_scheduler(self):
        """Test momentum scheduler."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)
        momentum_enc = MomentumEncoder(encoder, momentum=0.99)

        scheduler = MomentumScheduler(
            momentum_enc,
            initial_momentum=0.95,
            final_momentum=0.999,
            total_steps=1000
        )

        # Initial momentum
        assert scheduler.get_momentum() == 0.95

        # Step 500 times
        for _ in range(500):
            scheduler.step()

        # Should be approximately halfway
        mid_momentum = scheduler.get_momentum()
        assert 0.95 < mid_momentum < 0.999

        # Step to completion
        for _ in range(500):
            scheduler.step()

        # Should reach final momentum
        assert abs(scheduler.get_momentum() - 0.999) < 1e-6


class TestPredictor:
    """Tests for predictor network."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = Predictor(
            d_emb=64,
            hidden_dim=128,
            num_actions=2
        )
        assert predictor.d_emb == 64
        assert predictor.hidden_dim == 128

    def test_forward_with_single_embedding(self):
        """Test forward pass with single embedding."""
        predictor = Predictor(d_emb=64, num_actions=2)

        B = 4
        context_emb = torch.randn(B, 64)
        actions = torch.randint(0, 2, (B,))

        output = predictor(context_emb, actions)

        # Should output 3 target embeddings
        assert output.shape == (B, 3, 64)

    def test_forward_with_multiple_embeddings(self):
        """Test forward pass with multiple embeddings."""
        predictor = Predictor(d_emb=64, num_actions=2, num_context_frames=3)

        B = 4
        context_emb = torch.randn(B, 3, 64)
        actions = torch.randint(0, 2, (B,))

        output = predictor(context_emb, actions)

        assert output.shape == (B, 3, 64)

    def test_forward_with_onehot_actions(self):
        """Test forward pass with one-hot actions."""
        predictor = Predictor(d_emb=64, num_actions=2)

        B = 4
        context_emb = torch.randn(B, 64)
        actions = torch.nn.functional.one_hot(torch.randint(0, 2, (B,)), 2).float()

        output = predictor(context_emb, actions)

        assert output.shape == (B, 3, 64)

    def test_predictor_v2(self):
        """Test alternative predictor with attention."""
        predictor = PredictorV2(d_emb=64, num_actions=2)

        B = 4
        context_emb = torch.randn(B, 3, 64)
        actions = torch.randint(0, 2, (B,))

        output = predictor(context_emb, actions)

        assert output.shape == (B, 3, 64)


class TestActorCritic:
    """Tests for actor-critic network."""

    def test_initialization(self):
        """Test actor-critic initialization."""
        ac = ActorCritic(d_emb=64, num_actions=2)
        assert ac.d_emb == 64
        assert ac.num_actions == 2

    def test_forward_pass(self):
        """Test forward pass."""
        ac = ActorCritic(d_emb=64, num_actions=2)

        B = 4
        embeddings = torch.randn(B, 64)

        action_logits, values = ac(embeddings)

        assert action_logits.shape == (B, 2)
        assert values.shape == (B, 1)

    def test_get_action_and_value(self):
        """Test getting action and value."""
        ac = ActorCritic(d_emb=64, num_actions=2)

        B = 4
        embeddings = torch.randn(B, 64)

        action, log_prob, entropy, value = ac.get_action_and_value(embeddings)

        assert action.shape == (B,)
        assert log_prob.shape == (B,)
        assert entropy.shape == (B,)
        assert value.shape == (B, 1)

    def test_deterministic_action(self):
        """Test deterministic action selection."""
        ac = ActorCritic(d_emb=64, num_actions=2)

        embeddings = torch.randn(1, 64)

        # Get deterministic action multiple times
        actions = []
        for _ in range(5):
            action, _, _, _ = ac.get_action_and_value(embeddings, deterministic=True)
            actions.append(action.item())

        # Should always select same action
        assert len(set(actions)) == 1

    def test_evaluate_actions(self):
        """Test evaluating specific actions."""
        ac = ActorCritic(d_emb=64, num_actions=2)

        B = 4
        embeddings = torch.randn(B, 64)
        actions = torch.randint(0, 2, (B,))

        log_probs, entropy, values = ac.evaluate_actions(embeddings, actions)

        assert log_probs.shape == (B,)
        assert entropy.shape == (B,)
        assert values.shape == (B, 1)

    def test_with_hidden_layer(self):
        """Test actor-critic with hidden layer."""
        ac = ActorCritic(d_emb=64, num_actions=2, hidden_dim=128)

        B = 4
        embeddings = torch.randn(B, 64)

        action_logits, values = ac(embeddings)

        assert action_logits.shape == (B, 2)
        assert values.shape == (B, 1)

    def test_separate_actor_critic(self):
        """Test separate actor-critic architecture."""
        ac = SeparateActorCritic(d_emb=64, num_actions=2)

        B = 4
        embeddings = torch.randn(B, 64)

        action_logits, values = ac(embeddings)

        assert action_logits.shape == (B, 2)
        assert values.shape == (B, 1)


class TestGradientFlow:
    """Tests for gradient flow through models."""

    def test_gradient_flow_encoder_to_actor_critic(self):
        """Test gradients flow from actor-critic to encoder."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)
        ac = ActorCritic(d_emb=64, num_actions=2)

        # Create input
        B, N = 2, 100
        patches = torch.randn(B, N, 768)
        positions = torch.zeros(B, N, 3)

        # Forward pass
        embeddings = encoder(patches, positions)
        action_logits, values = ac(embeddings)

        # Compute loss
        loss = action_logits.mean() + values.mean()
        loss.backward()

        # Check that encoder has gradients
        has_gradients = False
        for param in encoder.parameters():
            if param.grad is not None:
                has_gradients = True
                break

        assert has_gradients, "No gradients in encoder"

    def test_no_gradient_through_momentum_encoder(self):
        """Test that no gradients flow through momentum encoder."""
        encoder = ViTEncoder(patch_dim=768, d_model=64)
        momentum_enc = MomentumEncoder(encoder, momentum=0.99)
        ac = ActorCritic(d_emb=64, num_actions=2)

        # Create input
        B, N = 2, 100
        patches = torch.randn(B, N, 768)
        positions = torch.zeros(B, N, 3)

        # Forward through momentum encoder
        target_embeddings = momentum_enc(patches, positions)

        # Forward through actor-critic
        action_logits, values = ac(target_embeddings)

        # Compute loss and backward
        loss = action_logits.mean() + values.mean()
        loss.backward()

        # Momentum encoder should have no gradients
        for param in momentum_enc.target_encoder.parameters():
            assert param.grad is None

    def test_gradient_flow_full_pipeline(self):
        """Test gradients in full JEPA pipeline."""
        # Create all components
        x_encoder = ViTEncoder(patch_dim=768, d_model=64)
        y_encoder_momentum = MomentumEncoder(x_encoder, momentum=0.99)
        predictor = Predictor(d_emb=64, num_actions=2, num_context_frames=1, num_target_frames=1)
        ac = ActorCritic(d_emb=64, num_actions=2)

        # Create inputs
        B, N = 2, 100
        patches = torch.randn(B, N, 768)
        positions = torch.zeros(B, N, 3)
        actions = torch.randint(0, 2, (B,))

        # Forward passes
        context_emb = x_encoder(patches, positions)  # (B, 64)
        target_emb = y_encoder_momentum(patches, positions)  # (B, 64)
        predicted_emb = predictor(context_emb, actions)  # Will be (B, 1, 64)

        # JEPA loss
        jepa_loss = torch.nn.functional.mse_loss(predicted_emb, target_emb.unsqueeze(1).detach())

        # RL losses
        action_logits, values = ac(context_emb)
        rl_loss = action_logits.mean() + values.mean()

        # Total loss
        total_loss = jepa_loss + rl_loss
        total_loss.backward()

        # Check gradients
        assert any(p.grad is not None for p in x_encoder.parameters()), "No gradients in x_encoder"
        assert any(p.grad is not None for p in predictor.parameters()), "No gradients in predictor"
        assert any(p.grad is not None for p in ac.parameters()), "No gradients in actor_critic"
        assert all(p.grad is None for p in y_encoder_momentum.target_encoder.parameters()), "Gradients in y_encoder!"


def test_integration():
    """Integration test: full forward pass through all models."""
    # Create models
    encoder = ViTEncoder(patch_dim=768, d_model=64, num_layers=2, num_heads=2)
    momentum_enc = MomentumEncoder(encoder, momentum=0.99)
    predictor = Predictor(d_emb=64, num_actions=2, num_context_frames=1, num_target_frames=1)
    ac = ActorCritic(d_emb=64, num_actions=2)

    # Create data
    B, N = 4, 2850
    patches = torch.randn(B, N, 768)
    positions = torch.zeros(B, N, 3)

    # Simulate temporal positions
    for i in range(N):
        positions[:, i, 0] = i % 25  # spatial i
        positions[:, i, 1] = i % 38  # spatial j
        positions[:, i, 2] = (i // 950) - 2  # temporal t

    actions = torch.randint(0, 2, (B,))

    # Forward passes
    context_emb = encoder(patches, positions)
    target_emb = momentum_enc(patches, positions)
    predicted_emb = predictor(context_emb, actions)
    action_logits, values = ac(context_emb)

    # Verify shapes
    assert context_emb.shape == (B, 64)
    assert target_emb.shape == (B, 64)
    assert predicted_emb.shape == (B, 1, 64)  # Updated to match num_target_frames=1
    assert action_logits.shape == (B, 2)
    assert values.shape == (B, 1)

    print("Integration test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
