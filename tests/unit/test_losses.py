"""Unit tests for loss functions."""

import pytest
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.losses.jepa_loss import JEPALoss, jepa_loss
from src.losses.ppo_loss import (
    PPOActorLoss, PPOCriticLoss,
    ppo_actor_loss, ppo_critic_loss,
    compute_gae
)
from src.losses.regularization_loss import (
    VarianceRegularizationLoss,
    variance_regularization_loss
)
from src.losses.combined_loss import CombinedLoss, compute_combined_loss


class TestJEPALoss:
    """Tests for JEPA loss."""

    def test_jepa_loss_basic(self):
        """Test basic JEPA loss computation."""
        B, N, D = 4, 3, 64
        predicted = torch.randn(B, N, D)
        target = torch.randn(B, N, D)

        loss = jepa_loss(predicted, target)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0  # MSE is always non-negative

    def test_jepa_loss_detaches_target(self):
        """Test that target is detached."""
        predicted = torch.randn(2, 64, requires_grad=True)
        target = torch.randn(2, 64, requires_grad=True)

        loss = jepa_loss(predicted, target)
        loss.backward()

        # Predicted should have gradients
        assert predicted.grad is not None

        # Target should not have gradients (was detached)
        assert target.grad is None

    def test_jepa_loss_module(self):
        """Test JEPA loss as module."""
        loss_fn = JEPALoss()

        predicted = torch.randn(4, 3, 64)
        target = torch.randn(4, 3, 64)

        loss = loss_fn(predicted, target)

        assert loss.shape == torch.Size([])


class TestPPOLoss:
    """Tests for PPO losses."""

    def test_ppo_actor_loss(self):
        """Test PPO actor loss."""
        B = 32
        log_probs = torch.randn(B)
        old_log_probs = torch.randn(B)
        advantages = torch.randn(B)

        loss = ppo_actor_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)

        assert loss.shape == torch.Size([])
        assert not torch.isnan(loss)

    def test_ppo_actor_loss_clipping(self):
        """Test that PPO clipping works."""
        B = 100
        # Create scenario where ratio is large
        log_probs = torch.ones(B) * 2.0
        old_log_probs = torch.zeros(B)
        advantages = torch.ones(B)

        loss = ppo_actor_loss(log_probs, old_log_probs, advantages, clip_epsilon=0.2)

        # Loss should be finite
        assert torch.isfinite(loss)

    def test_ppo_actor_loss_module(self):
        """Test PPO actor loss module."""
        loss_fn = PPOActorLoss(clip_epsilon=0.2, entropy_coef=0.01)

        B = 32
        log_probs = torch.randn(B)
        old_log_probs = torch.randn(B)
        advantages = torch.randn(B)
        entropy = torch.rand(B)

        loss, info = loss_fn(log_probs, old_log_probs, advantages, entropy)

        assert loss.shape == torch.Size([])
        assert 'policy_loss' in info
        assert 'entropy_loss' in info
        assert 'approx_kl' in info

    def test_ppo_critic_loss(self):
        """Test PPO critic loss."""
        B = 32
        values = torch.randn(B)
        returns = torch.randn(B)

        loss = ppo_critic_loss(values, returns)

        assert loss.shape == torch.Size([])
        assert loss.item() >= 0  # MSE is non-negative

    def test_ppo_critic_loss_module(self):
        """Test PPO critic loss module."""
        loss_fn = PPOCriticLoss(clip_value=False)

        B = 32
        values = torch.randn(B, 1)
        returns = torch.randn(B)

        loss, info = loss_fn(values, None, returns)

        assert loss.shape == torch.Size([])
        assert 'value_loss' in info
        assert 'explained_variance' in info

    def test_compute_gae(self):
        """Test GAE computation."""
        T = 100
        rewards = torch.ones(T)
        values = torch.zeros(T)
        dones = torch.zeros(T)
        next_value = torch.tensor(0.0)

        advantages, returns = compute_gae(
            rewards, values, dones, next_value,
            gamma=0.99, gae_lambda=0.95
        )

        assert advantages.shape == (T,)
        assert returns.shape == (T,)
        # With constant rewards and zero values, advantages should be positive
        assert advantages[0] > 0


class TestRegularizationLoss:
    """Tests for regularization loss."""

    def test_variance_regularization_basic(self):
        """Test basic variance regularization."""
        B, D = 32, 64
        embeddings = torch.randn(B, D)

        loss = variance_regularization_loss(embeddings, d_emb=D)

        assert loss.shape == torch.Size([])
        # Loss should be negative (encourages high variance)
        assert loss.item() <= 0

    def test_variance_regularization_high_variance(self):
        """Test with high variance embeddings."""
        B, D = 100, 64
        # High variance embeddings
        embeddings = torch.randn(B, D) * 10.0

        loss = variance_regularization_loss(embeddings, d_emb=D)

        # Loss should be capped at -1.0
        assert loss.item() >= -1.0

    def test_variance_regularization_low_variance(self):
        """Test with low variance embeddings (collapse)."""
        B, D = 100, 64
        # Low variance embeddings (collapsed)
        embeddings = torch.randn(B, D) * 0.01

        loss = variance_regularization_loss(embeddings, d_emb=D)

        # Loss should be close to zero (not capped)
        assert loss.item() > -0.1

    def test_variance_regularization_module(self):
        """Test variance regularization module."""
        loss_fn = VarianceRegularizationLoss(d_emb=64, min_variance=0.01)

        B, D = 32, 64
        embeddings = torch.randn(B, D)

        loss, stats = loss_fn(embeddings, return_stats=True)

        assert loss.shape == torch.Size([])
        assert 'avg_variance' in stats
        assert 'collapsed' in stats
        assert 'num_low_var_dims' in stats

    def test_collapse_detection(self):
        """Test collapse detection."""
        loss_fn = VarianceRegularizationLoss(d_emb=64, min_variance=0.01)

        # Create collapsed embeddings
        B, D = 100, 64
        collapsed_embeddings = torch.ones(B, D) * 0.5 + torch.randn(B, D) * 0.001

        loss, stats = loss_fn(collapsed_embeddings, return_stats=True)

        # Should detect collapse
        assert stats['collapsed'] == True
        assert stats['avg_variance'] < 0.01


class TestCombinedLoss:
    """Tests for combined loss with configuration switching."""

    def test_config1_baseline(self):
        """Test Config 1: No JEPA, only RL gradients."""
        loss_fn = CombinedLoss(config_type=1)

        assert not loss_fn.use_jepa
        assert loss_fn.use_rl_gradients
        assert not loss_fn.use_regularization

    def test_config2_best(self):
        """Test Config 2: JEPA + RL gradients."""
        loss_fn = CombinedLoss(config_type=2)

        assert loss_fn.use_jepa
        assert loss_fn.use_rl_gradients
        assert not loss_fn.use_regularization

    def test_config3_collapse(self):
        """Test Config 3: JEPA without RL gradients."""
        loss_fn = CombinedLoss(config_type=3)

        assert loss_fn.use_jepa
        assert not loss_fn.use_rl_gradients
        assert not loss_fn.use_regularization

    def test_config4_regularization(self):
        """Test Config 4: JEPA + regularization."""
        loss_fn = CombinedLoss(config_type=4)

        assert loss_fn.use_jepa
        assert not loss_fn.use_rl_gradients
        assert loss_fn.use_regularization

    def test_combined_loss_config2(self):
        """Test combined loss for config 2 (full)."""
        loss_fn = CombinedLoss(config_type=2)

        B = 32
        # Create dummy inputs
        predicted_emb = torch.randn(B, 3, 64)
        target_emb = torch.randn(B, 3, 64)
        context_emb = torch.randn(B, 64)
        log_probs = torch.randn(B)
        old_log_probs = torch.randn(B)
        advantages = torch.randn(B)
        entropy = torch.rand(B)
        values = torch.randn(B)
        returns = torch.randn(B)

        loss, loss_dict = loss_fn(
            predicted_embeddings=predicted_emb,
            target_embeddings=target_emb,
            context_embeddings=context_emb,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            entropy=entropy,
            values=values,
            returns=returns
        )

        assert loss.shape == torch.Size([])
        assert 'jepa_loss' in loss_dict
        assert 'actor_loss' in loss_dict
        assert 'critic_loss' in loss_dict
        assert loss_dict['jepa_loss'] > 0  # Should have JEPA loss
        assert loss_dict['reg_loss'] == 0  # No regularization in config 2

    def test_combined_loss_config3(self):
        """Test combined loss for config 3 (no RL gradients)."""
        loss_fn = CombinedLoss(config_type=3)

        B = 32
        predicted_emb = torch.randn(B, 3, 64, requires_grad=True)
        target_emb = torch.randn(B, 3, 64, requires_grad=True)
        context_emb = torch.randn(B, 64, requires_grad=True)
        log_probs = torch.randn(B, requires_grad=True)
        old_log_probs = torch.randn(B)
        advantages = torch.randn(B)
        values = torch.randn(B, requires_grad=True)
        returns = torch.randn(B)

        loss, loss_dict = loss_fn(
            predicted_embeddings=predicted_emb,
            target_embeddings=target_emb,
            context_embeddings=context_emb,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            returns=returns
        )

        # Backward pass
        loss.backward()

        # JEPA components should have gradients
        assert predicted_emb.grad is not None

        # RL losses are detached in config 3, so gradients limited
        # (This is the key feature of config 3)

    def test_combined_loss_config4(self):
        """Test combined loss for config 4 (with regularization)."""
        loss_fn = CombinedLoss(config_type=4)

        B = 32
        predicted_emb = torch.randn(B, 3, 64)
        target_emb = torch.randn(B, 3, 64)
        context_emb = torch.randn(B, 64)

        loss, loss_dict = loss_fn(
            predicted_embeddings=predicted_emb,
            target_embeddings=target_emb,
            context_embeddings=context_emb
        )

        assert loss_dict['reg_loss'] != 0  # Should have regularization
        assert 'avg_variance' in loss_dict

    def test_functional_interface(self):
        """Test functional interface."""
        B = 32
        predicted_emb = torch.randn(B, 3, 64)
        target_emb = torch.randn(B, 3, 64)

        loss, loss_dict = compute_combined_loss(
            config_type=2,
            predicted_embeddings=predicted_emb,
            target_embeddings=target_emb
        )

        assert loss.shape == torch.Size([])
        assert 'total_loss' in loss_dict


class TestGradientFlowInLosses:
    """Test gradient flow control in losses."""

    def test_gradient_flow_config2(self):
        """Test that gradients flow in config 2."""
        loss_fn = CombinedLoss(config_type=2)

        B = 16
        predicted_emb = torch.randn(B, 3, 64, requires_grad=True)
        target_emb = torch.randn(B, 3, 64)
        context_emb = torch.randn(B, 64, requires_grad=True)
        log_probs = torch.randn(B, requires_grad=True)
        old_log_probs = torch.randn(B)
        advantages = torch.randn(B)
        values = torch.randn(B, requires_grad=True)
        returns = torch.randn(B)

        loss, _ = loss_fn(
            predicted_embeddings=predicted_emb,
            target_embeddings=target_emb,
            context_embeddings=context_emb,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            returns=returns
        )

        loss.backward()

        # JEPA and RL components should have gradients in config 2
        assert predicted_emb.grad is not None
        assert log_probs.grad is not None
        assert values.grad is not None
        # Note: context_emb only gets gradients if used in regularization (config 4)
        # or if it's connected to predicted_emb (which happens in actual training)

    def test_no_gradient_through_target(self):
        """Test that no gradients flow through target embeddings."""
        loss_fn = CombinedLoss(config_type=2)

        predicted_emb = torch.randn(4, 3, 64, requires_grad=True)
        target_emb = torch.randn(4, 3, 64, requires_grad=True)

        loss, _ = loss_fn(
            predicted_embeddings=predicted_emb,
            target_embeddings=target_emb
        )

        loss.backward()

        # Target should not have gradients (detached in JEPA loss)
        assert target_emb.grad is None


def test_integration():
    """Integration test: all losses together."""
    # Test all 4 configurations
    for config_type in [1, 2, 3, 4]:
        loss_fn = CombinedLoss(config_type=config_type)

        B = 16
        predicted_emb = torch.randn(B, 3, 64) if loss_fn.use_jepa else None
        target_emb = torch.randn(B, 3, 64) if loss_fn.use_jepa else None
        context_emb = torch.randn(B, 64)
        log_probs = torch.randn(B)
        old_log_probs = torch.randn(B)
        advantages = torch.randn(B)
        values = torch.randn(B)
        returns = torch.randn(B)

        loss, loss_dict = loss_fn(
            predicted_embeddings=predicted_emb,
            target_embeddings=target_emb,
            context_embeddings=context_emb,
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages,
            values=values,
            returns=returns
        )

        # Should compute without errors
        assert loss.shape == torch.Size([])
        assert 'total_loss' in loss_dict
        assert loss_dict['config_type'] == config_type

        print(f"Config {config_type}: {loss_fn.get_config_description()}")
        print(f"  Total loss: {loss.item():.4f}")
        print(f"  Components: {loss_dict}")

    print("\nIntegration test passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
