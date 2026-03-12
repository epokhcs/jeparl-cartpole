# Phase 4 Complete: Loss Functions ✓

## Summary

Phase 4 has been successfully implemented and tested. All loss functions for JEPA-RL training are now complete, including the critical configuration-dependent combined loss that enables the 4 experimental setups from the paper.

## Components Implemented

### 1. JEPA Loss ([src/losses/jepa_loss.py](../src/losses/jepa_loss.py))

**L2 distance between predicted and target embeddings**:

```python
L_JEPA = ||s_y^pred - s_y||^2
```

**Key Features**:
- MSE loss between predictor output and target encoder output
- **Critical**: Target embeddings always detached (no gradients)
- Support for both per-sample and batch reduction
- Module and functional interfaces

**Additional Variants**:
- `ContrastiveJEPALoss`: Contrastive learning variant (experimental)
- `AdaptiveJEPALoss`: Learnable per-dimension/frame weighting (experimental)

**Tested**: ✓ All JEPA loss tests pass

### 2. PPO Losses ([src/losses/ppo_loss.py](../src/losses/ppo_loss.py))

**PPO Actor Loss** (Clipped Surrogate Objective):
```python
L_actor = -E[min(ratio × A, clip(ratio, 1-ε, 1+ε) × A)]
L_entropy = -E[H(π)]
```

**PPO Critic Loss** (Value Function):
```python
L_critic = MSE(V, R)
```

**Key Features**:
- Clipped probability ratio (ε = 0.2)
- Entropy bonus for exploration
- Advantage normalization
- Optional value function clipping
- Explained variance monitoring
- KL divergence approximation

**GAE Computation**:
- Generalized Advantage Estimation
- Configurable γ (gamma) and λ (lambda)
- Efficient reverse-order computation

**Tested**: ✓ All PPO loss tests pass

### 3. Regularization Loss ([src/losses/regularization_loss.py](../src/losses/regularization_loss.py))

**Variance-based Collapse Prevention** (as per paper):
```python
L_reg = -min(1, (1/d_emb) * Σ Var(s_x)_i)
```

**Purpose**: Encourages embeddings to use full dimensionality

**Key Features**:
- Computes variance per dimension across batch
- Negative loss to maximize variance (capped at -1.0)
- Collapse detection with configurable threshold
- Detailed statistics: min/max/std of variances, number of collapsed dims

**Monitoring**:
- Average variance tracking
- Per-dimension variance analysis
- Collapse flag (variance < threshold)

**Additional Variants**:
- `StdRegularizationLoss`: Standard deviation targeting
- `VICRegLoss`: VICReg-style (Variance-Invariance-Covariance)

**Tested**: ✓ All regularization loss tests pass, including collapse detection

### 4. Combined Loss ([src/losses/combined_loss.py](../src/losses/combined_loss.py))

**Critical Implementation** - Configuration-dependent loss composition:

#### Four Experimental Configurations:

**Config 1: Baseline** (J_hat, ∇, R_hat)
```python
use_jepa = False
use_rl_gradients = True
use_regularization = False

L = L_actor + L_critic
```
- No JEPA loss, only RL
- Gradients flow to encoder from RL losses
- Expected: Limited performance

**Config 2: Best** (J, ∇, R_hat)
```python
use_jepa = True
use_rl_gradients = True
use_regularization = False

L = L_JEPA + L_actor + L_critic
```
- JEPA + RL losses
- Gradients flow from both JEPA and RL
- Expected: **Best performance, no collapse**

**Config 3: Collapse** (J, ∇_hat, R_hat)
```python
use_jepa = True
use_rl_gradients = False
use_regularization = False

L = L_JEPA + L_actor.detach() + L_critic.detach()
```
- JEPA loss only (RL losses detached)
- No gradients to encoder from RL
- Expected: **Embedding collapse (variance → 0)**

**Config 4: Regularization** (J, ∇_hat, R)
```python
use_jepa = True
use_rl_gradients = False
use_regularization = True

L = L_JEPA + L_actor.detach() + L_critic.detach() + L_reg
```
- JEPA + regularization, RL detached
- No RL gradients, but regularization prevents collapse
- Expected: **Slower learning, but no collapse**

**Key Features**:
- Single class handles all 4 configurations
- Automatic gradient flow control
- Comprehensive loss tracking
- Weight configuration for each component
- Human-readable configuration descriptions

**Tested**: ✓ All configuration switching tests pass

## Test Results

### Comprehensive Testing: 25/25 Loss Tests Passing ✅

```
tests/unit/test_losses.py::TestJEPALoss::test_jepa_loss_basic PASSED
tests/unit/test_losses.py::TestJEPALoss::test_jepa_loss_detaches_target PASSED
tests/unit/test_losses.py::TestJEPALoss::test_jepa_loss_module PASSED
tests/unit/test_losses.py::TestPPOLoss::test_ppo_actor_loss PASSED
tests/unit/test_losses.py::TestPPOLoss::test_ppo_actor_loss_clipping PASSED
tests/unit/test_losses.py::TestPPOLoss::test_ppo_actor_loss_module PASSED
tests/unit/test_losses.py::TestPPOLoss::test_ppo_critic_loss PASSED
tests/unit/test_losses.py::TestPPOLoss::test_ppo_critic_loss_module PASSED
tests/unit/test_losses.py::TestPPOLoss::test_compute_gae PASSED
tests/unit/test_losses.py::TestRegularizationLoss::test_variance_regularization_basic PASSED
tests/unit/test_losses.py::TestRegularizationLoss::test_variance_regularization_high_variance PASSED
tests/unit/test_losses.py::TestRegularizationLoss::test_variance_regularization_low_variance PASSED
tests/unit/test_losses.py::TestRegularizationLoss::test_variance_regularization_module PASSED
tests/unit/test_losses.py::TestRegularizationLoss::test_collapse_detection PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_config1_baseline PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_config2_best PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_config3_collapse PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_config4_regularization PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_combined_loss_config2 PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_combined_loss_config3 PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_combined_loss_config4 PASSED
tests/unit/test_losses.py::TestCombinedLoss::test_functional_interface PASSED
tests/unit/test_losses.py::TestGradientFlowInLosses::test_gradient_flow_config2 PASSED
tests/unit/test_losses.py::TestGradientFlowInLosses::test_no_gradient_through_target PASSED
tests/unit/test_losses.py::test_integration PASSED

======================== 25 passed in 0.75s ==============================
```

### Overall Test Status: 65/65 Tests Passing ✅

- Environment tests: 14/14 ✓
- Model tests: 26/26 ✓
- Loss tests: 25/25 ✓

## Technical Specifications

### Loss Weights (Default Configuration)

```python
jepa_weight = 1.0      # JEPA reconstruction loss
actor_weight = 1.0     # Policy gradient loss
critic_weight = 0.5    # Value function loss
reg_weight = 0.1       # Variance regularization
entropy_coef = 0.01    # Entropy bonus
clip_epsilon = 0.2     # PPO clipping parameter
```

### PPO Hyperparameters

```python
gamma = 0.99           # Discount factor
gae_lambda = 0.95      # GAE lambda
clip_epsilon = 0.2     # Probability ratio clipping
entropy_coef = 0.01    # Exploration bonus
max_grad_norm = 0.5    # Gradient clipping
```

### Collapse Detection

```python
variance_threshold = 0.01  # Below this indicates collapse
d_emb = 64                 # Embedding dimension
```

## Usage Examples

### Basic Usage - Config 2 (Best)

```python
from src.losses.combined_loss import CombinedLoss

# Create combined loss for best configuration
loss_fn = CombinedLoss(config_type=2)

# Forward pass through models
context_emb = x_encoder(patches, positions)
target_emb = y_encoder_momentum(patches, positions)
predicted_emb = predictor(context_emb, actions)
action_logits, values = actor_critic(context_emb)

# Compute action probabilities and sample
probs = torch.softmax(action_logits, dim=-1)
dist = torch.distributions.Categorical(probs)
actions = dist.sample()
log_probs = dist.log_prob(actions)
entropy = dist.entropy()

# Compute returns and advantages
advantages, returns = compute_gae(rewards, values, dones, next_value)

# Combined loss
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

# Backward and optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Update momentum encoder
y_encoder_momentum.update()
```

### Configuration Switching

```python
# Test all 4 configurations
for config_type in [1, 2, 3, 4]:
    loss_fn = CombinedLoss(config_type=config_type)
    print(loss_fn.get_config_description())

    # Train with this configuration
    loss, loss_dict = loss_fn(...)
```

### Monitoring Collapse

```python
from src.losses.regularization_loss import VarianceRegularizationLoss

reg_loss_fn = VarianceRegularizationLoss(d_emb=64, min_variance=0.01)

loss, stats = reg_loss_fn(context_embeddings, return_stats=True)

if stats['collapsed']:
    print(f"WARNING: Embedding collapse detected!")
    print(f"  Average variance: {stats['avg_variance']:.6f}")
    print(f"  Low variance dims: {stats['num_low_var_dims']}/64")
```

## Verification Results

✅ **All loss tests pass** (25/25)
✅ **Configuration switching verified** for all 4 configs
✅ **Gradient flow control** working correctly
✅ **Target embeddings** always detached
✅ **PPO clipping** functioning properly
✅ **GAE computation** matches reference
✅ **Collapse detection** working with test data
✅ **Integration test** passes with all configs

## Key Implementation Details

### Gradient Flow Control

The critical distinction between configurations:

**Config 2** (Gradients flow):
```python
loss = jepa_loss + actor_loss + critic_loss
loss.backward()  # Gradients flow to encoder from all losses
```

**Config 3** (Gradients stopped):
```python
loss = jepa_loss + actor_loss.detach() + critic_loss.detach()
loss.backward()  # Only JEPA gradients flow to encoder
```

### Target Detachment

Always detach target embeddings in JEPA loss:
```python
def jepa_loss(predicted, target):
    target = target.detach()  # No gradients through Y-encoder
    return F.mse_loss(predicted, target)
```

### Advantage Normalization

Standard practice in PPO:
```python
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
```

## Next Steps: Phase 5

Phase 5 will implement the training pipeline:

1. **Rollout Buffer** (rollout_buffer.py)
   - Store transitions with frame sequences
   - Compute GAE
   - Handle episode boundaries

2. **Trainer** (trainer.py)
   - Main training loop
   - Rollout collection
   - Model updates
   - Momentum encoder updates

3. **Embedding Monitor** (embedding_monitor.py)
   - Track variance over time
   - Detect collapse
   - Visualization

4. **Training Scripts**
   - train.py: Single configuration training
   - run_experiments.py: All 4 configs × 5 seeds

## Files Created

### Source Code
- [src/losses/jepa_loss.py](../src/losses/jepa_loss.py)
- [src/losses/ppo_loss.py](../src/losses/ppo_loss.py)
- [src/losses/regularization_loss.py](../src/losses/regularization_loss.py)
- [src/losses/combined_loss.py](../src/losses/combined_loss.py)

### Tests
- [tests/unit/test_losses.py](../tests/unit/test_losses.py)

---

**Status**: Phase 4 Complete ✓
**Tests**: 65/65 Passing ✅ (25 loss + 26 model + 14 environment)
**Ready for**: Phase 5 - Training Pipeline
**Total Progress**: 4/6 Phases Complete (67%)
