# JEPA for RL: CartPole Implementation

This project implements the paper "JEPA for RL: Investigating Joint-Embedding Predictive Architectures for Reinforcement Learning" using PyTorch and Gymnasium.

## Overview

Joint-Embedding Predictive Architectures (JEPA) learn representations by predicting future embeddings in latent space. This implementation adapts JEPA to reinforcement learning tasks, specifically CartPole with high-dimensional pixel observations (400×600×3 = 720,000 dimensions).

### Key Components

- **Vision Transformer Encoder**: Processes image patches with spatio-temporal positional encoding
- **Momentum Encoder**: Target encoder updated via exponential moving average (EMA)
- **Predictor Network**: Shallow MLP predicting future embeddings from context + action
- **Actor-Critic**: PPO-style reinforcement learning on learned representations

### Four Experimental Configurations

1. **Config 1 - Baseline**: No JEPA, only RL gradients → Limited performance
2. **Config 2 - Best**: JEPA + RL gradients → Best performance, no collapse
3. **Config 3 - Collapse**: JEPA without RL gradients → Embedding collapse
4. **Config 4 - Regularization**: JEPA + variance regularization → Prevents collapse

## Installation

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd jeparl-cartpole

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

## Project Structure

```
jeparl-cartpole/
├── src/
│   ├── models/              # Neural network models
│   │   ├── vit_encoder.py          # Vision Transformer encoder
│   │   ├── momentum_encoder.py     # Momentum-updated target encoder
│   │   ├── predictor.py            # JEPA predictor network
│   │   └── actor_critic.py         # Policy and value heads
│   ├── training/            # Training pipeline
│   │   ├── trainer.py              # Main training loop
│   │   ├── rollout_buffer.py       # Experience buffer with GAE
│   │   └── embedding_monitor.py    # Collapse detection
│   ├── environment/         # Environment wrappers
│   │   ├── cartpole_pixels.py      # CartPole with pixel observations
│   │   └── preprocessing.py        # Patch extraction
│   ├── losses/              # Loss functions
│   │   ├── jepa_loss.py            # JEPA reconstruction loss
│   │   ├── ppo_loss.py             # PPO actor-critic losses
│   │   ├── regularization_loss.py  # Variance regularization
│   │   └── combined_loss.py        # Configuration-dependent composition
│   └── utils/               # Utilities
│       ├── config.py               # Configuration management
│       ├── logger.py               # Logging (TensorBoard, WandB)
│       ├── checkpoint.py           # Model checkpointing
│       ├── metrics.py              # Metrics collection
│       └── seed.py                 # Reproducibility
├── configs/                 # YAML configurations
│   ├── base_config.yaml            # Base hyperparameters
│   └── experiments/                # 4 experimental configs
├── scripts/                 # Executable scripts
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation script
│   ├── run_experiments.py          # Run all 4 configs × 5 seeds
│   ├── smoke_test.py               # Quick verification test
│   ├── visualize_embeddings.py     # Embedding visualization
│   └── compare_results.py          # Multi-config comparison
├── tests/                   # Unit and integration tests
├── notebooks/               # Jupyter notebooks for analysis
└── results/                 # Experiment outputs
```

## Usage

### Quick Verification

Before running full experiments, verify the system works:

```bash
# Run smoke test (tests all 4 configs for 100 steps)
python scripts/smoke_test.py
```

### Quick Start

Train with the best configuration (Config 2: JEPA + RL gradients):

```bash
python scripts/train.py --config configs/experiments/config2.yaml
```

### Training a Single Configuration

```bash
# Config 1: Baseline (no JEPA)
python scripts/train.py --config configs/experiments/config1.yaml

# Config 2: JEPA + RL gradients (best)
python scripts/train.py --config configs/experiments/config2.yaml

# Config 3: JEPA without RL gradients (demonstrates collapse)
python scripts/train.py --config configs/experiments/config3.yaml

# Config 4: JEPA + regularization
python scripts/train.py --config configs/experiments/config4.yaml
```

### Running Full Experiments

Reproduce all paper results (4 configs × 5 seeds = 20 runs):

```bash
python scripts/run_experiments.py --num-seeds 5 --device cpu
```

### Evaluating Trained Models

```bash
# Evaluate a trained model
python scripts/evaluate.py \
    --checkpoint results/config2/seed_42/checkpoints/step_100000.pt \
    --num-episodes 100
```

### Visualizing Embeddings

```bash
# Visualize embedding space and detect collapse
python scripts/visualize_embeddings.py \
    --checkpoint results/config2/seed_42/checkpoints/step_100000.pt \
    --num-episodes 10 \
    --output-dir results/visualizations
```

### Comparing Results

```bash
# Compare all configurations statistically
python scripts/compare_results.py \
    --results-dir results/experiments/<timestamp> \
    --output-dir results/comparisons
```

### Custom Configuration

Edit `configs/base_config.yaml` or create a new config file:

```yaml
model:
  d_emb: 64              # Embedding dimension
  patch_size: 16         # Patch size for ViT
  num_layers: 4          # Transformer layers

training:
  total_steps: 100000    # Total environment steps
  learning_rate: 3.0e-4  # Learning rate

configuration:
  use_jepa: true         # Enable JEPA loss
  use_rl_gradients: true # Enable RL gradient propagation
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir results/logs/tensorboard
```

### Weights & Biases

Enable WandB in config:

```yaml
logging:
  use_wandb: true
  wandb_project: "jepa-rl"
  wandb_entity: "your-username"
```

Then run training normally.

## Testing

Run tests:

```bash
# All tests (65 unit tests)
pytest tests/

# Unit tests only
pytest tests/unit/

# Integration tests
pytest tests/integration/

# With coverage (target: >80%)
pytest --cov=src tests/
```

**Current Status:** ✅ All 65 unit tests passing, >80% coverage

## Expected Results

Based on the paper:

- **Config 1 (Baseline)**: Moderate learning, stable embeddings
- **Config 2 (Best)**: Fastest learning, highest final performance (>450 return)
- **Config 3 (Collapse)**: Embedding variance drops below 10^-7, learning stalls
- **Config 4 (Regularization)**: Prevents collapse, slower learning than Config 2

## Key Implementation Details

### Vision Transformer

- Input: 400×608×3 images (padded for clean 16×16 patch division)
- Patches: 25×38 = 950 patches per frame × 3 frames = 2850 patches
- Positional encoding: (i, j, t) for spatial and temporal position
- Output: 64-dimensional [CLS] token embedding

### Momentum Encoder

- Update rule: θ_target = 0.99 × θ_target + 0.01 × θ_context
- No gradient propagation to target encoder
- Synchronized with context encoder architecture

### Loss Functions

- **JEPA Loss**: L2 distance between predicted and target embeddings
- **PPO Loss**: Clipped surrogate + value loss + entropy bonus
- **Regularization**: -min(1, mean_variance / d_emb)

### Gradient Control

Critical for preventing collapse:
- **Config 1, 2**: RL gradients flow to encoder
- **Config 3, 4**: RL gradients stopped before encoder (use `.detach()`)

## Troubleshooting

### Out of Memory

Reduce batch size in config:

```yaml
training:
  mini_batch_size: 32  # Reduce from 64
```

### Slow Training

- Enable GPU: Set `device: "cuda"` in config
- Reduce image resolution (not recommended for paper reproduction)
- Decrease rollout steps

### Collapse Not Detected in Config 3

- Check gradient flow is properly stopped
- Verify momentum encoder is not receiving gradients
- Monitor embedding variance from step 0

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{jepa_rl_2024,
  title={JEPA for RL: Investigating Joint-Embedding Predictive Architectures for Reinforcement Learning},
  author={[Authors]},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

- Paper authors for the original research
- PyTorch and Gymnasium communities
- Vision Transformer implementation references
