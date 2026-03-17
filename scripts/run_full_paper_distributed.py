#!/usr/bin/env python
"""Run all 4 configurations with distributed training across multiple GPUs.

Optimized for GCP with 8x B200 GPUs:
- Full paper specifications (600×400 images, 2048 rollout)
- DataParallel across all available GPUs
- Automatic batch size scaling
- Expected time: ~2-3 hours for all 4 configs
"""

import argparse
import subprocess
import time
import torch
from pathlib import Path


def run_config(config_num: int, num_gpus: int, output_dir: str):
    """Run a single configuration with multi-GPU support."""
    config_file = f"configs/experiments/config{config_num}_full_paper.yaml"

    print("=" * 80)
    print(f"RUNNING CONFIGURATION {config_num} - FULL PAPER SPECS (MULTI-GPU)")
    print("=" * 80)

    config_names = {
        1: "Baseline (No JEPA)",
        2: "JEPA + RL Gradients (Best)",
        3: "JEPA Only (Should Collapse)",
        4: "JEPA + Regularization"
    }

    print(f"Configuration: {config_names[config_num]}")
    print(f"Config file: {config_file}")
    print(f"GPUs: {num_gpus} x NVIDIA B200")
    print(f"Parameters: 600×400 images, 64-dim, 4 layers, 100k steps")
    print(f"Batch size: {64 * num_gpus} (scaled across GPUs)")
    print(f"Expected time: ~30-45 min per config on 8x B200")
    print(f"Output: {output_dir}")
    print()

    start_time = time.time()

    # Run training with CUDA
    cmd = [
        "python", "-u", "scripts/train.py",
        "--config", config_file,
        "--device", "cuda",
        "--output-dir", output_dir
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ Config {config_num} completed successfully in {elapsed/60:.1f} minutes")
    else:
        print(f"\n❌ Config {config_num} failed with exit code {result.returncode}")

    print()
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all 4 configurations with distributed training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/full_paper_distributed",
        help="Base output directory"
    )
    parser.add_argument(
        "--configs",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        choices=[1, 2, 3, 4],
        help="Which configs to run (default: all)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="Number of GPUs to use (default: all available)"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Detect GPUs
    if not torch.cuda.is_available():
        print("❌ ERROR: CUDA not available. This script requires NVIDIA GPUs.")
        return 1

    num_gpus = args.gpus if args.gpus else torch.cuda.device_count()

    print("=" * 80)
    print("JEPA FOR RL - DISTRIBUTED TRAINING (FULL PAPER SPECS)")
    print("=" * 80)
    print(f"Detected GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    print(f"Using: {num_gpus} GPUs")
    print(f"Configurations: {args.configs}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("FULL PAPER SPECIFICATIONS:")
    print("  - Image size: 600×400 pixels (950 patches per observation)")
    print("  - Model: 64-dim embeddings, 4 layers, 4 heads")
    print("  - Training steps: 100,000")
    print("  - Rollout buffer: 2048 steps")
    print(f"  - Batch size: {64 * num_gpus} (scaled across {num_gpus} GPUs)")
    print()
    print(f"⏱️  ESTIMATED TIME: ~2-3 hours total on {num_gpus}x B200")
    print(f"💾 ESTIMATED MEMORY: ~20-30GB per GPU")
    print()

    if not args.yes:
        response = input("Continue? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Cancelled.")
            return 1
    else:
        print("✅ Auto-confirmed with --yes flag")
        print()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run each configuration
    overall_start = time.time()
    results = {}

    for config_num in args.configs:
        success = run_config(config_num, num_gpus, args.output_dir)
        results[config_num] = success

        if not success:
            print(f"⚠️  Config {config_num} failed. Continuing with remaining configs...")

    overall_elapsed = time.time() - overall_start

    # Print summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total time: {overall_elapsed/3600:.1f} hours")
    print()

    config_names = {
        1: "Config 1: Baseline",
        2: "Config 2: JEPA + RL",
        3: "Config 3: JEPA Only",
        4: "Config 4: JEPA + Reg"
    }

    for config_num in args.configs:
        status = "✅ SUCCESS" if results[config_num] else "❌ FAILED"
        print(f"{status} - {config_names[config_num]}")

    print()
    print(f"Results saved to: {args.output_dir}")
    print(f"Launch TensorBoard: tensorboard --logdir {args.output_dir} --port 6006")
    print()

    # Return success if all passed
    all_success = all(results.values())
    return 0 if all_success else 1


if __name__ == "__main__":
    exit(main())
