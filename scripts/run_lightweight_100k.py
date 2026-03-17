#!/usr/bin/env python
"""Run all 4 configurations with LIGHTWEIGHT 100K specifications.

Optimized for Apple M4:
- Image size: 300×200 (proven to work - only 700MB memory)
- Model: Full capacity (64-dim, 4 layers)
- Training: 100,000 steps (same as paper)
- Memory: ~1-1.5GB per config (safe for M4)

Expected time: ~6-8 hours on Apple M4 (MPS)
"""

import argparse
import subprocess
import time
from pathlib import Path


def run_config(config_num: int, device: str, output_dir: str):
    """Run a single configuration."""
    config_file = f"configs/experiments/config{config_num}_lightweight_100k.yaml"

    print("=" * 80)
    print(f"RUNNING CONFIGURATION {config_num} - LIGHTWEIGHT 100K")
    print("=" * 80)

    config_names = {
        1: "Baseline (No JEPA)",
        2: "JEPA + RL Gradients (Best)",
        3: "JEPA Only (Should Collapse)",
        4: "JEPA + Regularization"
    }

    print(f"Configuration: {config_names[config_num]}")
    print(f"Config file: {config_file}")
    print(f"Device: {device}")
    print(f"Parameters: 300×200 images, 64-dim, 4 layers, 100k steps")
    print(f"Expected time: ~1.5-2 hours per config on M4")
    print(f"Memory: ~1-1.5GB (proven to work)")
    print(f"Output: {output_dir}")
    print()

    start_time = time.time()

    # Run training
    cmd = [
        "python", "-u", "scripts/train.py",
        "--config", config_file,
        "--device", device,
        "--output-dir", output_dir
    ]

    result = subprocess.run(cmd, capture_output=False, text=True)

    elapsed = time.time() - start_time

    if result.returncode == 0:
        print(f"\n✅ Config {config_num} completed successfully in {elapsed/3600:.1f} hours")
    else:
        print(f"\n❌ Config {config_num} failed with exit code {result.returncode}")

    print()
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all 4 configurations with LIGHTWEIGHT 100K specifications"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["cpu", "mps", "cuda"],
        help="Device to use for training"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/lightweight_100k",
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
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("JEPA FOR RL - LIGHTWEIGHT 100K SPECIFICATIONS")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Configurations: {args.configs}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("OPTIMIZED SPECIFICATIONS FOR M4:")
    print("  - Image size: 300×200 pixels (247 patches per observation)")
    print("  - Model: 64-dim embeddings, 4 layers, 4 heads")
    print("  - Training steps: 100,000 (same as paper)")
    print("  - Rollout buffer: 512 steps")
    print("  - Batch size: 32")
    print()
    print(f"⏱️  ESTIMATED TIME: ~6-8 hours total on Apple M4")
    print(f"💾 ESTIMATED MEMORY: ~1-1.5GB per config (PROVEN TO WORK)")
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
        success = run_config(config_num, args.device, args.output_dir)
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
