#!/usr/bin/env python
"""Run all 4 configurations with FULL PAPER specifications.

This matches the original paper's hyperparameters:
- Image size: 600×400 (vs 300×200 lightweight)
- Embedding dim: 64 (vs 32 lightweight)
- Transformer layers: 4 (vs 2 lightweight)
- Training steps: 100,000 (vs 10,000 lightweight)
- Rollout buffer: 2048 (vs 512 lightweight)
- Batch size: 64 (vs 32 lightweight)

Expected time: ~8-10 hours on Apple M4 (MPS)
Expected memory: ~2-3GB per config
"""

import argparse
import subprocess
import time
from pathlib import Path


def run_config(config_num: int, device: str, output_dir: str):
    """Run a single configuration.

    Args:
        config_num: Configuration number (1-4)
        device: Device to use (mps, cpu, cuda)
        output_dir: Output directory for results
    """
    config_file = f"configs/experiments/config{config_num}_full_paper.yaml"

    print("=" * 80)
    print(f"RUNNING CONFIGURATION {config_num} - FULL PAPER SPECS")
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
    print(f"Parameters: 64-dim, 4 layers, 600×400 images, 100k steps")
    print(f"Expected time: ~2-2.5 hours per config on M4")
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
        description="Run all 4 configurations with FULL PAPER specifications"
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
        default="results/all_configs_full_paper",
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
    print("JEPA FOR RL - FULL PAPER SPECIFICATIONS")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Configurations: {args.configs}")
    print(f"Output directory: {args.output_dir}")
    print()
    print("PAPER SPECIFICATIONS:")
    print("  - Image size: 600×400 pixels (2,850 patches per observation)")
    print("  - Embedding dimension: 64")
    print("  - Transformer: 4 layers, 4 heads")
    print("  - Training steps: 100,000")
    print("  - Rollout buffer: 2048 steps")
    print("  - Batch size: 64")
    print()
    print(f"⏱️  ESTIMATED TIME: ~8-10 hours total on Apple M4")
    print(f"💾 ESTIMATED MEMORY: ~2-3GB per config")
    print()

    if not args.yes:
        response = input("This will take several hours. Continue? (yes/no): ")
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

        # Save intermediate checkpoint
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
