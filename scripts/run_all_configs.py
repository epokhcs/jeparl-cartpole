#!/usr/bin/env python
"""Run all 4 experimental configurations sequentially with MPS backend."""

import argparse
import subprocess
import time
from pathlib import Path


def run_config(config_num: int, device: str, output_dir: str):
    """Run a single configuration.

    Args:
        config_num: Configuration number (1-4)
        device: Device to use (cpu, mps, cuda)
        output_dir: Output directory for results
    """
    config_file = f"configs/experiments/config{config_num}_lightweight.yaml"

    print("=" * 70)
    print(f"RUNNING CONFIGURATION {config_num}")
    print("=" * 70)

    config_names = {
        1: "Baseline (No JEPA)",
        2: "JEPA + RL Gradients (Best)",
        3: "JEPA Only (Should Collapse)",
        4: "JEPA + Regularization"
    }

    print(f"Configuration: {config_names[config_num]}")
    print(f"Config file: {config_file}")
    print(f"Device: {device}")
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
        print(f"\n✅ Config {config_num} completed successfully in {elapsed/60:.1f} minutes")
    else:
        print(f"\n❌ Config {config_num} failed with exit code {result.returncode}")

    print()
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Run all 4 experimental configurations"
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
        default="results/all_configs_lightweight",
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

    args = parser.parse_args()

    print("=" * 70)
    print("JEPA FOR RL - ALL CONFIGURATIONS TEST")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Configurations: {args.configs}")
    print(f"Output directory: {args.output_dir}")
    print()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run each configuration
    overall_start = time.time()
    results = {}

    for config_num in args.configs:
        success = run_config(config_num, args.device, args.output_dir)
        results[config_num] = success

    overall_elapsed = time.time() - overall_start

    # Print summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total time: {overall_elapsed/60:.1f} minutes")
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
