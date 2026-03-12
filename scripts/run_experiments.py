"""Run all 4 experimental configurations with multiple seeds.

This script reproduces the paper results by running all 4 configurations
with 5 different random seeds each (20 total runs).

Usage:
    python scripts/run_experiments.py --num-seeds 5
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_experiment(config_id: int, seed: int, output_dir: str, device: str):
    """Run a single experiment.

    Args:
        config_id: Configuration ID (1-4)
        seed: Random seed
        output_dir: Output directory
        device: Device to use

    Returns:
        subprocess result
    """
    config_file = f"configs/experiments/config{config_id}.yaml"

    print(f"\n{'='*60}")
    print(f"Running Configuration {config_id}, Seed {seed}")
    print(f"{'='*60}\n")

    # Run training script
    cmd = [
        sys.executable,  # Python executable
        "scripts/train.py",
        "--config", config_file,
        "--seed", str(seed),
        "--device", device,
        "--output-dir", output_dir
    ]

    result = subprocess.run(cmd, capture_output=False)

    return result.returncode


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Run all JEPA-RL experiments'
    )
    parser.add_argument(
        '--num-seeds',
        type=int,
        default=5,
        help='Number of random seeds per configuration'
    )
    parser.add_argument(
        '--configs',
        type=int,
        nargs='+',
        default=[1, 2, 3, 4],
        help='Which configurations to run (default: all)'
    )
    parser.add_argument(
        '--base-seed',
        type=int,
        default=42,
        help='Base random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/experiments',
        help='Base output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        help='Device (cuda/cpu/mps)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run experiments in parallel (not recommended unless you have multiple GPUs)'
    )

    args = parser.parse_args()

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("JEPA-RL Experiment Runner")
    print("="*60)
    print(f"Configurations: {args.configs}")
    print(f"Seeds per config: {args.num_seeds}")
    print(f"Base seed: {args.base_seed}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Total runs: {len(args.configs) * args.num_seeds}")
    print("="*60)

    # Generate experiment list
    experiments = []
    for config_id in args.configs:
        for seed_offset in range(args.num_seeds):
            seed = args.base_seed + seed_offset
            experiments.append((config_id, seed))

    print(f"\nPrepared {len(experiments)} experiments")

    # Run experiments
    results = []
    failed = []

    for i, (config_id, seed) in enumerate(experiments, 1):
        print(f"\n\nExperiment {i}/{len(experiments)}")

        return_code = run_experiment(
            config_id=config_id,
            seed=seed,
            output_dir=str(output_dir),
            device=args.device
        )

        if return_code == 0:
            results.append((config_id, seed, 'SUCCESS'))
            print(f"✓ Config {config_id}, Seed {seed}: SUCCESS")
        else:
            results.append((config_id, seed, 'FAILED'))
            failed.append((config_id, seed))
            print(f"✗ Config {config_id}, Seed {seed}: FAILED")

    # Summary
    print("\n\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for config_id in args.configs:
        config_results = [r for r in results if r[0] == config_id]
        success_count = sum(1 for r in config_results if r[2] == 'SUCCESS')
        print(f"Config {config_id}: {success_count}/{len(config_results)} successful")

    total_success = sum(1 for r in results if r[2] == 'SUCCESS')
    print(f"\nTotal: {total_success}/{len(results)} successful")

    if failed:
        print("\nFailed experiments:")
        for config_id, seed in failed:
            print(f"  - Config {config_id}, Seed {seed}")

    print(f"\nResults saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
