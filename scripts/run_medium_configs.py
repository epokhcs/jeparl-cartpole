#!/usr/bin/env python
"""Run all 4 configurations with MEDIUM specifications (memory-optimized)."""

import sys
sys.path.insert(0, '.')

# Import the run logic from full paper script
from pathlib import Path
exec(Path('scripts/run_full_paper_configs.py').read_text().replace(
    'config{config_num}_full_paper',
    'config{config_num}_medium'
).replace(
    'FULL PAPER',
    'MEDIUM'
).replace(
    'results/all_configs_full_paper',
    'results/all_configs_medium'
).replace(
    '~8-10 hours',
    '~4-5 hours'
).replace(
    '~2-2.5 hours per config',
    '~1-1.5 hours per config'
).replace(
    'Original paper specifications',
    'Memory-optimized specifications for M4'
).replace(
    '600×400 pixels (2,850 patches',
    '600×400 pixels, 512 rollout (vs 2048'
))
