#!/usr/bin/env bash
set -e

mkdir -p .mplcache

MPLCONFIGDIR=.mplcache python src/train.py --fd FD001 --epochs 30 --window-size 30 --batch-size 128 --output-dir outputs/FD001
MPLCONFIGDIR=.mplcache python src/analyze_outputs.py --fd FD001 --importance-model lstm --output-dir outputs/FD001
