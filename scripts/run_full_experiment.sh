#!/usr/bin/env bash
set -e

mkdir -p .mplcache

MPLCONFIGDIR=.mplcache python src/train.py --epochs 30 --window-size 30 --batch-size 128
MPLCONFIGDIR=.mplcache python src/analyze_outputs.py --importance-model lstm
