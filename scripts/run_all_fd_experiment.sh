#!/usr/bin/env bash
set -e

mkdir -p .mplcache

for fd in FD001 FD002 FD003 FD004; do
  echo "=== Running ${fd} ==="
  MPLCONFIGDIR=.mplcache python src/train.py \
    --fd "${fd}" \
    --epochs 30 \
    --window-size 30 \
    --batch-size 128 \
    --output-dir "outputs/${fd}"

  MPLCONFIGDIR=.mplcache python src/analyze_outputs.py \
    --fd "${fd}" \
    --output-dir "outputs/${fd}" \
    --importance-model lstm
done
