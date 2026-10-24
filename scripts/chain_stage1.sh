#!/bin/bash
set -e
cd "$(dirname "$0")"
for m in 0.5b 1.5b; do
  if [ $m = 1.5b ]; then export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.45; BS=4; else BS=8; fi
  [ -f ../results/$m/plain_items.jsonl ] || ../.venv/bin/python -u run_baseline.py --model $m --bs $BS
  [ -f ../data/acts/$m/steer_vecs.pt ] || ../.venv/bin/python -u steer.py build --model $m
done
echo CHAIN1 DONE
