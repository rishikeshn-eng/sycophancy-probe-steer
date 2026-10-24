#!/bin/bash
set -e
cd "$(dirname "$0")"
for m in 0.5b 1.5b; do
  ../.venv/bin/python -u run_baseline.py --model $m
  ../.venv/bin/python -u steer.py build --model $m
done
echo CHAIN1 DONE
