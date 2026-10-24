#!/bin/bash
# usage: chain_stage2.sh <model> <layers csv>   (sweep -> select -> final -> generations), run under gpulock
set -e
cd "$(dirname "$0")"
m=$1; L=$2
if [ $m = 1.5b ]; then export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.5 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.45; fi
PY=../.venv/bin/python
BS=16; [ $m = 1.5b ] && BS=8
$PY -u steer.py sweep --model $m --bs $BS --small 100 --kinds pushback --layers $L --alphas=-0.5,-0.25,-0.1
$PY -u steer.py sweep --model $m --bs $BS --small 100 --kinds opinion --layers $L --alphas=-0.5,-0.25
CFG=$($PY select_configs.py $m)
echo "SELECTED $CFG"
[ -f ../results/$m/steer_final.json ] || $PY -u steer.py final --bs $BS --model $m --configs "$CFG"
[ -f ../results/$m/generations.jsonl ] || $PY -u gen_eval.py gen --model $m --configs "$CFG"
echo "CHAIN2 $m DONE"
