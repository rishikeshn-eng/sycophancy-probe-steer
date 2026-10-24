#!/bin/bash
# End-to-end reproduction. Every GPU stage runs under the shared lock if present (falls back to running directly).
set -e
cd "$(dirname "$0")"
LOCK=~/ai-safety-portfolio/_shared/gpulock.sh; [ -x "$LOCK" ] || LOCK=""
mkdir -p data/raw && cd data/raw
for f in nlp_survey philpapers2020 political_typology_quiz; do
  [ -s sycophancy_on_$f.jsonl ] || curl -sSL --retry 5 -o sycophancy_on_$f.jsonl https://raw.githubusercontent.com/anthropics/evals/main/sycophancy/sycophancy_on_$f.jsonl
done
for f in answer are_you_sure; do
  [ -s mt_$f.jsonl ] || curl -sSL --retry 5 -o mt_$f.jsonl https://raw.githubusercontent.com/meg-tong/sycophancy-eval/main/datasets/$f.jsonl
done
cd ../../scripts
PY=../.venv/bin/python
$PY prepare_data.py
$LOCK ./chain_stage1.sh
for m in 0.5b 1.5b; do $PY probes.py --model $m; done
$LOCK ./chain_stage2.sh 0.5b 3,6,9,12,15,18,21
$LOCK ./chain_stage2.sh 1.5b 6,10,14,18,22
for m in 0.5b 1.5b; do $PY gen_eval.py judge --model $m; done
$PY make_site.py
