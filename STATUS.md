# STATUS — sycophancy-probe-steer (IN PROGRESS, not published)

Repo: not yet created (planned: rishikeshn-eng/sycophancy-probe-steer). Pages: not yet deployed.
Headline result: none yet.

## Done
- venv (.venv, uv with project-local cache), datasets in data/ (gitignored):
  Anthropic MWE sycophancy (NLP / PhilPapers from anthropics/evals, because the HF mirror's philpapers file is a byte-copy of nlp_survey / political),
  meg-tong are_you_sure + answer, TruthfulQA MC1, MMLU. Items: 3000 opinion, 3068 MC pushback, 600 MMLU, 300 TriviaQA.
- Scripts in scripts/: prepare_data.py (run), run_baseline.py, probes.py, steer.py (build/sweep/final), gen_eval.py (gen/judge), make_site.py.
- 0.5B opinion-stage baseline: syc rate nlp 0.53, political 0.53, philpapers 0.80 (results/0.5b/opinion_items.jsonl).
- Fixed MPS memory blow-up (10 GB cache → watermark caps + empty_cache in common.py).

## Running / queued
- `scripts/chain_stage1.sh` under gpulock (nohup, log logs/stage1.log): baseline + steering-vector build for 0.5b then 1.5b.
  It was still waiting for the GPU lock at 04:47 IST on 2026-10-24.

## Left to do (in order)
1. After stage 1: `.venv/bin/python scripts/probes.py --model 1.5b` (and 0.5b), CPU only.
2. Steering sweep under gpulock: `./run_job.sh sweep_1.5b steer.py sweep --model 1.5b --small 150`, pick configs,
   then `steer.py final --model 1.5b --configs "pushback:L:alpha;opinion:L:alpha"` and the same for 0.5b.
3. `gen_eval.py gen --configs ...` (GPU), then `gen_eval.py judge` (Gemini).
4. `make_site.py`; write docs/index.html (Plotly over docs/data.json) and README; create the GitHub repo and enable Pages.
