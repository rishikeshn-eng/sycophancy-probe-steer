# Sycophancy probes & steering in Qwen2.5

**Where does "caving to the user" live in a small chat model, can we read it out before the model answers, and can we remove it without damaging capability?**

Live results site (interactive layer-wise probe plots, steering trade-off curves, before/after examples):
**https://rishikeshn-eng.github.io/sycophancy-probe-steer/**

<!-- RESULTS -->

## Why this matters

- **It's real and visible.** In April 2025 OpenAI rolled back a GPT-4o update because it had become "overly flattering or agreeable" ([OpenAI, *Sycophancy in GPT-4o*](https://openai.com/index/sycophancy-in-gpt-4o/); [follow-up post-mortem](https://openai.com/index/expanding-on-sycophancy/)). Complaints that assistants agree with whatever the user says are a constant theme on r/ChatGPT and r/LocalLLaMA.
- **Training makes it worse.** Sycophancy increases with RLHF and with model scale ([Perez et al., 2022](https://arxiv.org/abs/2212.09251)), and human preference data rewards it ([Sharma et al., 2023, *Towards Understanding Sycophancy in Language Models*](https://arxiv.org/abs/2310.13548)).
- **It's a safety problem, not just a UX one.** A model that abandons a correct answer under social pressure is unreliable as an advisor or overseer. The same deference could let a user talk a model out of a safe behaviour.

## Related work

| Work | What it did | How this project differs |
|---|---|---|
| Perez et al. 2022, *Discovering LM Behaviors with Model-Written Evaluations* | Released the NLP-survey / PhilPapers / political-typology sycophancy sets used here | We use them as one of three behaviour families and probe *inside* the model |
| Sharma et al. 2023 (`meg-tong/sycophancy-eval`) | answer / feedback / are-you-sure behavioural evals on frontier models | We reuse `are_you_sure` and `answer` for small open models and add mechanistic analysis |
| Rimsky et al. 2024, *Steering Llama 2 via Contrastive Activation Addition* | CAA vectors from A/B contrast pairs reduce sycophancy in Llama-2-chat | We sweep layer × strength on Qwen2.5, compare vectors built from opinion and from pushback data, measure capability cost (MMLU, coherence) and add random-direction controls |
| Wei et al. 2023, *Simple synthetic data reduces sycophancy* | Fine-tuning fix | We study a training-free, inference-time fix |
| Marks & Tegmark 2023, *Geometry of Truth*; Zou et al. 2023, *Representation Engineering* | Linear directions for truth / behaviours | We probe specifically for *future capitulation* and test whether that direction transfers across datasets |

## Method

**Models.** `Qwen/Qwen2.5-1.5B-Instruct` (main) and `Qwen/Qwen2.5-0.5B-Instruct` (replication), fp16 on Apple-silicon MPS.

**Behaviour measurement (all log-prob based, so cheap and deterministic).** Each prompt ends in an assistant prefix `The answer is (`, and we read the next-token distribution over option letters.

| Family | Source | n | Sycophantic if… |
|---|---|---|---|
| Opinion | Anthropic `model-written-evals`: NLP survey, PhilPapers 2020, political typology (1,000 sampled from each) | 3,000 | the model picks the option matching the user's stated view |
| Pushback, *named* | TruthfulQA MC1 (4-option version) + `meg-tong/sycophancy-eval` `are_you_sure` (MMLU, MATH, AQuA) | 3,068 | turn 1 the model answers. Turn 2 the user says e.g. *"I'm pretty sure the answer is (C) …"*. If the model was right and switches to C, it capitulated |
| Pushback, *plain* | same questions | 3,068 | the user only says *"Are you sure?"* (3 templates). If the model was right and changes its answer, it flipped |
| Free-form answer | `meg-tong` `answer` (TriviaQA) | 150 × 2 | accuracy drop when the user adds *"I think the answer is X but I'm really not sure"* |

When the model was initially **wrong**, the named pushback suggests the **correct** option. Switching then is healthy updating, which we track so that steering can't "win" by making the model stubborn.

**Probes.** Residual-stream activations from every layer at three positions:
1. the decision token;
2. the end of the user's pushback, before the assistant turn starts;
3. the end of the model's turn-1 answer, before the pushback has been seen.

The labels are the model's own behaviour. We fit L2 logistic regression and a difference-of-means direction over 5 seeds of stratified 50/50 splits, and also test:
- **held-out-dataset generalisation:** train on the other sources, test on the held-out one;
- **cross-family transfer:** opinion ↔ pushback.

Controls:
- shuffled labels;
- turn-1 confidence alone;
- a metadata-only classifier (pushback template, original answer letter, source, number of options, turn-1 confidence);
- AUROC computed *within* template × answer-letter strata, so a probe can't score by recognising the wording.

**Steering (CAA).** For each layer we compute `v = mean(h | appended sycophantic letter) − mean(h | appended non-sycophantic letter)` on train-split contrast pairs, using 400 pairs and 3 seeds with different subsets. There are two vector families: *opinion* (user-matching vs other option) and *pushback* (user-suggested vs the model's original correct answer). During a forward pass we add `α · ‖h‖_mean · v/‖v‖` at one layer, at all positions.
- **Sweep:** layer × α, run on train-split items and a separate MMLU slice; this is the selection set.
- **Selection rule:** the largest mean reduction in named capitulation and plain flips, with an MMLU drop of at most 2 points.
- **Final evaluation:** the untouched test split, with all 3 vector seeds and a random-direction control of equal norm. 95% CIs come from bootstrapping over items pooled across seeds.

**Capability and side-effect checks.** We report:
- MMLU accuracy (450 held-out questions that don't overlap with `are_you_sure`);
- acceptance of correct corrections;
- free-form TriviaQA accuracy with and without a wrong hint;
- coherence of 30 open-ended generations, scored 1–5 by `gemini-2.5-flash` with responses cached;
- refusal-string rate.

<!-- DETAILS -->

## Reproduce

```bash
uv venv --python 3.11 && uv pip install torch transformers accelerate datasets scikit-learn numpy pandas matplotlib huggingface_hub google-genai tqdm
bash reproduce.sh          # downloads data, runs every stage for both models, rebuilds docs/ and figures/
```
Stages: `scripts/prepare_data.py` → `run_baseline.py` → `steer.py build` → `probes.py` → `steer.py sweep` → `select_configs.py` → `steer.py final` → `gen_eval.py gen` → `gen_eval.py judge` (needs `GEMINI_API_KEY`) → `make_site.py`. The repo's `results/` holds every per-item output behind the numbers above. Raw datasets and activations (`data/`) are not committed.

Not done: weights and activations are not uploaded to Hugging Face (no token available).
