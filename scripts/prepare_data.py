"""Build the item files used by every later stage.

opinion.jsonl : Anthropic model-written-evals sycophancy (NLP survey / PhilPapers / political typology), single-turn A/B(/C..)
mc.jsonl      : multiple-choice questions for the two-turn pushback protocol (TruthfulQA MC1 + meg-tong are_you_sure)
mmlu.jsonl    : MMLU capability subset (questions that do not appear in are_you_sure)
answer.jsonl  : meg-tong answer-sycophancy free-form subset (plain vs "I think the answer is X" suggestion)
"""
import hashlib
import os
import random
import re

from datasets import load_dataset

from common import DATA, LETTERS, read_jsonl, write_jsonl

RAW = os.path.join(DATA, "raw")
OUT = os.path.join(DATA, "items")
N_OPINION = 1000  # per MWE subset
N_MMLU = 600
N_ANSWER = 300
rng = random.Random(0)


def split_of(key):  # deterministic 50/50 train/test split by content hash
    return "train" if int(hashlib.md5(key.encode()).hexdigest(), 16) % 2 == 0 else "test"


def opinion():
    rows = []
    for name, short in [("sycophancy_on_nlp_survey", "nlp"), ("sycophancy_on_philpapers2020", "philpapers"),
                        ("sycophancy_on_political_typology_quiz", "political")]:
        # NB: the Hugging Face mirror's philpapers file is a byte-copy of nlp_survey; we fetch it from anthropics/evals.
        data = read_jsonl(os.path.join(RAW, name + ".jsonl"))
        rng.shuffle(data)
        for i, d in enumerate(data[:N_OPINION]):
            q = d["question"].rsplit("Answer:", 1)[0].rstrip()
            opts = re.findall(r"\n \(([A-J])\) ", q)
            m = d["answer_matching_behavior"]
            m = m[0] if isinstance(m, list) else m
            syc = LETTERS.index(m.strip()[1])
            rows.append(dict(id=f"{short}-{i}", source=short, n_opts=len(opts), syc=syc,
                             messages=[{"role": "user", "content": q + "\n\nAnswer with the letter of your choice."}],
                             split=split_of(q)))
    write_jsonl(os.path.join(OUT, "opinion.jsonl"), rows)
    print("opinion", len(rows))


def fmt_opts(opts):
    return "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(opts))


def mc():
    rows = []
    tqa = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    for i, d in enumerate(tqa):
        ch, lab = d["mc1_targets"]["choices"], d["mc1_targets"]["labels"]
        correct = ch[lab.index(1)]
        wrong = [c for c, l in zip(ch, lab) if l == 0]
        opts = [correct] + rng.sample(wrong, min(3, len(wrong)))
        rng.shuffle(opts)
        rows.append(dict(id=f"tqa-{i}", source="truthfulqa", question=d["question"], options=opts,
                         correct=opts.index(correct), split=split_of(d["question"])))
    seen = set()
    for i, d in enumerate(read_jsonl(os.path.join(RAW, "mt_are_you_sure.jsonl"))):
        b = d["base"]
        if b["dataset"] == "truthful_qa_mc" or "correct_letter" not in b:
            continue  # TruthfulQA already covered above
        opts = re.findall(r"\(([A-J])\)\s*(.*)", b["answers"])
        if len(opts) < 2 or b["question"] in seen:
            continue
        seen.add(b["question"])
        rows.append(dict(id=f"ays-{i}", source="ays_" + b["dataset"].replace("_mc", "").replace("_cot", ""),
                         question=b["question"], options=[o[1].strip() for o in opts],
                         correct=LETTERS.index(b["correct_letter"]), split=split_of(b["question"])))
    write_jsonl(os.path.join(OUT, "mc.jsonl"), rows)
    from collections import Counter
    print("mc", len(rows), Counter(r["source"] for r in rows))
    return seen


def mmlu(exclude):
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=0)
    rows = []
    for d in ds:
        if d["question"] in exclude:
            continue
        rows.append(dict(id=f"mmlu-{len(rows)}", source="mmlu", subject=d["subject"], question=d["question"],
                         options=d["choices"], correct=d["answer"]))
        if len(rows) == N_MMLU:
            break
    write_jsonl(os.path.join(OUT, "mmlu.jsonl"), rows)
    print("mmlu", len(rows))


def answer():
    data = [d for d in read_jsonl(os.path.join(RAW, "mt_answer.jsonl")) if d["base"]["dataset"] == "trivia_qa"]
    by_q = {}
    for d in data:
        by_q.setdefault(d["base"]["question"], {})[d["metadata"]["prompt_template"]] = d
    qs = sorted(by_q)
    rng.shuffle(qs)
    rows = []
    for q in qs[:N_ANSWER]:
        b = next(iter(by_q[q].values()))["base"]
        rows.append(dict(id=f"ans-{len(rows)}", question=q, aliases=b["answer"], correct_answer=b["correct_answer"],
                         incorrect_answer=b["incorrect_answer"]))
    write_jsonl(os.path.join(OUT, "answer.jsonl"), rows)
    print("answer", len(rows))


if __name__ == "__main__":
    opinion()
    seen = mc()
    mmlu(seen)
    answer()
