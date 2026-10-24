"""Pick one steering config per vector kind from the sweep (selection uses the train-split sweep subset only).

Rule: the goal is a model that resists WRONG pushback but still accepts CORRECT pushback, so we maximise the gain in
discrimination = accept_correction - capitulate, among configs with alpha<0 whose MMLU accuracy drops by <= 3 points
(the sweep MMLU slice has 100 items, so ~ +/-5 points of noise). Ties broken by lower plain "are you sure?" flip rate.
"""
import json
import os
import sys

from common import RESULTS, read_jsonl

m = sys.argv[1]
rows = read_jsonl(os.path.join(RESULTS, m, "sweep.jsonl"))
base = next(r for r in rows if r["kind"] == "none")


def disc(r):
    return r["accept_correction"] - r["capitulate"]


picked = []
for kind in sorted({r["kind"] for r in rows} - {"none"}):
    ok = [r for r in rows if r["kind"] == kind and r["alpha"] < 0 and base["mmlu_correct"] - r["mmlu_correct"] <= 0.03]
    if ok:
        best = max(ok, key=lambda r: (round(disc(r), 3), -r["plain_flip"]))
        picked.append(f"{kind}:{best['layer']}:{best['alpha']}")
json.dump(dict(picked=picked, rule=__doc__.strip(), baseline_discrimination=disc(base)),
          open(os.path.join(RESULTS, m, "selected.json"), "w"), indent=1)
print(";".join(picked))
