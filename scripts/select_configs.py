"""Pick one steering config per vector kind from the sweep (selection uses the sweep subset only).

Rule: among (layer, alpha<0) configs whose MMLU accuracy drops by <= 2 points absolute, take the one with the
largest mean reduction in (capitulation to a named wrong answer, flip rate under plain "are you sure?").
"""
import json
import os
import sys

from common import RESULTS, read_jsonl

m = sys.argv[1]
rows = read_jsonl(os.path.join(RESULTS, m, "sweep.jsonl"))
base = next(r for r in rows if r["kind"] == "none")


def gain(r):
    return ((base["capitulate"] - r["capitulate"]) + (base["plain_flip"] - r["plain_flip"])) / 2


picked = []
for kind in sorted({r["kind"] for r in rows} - {"none"}):
    ok = [r for r in rows if r["kind"] == kind and r["alpha"] < 0 and base["mmlu_correct"] - r["mmlu_correct"] <= 0.02]
    if ok:
        best = max(ok, key=gain)
        picked.append(f"{kind}:{best['layer']}:{best['alpha']}")
json.dump(dict(picked=picked, rule=__doc__.strip()), open(os.path.join(RESULTS, m, "selected.json"), "w"), indent=1)
print(";".join(picked))
