"""Aggregate results/ into docs/data.json (for the GitHub Pages site) and figures/*.png (for the README)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, ROOT, bootstrap_ci, read_jsonl

DOCS, FIG = os.path.join(ROOT, "docs"), os.path.join(ROOT, "figures")
os.makedirs(DOCS, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9})


def ms(a):
    a = np.asarray(a)
    return a.mean(0).round(4).tolist(), a.std(0).round(4).tolist()


def probes(m):
    p = os.path.join(RESULTS, m, "probes.json")
    if not os.path.exists(p):
        return None
    P = json.load(open(p))
    out = dict(counts=P["counts"], controls={}, curves={})
    for group in ("pooled", "lodo", "cross_kind"):
        for k, v in P[group].items():
            out["curves"][f"{group}:{k}"] = {pk: ms(pv) for pk, pv in v.items()}
    for k, v in P["controls"].items():
        out["controls"][k] = {pk: ms(pv) for pk, pv in v.items()} if isinstance(v, dict) else v
    out["cosine"] = P["cosine"]
    return out


def final(m):
    p = os.path.join(RESULTS, m, "steer_final.json")
    if not os.path.exists(p):
        return None
    F = json.load(open(p))
    metrics = ["opinion_syc", "capitulate", "pushback_still_correct", "accept_correction", "plain_flip",
               "mmlu_correct"]
    groups = {}
    for k, v in F.items():
        cfg = k.rsplit(":seed", 1)[0]
        groups.setdefault(cfg, []).append(v)
    out = {}
    for cfg, runs in groups.items():
        row = {}
        for mt in metrics:
            seed_means = [np.mean(r[mt]) for r in runs]
            pooled = np.concatenate([r[mt] for r in runs])  # items x seeds
            m_, lo, hi = bootstrap_ci(pooled)
            row[mt] = dict(mean=round(m_, 4), lo=round(lo, 4), hi=round(hi, 4),
                           seed_std=round(float(np.std(seed_means)), 4), n=len(runs[0][mt]), seeds=len(runs))
        out[cfg] = row
    return out


def gens(m):
    p = os.path.join(RESULTS, m, "generations.jsonl")
    if not os.path.exists(p):
        return None
    R = read_jsonl(p)
    out = dict(summary={}, examples={})
    for cfg in dict.fromkeys(r["cfg"] for r in R):
        rr = [r for r in R if r["cfg"] == cfg]
        a = [r for r in rr if r["task"] == "answer"]
        s = {}
        for cond in ("plain", "wrong_hint"):
            x = [r for r in a if r["cond"] == cond]
            s[f"answer_acc_{cond}"] = bootstrap_ci([r["correct"] for r in x])
            s[f"says_wrong_{cond}"] = bootstrap_ci([r["says_wrong"] for r in x])
        o = [r for r in rr if r["task"] == "open"]
        sc = [r.get("coherence") for r in o if r.get("coherence") is not None]
        s["coherence"] = bootstrap_ci(sc) if sc else None
        s["refusal"] = float(np.mean([r["refusal"] for r in o]))
        out["summary"][cfg] = s
        out["examples"][cfg] = dict(
            pushback=[dict(id=r["id"], prompt=r["prompt"], output=r["output"]) for r in rr
                      if r["task"] == "pushback_example"],
            open=[dict(prompt=r["prompt"][-1]["content"], output=r["output"], coherence=r.get("coherence"))
                  for r in o],
            answer=[dict(prompt=r["prompt"][-1]["content"], output=r["output"], correct=r["correct"])
                    for r in a if r["cond"] == "wrong_hint"][:15])
    return out


def fig_probes(m, P):
    nice = {"answer_prefix": "decision token", "end_of_pushback": "end of user pushback",
            "before_pushback": "end of turn-1 answer (before pushback seen)", "end_of_user_turn": "end of user turn"}
    titles = {"pushback": 'named pushback: "will capitulate"', "plain": 'plain "are you sure?": "will flip"',
              "opinion": 'opinion: "will agree with user"'}
    kinds = [k for k in ("pushback", "plain", "opinion") if any(c.startswith(f"pooled:{k}/") for c in P["curves"])]
    fig, axs = plt.subplots(1, len(kinds) + 1, figsize=(4.4 * (len(kinds) + 1), 3.6))
    for ax, kind in zip(axs, kinds):
        for i, (k, v) in enumerate((k, v) for k, v in P["curves"].items() if k.startswith(f"pooled:{kind}/")):
            mu, sd = np.array(v["logreg"][0]), np.array(v["logreg"][1])
            ax.plot(mu, color=f"C{i}", label=nice.get(k.split("/")[1], k))
            ax.fill_between(range(len(mu)), mu - sd, mu + sd, color=f"C{i}", alpha=0.2)
            if "within_strata" in v:
                ax.plot(v["within_strata"][0], color=f"C{i}", ls="-.", lw=0.9, label="  same, within template x letter")
        sh = P["controls"].get(f"{kind}/shuffled_labels")
        if sh:
            ax.plot(sh["logreg"][0], "k:", label="shuffled-label control")
        meta = P["controls"].get(f"{kind}/metadata_baseline_auroc")
        if meta:
            ax.axhline(np.mean(meta), color="gray", ls="--", label="metadata-only baseline")
        conf = P["controls"].get(f"{kind}/turn1_confidence_auroc")
        if conf is not None:
            ax.axhline(conf, color="gray", ls=":", label="turn-1 confidence only")
        ax.set_title(f"{titles[kind]} ({m})", fontsize=9)
        ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=6)
    ax = axs[-1]
    for k, v in P["curves"].items():
        if k.startswith("lodo:") or k.startswith("cross_kind:"):
            ls = "-" if k.startswith("lodo") else "--"
            ax.plot(v["logreg"][0], ls, lw=1, label=k.replace("lodo:", "held-out ").replace("cross_kind:", ""))
    ax.set_title("held-out dataset / cross-family", fontsize=9)
    ax.set_xlabel("layer")
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=5.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"probes_{m}.png"))
    plt.close(fig)


def fig_sweep(m, rows):
    if not rows:
        return
    base = next(r for r in rows if r["kind"] == "none")
    kinds = sorted({r["kind"] for r in rows} - {"none"})
    alphas = sorted({r["alpha"] for r in rows if r["kind"] != "none"})
    fig, axs = plt.subplots(len(kinds), 3, figsize=(13, 3.3 * len(kinds)), squeeze=False)
    for i, kind in enumerate(kinds):
        for j, (mt, lab) in enumerate([("capitulate", "capitulation to wrong pushback"),
                                       ("opinion_syc", "opinion sycophancy"), ("mmlu_correct", "MMLU accuracy")]):
            ax = axs[i, j]
            for a in alphas:
                rr = sorted([r for r in rows if r["kind"] == kind and r["alpha"] == a], key=lambda r: r["layer"])
                ax.plot([r["layer"] for r in rr], [r[mt] for r in rr], marker="o", ms=3, label=f"alpha={a}")
            ax.axhline(base[mt], color="k", ls=":", label="no steering")
            ax.set_title(f"{kind} vector: {lab}")
            ax.set_xlabel("layer")
            if j == 0:
                ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"sweep_{m}.png"))
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for kind in kinds:
        rr = [r for r in rows if r["kind"] == kind]
        ax.scatter([base["capitulate"] - r["capitulate"] for r in rr],
                   [base["mmlu_correct"] - r["mmlu_correct"] for r in rr], s=12, label=f"{kind} vector")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("reduction in capitulation rate (abs.)")
    ax.set_ylabel("MMLU accuracy lost (abs.)")
    ax.set_title(f"steering trade-off, every layer x alpha ({m})")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"tradeoff_{m}.png"))
    plt.close(fig)


def main():
    data = {}
    for m in ("1.5b", "0.5b"):
        d = os.path.join(RESULTS, m)
        if not os.path.exists(os.path.join(d, "baseline.json")):
            continue
        sweep = read_jsonl(os.path.join(d, "sweep.jsonl")) if os.path.exists(os.path.join(d, "sweep.jsonl")) else []
        meta = os.path.join(d, "steer_vecs_meta.json")
        data[m] = dict(baseline=json.load(open(os.path.join(d, "baseline.json"))), probes=probes(m), sweep=sweep,
                       final=final(m), gens=gens(m),
                       vec_meta=json.load(open(meta)) if os.path.exists(meta) else None)
        if data[m]["probes"]:
            fig_probes(m, data[m]["probes"])
        fig_sweep(m, sweep)
    json.dump(data, open(os.path.join(DOCS, "data.json"), "w"))
    print("wrote docs/data.json", {m: [k for k, v in d.items() if v] for m, d in data.items()})


if __name__ == "__main__":
    main()
