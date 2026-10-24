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




def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def ci(t):
    return "–" if not t else f"{pct(t[0])} [{pct(t[1])}, {pct(t[2])}]"


def tables(data):
    """Markdown tables for the README (written between the AUTO markers)."""
    out = []
    out.append("| model | opinion syc. (NLP / PhilPapers / political) | capitulates to named wrong answer | accepts named correct answer | flips when right under plain \"are you sure?\" | fixes when wrong under plain \"are you sure?\" |")
    out.append("|---|---|---|---|---|---|")
    for m, d in data.items():
        b = d["baseline"]
        op = " / ".join(pct(b["opinion_syc_rate"][s][0]) for s in ("nlp", "philpapers", "political"))
        out.append(f"| Qwen2.5-{m.upper()}-Instruct | {op} | {ci(b['capitulation_rate']['all'])} | "
                   f"{ci(b['accept_correction_rate']['all'])} | {ci(b['plain_flip_when_right']['all'])} | "
                   f"{ci(b['plain_fix_when_wrong']['all'])} |")
    out.append("")
    out.append("**Probes** (best layer; mean of 5 seeds; the decision-token probe partly reads the answer about to be produced, so the earlier positions are the more meaningful ones):")
    out.append("")
    out.append("| model | behaviour | read position | best layer | AUROC | AUROC within template x answer-letter | metadata-only baseline | turn-1 confidence only |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m, d in data.items():
        P = d.get("probes")
        if not P:
            continue
        for k, v in P["curves"].items():
            if not k.startswith("pooled:"):
                continue
            kind, pos = k[7:].split("/")
            mu = np.array(v["logreg"][0])
            b = int(mu.argmax())
            ws = v.get("within_strata")
            meta = P["controls"].get(f"{kind}/metadata_baseline_auroc")
            conf = P["controls"].get(f"{kind}/turn1_confidence_auroc")
            out.append(f"| {m} | {kind} | {pos} | {b} | {mu[b]:.3f} ± {v['logreg'][1][b]:.3f} | "
                       f"{ws[0][b]:.3f} | {np.mean(meta) if meta else float('nan'):.3f} | "
                       f"{'–' if conf is None else f'{conf:.3f}'} |")
    out.append("")
    out.append("**Steering, held-out test split** (pooled over 3 vector seeds; 95% bootstrap CI over items; random = random direction with the same norm):")
    out.append("")
    out.append("| model | config (vector:layer:alpha) | capitulates to wrong | accepts correct | discrimination (accept − capitulate) | plain flip | opinion syc. | MMLU |")
    out.append("|---|---|---|---|---|---|---|---|")
    for m, d in data.items():
        F = d.get("final")
        if not F:
            continue
        for cfg, v in F.items():
            disc = v["accept_correction"]["mean"] - v["capitulate"]["mean"]
            cell = lambda k: f"{pct(v[k]['mean'])} [{pct(v[k]['lo'])}, {pct(v[k]['hi'])}]"
            out.append(f"| {m} | {cfg} | {cell('capitulate')} | {cell('accept_correction')} | {100 * disc:+.1f} pts | "
                       f"{cell('plain_flip')} | {cell('opinion_syc')} | {cell('mmlu_correct')} |")
    out.append("")
    out.append("**Free-form generation checks** (TriviaQA answer sycophancy, 100 questions; coherence 1-5 on 30 open prompts judged by Gemini):")
    out.append("")
    out.append("| model | config | accuracy, no hint | accuracy with wrong hint | repeats the wrong hint | coherence | refusals |")
    out.append("|---|---|---|---|---|---|---|")
    for m, d in data.items():
        G = d.get("gens")
        if not G:
            continue
        for cfg, s in G["summary"].items():
            coh = f"{s['coherence'][0]:.2f} [{s['coherence'][1]:.2f}, {s['coherence'][2]:.2f}]" if s["coherence"] else "–"
            out.append(f"| {m} | {cfg} | {ci(s['answer_acc_plain'])} | {ci(s['answer_acc_wrong_hint'])} | "
                       f"{ci(s['says_wrong_wrong_hint'])} | {coh} | {pct(s['refusal'])} |")
    return "\n".join(out)


def write_readme_tables(data):
    p = os.path.join(ROOT, "README.md")
    s = open(p).read()
    a, b = "<!-- AUTO:START -->", "<!-- AUTO:END -->"
    if a in s and b in s:
        s = s[: s.index(a) + len(a)] + "\n" + tables(data) + "\n" + s[s.index(b):]
        open(p, "w").write(s)


if __name__ == "__main__":
    main()
    write_readme_tables(json.load(open(os.path.join(DOCS, "data.json"))))
