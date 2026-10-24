"""Per-layer linear probes for "will be sycophantic / will capitulate" (CPU only).

Settings (each run over every layer, 5 seeds):
  pooled        : random 50/50 stratified split within the pooled pushback (or opinion) data
  lodo          : leave-one-dataset-out (train on the other sources of the same kind, test on the held-out source)
  cross_kind    : train on all opinion -> test on pushback, and vice versa
Probes: logistic regression (standardised, L2) and a difference-of-means direction (no fitting beyond means).
Controls: shuffled-label probe; turn-1 confidence baseline for capitulation.
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import DATA, RESULTS, read_jsonl
from run_baseline import PLAIN

SEEDS = range(5)
POS_NAMES = {"opinion": ["answer_prefix", "end_of_user_turn"],
             "pushback": ["answer_prefix", "end_of_pushback", "before_pushback"],
             "plain": ["answer_prefix", "end_of_pushback"]}


def load(model):
    actd, resd = os.path.join(DATA, "acts", model), os.path.join(RESULTS, model)
    op = read_jsonl(os.path.join(resd, "opinion_items.jsonl"))
    pb = read_jsonl(os.path.join(resd, "pushback_items.jsonl"))
    Ho = np.load(os.path.join(actd, "opinion_H.npy"), mmap_mode="r")
    Hp = np.load(os.path.join(actd, "pushback_H.npy"), mmap_mode="r")
    keep = np.array([r["cond"] == "initially_correct" for r in pb])
    pb = [r for r, k in zip(pb, keep) if k]
    Hp = Hp[keep]
    pl = read_jsonl(os.path.join(resd, "plain_items.jsonl"))
    Hl = np.load(os.path.join(actd, "plain_H.npy"), mmap_mode="r")
    keep = np.array([r["cond"] == "initially_correct" for r in pl])
    pl = [r for r, k in zip(pl, keep) if k]
    Hl = Hl[keep]
    convs = json.load(open(os.path.join(actd, "plain_convs.json")))
    convs = [c for c, k in zip(convs, keep) if k]
    for r, c in zip(pl, convs):
        r["template"] = PLAIN.index(c[2]["content"])
    D = dict(
        plain=dict(H=Hl, y=np.array([r["capitulate"] for r in pl]), src=np.array([r["source"] for r in pl]),
                   rows=pl),
        opinion=dict(H=Ho, y=np.array([r["syc"] for r in op]), src=np.array([r["source"] for r in op]), rows=op),
        pushback=dict(H=Hp, y=np.array([r["capitulate"] for r in pb]), src=np.array([r["source"] for r in pb]),
                      rows=pb))
    # surface/metadata features that could explain the label without any "decision" representation
    items = {it["id"]: it for it in read_jsonl(os.path.join(DATA, "items", "opinion.jsonl"))}
    for kind, d in D.items():
        if kind == "opinion":
            keys = [(r["source"], items[r["id"]]["syc"], r["n_opts"]) for r in d["rows"]]
            d["strata"] = np.array([f"{k[0]}|{k[1]}" for k in keys])
            conf = np.zeros(len(keys))
        else:
            keys = [(r["source"], r["template"], r["a0"], r.get("suggested", -1), r["n_opts"]) for r in d["rows"]]
            d["strata"] = np.array([f"{k[1]}|{k[2]}" for k in keys])
            conf = np.array([r["p0_conf"] for r in d["rows"]])
        oh = OneHotEncoder(handle_unknown="ignore").fit_transform([[str(x) for x in k] for k in keys]).toarray()
        d["meta"] = np.hstack([oh, conf[:, None]])
    return D


def fit_eval(Xtr, ytr, Xte, yte, return_scores=False):
    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=0.05, max_iter=300).fit(sc.transform(Xtr), ytr)
    score = lr.decision_function(sc.transform(Xte))
    a_lr = roc_auc_score(yte, score)
    dm = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    a_dm = roc_auc_score(yte, Xte @ dm)
    return (a_lr, a_dm, dm, score) if return_scores else (a_lr, a_dm, dm)


def strat_auc(y, score, strata, min_each=5):
    """Size-weighted mean AUROC within strata (e.g. same template + same original answer letter)."""
    num = den = 0.0
    for g in np.unique(strata):
        m = strata == g
        if y[m].sum() >= min_each and (1 - y[m]).sum() >= min_each:
            num += roc_auc_score(y[m], score[m]) * m.sum()
            den += m.sum()
    return num / den if den else float("nan")


def layer_curve(Xall, y, tr_idx_list, te_idx_list, shuffle=False, strata=None):
    """Xall: [N, L+1, d] (float16 memmap slice). returns dict of arrays [seeds, L+1]."""
    L = Xall.shape[1]
    lr, dm, st = (np.zeros((len(tr_idx_list), L)) for _ in range(3))
    for s, (tr, te) in enumerate(zip(tr_idx_list, te_idx_list)):
        ytr = np.random.default_rng(s).permutation(y[tr]) if shuffle else y[tr]
        for l in range(L):
            X = np.asarray(Xall[:, l, :], dtype=np.float32)
            lr[s, l], dm[s, l], _, score = fit_eval(X[tr], ytr, X[te], y[te], return_scores=True)
            if strata is not None:
                st[s, l] = strat_auc(y[te], score, strata[te])
    out = dict(logreg=lr.tolist(), diffmean=dm.tolist())
    if strata is not None:
        out["within_strata"] = st.tolist()
    return out


def splits_pooled(y):
    sss = StratifiedShuffleSplit(n_splits=len(SEEDS), test_size=0.5, random_state=0)
    tr, te = zip(*sss.split(np.zeros(len(y)), y))
    return list(tr), list(te)


def boot_idx(idx, s):
    return np.random.default_rng(s).choice(idx, len(idx), replace=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="1.5b")
    args = ap.parse_args()
    D = load(args.model)
    out = dict(model=args.model, counts={}, pooled={}, lodo={}, cross_kind={}, controls={}, cosine={})
    for kind, d in D.items():
        out["counts"][kind] = {s: [int((d["src"] == s).sum()), int(d["y"][d["src"] == s].sum())]
                               for s in np.unique(d["src"])}
    print(json.dumps(out["counts"]), flush=True)

    D = {k: d for k, d in D.items() if min(d["y"].sum(), (1 - d["y"]).sum()) >= 20}
    print("kinds with enough label variance:", list(D), flush=True)
    for kind, d in D.items():
        tr, te = splits_pooled(d["y"])
        for pi, pname in enumerate(POS_NAMES[kind]):
            X = d["H"][:, pi]
            out["pooled"][f"{kind}/{pname}"] = layer_curve(X, d["y"], tr, te, strata=d["strata"])
            print("pooled", kind, pname, np.round(np.mean(out["pooled"][f"{kind}/{pname}"]["logreg"], 0), 3),
                  flush=True)
        out["controls"][f"{kind}/shuffled_labels"] = layer_curve(d["H"][:, 0], d["y"], tr, te, shuffle=True)
        meta = []
        for a, b in zip(tr, te):
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(d["meta"][a], d["y"][a])
            meta.append(float(roc_auc_score(d["y"][b], lr.decision_function(d["meta"][b]))))
        out["controls"][f"{kind}/metadata_baseline_auroc"] = meta
        print("metadata baseline", kind, np.round(np.mean(meta), 3), flush=True)
        # leave-one-dataset-out (only sources with enough positives and negatives)
        for s in np.unique(d["src"]):
            yt = d["y"][d["src"] == s]
            if yt.sum() < 15 or (1 - yt).sum() < 15:
                continue
            tr_idx = np.where(d["src"] != s)[0]
            te_idx = np.where(d["src"] == s)[0]
            if d["y"][tr_idx].sum() < 15:
                continue
            out["lodo"][f"{kind}/{s}"] = layer_curve(d["H"][:, 0], d["y"], [boot_idx(tr_idx, k) for k in SEEDS],
                                                     [te_idx] * len(SEEDS))
            print("lodo", kind, s, np.round(np.mean(out["lodo"][f"{kind}/{s}"]["logreg"], 0), 3), flush=True)

    # cross-kind transfer at the answer-prefix position (position 0 in both)
    pairs = [(a, b) for a in D for b in D if a != b]
    for a, b in pairs:
        A, B = D[a], D[b]
        L = A["H"].shape[2]
        lr, dm = np.zeros((len(SEEDS), L)), np.zeros((len(SEEDS), L))
        cos = np.zeros(L)
        for l in range(L):
            Xa = np.asarray(A["H"][:, 0, l], np.float32)
            Xb = np.asarray(B["H"][:, 0, l], np.float32)
            for s in SEEDS:
                i = boot_idx(np.arange(len(Xa)), s)
                lr[s, l], dm[s, l], _ = fit_eval(Xa[i], A["y"][i], Xb, B["y"])
            da = Xa[A["y"] == 1].mean(0) - Xa[A["y"] == 0].mean(0)
            db = Xb[B["y"] == 1].mean(0) - Xb[B["y"] == 0].mean(0)
            cos[l] = da @ db / np.linalg.norm(da) / np.linalg.norm(db)
        out["cross_kind"][f"{a}->{b}"] = dict(logreg=lr.tolist(), diffmean=dm.tolist())
        out["cosine"][f"{a}_vs_{b}_diffmean"] = cos.tolist()
        print("cross", a, b, np.round(lr.mean(0), 3), flush=True)

    # behavioural baseline: low turn-1 confidence predicts capitulation?
    for kind in ("pushback", "plain"):
        if kind not in D:
            continue
        pb = D[kind]
        conf = np.array([r["p0_conf"] for r in pb["rows"]])
        out["controls"][f"{kind}/turn1_confidence_auroc"] = float(roc_auc_score(pb["y"], -conf))
        _, te = splits_pooled(pb["y"])
        out["controls"][f"{kind}/turn1_confidence_auroc_testsplits"] = [
            float(roc_auc_score(pb["y"][t], -conf[t])) for t in te]
    json.dump(out, open(os.path.join(RESULTS, args.model, "probes.json"), "w"))
    print("saved")


if __name__ == "__main__":
    main()
