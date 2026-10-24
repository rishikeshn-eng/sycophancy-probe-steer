"""Shared helpers: model loading, chat formatting, letter-logit scoring, activation capture, steering hooks."""
import json
import os
import random
from contextlib import contextmanager

import numpy as np

# Cap the MPS caching allocator: six projects share 16 GB of unified memory.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.35")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.3")
import torch

torch.set_num_threads(4)
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
MODELS = {"1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "0.5b": "Qwen/Qwen2.5-0.5B-Instruct"}
LETTERS = "ABCDEFGHIJ"
ANSWER_PREFIX = "My answer is ("


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def load_model(key):
    name = MODELS[key]
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to("mps").eval()
    return tok, model


def letter_ids(tok):
    ids = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(i) == 1 for i in ids), ids
    return [i[0] for i in ids]


def render(tok, messages, prefix=ANSWER_PREFIX):
    """Chat-template the conversation and open an assistant turn that ends with `prefix`."""
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return text + prefix


def layers_of(model):
    return model.model.layers


class Steer:
    """Adds coef * vec to the residual stream output of one decoder layer (all positions)."""

    def __init__(self, model):
        self.model, self.handle = model, None

    @contextmanager
    def on(self, layer, vec, coef):
        if vec is None or coef == 0:
            yield
            return
        v = (coef * vec).to(self.model.device, torch.float16)

        def hook(_m, _i, out):
            if isinstance(out, tuple):
                return (out[0] + v,) + tuple(out[1:])
            return out + v

        h = layers_of(self.model)[layer].register_forward_hook(hook)
        try:
            yield
        finally:
            h.remove()


@torch.no_grad()
def score_letters(tok, model, prompts, n_opts, bs=8, want_hidden=False, positions=None):
    """Return (probs over first n_opts letters [N, max_opts]) and optionally hidden states.

    hidden: float16 array [N, n_layers+1, d] taken at the final token (or at token offsets from the end given
    by positions[i] list -> returns [N, n_pos, n_layers+1, d]).
    """
    lid = torch.tensor(letter_ids(tok), device=model.device)
    out_p, out_h = [], []
    order = np.argsort([-len(p) for p in prompts])  # length-sort for less padding
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    for b in range(0, len(prompts), bs):
        idx = order[b : b + bs]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True).to(model.device)
        o = model(**enc, output_hidden_states=want_hidden, logits_to_keep=1)
        logits = o.logits[:, -1, :].float()[:, lid]
        for j, i in enumerate(idx):
            k = n_opts[i]
            p = torch.softmax(logits[j, :k], -1).cpu().numpy()
            out_p.append(np.pad(p, (0, len(LETTERS) - k)))
        if want_hidden:
            T = o.hidden_states[0].shape[1]
            for j, i in enumerate(idx):
                sel = [T - 1] if positions is None else [T - 1 - off for off in positions[i]]
                h = torch.stack([hl[j, sel, :] for hl in o.hidden_states], 1)  # [n_pos, L+1, d]
                h = h.cpu().numpy().astype(np.float16)
                out_h.append(h[0] if positions is None else h)
        del o, enc
        if (b // bs) % 4 == 0:
            torch.mps.empty_cache()
        if (b // bs) % 50 == 0:
            print(f"  scored {b + len(idx)}/{len(prompts)}", flush=True)
    P = np.stack(out_p)[inv]
    H = np.stack(out_h)[inv] if want_hidden else None
    return P, H


def bootstrap_ci(x, n=2000, seed=0, stat=np.mean):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(stat(x)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
