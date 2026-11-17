"""
Gate 1.5: build the on-policy contrastive steering vector + lock (L*, alpha*).

Steps (per PLAN.md section 5.1-5.2):
  1. Extraction split: GSM8K-train[0:150]. For each question, greedily generate the model's
     own verbose ("sparse") trace y-. Build a "dense" y+ by deterministically compressing it
     (strip filler sentences, keep the equations/numbers/boxed answer).
  2. For each candidate layer L in {12,14,16,18,20}: capture last-token residual-stream
     activations for (question + y+) and (question + y-), compute mean-difference vector,
     L2-normalize, scale by the layer's mean residual-stream token norm.
  3. Tune split: GSM8K-train[150:350] (N=200) + WikiText-2[0:50]. Sweep alpha in
     {0.5,1.0,1.5,2.0} per layer; pick (L*, alpha*) maximizing tune-split accuracy subject to
     KL(base||steered) <= 0.15 nats/token on WikiText and truncation <= base + 0.5%.
  4. Save vectors (.npz, keys block_<b>) + locked_config.json (L*, alpha*, SHA-256 of vectors).

Only the on-policy arm is built here (no external teacher API). Scope cut is documented in
the report, not hidden.
"""
from __future__ import annotations
import argparse
import functools
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .steer import steering, save_vectors, sha256_file
from .extract_answer import gold_from_gsm8k

print = functools.partial(print, flush=True)  # noqa: A001 - force visible progress when redirected to a file

CANDIDATE_LAYERS = [12, 14, 16, 18, 20]
ALPHAS = [0.5, 1.0, 1.5, 2.0]
KL_CEILING = 0.15  # nats/token
TRUNC_SLACK = 0.005  # +0.5 percentage points over base


# ---------------------------------------------------------------- compression (on-policy y+)
_FILLER_PATTERNS = [
    r"^(let'?s|let me|first,? let'?s|to (determine|find|solve|calculate)).*?[:,]\s*",
    r"^(now|next|then|so|therefore|thus|finally),?\s*",
    r"\bwe (need to|will|can|should)\b[^.]*\.\s*",
    r"\bthis (means|tells us|gives us)\b[^.]*\.\s*",
]


def compress_trace(text: str) -> str:
    """Deterministic algebraic compression: keep lines with numbers/equations/boxed answer,
    drop pure prose filler sentences. Zero external API calls."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    kept = []
    for line in lines:
        has_math = bool(re.search(r"\d", line)) or "\\boxed" in line or "=" in line
        if not has_math:
            continue
        s = line
        for pat in _FILLER_PATTERNS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE)
        s = s.strip(" -*:")
        if s:
            kept.append(s)
    out = " ".join(kept)
    return out if out else text  # never return empty


# ---------------------------------------------------------------- activation capture
def get_blocks(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj, ok = model, True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise ValueError("cannot locate decoder blocks")


@torch.no_grad()
def last_token_activations(model, tok, texts: list[str], layers: list[int], batch_size: int = 8):
    """Returns {layer: np.ndarray [N, d_model]} — last real-token hidden state after each block."""
    blocks = get_blocks(model)
    out = {l: [] for l in layers}
    norms = {l: [] for l in layers}
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        res = model(**enc, output_hidden_states=True)
        # hidden_states[0] = embeddings, hidden_states[b+1] = output of block b
        lens = enc["attention_mask"].sum(dim=1) - 1  # last real token index (left-padded -> use last col)
        last_idx = enc["input_ids"].shape[1] - 1  # left-padded, so last column is always the last real token
        for l in layers:
            h = res.hidden_states[l + 1][:, last_idx, :].float().cpu().numpy()
            out[l].append(h)
            tok_norms = res.hidden_states[l + 1].float().norm(dim=-1)
            mask = enc["attention_mask"].bool()
            norms[l].append(tok_norms[mask].cpu().numpy())
    return ({l: np.concatenate(v, axis=0) for l, v in out.items()},
            {l: float(np.concatenate(v).mean()) for l, v in norms.items()})


def mean_diff_vectors(pos: dict, neg: dict) -> dict:
    """{layer: unit-norm mean-difference direction}"""
    vecs = {}
    for l in pos:
        mu_pos = pos[l].mean(axis=0)
        mu_neg = neg[l].mean(axis=0)
        v = mu_pos - mu_neg
        n = np.linalg.norm(v) + 1e-8
        vecs[l] = (v / n).astype(np.float32)
    return vecs


# ---------------------------------------------------------------- KL + accuracy on tune split
@torch.no_grad()
def kl_on_passages(model, tok, passages: list[str], vectors: dict, block: int, alpha: float,
                    max_len: int = 256) -> float:
    """Mean KL(base || steered) per token over a batch of WikiText passages."""
    kls = []
    for text in passages:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=max_len).to(model.device)
        if enc["input_ids"].shape[1] < 2:
            continue
        base_logits = model(**enc).logits[0, :-1]
        with steering(model, {block: vectors[block]}, alpha, "all"):
            steer_logits = model(**enc).logits[0, :-1]
        p = torch.log_softmax(base_logits.float(), dim=-1)
        q = torch.log_softmax(steer_logits.float(), dim=-1)
        kl = torch.sum(p.exp() * (p - q), dim=-1)  # KL(base || steered) per position
        kls.append(kl.mean().item())
    return float(np.mean(kls)) if kls else float("inf")


@torch.no_grad()
def greedy_accuracy(model, tok, questions: list[dict], instruction: str, vectors: dict | None,
                    block: int | None, alpha: float, max_new_tokens: int = 512, batch_size: int = 16):
    from .extract_answer import grade
    correct, trunc, n = 0, 0, 0
    eos = {tok.eos_token_id}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": f"{q['question']}\n{instruction}"}],
            add_generation_prompt=True, tokenize=False) for q in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        P = enc["input_ids"].shape[1]
        gc = dict(do_sample=False, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                  pad_token_id=pad)
        if vectors:
            with steering(model, vectors, alpha, "all"):
                gen = model.generate(**enc, **gc)
        else:
            gen = model.generate(**enc, **gc)
        gen = gen[:, P:].tolist()
        for q, ids in zip(batch, gen):
            pos = next((j for j, t in enumerate(ids) if t in eos), None)
            finished = pos is not None
            text = tok.decode(ids[:pos] if finished else ids, skip_special_tokens=True)
            gold = gold_from_gsm8k(q["answer"])
            g = grade(text, gold)
            correct += int(g["correct"])
            trunc += int(not finished)
            n += 1
    return correct / max(n, 1), trunc / max(n, 1)


# ---------------------------------------------------------------- main pipeline
def build_traces(model, tok, extract_qs: list[dict], max_new_tokens: int = 400, batch_size: int = 16):
    """Generate the model's own verbose trace (sparse, y-) for each extraction question,
    keep only ones that land on the correct final answer, then build the compressed y+."""
    from .extract_answer import grade
    DEFAULT_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
    kept = []
    eos = {tok.eos_token_id}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for i in range(0, len(extract_qs), batch_size):
        batch = extract_qs[i:i + batch_size]
        prompts = [tok.apply_chat_template(
            [{"role": "user", "content": f"{q['question']}\n{DEFAULT_INSTR}"}],
            add_generation_prompt=True, tokenize=False) for q in batch]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        P = enc["input_ids"].shape[1]
        gen = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             repetition_penalty=1.0, pad_token_id=pad)
        gen = gen[:, P:].tolist()
        for q, ids, prompt in zip(batch, gen, prompts):
            pos = next((j for j, t in enumerate(ids) if t in eos), None)
            text = tok.decode(ids[:pos] if pos is not None else ids, skip_special_tokens=True)
            gold = gold_from_gsm8k(q["answer"])
            g = grade(text, gold)
            if g["correct"]:  # only keep traces the model itself got right
                kept.append({"question": q["question"], "prompt": prompt, "sparse": text,
                            "dense": compress_trace(text)})
    return kept


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="vectors")
    ap.add_argument("--extract-n", type=int, default=150)
    ap.add_argument("--tune-n", type=int, default=200)
    ap.add_argument("--wikitext-n", type=int, default=50)
    ap.add_argument("--instruction", default="Please reason step by step, and put your final answer within \\boxed{}.")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--layers", default=None, help="comma-separated override of CANDIDATE_LAYERS, e.g. 14,16,18")
    ap.add_argument("--alphas", default=None, help="comma-separated override of ALPHAS, e.g. 1.0,1.5")
    ap.add_argument("--deviation-note", default=None,
                    help="free text documenting any shortcut from the frozen protocol (e.g. reduced tune-n for time)")
    args = ap.parse_args(argv)
    layers = [int(x) for x in args.layers.split(",")] if args.layers else CANDIDATE_LAYERS
    alphas = [float(x) for x in args.alphas.split(",")] if args.alphas else ALPHAS

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to("cuda").eval()

    train = list(load_dataset("openai/gsm8k", "main", split="train"))
    extract_qs = train[:args.extract_n]
    tune_qs = train[args.extract_n:args.extract_n + args.tune_n]
    wiki = list(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test"))
    passages = [w["text"] for w in wiki if len(w["text"].strip()) > 200][: args.wikitext_n]

    print(f"[extract] generating on-policy traces for {len(extract_qs)} questions...")
    traces = build_traces(model, tok, extract_qs)
    print(f"[extract] {len(traces)}/{len(extract_qs)} traces kept (model answered correctly)")
    if len(traces) < 20:
        print("WARNING: very few usable traces; vectors may be noisy. Consider raising --extract-n.")

    pos_texts = [t["prompt"] + t["dense"] for t in traces]
    neg_texts = [t["prompt"] + t["sparse"] for t in traces]

    print(f"[extract] capturing activations at layers {layers}...")
    pos_acts, pos_norms = last_token_activations(model, tok, pos_texts, layers)
    neg_acts, _ = last_token_activations(model, tok, neg_texts, layers)
    unit_vecs = mean_diff_vectors(pos_acts, neg_acts)

    print(f"[tune] sweeping (layer, alpha) on {len(tune_qs)} train-tune questions + {len(passages)} WikiText "
         f"passages... ({len(layers)}x{len(alphas)} = {len(layers) * len(alphas)} combos)")
    base_acc, base_trunc = greedy_accuracy(model, tok, tune_qs, args.instruction, None, None, 1.0)
    print(f"[tune] base accuracy on tune split: {base_acc:.3f} (truncation {base_trunc:.3f})")

    results = []
    t_sweep = time.time()
    for ci, l in enumerate(layers):
        for cj, a in enumerate(alphas):
            tc0 = time.time()
            # scale = (alpha/100) * mean_token_norm at this layer, per PLAN.md 5.2
            vectors = {l: unit_vecs[l] * (a / 100.0) * pos_norms[l]}
            kl = kl_on_passages(model, tok, passages, vectors, l, 1.0)  # lam=1.0 since vector already scaled
            acc, trunc = greedy_accuracy(model, tok, tune_qs, args.instruction, vectors, l, 1.0)
            ok = kl <= KL_CEILING and trunc <= base_trunc + TRUNC_SLACK
            results.append({"layer": l, "alpha": a, "tune_acc": acc, "tune_trunc": trunc, "kl": kl, "ok": ok})
            done = ci * len(alphas) + cj + 1
            print(f"  [{done}/{len(layers)*len(alphas)}] L={l:>2} a={a:<4} acc={acc:.3f} trunc={trunc:.3f} "
                 f"KL={kl:.4f} {'OK' if ok else 'REJECTED'}  ({time.time()-tc0:.0f}s, "
                 f"elapsed {time.time()-t_sweep:.0f}s)")

    valid = [r for r in results if r["ok"]]
    if valid:
        best = max(valid, key=lambda r: r["tune_acc"])
    else:
        print("WARNING: no (layer, alpha) passed the KL/truncation guard. Locking least-harmful config "
              "at 60% depth to formally test and document a likely NO-GO.")
        l60 = layers[min(range(len(layers)), key=lambda i: abs(layers[i] - round(0.6 * 28)))]
        best = next((r for r in results if r["layer"] == l60 and r["alpha"] == 1.0), results[0])
        best = dict(best, forced=True)

    L_star, alpha_star = best["layer"], best["alpha"]
    final_vector = {L_star: (unit_vecs[L_star] * (alpha_star / 100.0) * pos_norms[L_star]).astype(np.float32)}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / f"{args.model.replace('/', '_')}_onpolicy.npz"
    save_vectors(vec_path, final_vector)

    locked = {
        "model": args.model, "layer": L_star, "alpha": alpha_star,
        "vector_path": str(vec_path), "vector_sha256": sha256_file(vec_path),
        "base_tune_accuracy": base_acc, "sweep_results": results,
        "n_traces_used": len(traces), "extract_n": args.extract_n, "tune_n": args.tune_n,
        "layers_swept": layers, "alphas_swept": alphas,
        "deviation_note": args.deviation_note,
        "arm": "steered_onpolicy", "teacher_arm_cut": "no external LLM API available; teacher arm skipped, documented",
        "wall_seconds": time.time() - t0,
    }
    (out_dir / "locked_config.json").write_text(json.dumps(locked, indent=1, default=str))
    print(f"\nLOCKED: layer={L_star} alpha={alpha_star} -> {out_dir/'locked_config.json'}")
    return locked


if __name__ == "__main__":
    main()
