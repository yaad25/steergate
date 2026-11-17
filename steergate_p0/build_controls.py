"""
Build ctrl_random and ctrl_shuffled vectors at the Gate-1.5-locked layer, matched to the
locked steering vector's norm (PLAN.md section 5.4, arms 5 and 6).

Reuses the exact same deterministic (greedy) on-policy trace generation as extract.py, so the
118 kept traces are identical, then captures activations ONLY at the locked layer (cheap - a
single layer vs. the full 5-layer sweep) to get per-pair differences for the shuffled control.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .extract import build_traces, last_token_activations
from .controls import random_control, shuffled_pair_vector, match_norm
from .steer import save_vectors


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--locked-config", default="vectors/locked_config.json")
    ap.add_argument("--extract-n", type=int, default=150)
    ap.add_argument("--out-dir", default="vectors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args(argv)

    locked = json.loads(Path(args.locked_config).read_text())
    L_star = locked["layer"]
    target_norm = float(np.linalg.norm(np.load(locked["vector_path"])[f"block_{L_star}"]))
    print(f"[controls] locked layer {L_star}, target norm {target_norm:.4f}")

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=getattr(torch, args.dtype)).to("cuda").eval()

    train = list(load_dataset("openai/gsm8k", "main", split="train"))
    extract_qs = train[: args.extract_n]

    print(f"[controls] regenerating on-policy traces for {len(extract_qs)} questions (deterministic, "
         f"should reproduce the same {locked['n_traces_used']} kept traces from Gate 1.5)...")
    traces = build_traces(model, tok, extract_qs)
    print(f"[controls] {len(traces)} traces kept (Gate 1.5 had {locked['n_traces_used']})")

    pos_texts = [t["prompt"] + t["dense"] for t in traces]
    neg_texts = [t["prompt"] + t["sparse"] for t in traces]

    print(f"[controls] capturing activations at locked layer {L_star} only...")
    pos_acts, _ = last_token_activations(model, tok, pos_texts, [L_star])
    neg_acts, _ = last_token_activations(model, tok, neg_texts, [L_star])
    pair_diffs = pos_acts[L_star] - neg_acts[L_star]  # [N, d], NOT averaged

    out_dir = Path(args.out_dir)
    rand_vec = random_control(pair_diffs.shape[1], target_norm, args.seed)
    rand_path = out_dir / f"{args.model.replace('/', '_')}_ctrl_random.npz"
    save_vectors(rand_path, {L_star: rand_vec})
    print(f"[controls] ctrl_random -> {rand_path} (norm {np.linalg.norm(rand_vec):.4f})")

    shuf_vec = shuffled_pair_vector(pair_diffs, args.seed, target_norm)
    shuf_path = out_dir / f"{args.model.replace('/', '_')}_ctrl_shuffled.npz"
    save_vectors(shuf_path, {L_star: shuf_vec})
    print(f"[controls] ctrl_shuffled -> {shuf_path} (norm {np.linalg.norm(shuf_vec):.4f})")

    meta = {"layer": L_star, "target_norm": target_norm, "n_pairs": len(traces),
            "n_pairs_expected_from_gate1_5": locked["n_traces_used"],
            "match": len(traces) == locked["n_traces_used"],
            "random_vector_path": str(rand_path), "shuffled_vector_path": str(shuf_path)}
    (out_dir / "controls_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"\n{'OK' if meta['match'] else 'WARNING: trace count mismatch!'} -> {out_dir/'controls_meta.json'}")
    return meta


if __name__ == "__main__":
    main()
