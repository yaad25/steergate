"""Seeded, resumable, truncation-aware GSM8K sampler/greedy evaluator.

One call = one *arm* (base, steered, control, prompt variant). Output: JSONL, one row per (qid, sample).
Common random numbers: the sampling seed depends only on (seed_base, qid, sample_idx), never on the arm.
"""
from __future__ import annotations
import argparse, json, os, sys, time, platform
from pathlib import Path
import numpy as np
import torch

from .extract_answer import grade, gold_from_gsm8k
from .sampling import SeededSampler, row_seed
from .steer import steering, load_vectors, sha256_file
from .controls import random_control

DEFAULT_INSTR = "Please reason step by step, and put your final answer within \\boxed{}."


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", help="JSONL with fields question, answer (GSM8K format). Default: HF openai/gsm8k test.")
    ap.add_argument("--limit", type=int); ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--arm", required=True, help="label stored in every row")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8, help="samples per question (ignored for --greedy)")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed-base", type=int, default=1234)
    ap.add_argument("--instruction", default=DEFAULT_INSTR)
    ap.add_argument("--system", default=None, help="optional system prompt (else the template default)")
    ap.add_argument("--chat-template-kwargs", default="{}", help='JSON, e.g. {"enable_thinking": false}')
    ap.add_argument("--steer-vector", help=".npz/.safetensors with keys block_<b>")
    ap.add_argument("--block", type=int); ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--steer-mode", default="all", choices=["all", "decode_only"])
    ap.add_argument("--control", default="none", choices=["none", "random"],
                    help="random: replace the vector by a Gaussian direction with the same norm at the same block")
    ap.add_argument("--control-seed", type=int, default=0)
    return ap.parse_args(argv)


def load_questions(args):
    if args.data:
        rows = [json.loads(l) for l in open(args.data) if l.strip()]
    else:
        from datasets import load_dataset
        rows = list(load_dataset("openai/gsm8k", "main", split="test"))
    rows = [dict(r, qid=i) for i, r in enumerate(rows)][args.offset:]
    return rows[: args.limit] if args.limit else rows


def load_model(name, dtype, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(device).eval()
    return model, tok


def make_gen_config(tok, model, max_new_tokens):
    """Explicit config: some instruct models ship repetition_penalty/top_k defaults that would silently apply."""
    from transformers import GenerationConfig
    eos = set()
    ge = getattr(model.generation_config, "eos_token_id", None)
    for e in ([ge] if isinstance(ge, int) else (ge or [])):
        eos.add(int(e))
    if tok.eos_token_id is not None:
        eos.add(int(tok.eos_token_id))
    pad = tok.pad_token_id if tok.pad_token_id is not None else sorted(eos)[0]
    gc = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens, repetition_penalty=1.0,
                          eos_token_id=sorted(eos), pad_token_id=pad)
    return gc, eos, pad


def build_prompt(tok, question, args, tmpl_kwargs):
    content = f"{question}\n{args.instruction}"
    msgs = ([{"role": "system", "content": args.system}] if args.system else []) + [{"role": "user", "content": content}]
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, **tmpl_kwargs)


@torch.no_grad()
def run(args, model=None, tok=None):
    t_start = time.time()
    if model is None:
        model, tok = load_model(args.model, args.dtype, args.device)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    gc, eos, pad = make_gen_config(tok, model, args.max_new_tokens)
    model.generation_config = gc                      # no hidden repetition penalty / top-k from the checkpoint
    tmpl_kwargs = json.loads(args.chat_template_kwargs)
    qs = load_questions(args)
    k = 1 if args.greedy else args.k
    prompts = {q["qid"]: build_prompt(tok, q["question"], args, tmpl_kwargs) for q in qs}
    plen = {qid: len(tok(p, add_special_tokens=False).input_ids) for qid, p in prompts.items()}
    gold = {q["qid"]: gold_from_gsm8k(q["answer"]) for q in qs}

    vectors = None
    if args.steer_vector:
        allv = load_vectors(args.steer_vector)
        assert args.block in allv, f"--block must be one of {sorted(allv)}"
        v = allv[args.block]
        if args.control == "random":
            v = random_control(v.size, float(np.linalg.norm(v)), args.control_seed)
        vectors = {args.block: v}

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            if l.strip():
                r = json.loads(l); done.add((r["qid"], r["sample"]))
    todo = sorted(((qid, s) for qid in prompts for s in range(k) if (qid, s) not in done), key=lambda x: (plen[x[0]], x[0], x[1]))
    n_gen_tokens, t_gen = 0, 0.0
    fh = open(out, "a")
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        enc = tok([prompts[qid] for qid, _ in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        P = enc["input_ids"].shape[1]
        procs = None
        if not args.greedy:
            from transformers import LogitsProcessorList
            seeds = [row_seed(args.seed_base, qid, s) for qid, s in batch]
            procs = LogitsProcessorList([SeededSampler(seeds, P, args.temperature, args.top_p)])
        t0 = time.time()
        if vectors:
            with steering(model, vectors, args.lam, args.steer_mode):
                gen = model.generate(**enc, generation_config=gc, logits_processor=procs)
        else:
            gen = model.generate(**enc, generation_config=gc, logits_processor=procs)
        t_gen += time.time() - t0
        gen = gen[:, P:].tolist()
        for (qid, s), ids in zip(batch, gen):
            pos = next((j for j, t in enumerate(ids) if t in eos), None)
            finished = pos is not None
            body = ids[:pos] if finished else ids
            ntok = (pos + 1) if finished else len(ids)
            n_gen_tokens += ntok
            text = tok.decode(body, skip_special_tokens=True)
            g = grade(text, gold[qid])
            fh.write(json.dumps({"arm": args.arm, "qid": qid, "sample": s, "correct": g["correct"], "has_box": g["has_box"],
                                 "finished": finished, "ntok": ntok, "pred": g["pred"], "gold": gold[qid], "text": text}) + "\n")
        fh.flush()
        print(f"[{args.arm}] {min(i + args.batch_size, len(todo))}/{len(todo)} rows  "
              f"{n_gen_tokens / max(t_gen, 1e-9):.0f} gen tok/s", file=sys.stderr, flush=True)
    fh.close()
    meta = {"args": vars(args), "versions": {"torch": torch.__version__, "python": platform.python_version()},
            "generation_config": gc.to_dict(), "n_questions": len(prompts), "k": k, "rows_written": len(todo),
            "gen_tokens": n_gen_tokens, "gen_seconds": t_gen, "tok_per_s": n_gen_tokens / max(t_gen, 1e-9),
            "wall_seconds": time.time() - t_start,
            "steer": None if not vectors else {"vector_sha256": sha256_file(args.steer_vector), "block": args.block, "lam": args.lam,
                                               "mode": args.steer_mode, "control": args.control,
                                               "perturbation_norm": float(abs(args.lam) * np.linalg.norm(next(iter(vectors.values()))))}}
    try:
        import transformers; import subprocess
        meta["versions"]["transformers"] = transformers.__version__
    except Exception:
        pass
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, indent=1, default=str))
    return meta


def main(argv=None):
    run(parse_args(argv))


if __name__ == "__main__":
    main()
