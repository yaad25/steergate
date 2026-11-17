"""Paired analysis of arms produced by evaluate.py (plan v5 §6)."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import numpy as np
from .stats import paired_lift, mcnemar_exact, decision_boot, noise_floor


def load_arm(path):
    by = defaultdict(list)
    for l in open(path):
        if l.strip():
            r = json.loads(l); by[r["qid"]].append(r)
    return by


def per_question(arm, qids, field="correct"):
    return np.array([np.mean([float(r[field]) for r in arm[q]]) for q in qids])


def summarize(path):
    arm = load_arm(path); rows = [r for v in arm.values() for r in v]
    return {"file": path, "questions": len(arm), "rows": len(rows),
            "accuracy": float(np.mean([r["correct"] for r in rows])) * 100,
            "truncation_rate": 1 - float(np.mean([r["finished"] for r in rows])),
            "boxed_compliance": float(np.mean([r["has_box"] for r in rows])),
            "mean_tokens": float(np.mean([r["ntok"] for r in rows]))}


def compare(base_path, treat_path, n_boot=10_000, go_min=1.0, margin=1.0, trunc_tol=0.005):
    B, T = load_arm(base_path), load_arm(treat_path)
    qids = sorted(set(B) & set(T))
    assert qids, "no overlapping questions"
    yb, yt = per_question(B, qids), per_question(T, qids)
    pl = paired_lift(yt, yb, n_boot=n_boot)
    sb, st = summarize(base_path), summarize(treat_path)
    fb = per_question(B, qids, "finished"); ft = per_question(T, qids, "finished")
    both = (fb == 1) & (ft == 1)
    lc = (float((yt[both] - yb[both]).mean()) * 100) if both.any() else float("nan")
    trunc_ok = st["truncation_rate"] <= sb["truncation_rate"] + trunc_tol
    dec = decision_boot(pl["delta"], pl["ci95"][0], pl["ci90"][1], go_min, margin, trunc_ok)
    res = {"n_questions": len(qids), "base": sb, "treat": st, "lift_pts": pl["delta"], "se": pl["se"],
           "ci95": pl["ci95"], "ci90": pl["ci90"], "length_controlled_lift_pts": lc,
           "truncation_ok": trunc_ok, "decision": dec,
           "note": "Decision applies the v5 rule; it is a GO/NO-GO only for a primary k=8 sampled run on the full test set."}
    if sb["rows"] == len(B) and st["rows"] == len(T):      # both single-sample (greedy) -> exact McNemar
        res["mcnemar"] = mcnemar_exact(yt >= 0.5, yb >= 0.5)
    return res


def noise(path_a, path_b):
    A, B = load_arm(path_a), load_arm(path_b)
    qids = sorted(set(A) & set(B))
    a, b = per_question(A, qids) >= 0.5, per_question(B, qids) >= 0.5
    return {"n": len(qids), "discordant_fraction_d0": noise_floor(a, b), "acc_a": float(a.mean()) * 100, "acc_b": float(b.mean()) * 100}


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("summary"); s.add_argument("files", nargs="+")
    c = sub.add_parser("compare"); c.add_argument("--base", required=True); c.add_argument("--treat", required=True)
    c.add_argument("--boot", type=int, default=10_000); c.add_argument("--go-min", type=float, default=1.0); c.add_argument("--margin", type=float, default=1.0)
    n = sub.add_parser("noise"); n.add_argument("a"); n.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "summary":
        for f in a.files: print(json.dumps(summarize(f), indent=1))
    elif a.cmd == "compare":
        print(json.dumps(compare(a.base, a.treat, a.boot, a.go_min, a.margin), indent=1, default=float))
    else:
        print(json.dumps(noise(a.a, a.b), indent=1))


if __name__ == "__main__":
    main()
