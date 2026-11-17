"""Matched controls (plan v5 §5.6): C1 random direction, C2 shuffled-pair vector; both norm-matched."""
from __future__ import annotations
import argparse
import numpy as np
from .steer import load_vectors, save_vectors


def match_norm(v: np.ndarray, target: float) -> np.ndarray:
    n = np.linalg.norm(v)
    return v * (target / n) if n > 0 else v


def random_control(d: int, target_norm: float, seed: int) -> np.ndarray:
    return match_norm(np.random.default_rng(seed).standard_normal(d).astype(np.float32), target_norm)


def shuffled_pair_vector(pair_diffs: np.ndarray, seed: int, target_norm: float) -> np.ndarray:
    """pair_diffs [N,d] = h(y+) - h(y-). Random sign flip per pair destroys the consistent direction."""
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=pair_diffs.shape[0]).astype(np.float32)
    return match_norm((signs[:, None] * pair_diffs).mean(0), target_norm)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("random"); r.add_argument("--vectors", required=True); r.add_argument("--block", type=int, required=True)
    r.add_argument("--lam", type=float, default=1.0); r.add_argument("--seed", type=int, default=0); r.add_argument("-o", required=True)
    s = sub.add_parser("shuffled"); s.add_argument("--diffs", required=True, help="npy [N,d] pair differences at the block")
    s.add_argument("--target-norm", type=float, required=True); s.add_argument("--block", type=int, required=True)
    s.add_argument("--seed", type=int, default=0); s.add_argument("-o", required=True)
    a = ap.parse_args()
    if a.cmd == "random":
        v = load_vectors(a.vectors)[a.block]
        save_vectors(a.o, {a.block: random_control(v.size, a.lam * np.linalg.norm(v), a.seed) / a.lam})
    else:
        save_vectors(a.o, {a.block: shuffled_pair_vector(np.load(a.diffs), a.seed, a.target_norm)})


if __name__ == "__main__":
    main()
