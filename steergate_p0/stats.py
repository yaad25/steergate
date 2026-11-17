"""Reference statistics for SteerGate (v5). Pure numpy/scipy; no model code.

All lifts are in accuracy *points* (percentage points). Inputs are per-question
mean correctness in [0,1] (k samples averaged; k=1 for greedy), aligned by question.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import binomtest

Z975, Z95 = 1.959964, 1.644854

def paired_lift(y_t: np.ndarray, y_b: np.ndarray, n_boot: int = 10_000, seed: int = 0):
    """Question-clustered paired lift. Returns dict with delta, se, bootstrap CIs."""
    y_t, y_b = np.asarray(y_t, float), np.asarray(y_b, float)
    assert y_t.shape == y_b.shape and y_t.ndim == 1
    d = (y_t - y_b) * 100.0
    n = d.size
    delta, se = d.mean(), d.std(ddof=1) / np.sqrt(n)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    lo95, hi95 = np.percentile(boots, [2.5, 97.5])
    lo90, hi90 = np.percentile(boots, [5.0, 95.0])
    return dict(n=n, delta=delta, se=se, ci95=(lo95, hi95), ci90=(lo90, hi90))

def mcnemar_exact(y_t: np.ndarray, y_b: np.ndarray):
    """Exact two-sided McNemar on binary greedy outcomes (k=1)."""
    t, b = np.asarray(y_t).astype(int), np.asarray(y_b).astype(int)
    n01 = int(((t == 1) & (b == 0)).sum()); n10 = int(((t == 0) & (b == 1)).sum())
    n = n01 + n10
    p = 1.0 if n == 0 else binomtest(min(n01, n10), n, 0.5).pvalue
    return dict(n01=n01, n10=n10, discordant_frac=n / t.size, p=p)

def decision(delta, se, go_min=1.0, margin=1.0, z_go=Z975, z_nogo=Z95):
    """Pre-registered three-state rule on the primary endpoint (analytic SE form).
    GO: CI95 lower > 0 and delta >= go_min.  NO-GO: one-sided 95% upper < margin
    (i.e. effect >= margin rejected at alpha=.05). Else INCONCLUSIVE."""
    if delta - z_go * se > 0 and delta >= go_min:
        return "GO"
    if delta + z_nogo * se < margin:
        return "NO-GO"
    return "INCONCLUSIVE"

def decision_boot(delta, ci95_lower, ci90_upper, go_min=1.0, margin=1.0, trunc_ok=True):
    """v5 rule using bootstrap CIs. ci90_upper is the upper end of the two-sided 90% CI
    (= one-sided 95% upper bound). GO also requires a clean truncation check (E4)."""
    if ci95_lower > 0 and delta >= go_min and trunc_ok:
        return "GO"
    if ci90_upper < margin:
        return "NO-GO"
    return "INCONCLUSIVE"

def noise_floor(y1: np.ndarray, y2: np.ndarray):
    """Base-vs-base discordance (different batch composition). Greedy binary arrays."""
    a, b = np.asarray(y1).astype(int), np.asarray(y2).astype(int)
    return float((a != b).mean())

def truncation_report(finished_t, finished_b):
    ft, fb = np.asarray(finished_t, bool), np.asarray(finished_b, bool)
    return dict(trunc_t=1 - ft.mean(), trunc_b=1 - fb.mean(),
                both_finished=float((ft & fb).mean()))

if __name__ == "__main__":
    # --- self-tests ---
    # exact McNemar: n01=10, n10=2 -> p = 2*(1+12+66)/4096
    t = np.array([1] * 10 + [0] * 2 + [1] * 50); b = np.array([0] * 10 + [1] * 2 + [1] * 50)
    r = mcnemar_exact(t, b); assert abs(r["p"] - 2 * (1 + 12 + 66) / 4096) < 1e-12, r
    # analytic SE vs bootstrap agree
    rng = np.random.default_rng(1); yb = rng.binomial(1, .6, 1319).astype(float)
    yt = np.clip(yb + rng.choice([-1, 0, 0, 0, 1], 1319), 0, 1)
    pl = paired_lift(yt, yb, n_boot=4000)
    boot_se = (pl["ci95"][1] - pl["ci95"][0]) / (2 * Z975)
    assert abs(boot_se - pl["se"]) / pl["se"] < 0.08, (boot_se, pl["se"])
    # identical arms -> delta 0, never GO
    assert paired_lift(yb, yb)["delta"] == 0 and decision(0.0, 0.5) in ("NO-GO", "INCONCLUSIVE")
    # decision boundaries
    assert decision(1.5, 0.3) == "GO" and decision(-1.0, 0.4) == "NO-GO" and decision(0.8, 0.9) == "INCONCLUSIVE"
    assert decision_boot(1.5, 0.5, 2.5) == "GO" and decision_boot(1.5, 0.5, 2.5, trunc_ok=False) == "INCONCLUSIVE"
    assert decision_boot(-0.5, -1.2, 0.2) == "NO-GO"
    print("stats.py self-tests passed")
