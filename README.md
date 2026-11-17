# SteerGate

**A checker that tells you whether a steering vector's gain on a model is real, or just sampling noise.**

Steering-vector gains on reasoning benchmarks are small (+1 to +2 points at best) and easily faked by noise or by forgetting to compare against a plain "think step by step" prompt. SteerGate runs **paired, seeded** evaluations with a **random-direction control** and a **prompt-only baseline**, and returns **one verdict with a confidence interval**: GO, NO-GO, or INCONCLUSIVE.

> Philosophy: **validate first, build second.** Nothing here claims a gain until the frozen protocol says so. See [PLAN.md](PLAN.md).

By: **yaad25 / h2rn** · License: Apache-2.0

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24292138.svg)](https://doi.org/10.5281/zenodo.24292138)

**100% free, no commercial tier, no paid features — ever.**

---

## Status (honest)

**Gate 2 complete. Verdict: NO-GO.** On-policy dense-reasoning steering does not improve, and
significantly *hurts*, GSM8K accuracy on Qwen2.5-1.5B-Instruct under this protocol — it loses to
both the unsteered model and a random-direction control. Full writeup in [PAPER.md](PAPER.md),
full build log in [PROGRESS.md](PROGRESS.md).

| Part | State |
|---|---|
| `steergate_p0` evaluator (paired GSM8K, controls, stats) | ✅ built, 14 tests pass on a tiny model |
| Gate 1 (greedy baseline) | ✅ done — 75.28% accuracy |
| Gate 1.5 (hyperparameter lock) | ✅ done — layer 16, alpha 1.5 |
| Gate 2 (full 5-arm evaluation, N=1,319, k=8) | ✅ done — **NO-GO** (see [PAPER.md](PAPER.md)) |
| `steergate report` / `steergate gate` CLI | 🔒 not built — Gate 2 did not pass, so per PLAN.md this isn't built |
| GitHub Action, llama.cpp export | 🔒 not built — same reason |

Everything marked 🔒 is intentionally not built, per the frozen protocol in PLAN.md: those tools
only get built on a GO. This is a negative result, not an incomplete project.

---

## Quick start

```bash
pip install torch transformers datasets safetensors numpy scipy pytest
pytest -q tests            # ~6s CPU, proves the evaluator works

# Gate 1: 20-question dry run on a real model
M=Qwen/Qwen2.5-1.5B-Instruct
python -m steergate_p0.evaluate --model $M --arm base --out runs/dry.jsonl --limit 20 --k 2
python -m steergate_p0.analyze summary runs/dry.jsonl
```

Full gate commands are in [RUNBOOK.md](RUNBOOK.md). The frozen statistical protocol is in [PROTOCOL.md](PROTOCOL.md).

---

## What makes this different from existing tools

`repeng` and `llama.cpp` already *create and apply* steering vectors. SteerGate does the part they don't: **tells you if the result is trustworthy** — held-out layer selection, a fluency gate, a random-direction control, and a prompt baseline, with a verdict you can reproduce.

See [PLAN.md](PLAN.md) for the full plan, the kill-gates, and the honest self-rating.

---

SteerGate is free and always will be — no paid tier, no commercial license, no feature gating.
