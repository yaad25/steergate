# Frozen evaluation protocol (master version)

Frozen before the first steered run. Any deviation is written into the report. Full rationale in PLAN.md.

## Three-way disjoint split (zero test leakage)
- **Extract** — `GSM8K-train[0:150]`: build contrastive traces (on-policy self-compressed + teacher-compressed).
- **Tune** — `GSM8K-train[150:350]` (N=200) + `WikiText-2[0:50]`: pick layer L* and scale α* (Gate 1.5). Lock to `locked_config.json` with SHA-256 **before test is ever loaded**.
- **Test** — full `GSM8K-test` (1,319 q): untouched until Gate 2.

## Sampling / generation (frozen)
- Sampled **k = 8**, **T = 0.7**, **top-p = 0.8**, cap **1024**, rep-penalty **1.0**, no top-k.
- **Common Random Numbers**: step-indexed Gumbel-max seeded by (master_seed, qid, sample_idx, step, token) — batch-invariant, identical noise across arms. (Implemented in `steergate_p0/sampling.py`.)

## Hyperparameter lock (Gate 1.5)
- Sweep L ∈ {12,14,16,18,20}, α ∈ {0.5,1.0,1.5,2.0} on the **tune** split only.
- Fluency ceiling: KL(base ‖ steered) ≤ **0.15 nats/token** on WikiText; truncation ≤ base + 0.5%.
- Lock single (L*, α*); never change it on the test set.

## The 6 arms
1. **base** — CoT prompt, no vector
2. **prompt_dense** — CoT + 1-line dense-equation instruction (the AxBench baseline you must beat)
3. **steered_onpolicy** — vector from self-compressed traces (zero external API)
4. **steered_teacher** — vector from DenseSteer-style teacher traces
5. **ctrl_random** — random direction, equal L2 norm
6. **ctrl_shuffled** — shuffled +/- labels, equal L2 norm

## Statistics
- Per-question pass rate over k=8; per-question paired differences.
- **10,000-sample question-clustered paired bootstrap**; two-sided 95% CI + one-sided 95% bound.

## Four-way verdict
- **BROKEN (oversteered)** — truncation or parse-fail rate > base + 1.0%.
- **GO** — not BROKEN, point lift ≥ +1.0, and 95% CI lower bound > 0 vs **base, ctrl_random, and ctrl_shuffled**.
  - GO-SUPERIOR if it also beats `prompt_dense`; GO-REDUNDANT if the prompt matches/beats it.
- **NO-GO** — one-sided 95% upper bound < +1.0, OR fails to beat random/shuffled controls.
- **INCONCLUSIVE** — upper bound ≥ +1.0 but lower bound ≤ 0.

## Integrity rules
- No claim of a gain until `g2b_verdict.json` says GO.
- Once `locked_config.json` is hashed, L*, α*, prompts, and regex are frozen for the test set.
- No Ollama/LM Studio claims. `llama.cpp --control-vector` export only post-Gate-2 if GO + an outside user asks.
