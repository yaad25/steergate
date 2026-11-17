# Does Activation Steering Improve Mathematical Reasoning? A Controlled Replication on Qwen2.5-1.5B-Instruct

**Author:** yaad25 (h2rn) · **License:** Apache-2.0 · **Code & data:** this repository

## Abstract

Activation steering — adding a direction vector to a language model's residual stream at
inference time — has been proposed as a training-free way to improve multi-step reasoning in
small language models (e.g., DenseSteer; Ouyang et al., 2026). We run a pre-registered,
controlled replication of an on-policy steering recipe on `Qwen2.5-1.5B-Instruct` and GSM8K,
using a frozen statistical protocol with paired sampling, two independent controls (a random
direction and a shuffled-pair direction, both norm-matched), and a prompt-only baseline. Across
the full 1,319-question test set (k=8 samples, 10,000-resample paired bootstrap), the steering
vector does not improve accuracy over the unsteered model (lift −1.24 points, 95% CI
[−1.86, −0.64]) and fails to beat even a random-direction control (−1.15 points,
95% CI [−1.77, −0.50]). A secondary analysis uncovers a hidden ~4-point accuracy loss in the
prompt-only baseline from a formatting-compliance bug that disproportionately penalizes dense,
equation-heavy outputs; once corrected, the prompt-only baseline beats steering as well. We
release the full evaluation harness, raw rollout logs, and locked configuration so the result
is independently reproducible.

**Verdict: NO-GO.** This specific recipe does not improve reasoning accuracy on this model and
dataset, and the result is robust to a lenient grading sensitivity check.

---

## 1. Introduction

Small language models often under-perform on multi-step arithmetic reasoning relative to their
larger counterparts, even when the underlying knowledge is present. Two broad families of
training-free fixes have been proposed: **prompting** (asking the model to show dense,
equation-heavy work) and **activation steering** (nudging the model's internal representations
toward a "good reasoning" direction extracted from contrastive examples, without touching any
weights).

Recent work (DenseSteer; Ouyang, Lin, and Kim, 2026) reports that steering a small model's
residual stream toward a "dense reasoning" direction — extracted from pairs of verbose vs.
compressed solution traces — improves GSM8K accuracy by roughly +2 points over an unsteered
baseline. Separately, Stanford's AxBench (Wu et al., 2025) found that across a range of
concept-steering tasks, a one-sentence natural-language instruction frequently matches or beats
activation-steering methods, and that increasing the steering strength monotonically degrades
fluency. These two results are in tension: is a small, trained-free steering vector a reliable
win, or does it only look that way without a careful prompt baseline and proper noise controls?

Published steering gains on math reasoning are small (≈+2 points) and evaluation protocols in
this literature vary widely in sample size, control arms, and grading strictness — exactly the
conditions under which a real effect can be indistinguishable from sampling noise, or a null
effect can look real due to an uncontrolled confound (e.g., answer-formatting compliance). We
set out to test, under a frozen, adversarially-controlled protocol, whether an on-policy
version of this steering recipe reproduces on a popular small open model.

### Contributions
1. A fully pre-registered, controlled replication protocol for activation steering on math
   reasoning, with two independent null controls and a prompt-only baseline, released as a
   reusable open-source harness (`steergate_p0`).
2. An honest negative result: on-policy dense-reasoning steering does not improve, and in fact
   measurably *hurts*, GSM8K accuracy on `Qwen2.5-1.5B-Instruct` under our protocol.
3. A methodological finding independent of the steering question: naive strict-format grading
   (requiring a literal `\boxed{}` tag) can hide a multi-point accuracy difference between arms
   when one arm's prompting style changes how often the model remembers to format its answer —
   and this bias is *not* uniform across arms, so it can silently favor or disfavor whichever
   arm is being tested.

---

## 2. Related Work

- **DenseSteer** (Ouyang, Lin, and Kim, ICML 2026; arXiv:2605.29247). Proposes steering small
  models toward a "dense reasoning" direction extracted by rewriting a model's own verbose trace
  into a compressed one, reporting ≈+2 point GSM8K gains on Qwen2.5-1.5B/3B. Our on-policy arm
  (`steered_onpolicy`) directly implements the zero-external-API version of this recipe.
- **AxBench** (Wu et al., 2025; arXiv:2501.17148). A large-scale benchmark showing that simple
  prompting baselines frequently match or outperform steering-vector methods, and that steering
  strength has a fluency/efficacy trade-off. Motivates our `prompt_dense` baseline arm and our
  use of an explicit fluency/truncation guard.
- **Contrastive Activation Addition / Representation Engineering** (Rimsky et al.; Zou et al.).
  Establish the mean-difference extraction method we use (CAA) and the broader linear
  representation hypothesis underlying steering.
- **Arditi et al.** (refusal direction). Shows that *directional ablation* of a single linear
  direction can remove a specific behavior (refusal) without retraining — the same family of
  technique as the "abliteration" used to produce the uncensored model that powered the agentic
  tooling built alongside this project, and conceptually related to our random/shuffled control
  construction.

---

## 3. Method

### 3.1 Model and task
`Qwen2.5-1.5B-Instruct`, evaluated on the full GSM8K test set (1,319 questions). Generation:
temperature 0.7, top-p 0.8, k=8 samples per question, max 1024 new tokens, repetition penalty
1.0. Sampling uses a step-indexed Gumbel-max common-random-number scheme so that every arm sees
*identical* sampling noise for a given (question, sample) pair, independent of batch size —
this removes a major source of unpaired variance between arms.

### 3.2 Three-way disjoint data split
- **Extraction** — `GSM8K-train[0:150]`: 150 questions used to build the on-policy contrastive
  trace pairs (118 were kept, filtered to cases the model itself answered correctly).
- **Tune** — a held-out slice of `GSM8K-train` + 50 WikiText-2 passages: used once, before the
  test set was ever touched, to sweep candidate layers {12,14,16,18,20} × α ∈ {1.0, 1.5} under a
  KL-divergence fluency guard (KL ≤ 0.15 nats/token vs. unsteered) and a truncation-rate guard.
  **Locked result: layer 16, α = 1.5** (full sweep table in `vectors/locked_config.json`).
- **Test** — the full 1,319-question GSM8K test set, untouched until this evaluation.

### 3.3 Vector construction
For each of 118 kept trace pairs, the model's own verbose ("sparse") solution is deterministically
compressed into a dense version (filler sentences stripped, equations and the final boxed answer
kept) — zero external API calls. The steering vector is the normalized mean difference between
dense- and sparse-trace activations at the locked layer, scaled to a dosage matching the locked
α on each token's residual-stream norm.

### 3.4 Arms (all five are paired on identical sampling noise)
| Arm | Description |
|---|---|
| `base` | Unsteered model, standard CoT instruction |
| `prompt_dense` | Unsteered model, instructed to show explicit equations for every step |
| `steered_onpolicy` | Base instruction + the locked on-policy steering vector |
| `ctrl_random` | Base instruction + a random direction, L2-norm-matched to the real vector |
| `ctrl_shuffled` | Base instruction + a vector built from the same 118 pairs with randomly flipped +/- labels, norm-matched |

A sixth arm (`steered_teacher`, using a frontier-model-rewritten trace as in the original
DenseSteer recipe) was planned but cut: it requires an external LLM API we did not have access
to. This is a scope limitation, documented rather than silently dropped.

### 3.5 Statistics and the frozen decision rule
For each arm, per-question accuracy is the mean correctness over its k=8 samples. For a given
comparison, we compute the paired per-question difference and run a 10,000-sample
question-clustered bootstrap to get a point estimate and 95%/one-sided-95% confidence intervals.
**Decision rule (frozen before any steered result was seen):**
- **GO** requires the steered arm's lower 95% CI bound to exceed 0 against **all three** of
  base, `ctrl_random`, and `ctrl_shuffled** simultaneously, with point lift ≥ +1.0 points.
- **NO-GO** if the one-sided 95% upper bound is below +1.0, or the arm fails to beat either
  control's point estimate.
- **BROKEN** if truncation or box-parse failure rate rises more than 1 point above base
  (none of our arms triggered this).

---

## 4. Results

### 4.1 Primary result (strict grading, full protocol)

| Arm | Accuracy | Truncation | `\boxed{}` compliance |
|---|---|---|---|
| base | 73.25% | 0.33% | 98.64% |
| prompt_dense | 70.66% | 0.61% | 94.60% |
| **steered_onpolicy** | **72.01%** | 0.32% | 98.19% |
| ctrl_random | 73.15% | 0.32% | 98.90% |
| ctrl_shuffled | 72.42% | 0.33% | 98.09% |

| Comparison (steered_onpolicy vs.) | Lift | 95% CI | Decision |
|---|---|---|---|
| base | −1.24 pts | [−1.86, −0.64] | **NO-GO** |
| prompt_dense | +1.35 pts | [+0.42, +2.29] | (moot — see below) |
| ctrl_random | −1.15 pts | [−1.77, −0.50] | **NO-GO** |
| ctrl_shuffled | −0.42 pts | [−0.95, +0.10] | **NO-GO** |

Under the frozen rule, a single failed comparison against base or either control is sufficient
for an overall NO-GO. Steering fails against **base** and **ctrl_random**, with confidence
intervals excluding zero in both cases — this is not an inconclusive null result, it is a
*significant negative* effect. Steering does beat `prompt_dense` on strict grading, but this
does not change the overall verdict, and §4.2 shows this specific comparison is itself an
artifact.

### 4.2 Secondary finding: a formatting-compliance confound

While inspecting failure cases, we found instances where the model produced the exactly correct
final number but never wrapped it in `\boxed{}}`, causing the strict grader to mark a correct
answer as wrong. We quantified this across all five arms by checking, for every row without a
box, whether the last number in the generated text matches the gold answer:

| Arm | `\boxed{}` missing | Hidden accuracy loss |
|---|---|---|
| base | 1.36% | 0.60 pts |
| **prompt_dense** | **5.40%** | **3.20 pts** |
| steered_onpolicy | 1.81% | 0.97 pts |
| ctrl_random | 1.10% | 0.48 pts |
| ctrl_shuffled | 1.91% | 1.00 pts |

The bug disproportionately affects `prompt_dense` — its "explicit equations for every step"
instruction apparently makes the model more likely to end on a worked equation rather than a
boxed restatement. Re-grading all five arms with a lenient parser (credit a correct final number
even without a box) and recomputing the same four comparisons:

| Comparison (lenient grading) | Lift | 95% CI | Decision |
|---|---|---|---|
| vs. base | −0.87 pts | [−1.45, −0.27] | **NO-GO** |
| vs. prompt_dense | **−0.88 pts** | [−1.74, −0.01] | **NO-GO** (flipped from GO) |
| vs. ctrl_random | −0.64 pts | [−1.27, −0.03] | **NO-GO** |
| vs. ctrl_shuffled | −0.44 pts | [−0.94, +0.07] | **NO-GO** |

Under lenient grading, steering loses to **every single arm**, including `prompt_dense`. The
strict-grading "win" against `prompt_dense` in §4.1 was entirely an artifact of unfairly
penalizing the prompt baseline's formatting habits, not a genuine steering advantage. This is a
cautionary methodological result on its own: any steering-vs-prompt comparison that relies on
strict answer-format matching should separately verify that formatting compliance itself is not
confounded with the experimental manipulation.

---

## 5. Discussion

**Why might on-policy steering fail here where DenseSteer reports a gain?** Several
non-exclusive possibilities, none of which we can distinguish with this dataset alone:
1. **Trace quality.** Our on-policy traces are self-compressed by the model itself with zero
   external supervision; DenseSteer's main reported numbers may rely on more carefully curated
   or teacher-rewritten traces (our `steered_teacher` arm, which would test this directly, was
   cut for lack of API access — see §3.4).
2. **Harness/prompt differences.** Our base accuracy (73.25% k=8 sampled; 75.28% in an earlier
   greedy-only sanity check) differs from some externally reported baselines for this model,
   consistent with the well-documented sensitivity of GSM8K scoring to prompt template and
   extraction method across papers.
3. **A genuinely small or absent effect for this specific layer/scale/model combination.**
   Published steering gains in this space are already small (≈+2 points); a result this close
   to the noise floor can easily flip sign across model sizes, layers, or trace-construction
   choices.

We do not claim to have falsified DenseSteer's reported result — only that this specific,
carefully-controlled on-policy replication on this model does not reproduce a positive effect,
and in fact shows a significant negative one against two independent null controls.

### Limitations
- The teacher-distilled arm was cut (no external LLM API available); we only test the
  zero-API on-policy variant.
- The hyperparameter sweep in Gate 1.5 was trimmed (N=80 tune questions, α∈{1.0,1.5} instead of
  the originally planned N=200, α∈{0.5,1.0,1.5,2.0}) under same-day time pressure; a wider sweep
  might locate a better-performing (layer, α) pair. This is a documented deviation from the
  original plan, not a silent one.
- Single model, single dataset. We make no claim about transfer to other model families, sizes,
  or reasoning benchmarks.

---

## 6. Conclusion

Under a pre-registered, adversarially-controlled replication protocol — paired sampling, two
independent null controls, and a prompt-only baseline — on-policy dense-reasoning activation
steering does not improve GSM8K accuracy on `Qwen2.5-1.5B-Instruct`, and significantly
*underperforms* both the unsteered model and a random-direction control. This result is stable
under a lenient grading sensitivity check, which additionally reveals that a formatting-
compliance confound can mask a multi-point accuracy difference in strict-grading comparisons.
We release the full harness, locked configuration, and raw per-question logs so this result can
be independently verified, extended to other models, or falsified.

---

## Reproducibility

- Code: `steergate_p0/` (this repository)
- Frozen protocol: `PROTOCOL.md`
- Locked hyperparameters + extraction sweep: `vectors/locked_config.json`
- Raw rollout logs (all 5 arms, byte-verified complete): `runs_backup/*.jsonl`
- Re-run the primary verdict: `python -m steergate_p0.analyze compare --base runs_backup/base.jsonl --treat runs_backup/steered_onpolicy.jsonl` (repeat `--base` for each comparator)
- Re-run the lenient sensitivity check: `python lenient_sensitivity.py`
- Full build/debugging log, including every bug found and fixed during data collection:
  `PROGRESS.md`

## References

1. Ouyang, Y., Lin, S., & Kim, J. (2026). *DenseSteer: Steering Small Language Models towards
   Dense Math Reasoning.* ICML 2026 / arXiv:2605.29247.
2. Wu, Z., Arora, A., Geiger, A., Wang, Z., Huang, J., Jurafsky, D., Manning, C. D., & Potts, C.
   (2025). *AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders.*
   arXiv:2501.17148.
3. Rimsky, N. et al. *Steering Llama 2 via Contrastive Activation Addition.*
4. Zou, A. et al. *Representation Engineering: A Top-Down Approach to AI Transparency.*
5. Arditi, A. et al. *Refusal in Language Models Is Mediated by a Single Direction.*
