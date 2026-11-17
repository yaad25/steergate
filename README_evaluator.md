# steergate_p0 — Phase 0 evaluator for the SteerGate v5 plan

Paired, seeded, truncation-aware GSM8K evaluation. One run = one arm; analysis compares arms question-by-question.
Tested end-to-end (14 tests) on a tiny random Qwen2 model with a char tokenizer, then run for real on
Qwen2.5-1.5B-Instruct across all 5 arms (10,552 rows each) — see [PAPER.md](PAPER.md) for the full result
(verdict: NO-GO) and [PROGRESS.md](PROGRESS.md) for the build/run log. The commands below still work as
a from-scratch walkthrough if you want to reproduce or extend the run yourself.

## Install
    pip install torch transformers datasets safetensors numpy scipy pytest
    pytest -q tests          # ~6 s, CPU only

## Design guarantees (all unit-tested)
- Sampling = Gumbel-max with a counter-based hash: the noise for a row depends only on (seed_base, qid, sample_idx, step, token).
  Same questions/samples get the same noise in every arm (common random numbers) and results do not depend on batch size.
- Generation config is overridden explicitly (rep. penalty 1.0, no top-k), so checkpoint defaults can't leak in.
- Steering hook is on `model.model.layers[b]` (0-based block b); extraction convention is `hidden_states[b+1]` (== that block's output).
  `--steer-mode all` matches llama.cpp control vectors; `decode_only` steers generated tokens only.
- Every row stores `finished` (EOS before the cap), `has_box`, `ntok`; `analyze compare` refuses GO if truncation exceeds base's.
- Runs resume (rerun the same command) and write `<out>.meta.json` (versions, generation config, vector sha256, perturbation norm, tok/s).

## Commands (Qwen2.5-1.5B-Instruct shown)
    M=Qwen/Qwen2.5-1.5B-Instruct
    # dry run (20 questions)
    python -m steergate_p0.evaluate --model $M --arm base --out runs/dry.jsonl --limit 20 --k 2
    # P0-a: evaluator validity — greedy baseline accuracy should be ~60-66% on the full test set
    python -m steergate_p0.evaluate --model $M --arm base_g --greedy --max-new-tokens 2048 --out runs/base_g.jsonl
    python -m steergate_p0.analyze summary runs/base_g.jsonl
    # P0-b/c: noise floor — same greedy run at two batch sizes / dtypes
    python -m steergate_p0.evaluate --model $M --arm base_g2 --greedy --max-new-tokens 2048 --batch-size 8 --out runs/base_g2.jsonl
    python -m steergate_p0.analyze noise runs/base_g.jsonl runs/base_g2.jsonl
    # primary baseline (k=8 sampled), then treatment with a vector file containing block_<b>
    python -m steergate_p0.evaluate --model $M --arm base --k 8 --out runs/base.jsonl
    python -m steergate_p0.evaluate --model $M --arm steer --k 8 --steer-vector vec.npz --block 14 --lam 1.0 --out runs/steer.jsonl
    python -m steergate_p0.analyze compare --base runs/base.jsonl --treat runs/steer.jsonl
    # control: random direction, identical norm
    python -m steergate_p0.evaluate ... --arm c1 --steer-vector vec.npz --block 14 --lam 1.0 --control random --out runs/c1.jsonl

## P0-d: released DenseSteer vectors (security)
The released `.pt` files are pickles. Convert only inside a disposable, network-less sandbox:
    python -m steergate_p0.convert_vector vec.pt -o vec.npz        # weights_only=True by default
then use `vec.npz` anywhere. DenseSteer's block numbering is 0-based (`direction.N` <-> block N); the llama.cpp parity test in the plan settles it.

## Caveats
- `analyze compare` GO/NO-GO is meaningful only for the pre-registered primary run (full 1,319 questions, k=8, T=0.7, top-p 0.8, cap 1024).
- Add a temperature-matched baseline arm (T=0.7 base is already the default baseline) and a prompt-only arm via `--system`/`--instruction` (plan P2).
- Compute hooks on multi-GPU (`device_map`) models are untested; use a single device.
