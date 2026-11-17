# Runbook — the gates, in order

**All three gates have been run for real; final verdict is NO-GO.** See [PAPER.md](PAPER.md) for
the result and [PROGRESS.md](PROGRESS.md) for the full log. The commands below are kept as the
exact reproduction steps — follow them if you want to re-run or extend this yourself.

Run these on a GPU (free Kaggle T4 is fine). Each gate can kill or redirect the project — that's the point. Do NOT skip ahead to building the CLI.

```bash
M=Qwen/Qwen2.5-1.5B-Instruct
pip install torch transformers datasets safetensors numpy scipy pytest
pytest -q tests        # must pass first (~6s, CPU)
```

## Gate 1 (day 1–2) — evaluator validity
Pass bar: greedy baseline accuracy **60–66%**. If outside that, fix prompt/extraction (max 2 tries) then stop.
```bash
# 20-question dry run (pipeline sanity)
python -m steergate_p0.evaluate --model $M --arm base --out runs/dry.jsonl --limit 20 --k 2
python -m steergate_p0.analyze summary runs/dry.jsonl

# P0-a greedy baseline on the FULL test set
python -m steergate_p0.evaluate --model $M --arm base_g --greedy --max-new-tokens 2048 --out runs/base_g.jsonl
python -m steergate_p0.analyze summary runs/base_g.jsonl
```

## Gate 2 (day 3–10) — noise floor + replication
Pass bar: a verdict (GO / NO-GO / INCONCLUSIVE) with a CI, and the noise floor documented.
```bash
# noise floor: same greedy run at a second batch size, compare
python -m steergate_p0.evaluate --model $M --arm base_g2 --greedy --max-new-tokens 2048 --batch-size 8 --out runs/base_g2.jsonl
python -m steergate_p0.analyze noise runs/base_g.jsonl runs/base_g2.jsonl

# primary sampled baseline (k=8, full set)
python -m steergate_p0.evaluate --model $M --arm base --k 8 --out runs/base.jsonl

# treatment: a real published vector (see P0-d) in vec.npz, steered at its block
python -m steergate_p0.evaluate --model $M --arm steer --k 8 --steer-vector vec.npz --block 14 --lam 1.0 --out runs/steer.jsonl

# controls
python -m steergate_p0.evaluate --model $M --arm c1 --steer-vector vec.npz --block 14 --lam 1.0 --control random --out runs/c1.jsonl

# verdict
python -m steergate_p0.analyze compare --base runs/base.jsonl --treat runs/steer.jsonl
```

## Gate 3 (week 0, parallel) — demand
Send the message in PLAN.md to 10 people. Pass bar: 3 describe the problem unprompted AND 2 say they'd try the tool. If it fails: Goal A only (replication note, no CLI).

## P0-d security — handling released DenseSteer vectors
Their `.pt` files are pickles (can run code). Convert ONLY in a disposable, network-less sandbox:
```bash
python -m steergate_p0.convert_vector vec.pt -o vec.npz   # weights_only=True by default
```
Then use `vec.npz` anywhere.

## Only after Gate 2 passes
Build `steergate report`, then `steergate gate`. GitHub Action + llama.cpp export only if G2 = GO and someone asks. Publish a replication log with every result, including negatives.
