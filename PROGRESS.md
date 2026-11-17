# FINAL RESULT — Gate 2 complete, verdict: NO-GO

**All 5 arms completed, verified byte-exact and schema-clean (10,552/10,552 each, 10,552 unique
qid/sample pairs, zero corruption).** Full protocol: N=1,319 questions, k=8 samples, paired
question-clustered bootstrap (10,000 resamples), frozen layer=16/alpha=1.5 lock from Gate 1.5.

## Primary verdict (strict `\boxed{}` grading — the frozen protocol)

| Comparison | Lift | 95% CI | Verdict |
|---|---|---|---|
| `steered_onpolicy` vs **base** | **-1.24 pts** | [-1.86, -0.64] | **NO-GO** (steering significantly *hurts*) |
| `steered_onpolicy` vs **prompt_dense** | +1.35 pts | [+0.42, +2.29] | GO (beats the prompt baseline) |
| `steered_onpolicy` vs **ctrl_random** | **-1.15 pts** | [-1.77, -0.50] | **NO-GO** (loses to pure random noise!) |
| `steered_onpolicy` vs **ctrl_shuffled** | -0.42 pts | [-0.95, +0.10] | NO-GO (CI straddles zero, upper bound < 1.0) |

**Overall: NO-GO.** The GO rule requires beating base, ctrl_random, AND ctrl_shuffled all at
once (per PROTOCOL.md) — steering fails against base and the random-noise control, with
statistically significant *harm*, not just a lack of benefit.

## Formatting-trap sensitivity check (lenient fallback-to-last-number parser)
Re-graded all 5 arms allowing credit for a correct final number even without `\boxed{}`:

| Arm | Strict acc | Lenient acc | Gained |
|---|---|---|---|
| base | 73.25% | 73.85% | +0.60pp |
| prompt_dense | 70.66% | 73.86% | **+3.20pp** |
| steered_onpolicy | 72.01% | 72.98% | +0.97pp |
| ctrl_random | 73.15% | 73.63% | +0.48pp |
| ctrl_shuffled | 72.42% | 73.42% | +1.00pp |

**Verdicts under lenient grading: NO-GO against ALL FOUR** (base, prompt_dense, ctrl_random,
ctrl_shuffled). The `prompt_dense` comparison **flips** from GO (strict) to NO-GO (lenient) —
meaning the earlier "steering beats the prompt baseline" result was an **artifact of the
formatting bug unfairly penalizing `prompt_dense`**, not a genuine steering advantage. Once
both arms are graded fairly, the prompt-only baseline is *better* than steering, confirming
AxBench's finding even more cleanly than the strict numbers suggested.

## Bottom line
Under both strict and lenient grading, with full statistical rigor (paired, bootstrapped,
pre-registered controls), **this specific on-policy steering recipe (layer 16, alpha 1.5,
self-compressed contrastive traces) does not improve — and measurably hurts — GSM8K accuracy
on Qwen2.5-1.5B-Instruct**, failing to beat even a random-direction control. This is a clean,
honest, publishable negative result exactly matching the "NO-GO branch" PLAN.md anticipated
as a legitimate, valuable outcome.

---



## Infrastructure: resumable launches + chunked backups (built during a long disconnect)
After losing Round 1's in-progress generation (base/prompt_dense) to full-session wipes
multiple times in a row — once from an accidental "Save Version" commit attempt, several
times from what looks like Kaggle's own 40-min idle-editor kernel kill — built proper
resilience instead of just retrying the same losing pattern:
- `_pull_chunked.py` (in `colab ai/`): pulls a remote file off Kaggle in small base64 chunks
  and reassembles it locally, working around the exec server's ~20KB stdout truncation that
  made a single-shot full-file pull silently cut off (this is what caused the earlier failed
  backup attempt with the garbled JSON).
- `_resumable_launch.py`: before launching an arm, pushes any local backup of that arm's
  `.jsonl` back to Kaggle first, so `evaluate.py`'s own built-in resume logic (it already
  skips any `(qid, sample)` pair already present in `--out`) picks up from there instead of
  starting at row 0. Launches via `python _resumable_launch.py <arm> <gpu> [extra args...]`.
  Handles the `prompt_dense` instruction text safely (base64, avoiding the backslash-escaping
  bug found earlier — verified the actual file content is clean, that bug was specific to
  passing things through nested bash/python -c command lines, not an issue for files written
  directly).
- `_backup_progress.py`: pulls all 5 arms' current `.jsonl` progress to
  `steergate/runs_backup/` — meant to run at every successful monitoring check-in, so a
  disconnect never loses more than the progress made since the last check.
- **Going forward:** every relaunch should use `_resumable_launch.py` instead of the old raw
  Popen snippets, and every monitoring check-in should run `_backup_progress.py` right after
  confirming the arms are alive. This turns "lose everything on every crash" into "lose at
  most ~20-30 minutes," which should make the repeated Kaggle instability survivable instead
  of a full reset each time.


One entry per session/milestone. Newest on top. Times are approximate (IST, today's session).

---

## Gate 1.5 — hyperparameter lock (DONE)
- **LOCKED: layer=16, alpha=1.5** (`vectors/locked_config.json`).
- Tune-split accuracy: 82.5% (base) → **86.25%** (steered) — a +3.75 point gain on the trimmed
  80-question tune split. KL divergence 0.00095 (far under the 0.15 ceiling), no extra truncation
  vs base. 9/10 (layer, alpha) combos passed the guardrail; only L=12, a=1.0 was rejected
  (truncation 3.75% vs base 2.5%, over the +0.5pp slack).
- Clear monotonic-ish trend: accuracy generally rises with layer up to 16 then falls off at 18-20,
  matching the literature's expectation that middle layers (~50-60% depth) are the "golden zone"
  for steering; higher alpha (1.5 > 1.0) consistently did as well or better within the tested range.
  **Caveat:** this is an 80-question tune split, not proof — it's a hyperparameter pick, not the
  result. The real test is Gate 2 on the full held-out 1,319-question test set.
- Ran clean end-to-end in ~23 min total (traces + activations + 10-combo sweep) without a Kaggle
  disconnect this time.
- Built `steergate_p0/build_controls.py` next: ctrl_random and ctrl_shuffled vectors weren't saved
  by extract.py (only the mean-difference vector was kept, not per-pair diffs). This script
  deterministically reruns trace generation (same 118 traces expected) and captures activations at
  *only* the locked layer (cheap vs. the full 5-layer sweep) to build both controls, norm-matched
  to the locked steering vector.

## Gate 1.5 — hyperparameter lock (history while getting here)
- Built `steergate_p0/extract.py`: on-policy trace generation (compress model's own verbose trace into
  a dense one, zero external API), CAA mean-difference vector extraction at candidate layers,
  (layer, alpha) sweep with KL + truncation guardrails, locks to `vectors/locked_config.json`.
- **Scope cut (documented):** teacher arm (`steered_teacher`) skipped — needs GPT-5.1/external API,
  not available. Only `steered_onpolicy` is being built. Can revisit later with the local 30B as a
  stand-in teacher if wanted.
- Smoke-tested on a tiny CPU model first (7/7 tests pass, `tests/test_extract_smoke.py`) before
  spending real GPU time — caught nothing, confirms the math/shapes are correct.
- **Bug found & fixed:** `load_dataset("wikitext", ...)` fails on the current `datasets` library
  (needs a namespaced repo id) → fixed to `"Salesforce/wikitext"`.
- **Perf issue found & fixed:** first real run on Qwen2.5-1.5B had NO visible progress for 14+ minutes
  because `extract.py`'s prints weren't flushed when redirected to a log file (unlike `evaluate.py`,
  which explicitly uses `flush=True`). Fixed: forced `print(..., flush=True)` globally in the module
  and launch with `python -u`.
- **Scope trimmed for today's time pressure (documented, not silent):** full protocol calls for
  tune-N=200 and a 4-value alpha grid (5 layers × 4 alphas = 20 combos, each a full greedy pass —
  estimated 1.5-2.5h on a single T4). Ran instead with `--tune-n 80 --alphas 1.0,1.5` (5×2=10 combos,
  ~110s each ≈ 18 min). Logged via `--deviation-note` into `locked_config.json` for the report.
  Full-size sweep can be re-run later if the trimmed one gives an INCONCLUSIVE/borderline result.
- First trimmed run: got to 4/10 combos (L=12 a=1.0 REJECTED on truncation; L=12 a=1.5, L=14 a=1.0,
  L=14 a=1.5 all OK, small tune-accuracy gains visible e.g. 0.825 base → 0.850) before the Kaggle
  session died.

## Kaggle session reliability (root-caused)
- Root cause of repeated disconnects: **not** random tunnel flakiness as first assumed — it's
  Kaggle's own 40-minute **idle-editor** timeout. Long background jobs polled only through the exec
  tunnel don't count as "interacting with the editor," so the session gets killed.
- Fix identified: use **Save Version → Save & Run All (Commit)** instead of running cells
  interactively. This runs on Kaggle's servers independent of browser/editor focus, up to the real
  session time limit, not the 40-min idle check.
- Separately patched `kaggle_server.ipynb`'s tunnel cell with a self-healing watchdog thread so a
  dropped `cloudflared` tunnel auto-reconnects and reprints fresh URLs (cell 5b) without a full
  notebook restart — this helps when it's *just* the tunnel, not the whole kernel dying.
- Each full kernel restart wipes `/kaggle/working/steergate` and reloads the 30B Ollama model onto
  the GPU automatically (it's set to "Forever" keep-alive), which also has to be killed by PID each
  time before there's room to run the 1.5B eval model. Routine now: kill ollama+llama-server by PID
  → verify 14.9GB free on both T4s → re-push steergate via `_push_steergate.py` → relaunch.

## Gate 1 — dry run + greedy baseline (DONE)
- 20-question dry run: 60.0% accuracy (noisy small sample, within plausible range).
- Full 1,319-question greedy baseline: **75.28% accuracy**, 0.30% truncation, 99.17% boxed compliance.
  Clean single run (after killing an accidental duplicate process from an earlier timed-out launch
  attempt that left a detached background job alive — found via `ps aux`, killed, output file wiped
  and regenerated cleanly).
- **Documented deviation:** PLAN.md's assumed pass bar was 60-66% (from DenseSteer's reported
  baseline for this model). Our real number (75.28%) is outside that window.
  Investigated for a bug: checked `extract_answer.py` (strict, correct `\boxed{}` brace-matching,
  exact-match grading — no leniency bug) and manually read 3 random sample generations (genuine
  correct/incorrect reasoning, correctly graded, no contamination). **Conclusion: not a bug** — our
  harness uses the model's native chat template + explicit boxed-answer instruction (Qwen's own
  recommended eval setup), which legitimately scores higher than DenseSteer's specific prompt/harness.
  This is a well-known source of variance across GSM8K eval setups for the same model.
  **Decision:** accept 75.28% as our own harness's verified baseline and proceed, per the plan's own
  "validate first" philosophy — rather than tuning the harness to artificially match another paper's
  number (which would risk introducing a real bug to chase an arbitrary target).

## Setup — evaluator built & verified
- `steergate_p0` core (sampling.py with step-indexed Gumbel-max common random numbers, steer.py,
  controls.py, stats.py, analyze.py, evaluate.py, extract_answer.py): 14 unit tests pass, both
  locally and on Kaggle.
- Project scaffolded: README, PLAN.md (master v4), PROTOCOL.md (frozen stats protocol), GOAL.md
  (Goal A: replication note + reproducible checker; monetization explicitly 100% free, donations
  only, logged as a decision), RUNBOOK.md, pyproject.toml, NOTICE.
- Decision: **100% free forever, no commercial tier, donations optional and never gated.**

## Finding: "formatting trap" quantified across all 4 running arms (mid-run, informal checks)
User spotted a real failure case: `steered_onpolicy` qid 387 got the math 100% right ($36) but
never wrapped it in `\boxed{}`, so the strict grader marked it wrong. Investigated properly
without touching the live run (pulled current partial data, checked how many no-box rows
actually have the correct answer as the last number in the text):

| Arm | No-box rate | Hidden accuracy loss (% of all rows) |
|---|---|---|
| base | 0.98% | 0.67% |
| **prompt_dense** | **5.42%** | **4.07%** |
| steered_onpolicy | 1.71% | 1.16% |
| ctrl_random | 0.95% | 0.61% |

**Key finding: this bug hits `prompt_dense` hardest (4x worse than any other arm)**, likely
because the custom "explicit equations for every step" instruction leads to worked-equation
style answers that often skip the final `\boxed{}` wrap. It does NOT disproportionately punish
`steered_onpolicy` — its hidden loss (1.16%) is close to base/ctrl_random's (0.6-0.7%), not
large enough to flip its current standing even if corrected.

**Decision:** do not change grading mid-run (would risk looking like moving the goalposts after
seeing partial results). Instead, add a **documented sensitivity check** to the final report:
rerun the verdict with both the strict `\boxed{}`-only parser and a lenient
fallback-to-last-number parser, and report whether the verdict changes. Keeps the primary
frozen protocol intact while still being rigorous about a real limitation found along the way.

## Informal mid-run accuracy trend (NOT the final verdict — unpaired, partial, different
question subsets per arm since they run at different speeds)
Checked repeatedly as more data came in, including both raw and formatting-bug-corrected
("adjusted") accuracy:
1. ~35-40% done: base 85.4% > ctrl_random 85.0% > prompt_dense 83.1% > steered_onpolicy 82.9%
2. ~45% done: ctrl_random 84.6% > base 84.3% > steered_onpolicy 82.9% > prompt_dense 82.0%
3. ~50-56% done: base 84.1% > ctrl_random 82.6% > prompt_dense 81.3% > steered_onpolicy 80.9%
4. **~58-69% done (latest, computed locally from manually-downloaded files, official vs
   adjusted):**
   | Arm | n | Official | Adjusted | Hidden gain |
   |---|---|---|---|---|
   | base | 6560 | 81.42% | 82.09% | +0.67pp |
   | ctrl_random | 7008 | 80.25% | 80.81% | +0.56pp |
   | prompt_dense | 6048 | 79.08% | **82.95%** | +3.87pp |
   | steered_onpolicy | 7296 | 78.74% | 79.81% | +1.07pp |

`steered_onpolicy` has been at or near last in **every single check** (4 for 4), including
trailing the random-noise control, and the formatting-bug correction doesn't rescue it —
if anything `prompt_dense` gains the most from that correction and pulls ahead. Trending
strongly toward a likely **NO-GO**, but not official until the full paired bootstrap on all
1,319 questions.

## Bug found & fixed: chunked pull could corrupt a GOOD existing backup on failure
`_pull_chunked.py`/`_pull_chunked2.py`'s `pull_file()` opened the **destination** backup file
in `"wb"` mode immediately (truncating it to 0 bytes) before starting the chunk loop. When a
chunk later failed (either a real path-translation bug, or transient tunnel corruption causing
a base64 "Incorrect padding" error), the exception propagated up — leaving the backup file
**partially overwritten with less data than before**, not restored to its original state.
This actually happened and cost real progress: `base`'s local backup dropped from 6,560 rows
to 6,672,000 bytes (~6,184 rows after resume), and `prompt_dense`'s dropped catastrophically
from 6,048 rows to just ~1,087 rows. **Fixed:** both pullers now write to a `<path>.partial`
temp file and only `os.replace()` it onto the real backup path after confirming the full size
matches — a failed/partial pull can never again corrupt a previously-good backup. Also added
retry-with-backoff (4 attempts) on each individual chunk, since the root cause of the corrupt
chunk was transient tunnel flakiness, not a permanent fault.

## Separately: BOTH Kaggle accounts fully restarted around the same time
Account #1 restarted (fresh session, confirmed via empty `/kaggle/working`) while account #2
was already down from its own earlier restart. This meant the **live 100% complete** versions
of `base`, `prompt_dense` (account #1) and `steered_onpolicy`, `ctrl_random` (account #2) were
all lost from the servers at the same time — only the local ~70% backups (and now, after the
corruption bug above, less than that for two of them) remain. `base`/`prompt_dense` relaunched
on the fresh account #1 session via `_resumable_launch.py`, resuming from whatever survived
locally. Account #2 still needs the user to restart it before `steered_onpolicy`/`ctrl_random`
can resume and `ctrl_shuffled` can be redone.

**Lesson, stated plainly:** pull the FULL backup the moment an arm hits 100%, not just check
its row count remotely and assume the Kaggle-side copy is safe. The window between "arm
finished" and "something restarts" was the exact gap that caused tonight's losses.

## Recovery: one session never actually died, redundant work found and redirected
Reconnecting to what the user said was "account #2" with URLs that looked suspiciously like
account #1's earlier ones turned out to be correct — it was a session that had been running
the whole time, completely unaffected by whatever made an earlier check look like a fresh
restart (likely a transient tunnel blip at that exact moment, not a real kernel death). This
session had **`base` and `prompt_dense` both genuinely 100% complete**, GPUs idle. Meanwhile
the `super_agent.py`-connected session had been duplicating effort, re-running those same two
arms from the corrupted 70% backups for ~15 minutes. Killed the redundant processes, pulled
the real complete files, and redeployed all 4 now-free GPUs (2 from each session) onto the
3 still-needed arms: `steered_onpolicy` + `ctrl_random` resumed cleanly from their clean 70%
FileLink backups (no corruption — the manual-download backups were never touched by the
buggy chunked-pull incident), and `ctrl_shuffled` relaunched fresh, split across the other
session's two freed GPUs.
**Lesson:** a tunnel error at one moment doesn't always mean the kernel died — re-check before
assuming total loss and relaunching from scratch; it cost real duplicated compute tonight.

## Both sessions restarted again; recovered cleanly this time
Both accounts went fresh-session again (as has happened many times tonight). `base`/`prompt_dense`
were already 100% safe locally regardless. Resumed `steered_onpolicy`/`ctrl_random` cleanly from
the clean 70% FileLink backups (verified exact byte match, no corruption, both confirmed alive via
`ps`). `ctrl_shuffled` restarted fresh (no backup existed yet) — **applying today's lesson**: now
backing it up incrementally from the start instead of waiting for 100%.

## Follow-on cascade from the corrupted-backup bug (all resolved)
1. Resuming `base`/`prompt_dense` from the corrupted backups crashed both processes instantly —
   `evaluate.py`'s resume logic reads every existing line to build its "done" set, and hit a
   truncated trailing JSON line. Fixed by stripping unparseable lines from both the Kaggle-side
   and local copies.
2. A second, subtler case: some corrupted lines still parsed as *valid* JSON but with the wrong
   shape (missing `qid`), crashing with `KeyError`. Re-cleaned with full schema validation
   (checking all 10 expected keys are present), not just "does `json.loads()` succeed."
3. Relaunching `prompt_dense` hit the **same backslash-escaping bug as earlier tonight**
   (`\boxed{}` → `\x08oxed{}`) — caused by inlining the instruction text directly into a new
   ad-hoc bash command instead of using the already-correct, file-based `_resumable_launch.py`
   (whose `PROMPT_DENSE_INSTRUCTION` constant is written via the Write tool, immune to shell
   escaping). Lesson re-confirmed: **never inline backslash-containing strings into a Bash
   command — always put them in a committed `.py` file and reference that.**
4. Final state: `base` pid 1689 (climbing cleanly), `prompt_dense` pid 1801 (verified correct
   instruction via `/proc/<pid>/cmdline`, climbing from ~1080 rows).

## Workaround: manual FileLink download beats the chunked tunnel pull once files get large
Once `runs/*.jsonl` files hit ~6-7MB, pulling them through the exec tunnel in ~12KB chunks
(`_pull_chunked.py`) started taking 10-15 minutes per file — too slow to be a good "every
check-in" habit anymore. Switched to: add a new cell in each Kaggle notebook (doesn't touch
the running processes) with `IPython.display.FileLink(...)` for each arm's output file, click
the links, and merge the downloaded files straight into `runs_backup/` locally. Much faster,
and lets analysis (official vs formatting-corrected accuracy) run **entirely locally** with
zero tunnel calls.

## Discovery: the Kaggle kernel survives even when my tunnel connection dies
Account #2's exec tunnel gave a stale-token 403 for a while (no fresh URLs available), but the
Kaggle session info panel showed the GPU still at 99%/99% utilization — meaning the underlying
kernel and background `evaluate.py` processes were **still alive and working** the whole time;
only *my* ability to check on them had broken. Confirmed by having the user manually download
the in-progress files straight through Kaggle's own notebook UI (`IPython.display.FileLink` in
a fresh, untouched cell — doesn't disturb the running processes) instead of waiting for the
tunnel: `ctrl_random` had advanced from 1,120 to **3,104 rows** and `steered_onpolicy` from
1,184 to **3,392 rows** in the time my connection was down. Merged into `runs_backup/`.
**Lesson:** a dead tunnel does NOT mean dead compute. Check the Kaggle session's own GPU%
panel before assuming an arm needs restarting — and the manual "download via FileLink in a new
cell" trick is a solid fallback whenever the tunnel can't be revived quickly.

## Gate 2 — steered_onpolicy COMPLETE, ctrl_shuffled launched (4/5 arms done or running)
`steered_onpolicy` finished: 10,552/10,552 rows, `runs/steered_onpolicy.jsonl.meta.json` written.
The moment its GPU (account #2, GPU0) freed, launched the 5th and final arm, `ctrl_shuffled`,
there (`_launch2.py ctrl_shuffled 0 --steer-vector vectors/Qwen_Qwen2.5-1.5B-Instruct_ctrl_shuffled.npz
--block 16 --lam 1.0`). Confirmed running cleanly, no errors, climbing at ~64 rows/min (faster
than the other arms' pace, likely because the model/cache was already warm on that GPU).
`ctrl_random` finished: 10,552/10,552. The moment GPU1 freed, split `ctrl_shuffled`'s remaining
work across both GPUs to roughly halve the wait: restarted the GPU0 instance with `--limit 660`
(questions 0-659, resumed cleanly from its existing 1,440 rows via `--out runs/ctrl_shuffled.jsonl`)
and launched a second instance on GPU1 with `--offset 660` covering questions 660-1318, writing
to a separate `runs/ctrl_shuffled_b.jsonl` (qids assigned before slicing, so no overlap/duplication
between the two halves). Will concatenate both files once complete before running the final
analysis.

## Gate 2 — Round 2 parallelized on a second Kaggle account (RUNNING)
- User set up a second Kaggle account to cut total time: Round 2 (`steered_onpolicy`,
  `ctrl_random`) now runs there in parallel with Round 1 (`base`, `prompt_dense`) on account #1,
  instead of waiting for Round 1 to free its GPUs.
- Built a parallel toolchain in `colab ai/`, parameterized separately so neither account's
  config touches the other: `_kaggle2.py` (explicit exec_url/token, not the global
  `super_agent.py` config), `_push_steergate2.py`, `_launch2.py`, `_pull_chunked2.py`,
  `_backup_progress2.py`. Mirrors the account-#1 tooling exactly.
- Pushed the full project (code + the 3 vector files) to account #2, freed its GPU (killed the
  auto-loaded Ollama), and launched: `steered_onpolicy` (pid 1498, GPU0, using
  `vectors/..._onpolicy.npz` block 16 lam 1.0) and `ctrl_random` (pid 1506, GPU1, using
  `..._ctrl_random.npz` block 16 lam 1.0). Both confirmed alive with no errors after 30s.
- **Known risk (flagged to user):** running 2 Kaggle accounts to combine GPU quota/sessions is
  against Kaggle's ToS and carries some ban risk if detected. User made the call to proceed
  anyway; noting it here for the record.
- `ctrl_shuffled` is still unassigned — will go on whichever account frees up a GPU first.

## Gate 2 — Round 1 relaunch #5 using the new resumable infrastructure (RUNNING)
- Reconnected via a fresh Kaggle session (expected — every full restart is fresh regardless of
  mode). GPU freed, full project pushed cleanly this time (no mid-push drop).
- Launched via the new `_resumable_launch.py`: `base` (pid 1529, GPU0), `prompt_dense` (pid 1537,
  GPU1). Verified `prompt_dense`'s `\boxed{}` instruction is correct this time by reading the
  live process's actual `/proc/<pid>/cmdline` (not just trusting the launch code).
- This is Round 1's 5th total launch attempt today, after losing it 4 times to full-session
  wipes (one from an accidental Kaggle "Save Version" commit conflict, the rest from what
  looks like the 40-min idle-editor kernel kill). From here on, `_backup_progress.py` runs at
  every monitoring check-in so a disconnect costs at most ~20-30 min of regenerated rows, not
  the whole arm.

## Gate 2 — full locked replication (STARTED)
- Launched sharded across both T4s: `base` on GPU0, `prompt_dense` on GPU1. Full protocol fidelity
  kept (N=1,319, k=8) — not cutting the test set size, since that's what makes the verdict
  trustworthy. Documented cut: no explicit G2-a seed-drift noise-floor check today (time pressure);
  `ctrl_random`/`ctrl_shuffled` give a partial sanity signal in its place.
- **Bug found & fixed:** the custom `--instruction` text for `prompt_dense` (containing `\boxed{}`)
  got corrupted into a backspace character (`\x08oxed{}`) somewhere across nested
  bash-heredoc/python-string escaping layers when launching via inline code. Caught it by reading
  `/proc/<pid>/cmdline` directly before letting the run waste hours on a broken instruction — the
  strict extractor requires a literal `\boxed{}`, so this would have artificially tanked
  `prompt_dense`'s accuracy and invalidated the one baseline that matters most (AxBench: a good
  prompt often beats steering). Fixed by base64-encoding the instruction text before sending it
  through the exec pipe, sidestepping the escaping entirely. **Lesson: verify any launch that embeds
  special characters (backslashes, `\b`, `\t` etc.) by reading the live process's actual argv, not
  just trusting it looks right in the launch code.**
- `base` arm was unaffected (uses the script's own default instruction, no custom text passed) and
  kept running the whole time.
- Vectors in place for the steered/control arms: `vectors/Qwen_Qwen2.5-1.5B-Instruct_onpolicy.npz`
  (layer 16), `_ctrl_random.npz`, `_ctrl_shuffled.npz` (all norm-matched to 1.3733).
- Still to launch: `steered_onpolicy`, `ctrl_random`, `ctrl_shuffled` (next pair/solo rounds as GPUs
  free up).

## Still to do
- [ ] Finish Gate 1.5 (relaunching now after latest Kaggle restart)
- [ ] Gate 2-a: noise floor (same arm, two seeds, check CI covers 0)
- [ ] Gate 2-b: full locked run — base / prompt_dense / steered_onpolicy / ctrl_random /
      ctrl_shuffled (teacher arm cut) on full 1,319 test questions, k=8, 10,000-sample paired
      bootstrap → GO / NO-GO / INCONCLUSIVE / BROKEN verdict
- [ ] Gate 3: send the 25 outreach DMs + 2 community posts (PLAN.md section 3.2) — this is on the
      user, can't be done by the agent, and is the slowest part (depends on other people replying)
- [ ] Write up the final report (positive or negative result — both are valid outcomes per PLAN.md)
- [ ] If GO + Gate 3 passes: minimal `steergate report`/`steergate gate` CLI
- [ ] Donation ask removed from README for now (2026-11-17) — revisit later if/when wanted
