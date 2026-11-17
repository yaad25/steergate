"""Formatting-trap sensitivity check: re-grade all 5 arms with a LENIENT parser
(falls back to the last number in the text if no \\boxed{} is found) instead of the
strict \\boxed{}-only grader, then recompute the same 4 comparisons. If the verdict
doesn't change, the strict grader's formatting-trap bias wasn't material to the result.
"""
import json
import re

NUM = re.compile(r"-?\d+(?:\.\d+)?")
ARMS = ["base", "prompt_dense", "steered_onpolicy", "ctrl_random", "ctrl_shuffled"]


def lenient_grade(text, gold, strict_correct, strict_has_box):
    if strict_correct:
        return True  # already correct under strict grading
    if strict_has_box:
        return False  # had a box but wrong number -> no amount of leniency fixes that
    nums = NUM.findall(text.replace(",", ""))
    if nums:
        try:
            return abs(float(nums[-1]) - gold) <= 1e-6 * max(1.0, abs(gold))
        except ValueError:
            return False
    return False


def regrade(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    correct = sum(1 for r in rows if lenient_grade(r["text"], r["gold"], r["correct"], r["has_box"]))
    return correct, len(rows)


def paired_bootstrap_verdict(base_scores, treat_scores, n_boot=10_000, seed=0):
    import random
    n = len(base_scores)
    diffs = [t - b for t, b in zip(treat_scores, base_scores)]
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    hi95 = boots[int(0.95 * n_boot)]
    return point, lo, hi, hi95


def per_question_scores(path, lenient=True):
    from collections import defaultdict
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    by_q = defaultdict(list)
    for r in rows:
        c = lenient_grade(r["text"], r["gold"], r["correct"], r["has_box"]) if lenient else r["correct"]
        by_q[r["qid"]].append(1.0 if c else 0.0)
    qids = sorted(by_q)
    return [sum(by_q[q]) / len(by_q[q]) for q in qids]


if __name__ == "__main__":
    print("=== Lenient-parser accuracy per arm ===")
    lenient_acc = {}
    for arm in ARMS:
        c, n = regrade(f"runs_backup/{arm}.jsonl")
        lenient_acc[arm] = 100 * c / n
        print(f"{arm:20} lenient accuracy: {lenient_acc[arm]:.2f}%  (n={n})")

    print("\n=== Verdict under LENIENT grading (steered_onpolicy vs each) ===")
    treat_scores = per_question_scores("runs_backup/steered_onpolicy.jsonl")
    for arm in ["base", "prompt_dense", "ctrl_random", "ctrl_shuffled"]:
        base_scores = per_question_scores(f"runs_backup/{arm}.jsonl")
        point, lo, hi, hi95 = paired_bootstrap_verdict(base_scores, treat_scores)
        if hi95 < 0.01:
            verdict = "NO-GO"
        elif lo > 0:
            verdict = "GO"
        else:
            verdict = "INCONCLUSIVE"
        print(f"vs {arm:16} lift={point*100:+.2f}pts  CI95=[{lo*100:+.2f}, {hi*100:+.2f}]  -> {verdict}")
