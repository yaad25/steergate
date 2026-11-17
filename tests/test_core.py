import json, numpy as np, torch
from steergate_p0.extract_answer import last_boxed, parse_number, grade, gold_from_gsm8k
from steergate_p0 import sampling as S
from steergate_p0.controls import random_control, shuffled_pair_vector, match_norm
from steergate_p0.stats import decision_boot, paired_lift


def test_extract_answers():
    assert last_boxed(r"so \boxed{18}.") == "18"
    assert last_boxed(r"\boxed{\frac{1}{2}} then \boxed{7}") == "7"
    assert last_boxed(r"\boxed{\text{18 dollars}}") == r"\text{18 dollars}"
    assert last_boxed(r"truncated \boxed{18") is None and last_boxed("no box") is None
    assert grade(r"x \boxed{\$1,234}", 1234.0)["correct"]
    assert grade(r"\boxed{18.00}", 18.0)["correct"] and grade(r"\boxed{-5}", -5.0)["correct"]
    assert grade(r"\boxed{\text{18 dollars}}", 18.0)["correct"]
    assert not grade(r"\boxed{19}", 18.0)["correct"] and not grade("answer is 18", 18.0)["has_box"]
    assert gold_from_gsm8k("blah\n#### 1,234") == 1234.0


def test_hash_uniform_quality_and_batch_invariance():
    V = 4000; th = S.token_hash(V, "cpu")
    seeds = torch.arange(64, dtype=torch.int64) * 7919 + 13
    us = torch.stack([S.hash_uniform(seeds, st, th) for st in range(4)])           # [4,64,V]
    assert abs(us.mean().item() - 0.5) < 0.01 and abs(us.var().item() - 1 / 12) < 0.004
    assert (us > 0).all() and (us < 1).all()
    flat = us.reshape(-1, V)
    rowcorr = np.corrcoef(flat[0].numpy(), flat[1].numpy())[0, 1]
    lagcorr = np.corrcoef(flat[:, :-1].reshape(-1).numpy(), flat[:, 1:].reshape(-1).numpy())[0, 1]
    assert abs(rowcorr) < 0.05 and abs(lagcorr) < 0.02
    solo = S.hash_uniform(seeds[5:6], 2, th)[0]
    inbatch = S.hash_uniform(torch.cat([seeds[:5], seeds[5:6], seeds[6:9]]), 2, th)[5]
    assert torch.equal(solo, inbatch)                                                # row noise independent of batch


def test_gumbel_max_matches_softmax():
    logits = torch.tensor([2.0, 1.0, 0.0, -1.0, 0.5]); N = 200_000
    proc = S.SeededSampler(torch.arange(N, dtype=torch.int64) * 2654435761 % (1 << 32), prompt_len=3, temperature=1.0, top_p=1.0)
    out = proc(torch.zeros(N, 3, dtype=torch.long), logits.repeat(N, 1))
    freq = torch.bincount(out.argmax(-1), minlength=5).double() / N
    p = torch.softmax(logits, -1).double()
    assert (freq - p).abs().max() < 0.006, (freq, p)


def test_top_p_excludes_tail():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05])); N = 20000
    proc = S.SeededSampler(torch.arange(N, dtype=torch.int64) + 99, prompt_len=1, temperature=1.0, top_p=0.7)
    pick = proc(torch.zeros(N, 1, dtype=torch.long), logits.repeat(N, 1)).argmax(-1)
    assert set(pick.tolist()) <= {0, 1} and 1 in set(pick.tolist())     # nucleus = {0.5, 0.3} (cum-before-token <= 0.7)


def test_controls_norm_matching():
    v = np.random.default_rng(0).standard_normal(64).astype(np.float32)
    c1 = random_control(64, 3.0, 1); assert abs(np.linalg.norm(c1) - 3.0) < 1e-4
    diffs = np.random.default_rng(1).standard_normal((200, 64)).astype(np.float32) + v
    c2 = shuffled_pair_vector(diffs, 2, 3.0); assert abs(np.linalg.norm(c2) - 3.0) < 1e-4
    cos = abs(float(c2 @ v) / (np.linalg.norm(c2) * np.linalg.norm(v)))
    assert cos < 0.5                                                       # sign-flipped mean loses the true direction
    assert np.allclose(np.linalg.norm(match_norm(v, 2.0)), 2.0, atol=1e-5)


def _write_arm(path, correct, finished=None, ntok=300, k=8, arm="x"):
    with open(path, "w") as f:
        for q in range(correct.shape[0]):
            for s in range(k):
                f.write(json.dumps({"arm": arm, "qid": q, "sample": s, "correct": bool(correct[q, s]),
                                    "has_box": True, "finished": True if finished is None else bool(finished[q, s]),
                                    "ntok": ntok, "pred": 1.0, "gold": 1.0, "text": ""}) + "\n")


def test_analyze_decisions(tmp_path):
    from steergate_p0.analyze import compare
    rng = np.random.default_rng(0); N, k = 1319, 8
    p = np.clip(rng.beta(0.5, 0.34, N), 1e-4, 1 - 1e-4)
    logit = lambda x: np.log(x / (1 - x)); expit = lambda x: 1 / (1 + np.exp(-x))
    yb = rng.random((N, k)) < p[:, None]
    _write_arm(tmp_path / "b.jsonl", yb)
    for shift, want in ((1.6, "GO"), (0.0, "NO-GO"), (-1.5, "NO-GO")):
        yt = rng.random((N, k)) < expit(logit(p) + shift)[:, None]
        _write_arm(tmp_path / "t.jsonl", yt)
        r = compare(str(tmp_path / "b.jsonl"), str(tmp_path / "t.jsonl"), n_boot=2000)
        if want == "GO": assert r["decision"] == "GO" and r["lift_pts"] > 1.0, r
        else: assert r["decision"] == want, r
    # truncation guard: a big lift with more truncation than base must not be GO
    yt = rng.random((N, k)) < expit(logit(p) + 1.6)[:, None]
    fin = rng.random((N, k)) > 0.05
    _write_arm(tmp_path / "t.jsonl", yt, finished=fin)
    assert compare(str(tmp_path / "b.jsonl"), str(tmp_path / "t.jsonl"), n_boot=2000)["decision"] != "GO"


def test_decision_boot_edges():
    assert decision_boot(1.0, 0.1, 2.0) == "GO" and decision_boot(0.99, 0.1, 2.0) == "INCONCLUSIVE"
    assert decision_boot(0.0, -1, 0.99) == "NO-GO" and decision_boot(0.0, -1, 1.0) == "INCONCLUSIVE"
