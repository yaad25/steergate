import json, numpy as np
from pathlib import Path
from steergate_p0 import evaluate as E, analyze as A
from steergate_p0.steer import save_vectors
from .conftest import make_tiny


def _data(path, n=7):
    rows = [{"question": f"What is {i}+{i + 1}? Show work.", "answer": f"work\n#### {2 * i + 1}"} for i in range(n)]
    Path(path).write_text("\n".join(json.dumps(r) for r in rows))


def _args(tmp, arm, out, **kw):
    argv = ["--model", "tiny", "--data", str(tmp / "d.jsonl"), "--arm", arm, "--out", str(out), "--device", "cpu",
            "--dtype", "float32", "--max-new-tokens", "16", "--k", "3", "--temperature", "1.0", "--top-p", "1.0"]
    for k_, v_ in kw.items():
        argv += [f"--{k_.replace('_', '-')}"] + ([] if v_ is True else [str(v_)])
    return E.parse_args(argv)


def rows(p):
    return sorted(((r["qid"], r["sample"], r["text"], r["ntok"], r["finished"]) for r in map(json.loads, open(p))))


def test_end_to_end(tmp_path):
    model, tok = make_tiny(layers=4)
    _data(tmp_path / "d.jsonl")
    # 1) sampled outputs are independent of batch size (CRN / counter-based noise)
    E.run(_args(tmp_path, "base", tmp_path / "a.jsonl", batch_size=2), model, tok)
    E.run(_args(tmp_path, "base", tmp_path / "b.jsonl", batch_size=8), model, tok)
    ra, rb = rows(tmp_path / "a.jsonl"), rows(tmp_path / "b.jsonl")
    assert len(ra) == 7 * 3 and sum(x == y for x, y in zip(ra, rb)) >= 0.95 * len(ra)
    # 2) resume: drop the last 5 rows, rerun, get the same file content
    lines = open(tmp_path / "a.jsonl").read().splitlines()
    open(tmp_path / "a.jsonl", "w").write("\n".join(lines[:-5]) + "\n")
    E.run(_args(tmp_path, "base", tmp_path / "a.jsonl", batch_size=2), model, tok)
    assert rows(tmp_path / "a.jsonl") == ra
    # 3) steering changes outputs; random control with same norm also changes but differently; metadata recorded
    v = {2: np.random.default_rng(123).standard_normal(32).astype(np.float32)}; save_vectors(tmp_path / "v.npz", v)
    E.run(_args(tmp_path, "t", tmp_path / "t.jsonl", batch_size=4, steer_vector=tmp_path / "v.npz", block=2, lam=20.0), model, tok)
    E.run(_args(tmp_path, "c1", tmp_path / "c.jsonl", batch_size=4, steer_vector=tmp_path / "v.npz", block=2, lam=20.0, control="random"), model, tok)
    rt, rc = rows(tmp_path / "t.jsonl"), rows(tmp_path / "c.jsonl")
    nd = lambda x, y: sum(p != q for p, q in zip(x, y))
    print("diff base-t", nd(ra, rt), "t-c", nd(rt, rc), "base-c", nd(ra, rc))
    assert nd(ra, rt) > 0 and nd(rt, rc) > 0
    meta = json.load(open(str(tmp_path / "t.jsonl") + ".meta.json"))
    assert meta["steer"]["block"] == 2 and meta["generation_config"]["repetition_penalty"] == 1.0 and meta["tok_per_s"] > 0
    nt = np.linalg.norm(v[2]) * 20.0
    assert abs(json.load(open(str(tmp_path / "c.jsonl") + ".meta.json"))["steer"]["perturbation_norm"] - nt) / nt < 1e-4
    # 4) greedy runs at two batch sizes -> noise floor ~ 0; analysis CLI paths run
    E.run(_args(tmp_path, "g1", tmp_path / "g1.jsonl", batch_size=2, greedy=True), model, tok)
    E.run(_args(tmp_path, "g2", tmp_path / "g2.jsonl", batch_size=7, greedy=True), model, tok)
    assert A.noise(tmp_path / "g1.jsonl", tmp_path / "g2.jsonl")["discordant_fraction_d0"] == 0.0
    res = A.compare(str(tmp_path / "a.jsonl"), str(tmp_path / "t.jsonl"), n_boot=500)
    assert res["decision"] in ("GO", "NO-GO", "INCONCLUSIVE") and res["n_questions"] == 7
    assert "mcnemar" in A.compare(str(tmp_path / "g1.jsonl"), str(tmp_path / "g2.jsonl"), n_boot=200)
