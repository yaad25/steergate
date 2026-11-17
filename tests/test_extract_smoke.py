"""Smoke test for extract.py's core functions against the tiny random Qwen2 test model.
Not a correctness-of-math proof, but catches shape bugs, crashes, and obvious nonsense
before burning real GPU time on Kaggle."""
import numpy as np
import pytest

from steergate_p0 import extract as ex


def test_compress_trace_keeps_math_drops_filler():
    text = ("Let's figure this out. First, we need to compute 2 + 2.\n"
            "This means the total is 4.\n"
            "So the answer is \\boxed{4}.")
    out = ex.compress_trace(text)
    assert "4" in out
    assert "\\boxed{4}" in out
    # never returns empty
    assert ex.compress_trace("no numbers here at all") != ""


def test_get_blocks_tiny(tiny):
    model, tok = tiny
    blocks = ex.get_blocks(model)
    assert len(blocks) == 4  # matches layers=4 in conftest.make_tiny


def test_last_token_activations_shapes(tiny):
    model, tok = tiny
    texts = ["user: hello\nassistant: hi there", "user: 2+2?\nassistant: 4"]
    layers = [0, 1, 2, 3]
    acts, norms = ex.last_token_activations(model, tok, texts, layers, batch_size=2)
    for l in layers:
        assert acts[l].shape == (2, model.config.hidden_size)
        assert np.isfinite(acts[l]).all()
        assert norms[l] > 0


def test_mean_diff_vectors_unit_norm(tiny):
    model, tok = tiny
    pos = ["user: a\nassistant: dense one", "user: b\nassistant: dense two"]
    neg = ["user: a\nassistant: a much longer verbose rambling trace one", "user: b\nassistant: verbose trace two"]
    layers = [0, 1]
    pos_acts, _ = ex.last_token_activations(model, tok, pos, layers, batch_size=2)
    neg_acts, _ = ex.last_token_activations(model, tok, neg, layers, batch_size=2)
    vecs = ex.mean_diff_vectors(pos_acts, neg_acts)
    for l in layers:
        n = np.linalg.norm(vecs[l])
        assert abs(n - 1.0) < 1e-3, f"layer {l} norm {n} not unit"


def test_kl_near_zero_for_zero_alpha(tiny):
    model, tok = tiny
    rng = np.random.default_rng(0)
    vec = rng.normal(size=model.config.hidden_size).astype(np.float32)
    vec /= np.linalg.norm(vec)
    passages = ["the quick brown fox jumps over the lazy dog repeatedly for testing purposes"]
    kl0 = ex.kl_on_passages(model, tok, passages, {0: vec}, 0, alpha=0.0, max_len=32)
    assert kl0 < 1e-4, f"KL should be ~0 at alpha=0, got {kl0}"
    kl_big = ex.kl_on_passages(model, tok, passages, {0: vec}, 0, alpha=50.0, max_len=32)
    assert kl_big > kl0, "a large perturbation should increase KL vs. alpha=0"


def test_greedy_accuracy_runs(tiny):
    model, tok = tiny
    qs = [{"question": "2+2?", "answer": "#### 4"}, {"question": "3+3?", "answer": "#### 6"}]
    acc, trunc = ex.greedy_accuracy(model, tok, qs, "answer please", None, None, 1.0,
                                    max_new_tokens=8, batch_size=2)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= trunc <= 1.0


def test_build_traces_runs(tiny):
    model, tok = tiny
    qs = [{"question": "2+2?", "answer": "#### 4"}, {"question": "3+3?", "answer": "#### 6"}]
    traces = ex.build_traces(model, tok, qs, max_new_tokens=8, batch_size=2)
    # with a random tiny model these almost certainly won't match gold -> list may be empty, that's fine;
    # the real assertion is that it doesn't crash and returns a list.
    assert isinstance(traces, list)
