import numpy as np, pytest, torch
from steergate_p0.steer import steering, get_blocks
from steergate_p0.sampling import SeededSampler
from steergate_p0.evaluate import make_gen_config


def ids(n, seed=0, V=60):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(3, V, (1, n), generator=g)


def test_lambda_zero_identity_and_hook_removed(tiny):
    model, _ = tiny; x = ids(10)
    base = model(x).logits
    v = {1: np.random.default_rng(0).standard_normal(32).astype(np.float32)}
    with steering(model, v, 0.0):
        assert torch.equal(model(x).logits, base)
    with steering(model, v, 5.0):
        assert not torch.allclose(model(x).logits, base)
    assert torch.equal(model(x).logits, base)                   # hooks removed
    assert all(len(b._forward_hooks) == 0 for b in get_blocks(model))


def test_block_index_mapping(tiny):
    """Steering at 0-based block b adds lam*v to layers[b]'s output (== unsteered hidden_states[b+1]) and
    leaves every earlier block output untouched. Probes are registered after the steering hook so they see its effect."""
    model, _ = tiny; x = ids(9); lam = 2.5; b = 1
    v = np.random.default_rng(1).standard_normal(32).astype(np.float32)
    blocks = get_blocks(model)

    def run(steer):
        outs = {}
        hs = [blk.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, (o[0] if isinstance(o, tuple) else o).detach().clone()))
              for k, blk in enumerate(blocks)] if not steer else []
        try:
            if steer:
                with steering(model, {b: v}, lam):
                    hs = [blk.register_forward_hook(lambda m, i, o, k=k: outs.__setitem__(k, (o[0] if isinstance(o, tuple) else o).detach().clone()))
                          for k, blk in enumerate(blocks)]
                    model(x)
                    for h in hs: h.remove()
                    hs = []
            else:
                model(x)
        finally:
            for h in hs: h.remove()
        return outs

    o0, o1 = run(False), run(True)
    for k in range(b):
        assert torch.allclose(o0[k], o1[k], atol=1e-6), k               # earlier blocks untouched
    assert torch.allclose(o1[b] - o0[b], (lam * torch.tensor(v)).expand(1, 9, 32), atol=1e-5)
    assert (o1[b + 1] - o0[b + 1]).abs().max() > 1e-3                    # propagates downstream
    hs0 = model(x, output_hidden_states=True).hidden_states
    assert torch.allclose(hs0[b + 1], o0[b], atol=1e-6)                  # extraction convention: hidden_states[b+1] == layers[b] output


def test_decode_only_skips_prefill(tiny):
    model, _ = tiny; v = {1: np.ones(32, dtype=np.float32)}
    x3, x1 = ids(3), ids(1)
    b3, b1 = model(x3).logits, model(x1).logits
    with steering(model, v, 4.0, mode="decode_only"):
        assert torch.equal(model(x3).logits, b3) and not torch.allclose(model(x1).logits, b1)


def test_delimiter_reading_is_degenerate(tiny):
    """Causal LM: the activation at the last prompt token cannot depend on the response that follows."""
    model, _ = tiny; prompt = ids(8, 3)
    ya, yb = ids(5, 4), ids(5, 5)
    ha = model(torch.cat([prompt, ya], 1), output_hidden_states=True).hidden_states
    hb = model(torch.cat([prompt, yb], 1), output_hidden_states=True).hidden_states
    for l in range(len(ha)):
        assert torch.allclose(ha[l][0, 7], hb[l][0, 7], atol=1e-5)    # position 7 = last prompt token
    assert not torch.allclose(ha[-1][0, -1], hb[-1][0, -1], atol=1e-3)  # while the final response token does differ


def test_sampler_batch_invariance(tiny):
    model, tok = tiny
    gc, eos, pad = make_gen_config(tok, model, 12); model.generation_config = gc
    x = torch.cat([ids(7, 1), ids(7, 2), ids(7, 3)], 0); seeds = [11, 22, 33]
    def go(rows, sd):
        p = SeededSampler(sd, rows.shape[1], 1.0, 1.0)
        from transformers import LogitsProcessorList
        return model.generate(input_ids=rows, attention_mask=torch.ones_like(rows), generation_config=gc,
                              logits_processor=LogitsProcessorList([p]))
    together = go(x, seeds)
    for i in range(3):
        solo = go(x[i:i + 1], [seeds[i]])[0]; n = min(solo.numel(), together.shape[1])
        assert torch.equal(solo[:n], together[i][:n])


def test_generation_config_has_no_hidden_penalty(tiny):
    model, tok = tiny; model.generation_config.repetition_penalty = 1.05
    gc, eos, pad = make_gen_config(tok, model, 8)
    assert gc.repetition_penalty == 1.0 and gc.do_sample is False and 1 in eos
