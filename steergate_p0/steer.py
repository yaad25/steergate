"""Activation steering by forward hooks on decoder-block outputs.

Block index b is 0-based: the hook sits on model.model.layers[b], whose output equals HF
hidden_states[b+1] (except the last block, whose hidden_states entry is post-norm).
mode="all": add lam*v at every position (matches llama.cpp control vectors).
mode="decode_only": skip multi-token (prefill) passes, i.e. steer generated tokens only.
"""
from __future__ import annotations
import hashlib
from contextlib import contextmanager
import numpy as np
import torch


def get_blocks(model):
    for path in ("model.layers", "transformer.h", "gpt_neox.layers", "model.decoder.layers"):
        obj, ok = model, True
        for part in path.split("."):
            if hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                ok = False
                break
        if ok:
            return obj
    raise ValueError("cannot locate decoder blocks")


@contextmanager
def steering(model, vectors: dict, lam: float, mode: str = "all"):
    assert mode in ("all", "decode_only")
    blocks = get_blocks(model)
    p = next(model.parameters())
    handles = []
    try:
        for b, v in vectors.items():
            assert 0 <= b < len(blocks), f"block {b} out of range"
            delta = (lam * torch.as_tensor(v, dtype=torch.float32)).to(device=p.device, dtype=p.dtype)

            def hook(mod, inp, out, delta=delta):
                h = out[0] if isinstance(out, tuple) else out
                if mode == "decode_only" and h.shape[1] > 1:
                    return out
                h2 = h + delta
                return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

            handles.append(blocks[b].register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


def save_vectors(path, vectors: dict):
    np.savez(path, **{f"block_{b}": np.asarray(v, dtype=np.float32) for b, v in vectors.items()})


def load_vectors(path) -> dict:
    path = str(path)
    if path.endswith(".safetensors"):
        from safetensors.numpy import load_file
        raw = load_file(path)
    else:
        z = np.load(path)
        raw = {k: z[k] for k in z.files}
    return {int(k.split("_")[1]): v.astype(np.float32) for k, v in raw.items() if k.startswith("block_")}


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
