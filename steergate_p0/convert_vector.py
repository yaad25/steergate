"""Convert a pickle-based released steering vector (.pt) to a safe .npz.

RUN ONLY INSIDE A DISPOSABLE, NETWORK-LESS SANDBOX. torch.load on untrusted files can execute code.
Default uses weights_only=True; pass --allow-unsafe-pickle only inside the sandbox if that fails.
"""
import argparse
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt"); ap.add_argument("-o", required=True)
    ap.add_argument("--allow-unsafe-pickle", action="store_true")
    a = ap.parse_args()
    obj = torch.load(a.pt, map_location="cpu", weights_only=not a.allow_unsafe_pickle)
    acts = getattr(obj, "layer_activations", obj)
    assert isinstance(acts, dict), f"unexpected object {type(obj)}"
    out = {}
    for k, t in acts.items():
        v = t.float().numpy() if hasattr(t, "float") else np.asarray(t, dtype=np.float32)
        out[f"block_{int(k)}"] = v
        print(f"block {int(k):3d}  dim {v.shape}  ||v|| = {np.linalg.norm(v):.4f}")
    np.savez(a.o, **out)
    print("wrote", a.o)


if __name__ == "__main__":
    main()
