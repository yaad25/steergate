"""Counter-based seeded sampling via the Gumbel-max trick.

Noise for (row_seed, step, token) is a pure function of those three numbers, so:
  * sampled outputs do not depend on batch composition or row position (up to float noise),
  * the same (question, sample_idx) seed is reused across arms => common random numbers.
"""
from __future__ import annotations
import hashlib
import torch
from transformers import LogitsProcessor

M32 = 0xFFFFFFFF


def row_seed(seed_base: int, qid: int, sample_idx: int) -> int:
    h = hashlib.blake2b(f"{seed_base}|{qid}|{sample_idx}".encode(), digest_size=4).digest()
    return int.from_bytes(h, "big")


def mix32(x: torch.Tensor) -> torch.Tensor:
    """32-bit integer hash on int64 tensors (all products < 2**59, so no overflow)."""
    x = ((x >> 16) ^ x) * 0x45D9F3B & M32
    x = ((x >> 16) ^ x) * 0x45D9F3B & M32
    return (x >> 16) ^ x


def token_hash(V: int, device) -> torch.Tensor:
    t = torch.arange(V, device=device, dtype=torch.int64)
    return mix32((t * 2654435761) & M32)


def hash_uniform(seeds: torch.Tensor, step: int, th: torch.Tensor) -> torch.Tensor:
    """seeds [B] int64 in [0,2^32); returns float64 uniforms in (0,1) of shape [B, V]."""
    h = mix32(seeds & M32)
    h = mix32((h + (step + 1) * 0x9E3779B1) & M32)
    g = mix32(h[:, None] ^ th[None, :])
    g = mix32((g + 0x7F4A7C15) & M32)
    return (g.double() + 0.5) / 4294967296.0


def top_p_mask(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sl, idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sl, dim=-1)
    remove = (probs.cumsum(-1) - probs) > top_p       # first token is always kept
    sl = sl.masked_fill(remove, float("-inf"))
    return torch.full_like(logits, float("-inf")).scatter(-1, idx, sl)


class SeededSampler(LogitsProcessor):
    """Use with do_sample=False: returns scores whose argmax is a sample from softmax(logits/T) (top-p filtered)."""

    def __init__(self, seeds, prompt_len: int, temperature: float = 0.7, top_p: float = 0.8):
        self.seeds = torch.as_tensor(seeds, dtype=torch.int64)
        self.prompt_len, self.temperature, self.top_p = prompt_len, temperature, top_p
        self._th = None

    def __call__(self, input_ids, scores):
        V = scores.shape[-1]
        if self._th is None or self._th.numel() != V or self._th.device != scores.device:
            self._th = token_hash(V, scores.device)
            self.seeds = self.seeds.to(scores.device)
        step = input_ids.shape[1] - self.prompt_len
        logits = top_p_mask(scores.float() / self.temperature, self.top_p)
        u = hash_uniform(self.seeds, step, self._th)
        gumbel = -torch.log(-torch.log(u))
        return (logits.double() + gumbel).float()
