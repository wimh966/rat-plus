import math
import torch
import torch.nn as nn
from ...utils.registry import pe_registry


@pe_registry.register("empty")
class Empty(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, seq_start, seq_end, device, dtype):
        return None, {}

    def step(self, seq_start, seq_end, device, dtype):
        return None, {}


@pe_registry.register("rope")
class RoPE(nn.Module):

    def __init__(self, dim, max_seq_len=2048, base=10000, device="cuda", **kwargs):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device=device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=self.max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, seq_start, seq_end, device, dtype): # [seq_start, seq_end)
        if seq_end > self.max_seq_len:
            self.max_seq_len = seq_end
            self._set_cos_sin_cache(self.max_seq_len, device, dtype)
        return None, \
            {"rope": (self.cos_cached[seq_start: seq_end].to(device=device, dtype=dtype),
                                self.sin_cached[seq_start: seq_end].to(device=device, dtype=dtype))}

    def step(self, seq_start, seq_end, device, dtype):
        return self.forward(seq_start, seq_end, device, dtype)
