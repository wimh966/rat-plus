import torch
import torch.nn as nn
from einops import repeat, rearrange
from ...utils import config as util_config
from ...utils.registry import init_registry


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin): # cos and sin has been taken out based on the position ids
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def segsum(x: torch.Tensor, y: torch.Tensor=None):  # produce sum in the manner (], the first axis indicates the end point!
    T = x.shape[-1]
    x = repeat(x, "... t -> ... t e", e=T)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    if y is not None: # TODO: review the code here
        x_segsum = x_segsum + y.unsqueeze(dim=-2)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum
