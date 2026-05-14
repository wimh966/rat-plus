# Copyright (c) 2026 Xiuying Wei, EPFL CLAIRE lab
# All rights reserved.
# Licensed under the Apache License 2.0.


import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import flex_attention as fla


def merge_last_token_naive(inter_out, intra_out, inter_lse, intra_lse) -> torch.Tensor:
    max_lse = torch.max(inter_lse, intra_lse)
    inter_lse_exp = torch.exp(inter_lse - max_lse)
    intra_lse_exp = torch.exp(intra_lse - max_lse)
    intra_adjust = (intra_lse_exp / (intra_lse_exp + inter_lse_exp)).to(intra_out.dtype).unsqueeze(-1)
    inter_adjust = (inter_lse_exp / (intra_lse_exp + inter_lse_exp)).to(inter_out.dtype).unsqueeze(-1)
    return inter_out * inter_adjust + intra_adjust * intra_out


class ChunkAttentionLocalPrefixWrapper(nn.Module):

    def __init__(self, softmax_scale, dilation_size, local_size=0, initial_size=0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dilation_size = dilation_size
        self.local_size = local_size
        self.initial_size = initial_size

    def forward(self, q, gated_k, gated_x) -> torch.Tensor:
        # (b a l h)
        seq_len = q.shape[-2]
        def block_local_prefix_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = (q_idx >= kv_idx)
            block_mask = ((kv_idx + 1) % self.dilation_size == 0)
            local_prefix_mask = ((q_idx - kv_idx <= self.local_size) | (kv_idx < self.initial_size))
            return (local_prefix_mask | block_mask) & causal_mask
        bs, _, kv_length, _ = gated_k.shape
        block_mask = fla.create_block_mask(block_local_prefix_causal_mask, 1, 1, q.shape[2], kv_length, device="cuda")
        out = fla.flex_attention(q, gated_k, gated_x, scale=self.softmax_scale, block_mask=block_mask)
        return out.transpose(1, 2).contiguous().view(bs, seq_len, -1)


class ChunkAttentionLocalPrefixWrapperFast(nn.Module):

    def __init__(self, softmax_scale, dilation_size, local_size=0, initial_size=0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dilation_size = dilation_size
        self.local_size = local_size
        self.initial_size = initial_size
        self.d_st = self.initial_size + ((self.dilation_size - 1 - self.initial_size) % self.dilation_size)
        self.num_kv_shift = self.initial_size // self.dilation_size

    def forward(self, q, gated_k, gated_x) -> torch.Tensor:
        seq_len = q.shape[-2]
        def block_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx - self.local_size) // self.dilation_size > kv_idx + self.num_kv_shift
        def local_prefix_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & ((q_idx - kv_idx <= self.local_size) | (kv_idx < self.initial_size))
        chunk_gated_k = gated_k[..., self.d_st: : self.dilation_size, :]
        chunk_gated_x = gated_x[..., self.d_st: : self.dilation_size, :]
        bs, _, num_chunk, _ = chunk_gated_x.shape
        block_mask = fla.create_block_mask(block_causal_mask, 1, 1, q.shape[2], num_chunk, device="cuda")
        inter_out, inter_lse = fla.flex_attention(q, chunk_gated_k, chunk_gated_x, scale=self.softmax_scale, block_mask=block_mask, return_lse=True)
        local_prefix_mask = fla.create_block_mask(local_prefix_causal_mask, 1, 1, q.shape[2], gated_k.shape[-2], device="cuda")
        local_prefix_out, local_prefix_lse = fla.flex_attention(q, gated_k, gated_x, scale=self.softmax_scale, block_mask=local_prefix_mask, return_lse=True)
        out = merge_last_token_naive(inter_out, local_prefix_out, inter_lse, local_prefix_lse)
        return out.transpose(1, 2).contiguous().view(bs, seq_len, -1)


class LocalAttentionPrefixWrapper(nn.Module):

    def __init__(self, softmax_scale, window_size, initial_size):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.window_size = window_size
        self.initial_size = initial_size

    def forward(self, q, gated_k, gated_x) -> torch.Tensor:
        seq_len = q.shape[-2]
        def local_prefix_causal_mask(b, h, q_idx, kv_idx):
            causal_mask = (q_idx >= kv_idx)
            local_prefix_mask = ((q_idx - kv_idx <= self.window_size) | (kv_idx < self.initial_size))
            return causal_mask & local_prefix_mask

        bs, _, kv_length, _ = gated_k.shape
        block_mask = fla.create_block_mask(local_prefix_causal_mask, 1, 1, q.shape[-2], kv_length, device="cuda")
        out = fla.flex_attention(q, gated_k, gated_x, scale=self.softmax_scale, block_mask=block_mask)
        return out.transpose(1, 2).contiguous().view(bs, seq_len, -1)


def get_eff_attention(softmax_scale, dilation_size=-1, local_size=0, initial_size=0):
    if dilation_size != -1:
        eff_attention = ChunkAttentionLocalPrefixWrapperFast(softmax_scale, dilation_size, local_size, initial_size)
    else:
        eff_attention = LocalAttentionPrefixWrapper(softmax_scale, window_size=local_size, initial_size=initial_size)
    return eff_attention