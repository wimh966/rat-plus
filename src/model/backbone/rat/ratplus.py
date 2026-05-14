# Copyright (c) 2026 Xiuying Wei, EPFL CLAIRE lab
# All rights reserved.
# Licensed under the Apache License 2.0.

import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import torch.nn.functional as F
from ...op import get_eff_attention, ascan
from ..cache import RATPlusSingleLayerCache, RATPlusFullSingleLayerCache


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin): # cos and sin has been taken out based on the position ids
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RATPlus(nn.Module):
    """
    RAT+ module.

    Args:
        d_model (int): Hidden dimension of the model.
        num_head (int): Number of attention heads.
        dilation_size (int): Dilation size, used for jointly training the sparse configuration or during inference.
        initial_size (int): Number of initial tokens invoked during inference.
        local_size (int): Local window size for sparse attention connections during inference.
        apply_re (bool): Whether to apply the recurrence mechanism.
        joint_train (bool): If True, trains both dense and sparse configurations. This option should not be used after pretraining.
    """
    def __init__(
        self,
        d_model,
        num_head=16,
        dilation_size=64,
        initial_size=0,
        local_size=0,
        apply_re=True,
        joint_train=True,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {"device": kwargs.get("device", "cuda"),
                          "dtype": kwargs.get("dtype", torch.float32)}
        self.layer_id = kwargs.get("layer_id", 0)
        self.d_model = d_model
        self.num_head = num_head
        assert self.d_model % self.num_head == 0
        self.d_head = self.d_model // self.num_head
        self.softmax_scale = self.d_head ** -0.5
        # q, k, v, f, g
        # Note that we add the output gating for both RAT+ and attention-only baseline since it has been known to be effective. 
        # Efficiency and accuracy studies are conducted consistently. We leave saving parameters for future work.
        self.in_proj = nn.Linear(d_model, 5 * self.d_model if apply_re else 4 * self.d_model, bias=False, **factory_kwargs)
        self.input_norm = nn.RMSNorm(self.d_model, eps=1.0e-6, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False, **factory_kwargs)
        self.joint_train = joint_train
        self.apply_re = apply_re
        self.dilation_size = dilation_size if type(dilation_size) is int else dilation_size[self.layer_id]
        self.initial_size = initial_size if type(initial_size) is int else initial_size[self.layer_id]
        self.local_size = local_size if type(local_size) is int else local_size[self.layer_id]
        self.eff_attention = get_eff_attention(softmax_scale=self.softmax_scale, dilation_size=self.dilation_size, local_size=self.local_size, initial_size=self.initial_size)
        # Parameter initialization called at higher level. Gaussian initialization; nothing special.

    def apply_rope(self, q, k, **kwargs):
        rotary_pos_emb = kwargs.get(f"rope", None)
        if rotary_pos_emb is None:
            raise NotImplementedError
        q_rope, k_rope = apply_rotary_pos_emb(q, k, rotary_pos_emb[0][None, None, :, :], rotary_pos_emb[1][None, None, :, :])
        return q_rope.to(k.dtype), k_rope.to(k.dtype)

    def prepare_input(self, hidden_states) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        inp = self.in_proj(hidden_states)
        o, q, k, x, g = torch.split(inp, [self.d_model, self.d_model, self.d_model, self.d_model, self.d_model if self.apply_re else 0], dim=-1)
        return o, q, k, x, g

    def forward(self, hidden_states: torch.Tensor, cache: Optional[Union[RATPlusFullSingleLayerCache, RATPlusSingleLayerCache]], **kwargs):
        bs, seq_len, _ = hidden_states.shape
        # Graph begins
        shortcut = hidden_states
        hidden_states = self.input_norm(hidden_states)
        o, q, k, x, g = self.prepare_input(hidden_states)
        # RNN begins
        if self.apply_re:
            g = torch.sigmoid(g.to(torch.float32))
            q, k, x, g = [m.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2) for m in (q, k, x, g)]
            g = g.repeat(1, 1, 1, 2)
            gated_kx = ascan(g, (1.0 - g) * torch.cat([k, x], dim=-1))
            gated_k, gated_x = gated_kx[..., :self.d_head].to(torch.bfloat16), gated_kx[..., self.d_head:].to(torch.bfloat16)
            if cache is not None:
                seq_pos = kwargs.get("seq_pos", seq_len - 1)
                cache.lastkcache[cache.bs_start: cache.bs_start + bs].copy_(gated_k[:, :, seq_pos: seq_pos + 1])
                cache.lastvcache[cache.bs_start: cache.bs_start + bs].copy_(gated_x[:, :, seq_pos: seq_pos + 1])
        else:
            q, k, x = [m.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2).contiguous() for m in (q, k, x)]
            gated_k, gated_x = k, x
        q, gated_k = self.apply_rope(q, gated_k, **kwargs)

        if cache is not None:
            seq_pos = kwargs.get("seq_pos", seq_len - 1)
            cache.update_kv_prefill(seq_pos, bs, gated_k, gated_x)
        if self.joint_train and self.training:
            # joint train
            mode = kwargs.get("mode", "dense")
            if mode == "sparse":
                out = self.eff_attention(q, gated_k, gated_x)
            else:
                out = F.scaled_dot_product_attention(q, gated_k, gated_x, is_causal=True).transpose(1, 2).reshape(bs, seq_len, -1)
        else:
            out = self.eff_attention(q, gated_k, gated_x)
        out = out * torch.sigmoid(o)
        final_out = self.out_proj(out) + shortcut
        return final_out

    def step(self, hidden_states, cache: Union[RATPlusSingleLayerCache, RATPlusFullSingleLayerCache], seq_pos=0, **kwargs): # (b, 1, d)
        bs, seq_len, _ = hidden_states.shape
        # Graph begins
        shortcut = hidden_states
        hidden_states = self.input_norm(hidden_states)
        o, q, k, x, g = self.prepare_input(hidden_states)
        # RNN begins
        if self.apply_re:
            g = torch.sigmoid(g.to(torch.float32))
            q, k, x, g = [m.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2) for m in (q, k, x, g)]
            lastkcache, lastvcache = cache.lastkcache[cache.bs_start: cache.bs_start + bs], cache.lastvcache[cache.bs_start: cache.bs_start + bs]
            gated_k = (g * lastkcache + (1.0 - g) * k).to(torch.bfloat16)
            gated_x = (g * lastvcache + (1.0 - g) * x).to(torch.bfloat16)
            lastkcache.copy_(gated_k)
            lastvcache.copy_(gated_x)
        else:
            q, k, x  = [m.reshape(bs, seq_len, self.num_head, self.d_head).transpose(1, 2) for m in (q, k, x)]
            gated_k, gated_x = k, x
        q, gated_k = self.apply_rope(q, gated_k, **kwargs)
        kcache, vcache = cache.get_kv_step(seq_pos, bs, gated_k, gated_x)
        out = F.scaled_dot_product_attention(q, kcache, vcache, is_causal=False).transpose(1, 2).reshape(bs, seq_len, self.d_model)
        # Update cache, update the new token, and move seq_start and seq_end
        cache.update_kv_step(seq_pos, bs, gated_k, gated_x)
        out = out * torch.sigmoid(o)
        final_out = self.out_proj(out) + shortcut
        return final_out

    @staticmethod
    def get_ckpt_name(model_config):
        chunk_size1 = model_config.chunk_size1
        local_size = model_config.local_size
        if type(chunk_size1) is list and len(chunk_size1) > 2:
            chunk_size1 = str(chunk_size1[0]) + str(chunk_size1[1])
        if type(local_size) is list and len(local_size) > 2:
            local_size = str(local_size[0]) + str(local_size[1])
        return model_config._name_ + f"l{model_config.chunk_size}l{chunk_size1}p{model_config.initial_size}w{local_size}" + f"re{model_config.apply_re}" + f"jointtrain{model_config.joint_train}"

    def extra_repr(self):
        return f"d_model={self.d_model}, nhead={self.num_head}, dilation_size={self.dilation_size}, initial_size={self.initial_size}, local_size={self.local_size}, re={self.apply_re}, mixtrain={self.joint_train}"