import os
import sys
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
import math
import argparse
import random
import gc
from triton.testing import do_bench
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
from src.utils.registry import get_all_registries
registries = get_all_registries()
import src.model
import src.task
import src.optim
import src.data
for registry in registries:
    registry._is_register = False

from src.model.embedding.pe import RoPE
from src.model.backbone.cache import RATPlusSingleLayerCache
from config import *
from src.model.op import ascan, get_eff_attention
from src.model.backbone.util import apply_rotary_pos_emb


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RATPlusCoreModule(nn.Module):

    def __init__(self, chunk_size, local_size, prefix_size, softmax_scale, apply_re=True, d_head=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.local_size = local_size
        self.prefix_size = prefix_size
        self.eff_attention = get_eff_attention(softmax_scale, chunk_size, local_size, prefix_size)
        self.apply_re = apply_re
        self.d_head = d_head

    def apply_rope(self, q, k, **kwargs):
        rotary_pos_emb = kwargs.get(f"rope", None)
        q_rope, k_rope = apply_rotary_pos_emb(q, k, rotary_pos_emb[0][None, None, :, :], rotary_pos_emb[1][None, None, :, :])
        return q_rope.to(k.dtype), k_rope.to(k.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor, g: torch.Tensor, cache=Optional[RATPlusSingleLayerCache], **kwargs):
        bs, _, seq_len, _ = q.shape
        if self.apply_re:
            g = g.repeat(1, 1, 1, 2)
            gated_kx = ascan(g, (1.0 - g) * torch.cat([k, x], dim=-1))
            gated_k, gated_x = gated_kx[..., :self.d_head].to(torch.bfloat16), gated_kx[..., self.d_head:].to(torch.bfloat16)
            if cache is not None:
                seq_pos = kwargs.get("seq_pos", seq_len - 1)
                cache.lastkcache[cache.bs_start: cache.bs_start + bs].copy_(gated_k[:, :, seq_pos: seq_pos + 1])
                cache.lastvcache[cache.bs_start: cache.bs_start + bs].copy_(gated_x[:, :, seq_pos: seq_pos + 1])
        else:
            gated_k, gated_x = k, x
        q, gated_k = self.apply_rope(q, gated_k, **kwargs)

        if cache is not None:
            seq_pos = kwargs.get("seq_pos", seq_len - 1)
            cache.update_kv_prefill(seq_pos, bs, gated_k, gated_x)
        out = self.eff_attention(q, gated_k, gated_x)
        return out

    def step(self, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor, g: torch.Tensor, cache: Optional[RATPlusSingleLayerCache], seq_pos=0, **kwargs): # (b, 1, d)
        bs, _, seq_len, _ = q.shape
        # RNN begins
        if self.apply_re:
            lastkcache, lastvcache = cache.lastkcache[cache.bs_start: cache.bs_start + bs], cache.lastvcache[cache.bs_start: cache.bs_start + bs]
            gated_k = (g * lastkcache + (1.0 - g) * k).to(torch.bfloat16)
            gated_x = (g * lastvcache + (1.0 - g) * x).to(torch.bfloat16)
            lastkcache.copy_(gated_k)
            lastvcache.copy_(gated_x)
        else:
            gated_k, gated_x = k, x
        q, gated_k = self.apply_rope(q, gated_k, **kwargs)
        kcache, vcache = cache.get_kv_step(seq_pos, bs, gated_k, gated_x)
        out = F.scaled_dot_product_attention(q, kcache, vcache, is_causal=False).transpose(1, 2).reshape(bs, seq_len, -1)
        # Update cache, update the new token, and move seq_start and seq_end
        cache.update_kv_step(seq_pos, bs, gated_k, gated_x)
        return out


class RATPlusCoreEff:

    def __init__(self, bs, seq_len, chunk_size1, local_size, prefix_size, apply_re, config: BaseConfig, dtype=torch.bfloat16):
        self.bs = bs
        self.seq_len = seq_len
        self.chunk_size1 = chunk_size1
        self.local_size = local_size
        self.prefix_size = prefix_size
        self.config = config.to_dict()
        self.softmax_scale = self.config.d_head ** (-0.5)
        self.apply_re = apply_re
        self.dtype = dtype

    def build(self, return_cache=False):
        ratplus = RATPlusCoreModule(self.chunk_size1, self.local_size, self.prefix_size, self.softmax_scale, self.apply_re, self.config.d_head).cuda()
        torch._dynamo.reset()
        ratplus = torch.compile(ratplus)
        rope = RoPE(dim=self.config.d_head, max_seq_len=self.seq_len, device="cuda")
        if return_cache:
            cache = RATPlusSingleLayerCache(0, self.bs, self.seq_len, self.prefix_size, self.local_size, self.chunk_size1, self.config.num_head, self.config.d_head, self.config.d_model, dtype=self.dtype, device="cuda")
            return ratplus, rope, cache
        return ratplus, rope

    def bench_train(self, seed=1005):
        seed_everything(seed)
        q, k, x, g = [(torch.randn(self.bs, self.config.num_head, self.seq_len, self.config.d_head) * 0.02).cuda().to(torch.bfloat16) for i in range(4)]
        g = torch.sigmoid(g.to(torch.float32))
        model, rope = self.build()
        model.train()
        _, rope_kwargs = rope(0, self.seq_len, "cuda", self.dtype)
        def train():
            with torch.amp.autocast("cuda", enabled=True, dtype=self.dtype):
                out = model(q, k, x, g, cache=None, **rope_kwargs)
            out.mean().backward() # also include backward time
        return do_bench(train)

    @torch.no_grad()
    def bench_context(self, seed=1005):
        seed_everything(seed)
        q, k, x, g = [(torch.randn(self.bs, self.config.num_head, self.seq_len, self.config.d_head) * 0.02).cuda().to(torch.bfloat16) for i in range(4)]
        g = torch.sigmoid(g.to(torch.float32))
        model, rope, cache = self.build(return_cache=True)
        model.eval()
        _, rope_kwargs = rope(0, self.seq_len, "cuda", self.dtype)
        @torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
        def context():
            cache.reset_cache()
            model(q, k, x, g, cache=cache, seq_pos=self.seq_len - 1, **rope_kwargs)
        return do_bench(context)

    @torch.no_grad()
    def bench_gen(self, seed=1005):
        seed_everything(seed)
        q, k, x, g = [(torch.randn(self.bs, self.config.num_head, 1, self.config.d_head) * 0.02).cuda().to(torch.bfloat16) for i in range(4)]
        g = torch.sigmoid(g.to(torch.float32))
        model, rope, cache = self.build(return_cache=True)
        model.eval()
        cache.reset_cache()
        cache.update_kv_fake_prefill(self.seq_len - 2)
        seq_start = cache.seq_start
        seq_end = cache.seq_end
        _, rope_kwargs = rope.step(self.seq_len - 1, self.seq_len, "cuda", self.dtype)
        @torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
        def gen():
            cache.seq_start = seq_start
            cache.seq_end = seq_end
            model.step(q, k, x, g, cache=cache, seq_pos=self.seq_len - 1, **rope_kwargs)
        return do_bench(gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk_size1', type=int, default=1)
    parser.add_argument('--local_size', type=int, default=0)
    parser.add_argument('--prefix_size', type=int, default=4)
    parser.add_argument('--apply_re', action='store_true', help='whether to apply the recurrence')
    args = parser.parse_args()

    config = Config3()
    seq_list = [4096, 8192, 16384, 32768, 65536, 131072, 262144]
    bs_list = [64, 32, 16, 8, 4, 2, 1]
    train_ms_list, context_ms_list = [], []
    for i in range(7):
        print("seq: {}, bs: {}".format(seq_list[i], bs_list[i]))
        if args.chunk_size1 == -1 and args.local_size == 0:
            args.prefix_size = seq_list[i] # then it should be attention, we use prefix only to simplify it.
        model = RATPlusCoreEff(bs_list[i], seq_list[i], args.chunk_size1, args.local_size, args.prefix_size, args.apply_re, config)
        # train_ms = model.bench_train()
        context_ms = model.bench_context()
        gen_ms = model.bench_gen()
        # train_ms_list.append(train_ms)
        context_ms_list.append(context_ms)
        train_ms = 0
        print("train: {:.2f}, context: {:.2f}, gen: {:.4f}".format(train_ms, context_ms, gen_ms))
    print(train_ms_list)
    print(context_ms_list)
    bs_gen_list = [64, 128, 256, 512, 1024, 4096]
    for bs in bs_gen_list:
        result = []
        for seq in seq_list:
            if args.chunk_size1 == -1 and args.local_size == 0:
                args.prefix_size = seq # then it should be attention, we use prefix only to simplify it.
            model = RATPlusCoreEff(bs, seq, args.chunk_size1, args.local_size, args.prefix_size, args.apply_re, config)
            try:
                gen_ms = model.bench_gen()
                result.append(str(round(gen_ms, 2)))
            except Exception as e:
                result.append("OOM")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"bs: {bs}, bs: {','.join(result)}")
