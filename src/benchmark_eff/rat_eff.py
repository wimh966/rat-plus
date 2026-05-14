import os
import sys
import torch
import math
import gc
import argparse
import numpy as np
import random
from triton.testing import do_bench
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

from SSM.sq_pretrain.src.model.backbone.rat.ratplus import RATPlus16LocalPrefixFgateSimple as RATPlus
from src.model.embedding.pe import RoPE
from src.model.backbone.cache import RATPlusSingleLayerCache
from config import *
# python rat_eff.py --chunk_size1=-1 --local_size=0 --prefix_size=4096 --apply_re

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RATPlusEff:

    def __init__(self, bs, seq_len, chunk_size1, local_size, prefix_size, apply_re, config: BaseConfig, dtype=torch.bfloat16):
        self.bs = bs
        self.seq_len = seq_len
        self.chunk_size1 = chunk_size1
        self.local_size = local_size
        self.prefix_size = prefix_size
        self.apply_re = apply_re
        self.config = config.to_dict()
        self.dtype = dtype

    def build(self, return_cache=False):
        ratplus = RATPlus(self.config.d_model, self.config.num_head, self.config.bias, self.config.init, self.config.ln,
                          1, self.chunk_size1, self.prefix_size, self.local_size, self.apply_re, False, layer_id=0).cuda()
        torch._dynamo.reset()
        ratplus = torch.compile(ratplus)
        rope = RoPE(dim=self.config.d_head, max_seq_len=self.seq_len, device="cuda")
        if return_cache:
            cache = RATPlusSingleLayerCache(0, self.bs, self.seq_len, self.prefix_size, self.local_size, self.chunk_size1, self.config.num_head, self.config.d_head, self.config.d_model, dtype=self.dtype, device="cuda")
            return ratplus, rope, cache
        return ratplus, rope

    def bench_train(self, seed=1005):
        seed_everything(seed)
        inp = (torch.randn(self.bs, self.seq_len, self.config.d_model) * 0.02).cuda()
        model, rope = self.build()
        model.train()
        _, rope_kwargs = rope(0, self.seq_len, "cuda", self.dtype)
        def train():
            with torch.amp.autocast("cuda", enabled=True, dtype=self.dtype):
                out = model(inp, cache=None, **rope_kwargs)
            out.mean().backward() # also include backward time
        return do_bench(train)

    @torch.no_grad()
    def bench_context(self, seed=1005):
        seed_everything(seed)
        inp = (torch.randn(self.bs, self.seq_len, self.config.d_model) * 0.02).cuda()
        model, rope, cache = self.build(return_cache=True)
        model.eval()
        _, rope_kwargs = rope(0, self.seq_len, "cuda", self.dtype)
        @torch.amp.autocast("cuda", enabled=True, dtype=self.dtype)
        def context():
            cache.reset_cache()
            model(inp, cache=cache, seq_pos=self.seq_len - 1, **rope_kwargs)
        return do_bench(context)

    @torch.no_grad()
    def bench_gen(self, seed=1005):
        seed_everything(seed)
        inp = (torch.randn(self.bs, 1, self.config.d_model) * 0.02).cuda()
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
            model.step(inp, cache=cache, seq_pos=self.seq_len - 1, **rope_kwargs)
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
        model = RATPlusEff(bs_list[i], seq_list[i], args.chunk_size1, args.local_size, args.prefix_size, args.apply_re, config)
        # train_ms = model.bench_train()
        train_ms = 0
        context_ms = model.bench_context()
        gen_ms = model.bench_gen()
        context_ms_list.append(context_ms)
        print("train: {:.2f}, context: {:.2f}, gen: {:.4f}".format(train_ms, context_ms, gen_ms))
    print(context_ms_list)
    bs_gen_list = [64, 128, 256, 512, 1024, 4096]
    for bs in bs_gen_list:
        result = []
        for seq in seq_list:
            if args.chunk_size1 == -1 and args.local_size == 0:
                args.prefix_size = seq # then it should be attention, we use prefix only to simplify it.
            model = RATPlusEff(bs, seq, args.chunk_size1, args.local_size, args.prefix_size, args.apply_re, config)
            try:
                gen_ms = model.bench_gen()
                result.append(str(round(gen_ms, 2)))
            except Exception as e:
                result.append("OOM")
            # delete
            del model
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"bs: {bs}, seq: {','.join(result)}")
