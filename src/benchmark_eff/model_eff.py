import os
import sys
import hydra
import wandb
import math
import torch
import random
import time
import gc
import numpy as np
from easydict import EasyDict
from triton.testing import do_bench
import torch.distributed as dist
from omegaconf import OmegaConf
from omegaconf.listconfig import ListConfig
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(project_root)
sys.path.insert(0, project_root)
from src.trainer.lm_trainer import LMTrainer
from src.trainer.lm_fsdp_trainer import LMFSDPTrainer
from src.utils.registry import get_all_registries
registries = get_all_registries()
import src.model
import src.task
from src.task.task import LMTask
import src.optim
import src.data  # to load all the things into registries
for registry in registries:
    registry._is_register = False
from src.utils import config as util_config
from src.model.backbone.cache import AttentionCache, LocalAttentionCache, RNNCache
from src.utils.registry import (
    data_registry,
    task_registry,
    lr_scheduler_registry,
    optimizer_registry,
    metric_registry
)
import src.utils.gen as gen_util

"""
torchrun --nnodes=1 --nproc-per-node=1 model_eff.py experiment=fineweb_edu/ratplus/ratplus-1b model.backbone.seq_cell.chunk_size1=16 model.backbone.seq_cell.prefix_size=4 model.backbone.seq_cell.local_size=0
"""

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
@torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
def bench_gen(seq_len, batch_size, config, task: LMTask, seq_start: int):
    task.eval()
    config.data.batch_size = batch_size
    cache = gen_util.get_cache(config)
    for layer_cache in cache.values():
        layer_cache.reset_cache()
        layer_cache.update_kv_fake_prefill(seq_start - 1)
    # warmup
    cur_inp = torch.ones(batch_size, 1, dtype=torch.long).cuda()
    for i in range(seq_start, seq_len):
        task.step(cur_inp, seq_pos=i, cache=cache)
    print("finish warmup!")
    for layer_cache in cache.values():
        layer_cache.reset_cache()
        layer_cache.update_kv_fake_prefill(seq_start - 1)

    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(seq_start, seq_len):
        task.step(cur_inp, seq_pos=i, cache=cache)
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    return total_time


@torch.no_grad()
@torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
def check_mx_bs(task: LMTask, config, seq_start):
    task.eval()
    for batch_size in range(16, 8192, 16): # range(16, 8192, 16):
        torch._dynamo.reset()
        try:
            config.data.batch_size = batch_size
            cache = gen_util.get_cache(config)
            for layer_cache in cache.values():
                layer_cache.reset_cache()
                layer_cache.update_kv_fake_prefill(seq_start - 1)
            cur_inp = torch.ones(batch_size, 1, dtype=torch.long).cuda()
            task.step(input_ids=cur_inp, seq_pos=seq_start, cache=cache)
            del cache
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except RuntimeError as e:
            torch.cuda.empty_cache()
            print("the batch size is {}".format(batch_size - 16))
            raise e


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(config):
    print(config)
    set_seed(1005)
    config = EasyDict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    task = (util_config.instantiate(task_registry,
                                    config=config.task,
                                    model_config=config.model,
                                    device="cuda",
                                    dtype=torch.float32))
    print(task)
    task = task.to("cuda").to(torch.bfloat16)
    torch._dynamo.reset()
    task = torch.compile(task)
    seq_start = config.data.seq_len - 1024
    # check_mx_bs(task, config, seq_start)
    torch.cuda.empty_cache()
    gen_ms = bench_gen(config.data.seq_len, batch_size=config.data.batch_size, config=config, task=task, seq_start=seq_start)
    print("the gen is {:.2f}s".format(gen_ms))


if __name__ == "__main__":
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("int", int)
    gpu_id = int(os.getenv("RANK", -1))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", -1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", world_size=world_size, rank=gpu_id, init_method="env://")
    main()
    dist.destroy_process_group()
