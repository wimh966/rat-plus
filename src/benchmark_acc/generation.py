import os
import sys
import hydra
import wandb
import torch
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
import src.optim
import src.data  # to load all the things into registries

for registry in registries:
    registry._is_register = False

@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="config",
)
def main(config):
    print(config)
    gpu_id = int(os.getenv("RANK", -1))
    trainer = LMTrainer(config)
    dist.barrier()
    print("Begin to generate!")
    trainer.generate()


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
