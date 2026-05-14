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
    if hasattr(config.trainer, "fsdp"):
        trainer = LMFSDPTrainer(config)
    else:
        trainer = LMTrainer(config)

    # trainer.one_batch()
    ### Train ####
    trainer.train()
    dist.barrier()
    print("Finish Training!")
    print("Begin to validate!")
    loss, val_metric = trainer.validate()
    final_val_results = {}
    final_val_results["result_loss"] = round(loss, 4)
    final_val_results[f"result_metric"] = val_metric

    if gpu_id in [-1, 0] and config.wandb_use:
        wandb.log(final_val_results)
        wandb.finish()

    print("validation loss is {:.4f} and metric is {}".format(loss, val_metric))


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
