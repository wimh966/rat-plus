# Copyright (c) 2026 Xiuying Wei, EPFL CLAIRE lab
# All rights reserved.
# Licensed under the Apache License 2.0.

import random
import os
import glob
import time
from tqdm import tqdm
import wandb
from contextlib import nullcontext
import shutil
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict
from omegaconf import OmegaConf
import torch.distributed as dist
import torch.profiler as tp
from ..utils import config as util_config
from ..utils import convert_load_ckpt
from ..utils.registry import (
    data_registry,
    task_registry,
    lr_scheduler_registry,
    optimizer_registry,
    metric_registry
)
from ..task.task import BaseTask
name2torchdtype = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class WandbLog:

    def __init__(self, config, metric, x_axis="epoch"):
        self.config = config
        for k, v in metric.items():
            if k == x_axis:
                wandb.define_metric(x_axis)
            else:
                wandb.define_metric(k, step_metric=x_axis)

    def record(self, item):
        wandb.log(item)


class Trainer:

    task: BaseTask

    def __init__(self, config):
        # configs
        self.config = EasyDict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
        self.train_dtype = name2torchdtype.get(self.config.trainer.dtype)
        # gpus
        self.gpu_id = int(os.getenv("RANK", -1))  # gpu_id means global rank
        assert self.gpu_id != -1, "we only support torchrun in job submission"
        self.local_rank = int(os.getenv("LOCAL_RANK", -1))
        self.device = (torch.device("cuda", self.local_rank) if self.local_rank != -1 else torch.device("cuda"))
        self.ngpus = dist.get_world_size() if self.gpu_id != -1 else 1
        print("The device is {} out of {}".format(self.gpu_id, self.ngpus))
        # set seed
        self.set_seed(config.trainer.seed)
        self.build_dataloader()
        self.step = -1
        self.max_epoch = self.config.trainer.max_epoch
        self.global_batch_size = self.config.trainer.global_batch_size
        assert (self.global_batch_size % (self.ngpus * self.config.data.batch_size) == 0)
        self.gradient_accumulation_steps = self.global_batch_size // (self.ngpus * self.config.data.batch_size)
        self.max_step = self.max_epoch * len(self.train_loader) // self.gradient_accumulation_steps # before optimizers
        self.set_load_save()
        self.build_task()
        self.build_optimizers()
        # load checkpoint and consider ddp
        self.resume_kwargs = self.load_checkpoint()
        if self.gpu_id != -1:
            self.task_wrapper = torch.nn.parallel.DistributedDataParallel(
                self.task, device_ids=[self.local_rank], output_device=self.local_rank,
                find_unused_parameters=True)
            if self.config.trainer.torch_compile:
                self.task_wrapper = torch.compile(self.task_wrapper)
        self.set_logging()
        self.print_info()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def set_logging(self,):
        # set logging
        self.log_interval = getattr(self.config.trainer, "log_interval", False)
        self.logging_metrics = {"train_loss": 0.0, "val_loss": 0.0, "step": 0, "lr": 0.0, "fwd+bwd": 0.0}
        self.logging_metrics[f"val_{self.config.task.metric._name_}"] = 0.0

        # set wandb
        if self.config.wandb_use and self.gpu_id in [-1, 0]:
            if not os.path.exists(self.config.wandb.dir):
                os.makedirs(self.config.wandb.dir)
            wandb.init(
                config=self.config,
                entity=self.config.wandb.entity,
                project=self.config.wandb.project,
                resume=None,
                anonymous=self.config.wandb.anonymous,
                mode=self.config.wandb.mode,
                dir=self.config.wandb.dir,
            )
            self.wandblog = WandbLog(self.config.wandb, self.logging_metrics, x_axis="step")

    def set_load_save(self,):
        self.is_save_checkpoint = getattr(self.config.trainer, "save_checkpoint", False)
        self.is_load_checkpoint = getattr(self.config.trainer, "load_checkpoint", False)
        self.save_interval = int(self.max_step * getattr(self.config.trainer, "save_interval", 0.1))
        task_long_name = task_registry.get(self.config.task._name_).get_ckpt_name(self.config.task, self.config.model)
        data_long_name = self.config.data._name_ + str(self.config.data.seq_len)
        self.save_dir = os.path.join(self.config.trainer.save_dir, data_long_name + "-" + task_long_name,
                                     str(self.config.optim.optimizer.lr).replace(".", "x") + "_" + str(self.config.data.global_batch_size) + "_" + str(self.config.trainer.max_epoch))
        print("plan to save or load checkpoint in {} for each {} step".format(self.save_dir, self.save_interval))
        if not os.path.exists(self.save_dir) and self.is_save_checkpoint and self.gpu_id == 0:
            os.makedirs(self.save_dir)

    def build_dataloader(self):
        self.train_loader = util_config.instantiate(registry=data_registry, config=self.config.data.train)
        self.val_loader = util_config.instantiate(registry=data_registry, config=self.config.data.val)

    def build_task(self,):
        self.task = (util_config.instantiate(task_registry,
                                             config=self.config.task,
                                             model_config=self.config.model,
                                             device=self.device,
                                             dtype=torch.float32))
        if self.config.trainer.pretrained_path is not None:
            convert_load_ckpt.convert(self.task, self.config.trainer.pretrained_path)
        self.task = self.task.to(self.device).to(torch.float32)

    def build_optimizers(self,):
        all_params = list(self.task.parameters())
        self.optimizer = util_config.instantiate(optimizer_registry, self.config.optim.optimizer, all_params)
        if self.config.optim.lr_scheduler._name_ in ["cosine", "constant"]:
            self.config.optim.lr_scheduler.T_max = self.max_step

        self.lr_scheduler = util_config.instantiate(lr_scheduler_registry, self.config.optim.lr_scheduler, optimizer=self.optimizer) if self.max_step != 0 else None
        self.gradient_clipping = getattr(self.config.trainer, "gradient_clipping", False)

    def print_info(self, type="init"):
        if self.gpu_id not in [-1, 0]:
            return
        if type == "init":
            print("the model size is {:.2f}M".format(sum([w.numel() for w in self.task.parameters()]) / 10 ** 6))
            print("the config is {}".format(self.config))
            print("train loader: length {}, examples {}".format(len(self.train_loader), self.train_loader.nsamples))
            print("val loader: length {}, examples {}".format(len(self.val_loader), self.val_loader.nsamples))
            print("the task is {}".format(self.task))
        elif type == "train":
            print("***** Running training *****")
            print("Global batch size = {}".format(self.global_batch_size))
            print("Gradient Accumulation steps = {}".format(self.gradient_accumulation_steps))
            print("Resume from step {} in total {} step".format(self.step, self.max_step))
        elif type == "validate":
            pass
        else:
            raise NotImplementedError

    def save_checkpoint(self, **resume_kwargs):
        # save checkpoint by step
        if not self.is_save_checkpoint:
            return
        if ((self.step + 1) % self.save_interval == 0 or self.step + 1 == self.max_step):
            if self.gpu_id in [-1, 0]:
                ckpt_path = os.path.join(self.save_dir, f"{self.step}.pth")
                ckpt = {
                    "task": self.task.state_dict(),
                    "config": self.config,
                    "step": self.step,
                    "optimizer": self.optimizer.state_dict(),
                    "lr_scheduler": self.lr_scheduler.state_dict(),
                    "resume_kwargs": resume_kwargs,
                }
                torch.save(ckpt, ckpt_path)
            dist.barrier()

    def load_checkpoint(self):
        if not self.is_load_checkpoint:
            return {}

        def find_latest_checkpoint():
            checkpoint_files = glob.glob(os.path.join(self.save_dir, f"*.pth"))
            return None if not checkpoint_files else max(checkpoint_files, key=os.path.getctime)

        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint is not None:
            print("load checkpoint from {}".format(latest_checkpoint))
            torch.serialization.add_safe_globals([EasyDict])
            ckpt = torch.load(latest_checkpoint, map_location=self.device)
            state_dict = ckpt["task"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if "_orig_mod." in k:
                    new_state_dict[k.replace("_orig_mod.", "")] = v
                else:
                    new_state_dict[k] = v
            self.task.load_state_dict(new_state_dict)
            self.optimizer.load_state_dict(ckpt["optimizer"])
            if self.lr_scheduler is not None:
                self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
            self.step = ckpt["step"]
            return ckpt["resume_kwargs"]
        return {}

    def forward(self, **kwargs):
        pass

    def validate(self, ):
        pass

    @torch.no_grad()
    def one_batch(self, ):
        # for draw, logging, ....
        self.task_wrapper.eval()
        trainloader_iter = iter(self.train_loader)
        inputs, labels, *extra_args = next(trainloader_iter)
        _, _ = self.forward(inputs, labels)

    def train(self, ):
        if self.step >= self.max_step - 1:
            return
        self.print_info("train")
        self.task_wrapper.train()
        self.optimizer.zero_grad()
        train_iterator = tqdm(range(self.step + 1, self.max_step), desc="Steps", disable=self.gpu_id not in [-1, 0])
        trainloader_iter = iter(self.train_loader)
        for i in range(0, self.step + 1):
            for j in range(self.gradient_accumulation_steps):
                try:
                    next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(self.train_loader)
                    next(trainloader_iter)

        for i in train_iterator:
            torch.cuda.synchronize()
            t0 = time.time()
            train_loss = 0.0
            train_loss1 = 0.0
            cached_inputs, cached_labels = [], []
            for micro_step in range(self.gradient_accumulation_steps):
                try:
                    inputs, labels, *extra_args = next(trainloader_iter)
                except StopIteration:
                    trainloader_iter = iter(self.train_loader)
                    inputs, labels, *extra_args = next(trainloader_iter)
                cached_inputs.append(inputs)
                cached_labels.append(labels)
                ctx_fn = self.task_wrapper.no_sync if micro_step < self.gradient_accumulation_steps - 1 else nullcontext
                with ctx_fn():
                    inputs = inputs.to("cuda", non_blocking=True)
                    labels = labels.to("cuda", non_blocking=True)
                    if self.config.model.backbone.seq_cell._name_ in ["ratplus"]:
                        loss, _ = self.forward(inputs, labels, mode="sparse")
                        loss = loss / self.gradient_accumulation_steps
                        train_loss1 = train_loss1 + loss.detach().item()
                        loss.backward()
                    else:
                        loss, _ = self.forward(inputs, labels)
                        loss = loss / self.gradient_accumulation_steps
                        train_loss = train_loss + loss.detach().item()
                        loss.backward()
            # finish the step
            if self.gradient_clipping is not False:
                torch.nn.utils.clip_grad_norm_(self.task.parameters(), self.gradient_clipping)

            self.optimizer.step()
            if self.config.model.backbone.seq_cell._name_ in ["ratplus"] and self.config.model.backbone.seq_cell.joint_train is True:
                self.optimizer.zero_grad()
                for micro_step in range(self.gradient_accumulation_steps):
                    ctx_fn = self.task_wrapper.no_sync if micro_step < self.gradient_accumulation_steps - 1 else nullcontext
                    with ctx_fn():
                        inputs = cached_inputs[micro_step].to("cuda", non_blocking=True)
                        labels = cached_labels[micro_step].to("cuda", non_blocking=True)
                        loss, _ = self.forward(inputs, labels)
                        loss = loss / self.gradient_accumulation_steps
                        train_loss = train_loss + loss.detach().item()
                        loss.backward()
                if self.gradient_clipping is not False:
                    torch.nn.utils.clip_grad_norm_(self.task.parameters(), self.gradient_clipping)
                self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            torch.cuda.synchronize()
            t2 = time.time()
            self.step += 1
            if (self.step + 1) % self.log_interval == 0:
                val_loss, val_metric = (self.validate() if self.config.trainer.eval_when_log else (0.0, 0.0))
                train_loss = torch.tensor(train_loss, device="cuda")
                dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
                train_loss1 = torch.tensor(train_loss1, device="cuda")
                dist.all_reduce(train_loss1, op=dist.ReduceOp.SUM)
                self.logging_metrics.update({
                    "train_loss": round(train_loss.item() / self.ngpus, 4),
                    "train_loss1": round(train_loss1.item() / self.ngpus, 4),
                    "val_loss": round(val_loss, 4),
                    f"val_{self.config.task.metric._name_}": val_metric,
                    "step": self.step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "fwd+bwd": (t2 - t0),
                    }
                )
                if self.gpu_id in [-1, 0]:
                    self.wandblog.record(self.logging_metrics) if self.config.wandb_use else print(self.logging_metrics)
                self.task_wrapper.train()

            self.save_checkpoint(**{"resume_step": self.step})
