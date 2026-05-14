# Copyright (c) 2026 Xiuying Wei, EPFL CLAIRE lab
# All rights reserved.
# Licensed under the Apache License 2.0.

import math
import numpy as np
import time
from tqdm import tqdm
import torch
from collections import defaultdict
from typing import Union
import torch.distributed as dist
from torchmetrics.text import Perplexity
from . import fsdp_trainer
from ..data.lm_dataloader import LMOrderedDataloader, LMRandomDataloader
from ..utils import config as util_config
from ..utils.registry import (
    data_registry,
    task_registry,
    lr_scheduler_registry,
    optimizer_registry,
)
from ..task.task import LMTask


class LMFSDPTrainer(fsdp_trainer.FSDPTrainer):
    task: LMTask

    def __init__(self, config):
        super().__init__(config)
        assert self.config.task._name_ == "lm"

    def forward(self, input_ids, labels, **kwargs):
        with torch.amp.autocast("cuda", enabled=True, dtype=self.train_dtype):
            preds = self.task_wrapper(input_ids=input_ids, seq_start=0, cache=None, **kwargs).to(torch.float32)
        loss = self.task.get_loss(preds, labels)
        return loss, preds

    @torch.no_grad()
    def validate(self):
        self.task_wrapper.eval()
        ddp_loss = torch.tensor(0.0).to(self.device)
        ddp_samples = torch.tensor(0.0).to(self.device)
        self.task.metric.reset()
        for i, (inputs, labels, *extra_args) in enumerate(self.val_loader):
            inputs = inputs.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)
            loss, preds = self.forward(inputs, labels)
            cnt = (labels != self.config.task.ignore_index).sum().item()
            ddp_loss += loss * cnt
            ddp_samples += cnt
            self.task.metric(preds, labels)

        dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(ddp_samples, op=dist.ReduceOp.SUM)
        val_loss = (ddp_loss / ddp_samples).item()
        val_metric = self.task.metric.compute()
        return val_loss, val_metric.item()

    def print_info(self, type="init"):
        if self.gpu_id not in [-1, 0]:
            return
        if type == "train":
            super().print_info(type)
            print("Num Examples = {}".format(self.train_loader.nsamples))
            print("Num Tokens = {}".format(self.train_loader.nsamples * self.config.data.seq_len))
        else:
            super().print_info(type)