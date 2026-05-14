# Copyright (c) 2026 Xiuying Wei, EPFL CLAIRE lab
# All rights reserved.
# Licensed under the Apache License 2.0.

import math
import numpy as np
import time
from tqdm import tqdm
import torch
import json
import os
from collections import defaultdict
from typing import Union
import torch.distributed as dist
from torchmetrics.text import Perplexity
from . import trainer
from ..data.lm_dataloader import LMOrderedDataloader, LMRandomDataloader
from ..utils import config as util_config
from ..utils.registry import (
    data_registry,
    task_registry,
    lr_scheduler_registry,
    optimizer_registry,
)
from ..utils import gen as gen_util
from ..task.task import LMTask


class LMTrainer(trainer.Trainer):
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

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def generate(self):
        self.task_wrapper.eval()
        cache: dict = gen_util.get_cache(self.config)
        # generate sample one by one
        with open(os.path.join(self.save_dir, f"{self.config.data._name_}_rank{self.gpu_id}.jsonl"), "w", encoding="utf-8") as f:
            test_data_path = os.path.join(os.path.dirname(self.config.data.val.tokenized_input_path), f"llama-validation-{self.config.data.seq_len}-inputs.bin")
            test_data = np.memmap(test_data_path, mode="r", dtype=np.int16)
            nsamples = len(test_data) // self.config.data.seq_len
            nsamples_per_gpu = nsamples // self.ngpus
            sample_st = nsamples_per_gpu * self.gpu_id
            sample_ed = nsamples_per_gpu * (self.gpu_id + 1)
            if sample_ed + nsamples_per_gpu > nsamples:
                sample_ed = nsamples
            for i in range(sample_st, sample_ed):
                inputs = test_data[i * self.config.data.seq_len: (i + 1) * self.config.data.seq_len]
                inputs = torch.from_numpy(inputs.astype(np.int64)).pin_memory().to("cuda", non_blocking=True).unsqueeze(0)
                for layer_id in cache:
                    cache[layer_id].reset_cache()
                inputs = inputs.to("cuda", non_blocking=True)
                pos = inputs[0].tolist().index(self.config.data.ignore_input_index) - 1
                preds = self.task(input_ids=inputs, seq_start=0, cache=cache, seq_pos=pos).to(torch.float32)
                start_token = torch.argmax(preds[:, pos: pos + 1], dim=-1)
                generated = gen_util.generate_greedy_search(self.task, cache, self.config, start_token, pos + 1)
                json.dump({"pred": generated}, f, ensure_ascii=False)
                f.write('\n')
        dist.barrier()
        if self.gpu_id == 0:
            # merge results
            answer_path = os.path.join(os.path.dirname(self.config.data.val.tokenized_input_path), "test.jsonl")
            answer_list = []
            with open(answer_path, "r", encoding="utf-8") as f_ans:
                for line in f_ans:
                    ans_item = json.loads(line)
                    answer_list.append(ans_item.get("outputs", None))  # or .get("answers")

            output_path = os.path.join(self.save_dir, f"{self.config.data._name_}.jsonl")
            with open(output_path, "w", encoding="utf-8") as fout:
                index = 0
                for r in range(self.ngpus):
                    file_path = os.path.join(self.save_dir, f"{self.config.data._name_}_rank{r}.jsonl")
                    with open(file_path, "r", encoding="utf-8") as fin:
                        for line in fin:
                            item = json.loads(line)
                            item["outputs"] = answer_list[index]
                            json.dump(item, fout, ensure_ascii=False)
                            fout.write("\n")
                            index += 1
        dist.barrier()

    def print_info(self, type="init"):
        if self.gpu_id not in [-1, 0]:
            return
        if type == "train":
            super().print_info(type)
            print("Num Examples = {}".format(self.train_loader.nsamples))
            print("Num Tokens = {}".format(self.train_loader.nsamples * self.config.data.seq_len))
        else:
            super().print_info(type)
