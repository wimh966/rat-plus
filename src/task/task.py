"""Implements Task interface, which consists of embedding + backbone + head, and loss/metrics."""

import torch
from typing import Optional, List, Tuple
import torch.nn as nn
import functools
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.nn.functional as F
from einops import rearrange
from omegaconf import ListConfig
from ..utils import config as util_config
from ..utils.registry import (
    task_registry,
    backbone_registry,
    head_registry,
    embedding_registry,
    metric_registry,
)
from ..model.base import Base


@task_registry.register("base")
class BaseTask(nn.Module):
    """Abstract class for all task.
    """

    embedding: Base
    head: Base
    backbone: Base

    def __init__(
        self,
        loss=None,
        metric=None,
        model_config=None,
        **kwargs,  # device and dtype
    ):
        super().__init__()
        self.model_config = model_config
        self.loss = util_config.instantiate(metric_registry, loss)
        self.metric = util_config.instantiate(metric_registry, metric)
        self.embedding = util_config.instantiate(embedding_registry, self.model_config.embedding, **kwargs)
        self.backbone = util_config.instantiate(backbone_registry, self.model_config.backbone, **kwargs)
        self.head = util_config.instantiate(head_registry, self.model_config.head, **kwargs)

    @property
    def nparams(self, ):
        return self.embedding.nparams

    def nflops(self, ):
        return self.embedding.nflops() + self.backbone.nflops() + self.head.nflops()

    def forward(self, inputs, **kwargs):
        pass

    @staticmethod
    def get_ckpt_name(task_config, model_config):
        embedding_name = model_config.embedding._name_
        backbone_name = model_config.backbone._name_
        head_name = model_config.head._name_
        return (
            f"{task_config._name_}"
            + "-" + embedding_registry.get(embedding_name).get_ckpt_name(model_config.embedding)
            + "-" + backbone_registry.get(backbone_name).get_ckpt_name(model_config.backbone)
            + "-" + head_registry.get(head_name).get_ckpt_name(model_config.head)
        )

    def print_info(self,):
        pass


@task_registry.register("lm")
class LMTask(BaseTask):

    def __init__(self, loss=None, metric=None, model_config=None, **kwargs):
        super().__init__(loss, metric, model_config, **kwargs)
        self.tie_embedding = kwargs.get("tie_embedding", False)
        if self.tie_embedding:
            assert hasattr(self.head, "vocab_head") and hasattr(self.embedding, "vocab_embedding")
            assert self.head.d_output == self.embedding.d_input
            self.head.vocab_head.weight = self.embedding.vocab_embedding.weight

    def forward(self, input_ids: torch.LongTensor = None, seq_start=0, cache=None, **kwargs):
        embedding_out, emb_kwargs = self.embedding(input_ids, seq_start)
        backbone_out = self.backbone(embedding_out, cache, **kwargs, **emb_kwargs)
        final_out = self.head(backbone_out)
        return final_out

    def step(self, input_ids: torch.LongTensor, seq_start=0, cache=None, **kwargs):
        embedding_out, emb_kwargs = self.embedding.step(input_ids, seq_start)
        backbone_out = self.backbone.step(embedding_out, cache, **kwargs, **emb_kwargs)
        final_out = self.head.step(backbone_out)
        return final_out

    @torch.compile
    def get_loss(self, preds, labels):
        preds = preds.view(-1, preds.shape[-1])
        labels = labels.view(-1)
        return self.loss(preds, labels)

    def get_fsdp_plicy(self,): # for fsdp1
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=self.backbone.get_fsdp_policy(),
        )
        return auto_wrap_policy

    @property
    def nparams(self, ):
        return self.embedding.nparams + self.backbone.nparams + self.head.nparams - (self.head.vocab_head.weight.numel() if self.tie_embedding else 0)

    def nflops(self, bs, seq_len):
        return self.embedding.nflops(bs, seq_len) + self.backbone.nflops(bs, seq_len) + self.head.nflops(bs, seq_len)

