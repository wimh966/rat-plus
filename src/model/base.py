import torch.nn as nn
import torch
import random
import os
from ..utils.registry import init_registry
from ..utils import config as util_config


class Base(nn.Module):

    def __init__(self,):
        super().__init__()

    @torch.no_grad()
    def init_weights(self, init_config):
        init_func = util_config.instantiate(init_registry, init_config, partial=True)
        for m in self.modules():
            self._init_weights(m, init_func)

    @torch.no_grad()
    def _init_weights(self, m, init_func=None): # basic cell for init wegihts
        if isinstance(m, (nn.Linear, nn.Embedding)):
            init_func(weight=m.weight, bias=m.bias if hasattr(m, "bias") else None)
        elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
            m.weight.data.fill_(1.0)
            if getattr(m, "bias", None) is not None:
                m.bias.data.zero_()
        else:
            pass

    def forward(self, x, *args, **kwargs):
        return x

    def step(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    @property
    def d_input(self):
        pass

    @property
    def d_output(self):
        pass

    def nflops(self):
        pass

    @property
    def nparams(self):
        pass

    @staticmethod
    def get_ckpt_name(model_config):
        return model_config._name_


class Identity(Base):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def nparams(self):
        return 0

    def nflops(self, *args, **kwargs):
        return 0

    @staticmethod
    def get_ckpt_name(model_config):
        return model_config._name_
