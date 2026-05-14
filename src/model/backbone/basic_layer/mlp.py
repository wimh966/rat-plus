import torch.nn as nn
import torch
from ....utils.registry import (
    layer_registry,
    activation_registry,
    norm_registry,
)
from ...base import Base


@layer_registry.register("ffn")
class FFN(Base):  # two layers + ln + residual connection

    def __init__(
        self,
        d_model,
        d_ffn,
        bias=True,
        act_fn="gelu",
        ln="identity",
        dropout=0.0,
        init: dict = None,
        **kwargs,
    ):
        super().__init__()
        factory_kwargs = {
            "device": kwargs.get("device", "cuda"),
            "dtype": kwargs.get("dtype", torch.float32),
        }
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.act_func = activation_registry.get(act_fn)
        if act_fn in ["swiglu"]:
            self.d_ffn *= 2
        self.fc1 = nn.Linear(self.d_model, self.d_ffn, bias=bias, **factory_kwargs)
        self.fc2 = nn.Linear(d_ffn, self.d_model, bias=bias, **factory_kwargs)
        self.ln = norm_registry[ln](self.d_model, eps=1.0e-6)  # only support pre-normalization here
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input):
        shortcut = input
        input = self.ln(input)
        fc1_out = self.fc1(input)
        act_out = self.act_func(fc1_out)
        fc2_outs = self.dropout(self.fc2(act_out))
        return fc2_outs + shortcut

    def nflops(self, bs):
        return 2 * bs * self.nparams

    @property
    def nparams(self,):
        return sum([w.numel() for w in self.fc1.parameters()]) + sum(
            [w.numel() for w in self.fc2.parameters()])

    @staticmethod
    def get_ckpt_name(model_config):
        return model_config._name_
