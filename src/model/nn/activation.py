import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import Identity
from ...utils.registry import activation_registry as registry


class SwiGLU(nn.Module):

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class ReLUSquared(nn.Module):
    def forward(self, x, *args, **kwargs):
        return F.relu(x) ** 2

# no _name_ here, all functions here
registry.register("tanh", torch.tanh)
registry.register("silu", torch.nn.SiLU())
registry.register("sigmoid", torch.sigmoid)
registry.register("exp", torch.exp)
registry.register("relu", torch.relu)
registry.register("relusquared", ReLUSquared())
registry.register("softmax", F.softmax)
registry.register("gelu", torch.nn.GELU())
registry.register("fastgelu", torch.nn.GELU(approximate="tanh"))
registry.register("swiglu", SwiGLU())
registry.register("identity", Identity())
