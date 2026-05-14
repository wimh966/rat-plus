import torch
import torch.nn as nn
from einops import rearrange
from ..base import Identity
from ...utils.registry import norm_registry as registry

# get class
registry.register("layernorm", torch.nn.LayerNorm)
registry.register("identity", Identity)
registry.register("rmsnorm", torch.nn.RMSNorm)
