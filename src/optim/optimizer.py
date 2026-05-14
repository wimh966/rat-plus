import torch
from ..utils.registry import optimizer_registry as registry

registry.register("adam", torch.optim.Adam)
registry.register("adamw", torch.optim.AdamW)
registry.register("rmsprop", torch.optim.RMSprop)
registry.register("sgd", torch.optim.SGD)
