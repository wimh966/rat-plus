import torch.nn as nn
from ...utils.registry import init_registry


@init_registry.register()
def xavier(weight, bias=None, **kwargs):
    nn.init.normal_(weight, mean=0.0, std=weight.data.shape[-1] ** -0.5)
    if bias is not None:
        nn.init.zeros_(bias)


@init_registry.register()
def uniform(weight, bias=None, **kwargs):
    r = weight.data.shape[-1] ** -0.5
    nn.init.uniform_(weight, -r, r)
    if bias is not None:
        nn.init.zeros_(bias)

@init_registry.register()
def torch_uniform(weight, bias=None, **kwargs): # make weight and bias all uniform
    r = weight.data.shape[-1] ** -0.5
    nn.init.uniform_(weight, -r, r)
    if bias is not None:
        nn.init.uniform_(bias, -r, r)


@init_registry.register()
def fixed(weight, bias=None, **kwargs):
    nn.init.normal_(weight, mean=0.0, std=kwargs["initializer_range"])
    if bias is not None:
        nn.init.zeros_(bias)


@init_registry.register()
def torch(weight, bias=None, **kwargs):
    pass