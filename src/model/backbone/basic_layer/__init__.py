from .mlp import FFN
from ...base import Identity
from ....utils.registry import layer_registry
layer_registry.register("identity", Identity)