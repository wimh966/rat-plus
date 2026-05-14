from .mlp import FFN
from .attention import Attention
from .local_attention import LocalAttention
from .attentionswa_infer import AttentionSWAInfer
from .attention_im import AttentionIm
from ...base import Identity
from ....utils.registry import layer_registry
layer_registry.register("identity", Identity)