import torch
import torch.nn as nn
from ...utils.registry import head_registry
from ..base import Base


@head_registry.register("lm")
class LMHead(Base):

    def __init__(self, vocab_size, d_model, init: dict = None, num_length=0, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False,
                                 device=kwargs.get("device", "cuda"), 
                                 dtype=kwargs.get("dtype", torch.float32))
        self.num_length = num_length # 0 means all, 1 means the last, ...
        self.init_weights(init)

    def forward(self, input: torch.Tensor):
        if self.num_length > 0:
            input = input[..., -self.num_length:,:]
        lm_logits = self.lm_head(input)
        return lm_logits

    @property
    def d_input(self):
        return self.d_model

    @property
    def d_output(self):
        return self.vocab_size

    def nflops(self, bs, seq_len):
        return 2 * bs * seq_len * self.d_input * self.d_output

    @property
    def nparams(self,):
        return self.d_input * self.d_output

    @property
    def vocab_head(self):
        return self.lm_head