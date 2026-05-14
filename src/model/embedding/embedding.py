import torch
import torch.nn as nn
from ...utils.registry import embedding_registry, pe_registry
from ...utils import config as util_config
from ..base import Base


@embedding_registry.register("lm")
class LMEmbedding(Base):

    def __init__(
        self,
        vocab_size,
        d_model,
        seq_len,
        dropout=0.0,
        pe: dict = None,
        init: dict = None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.pe = pe
        self.wte = nn.Embedding(vocab_size, d_model, device=kwargs.get("device", "cuda"), dtype=kwargs.get("dtype", torch.float32))
        self.wpe = util_config.instantiate(pe_registry, pe, **kwargs)
        self.init_weights(init)

    def forward(self, input_ids, seq_start=0):  # the position of seq_start
        bs, seq = input_ids.shape
        seq_end = seq_start + seq # [seq_start, seq_end)
        hidden_states = self.wte(input_ids)
        pos_emb_cur, pos_emb_later = self.wpe(seq_start, seq_end, hidden_states.device, hidden_states.dtype)
        if pos_emb_cur is not None:
            hidden_states = hidden_states + pos_emb_cur
        return hidden_states, pos_emb_later # dict

    def step(self, input_ids, seq_start=0):
        bs, seq = input_ids.shape
        seq_end = seq_start + seq # [seq_start, seq_end)
        hidden_states = self.wte(input_ids)
        pos_emb_cur, pos_emb_later = self.wpe.step(seq_start, seq_end, hidden_states.device, hidden_states.dtype)
        if pos_emb_cur is not None:
            hidden_states = hidden_states + pos_emb_cur
        return hidden_states, pos_emb_later # dict

    @property
    def d_input(self):
        return self.vocab_size

    @property
    def d_output(self):
        return self.d_model

    def nflops(self, bs, seq_len):
        return 0

    @property
    def nparams(self):
        return self.d_input * self.d_output

    @property
    def vocab_embedding(self):
        return self.wte

    @staticmethod
    def get_ckpt_name(model_config):
        if hasattr(model_config.pe, "base"):
            return model_config._name_ + f"pos{model_config.pe._name_}" + f"{model_config.pe.base}"
        return model_config._name_ + f"pos{model_config.pe._name_}"
