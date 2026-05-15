import torch
import math
import copy
from datasets import load_dataset
from transformers import LlamaTokenizer
import torch.nn.functional as F
from ..model.backbone.cache import RATPlusSingleLayerCache, RATPlusFullSingleLayerCache


def get_cache(config):

    def get_ratplus_fullcache(bs, seq_len, num_layers, seq_cell):
        cache = dict()
        for i in range(num_layers):
            cache[i] = RATPlusFullSingleLayerCache(i, bs, seq_len,
                                                   initial_size=seq_cell.initial_size[i] if type(seq_cell.initial_size) is list else seq_cell.initial_size,
                                                   local_size=seq_cell.local_size[i] if type(seq_cell.local_size) is list else seq_cell.local_size,
                                                   dilation_size=seq_cell.dilation_size[i] if type(seq_cell.dilation_size) is list else seq_cell.dilation_size,
                                                   num_head=seq_cell.num_kv_head if hasattr(seq_cell, "num_kv_head") else seq_cell.num_head,
                                                   d_head=seq_cell.d_head,
                                                   d_model=seq_cell.d_model, 
                                                   dtype=torch.bfloat16, 
                                                   device="cuda")
        return cache


    def get_ratplus_cache(bs, seq_len, num_layers, seq_cell):
        cache = dict()
        for i in range(num_layers):
            cache[i] = RATPlusSingleLayerCache(i, bs, seq_len,
                                               initial_size=seq_cell.initial_size[i] if type(seq_cell.initial_size) is list else seq_cell.initial_size,
                                               local_size=seq_cell.local_size[i] if type(seq_cell.local_size) is list else seq_cell.local_size,
                                               dilation_size=seq_cell.dilation_size[i] if type(seq_cell.dilation_size) is list else seq_cell.dilation_size,
                                               num_head=seq_cell.num_kv_head if hasattr(seq_cell, "num_kv_head") else seq_cell.num_head,
                                               d_head=seq_cell.d_head,
                                               d_model=seq_cell.d_model,
                                               dtype=torch.bfloat16,
                                               device="cuda")
        return cache

    cache_factory = {
        "ratplus": get_ratplus_cache,
    }
    cache_fn = cache_factory[config.model.backbone.seq_cell._name_]
    cache = cache_fn(config.data.batch_size, config.data.seq_len, config.model.backbone.num_layers, config.model.backbone.seq_cell)
    print(cache)
    return cache


max_new_tokens_dict = {
    "narrativeqa_summary": 128,
    "narrativeqa_text": 128,
    "narrativeqa_full": 128,
    "qmsum": 384,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "wikisum": 256,
    "dureader": 128,
    "govreport": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64
}


cache_dir = "/home/xwei/fake_path/huggingface_home"


def get_test_set(data):
    return load_dataset('THUDM/LongBench', data, split="test", cache_dir=cache_dir)


def apply_repetition_penalty(logits, input_ids, penalty=1.2):
    token_tensor = torch.tensor(input_ids, dtype=torch.long, device=logits.device).unsqueeze(0).unsqueeze(1)
    score = logits.gather(2, token_tensor)  # shape: (B, 1, D)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(2, token_tensor, score)


@torch.no_grad()
def generate_greedy_search(
    model,
    cache,
    config,
    input_ids: torch.Tensor,
    seq_start: int = 2048,
    max_new_tokens: int=-1,
    enc=None,
    repetition_penalty=1.0,
):
    """
    Beam search decoding.
    model: your model, must return logits of shape [B, 1, V]
    input_ids: [B, 1]
    returns: [B, 1 + max_new_tokens]
    """
    if enc is None:
        enc = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", cache_dir=cache_dir, local_files_only=True)
    if max_new_tokens == -1:
        max_new_tokens = max_new_tokens_dict.get(config.data._name_, 128)
    generated = []
    generated.append(input_ids.item())
    for step in range(seq_start, seq_start + max_new_tokens):
        logits = model.step(input_ids=input_ids, seq_start=step, cache=cache, seq_pos=step).to(torch.float32).contiguous()
        if repetition_penalty != 1.0:
            apply_repetition_penalty(logits, generated, repetition_penalty)
        next_token = torch.argmax(logits, dim=-1)  # [B, 1]
        input_ids = next_token
        generated.append(next_token.item())
        if torch.all(next_token.squeeze(1) == config.data.ignore_input_index):
            break
    generated = enc.decode(generated, skip_special_tokens=True)
    return generated