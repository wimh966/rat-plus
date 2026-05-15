import os
import torch
import json
import sys
import os
import glob
import numpy as np
import random
import argparse
from tqdm import tqdm
from easydict import EasyDict
from datasets import load_dataset
from omegaconf import OmegaConf
from transformers import LlamaTokenizer, AutoTokenizer
import hydra
from config.config import dataset2maxlen_dict, dataset2prompt_dict, datasettype_dict
repo_name = "ratplus_clean"
model_root = f"/home/xwei/{repo_name}"
sys.path.insert(0, model_root)
import src.utils.config as util_config
from src.utils.registry import task_registry
from src.utils import convert_load_ckpt
from src.utils import gen as gen_util


def get_hydra_config(hydra_overrides):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("int", int)
    with hydra.initialize(config_path=f"../{repo_name}/configs/", version_base=None):
        config = hydra.compose(
            config_name="config",
            overrides=[x for x in hydra_overrides.split(';')]
        )
    config = EasyDict(OmegaConf.to_container(config, resolve=True, throw_on_missing=True))
    return config


class ModelWrapper:

    def __init__(self, hydra_overrides):
        self.config = get_hydra_config(hydra_overrides)
        torch.serialization.add_safe_globals([EasyDict])
        self.model = (util_config.instantiate(task_registry,
                                              config=self.config.task,
                                              model_config=self.config.model,
                                              device="cuda",
                                              dtype=torch.float32))
        self.save_dir = self.config.trainer.save_dir
        print("load ckpt from {}".format(self.config.trainer.pretrained_path))
        convert_load_ckpt.convert(self.model, self.config.trainer.pretrained_path)

        self.model = self.model.to("cuda").to(torch.float32)
        self.model = torch.compile(self.model)
        self.model.eval()
        print(self.model)
        self.repetition_penalty = 1.0
        self.cache: dict = gen_util.get_cache(self.config)

    @torch.no_grad()
    @torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16)
    def generate_single(self, input_ids, max_new_tokens, enc):
        for layer_id in self.cache:
            self.cache[layer_id].reset_cache()

        pos = input_ids[0].tolist().index(self.config.data.ignore_input_index) - 1
        preds = self.model(input_ids=input_ids, seq_start=0, cache=self.cache, seq_pos=pos).to(torch.float32)
        start_token = torch.argmax(preds[:, pos: pos + 1], dim=-1)
        generated = gen_util.generate_greedy_search(self.model, self.cache, self.config, start_token, pos + 1, max_new_tokens, enc, self.repetition_penalty)
        return generated

def get_pred(data, model: ModelWrapper, enc, prompt_template: str, max_prefill, max_gen, max_len, out_path, eos_token_id: int, test_mode: bool=False):
    print(f"max_prefill {max_prefill}, max_gen {max_gen}, max gen {max_gen}")
    fout = open(out_path, "w", encoding="utf-8")
    seed_everything(1003)
    num = 0
    for json_obj in tqdm(data):
        prompt = prompt_template.format_map(json_obj)
        ids = enc(prompt, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
        # truncation the middle part
        if len(ids) >= max_prefill:
            half = (max_prefill - 1) // 2
            second_half = max_prefill - 1 - half
            ids = ids[:half] + ids[-second_half:]
            assert len(ids) < max_prefill, len(ids)
        # we run prefill by using max len to avoid triton compile error
        input_ids = ids + [eos_token_id] * (max_len - len(ids))
        input_ids = torch.tensor(input_ids).unsqueeze(0).cuda()
        pred = model.generate_single(input_ids=input_ids, max_new_tokens=max_gen, enc=enc) # we do post process in evaluation stage
        json.dump({"pred": pred, "answers":  json_obj["answers"]}, fout, ensure_ascii=False)
        fout.write('\n')
        num += 1
        if test_mode and num >= 1:
            break
    fout.close()


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hydra_overrides', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--data_type', type=int, default=1)
    parser.add_argument('--tokenizer', type=str, default="llama")
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode')
    args = parser.parse_args()
    seed_everything(42)
    # build model
    model = ModelWrapper(args.hydra_overrides)
    if args.tokenizer == "llama":
        enc = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
    else:
        raise NotImplementedError
    eos_token_id = enc.eos_token_id
    datasets = []
    datasets = datasettype_dict.get(args.data_type)
    # for i in range(1, 6):
    #     datasets = datasets + datasettype_dict.get(i)
    if args.data_type == 3:
        model.repetition_penalty = 1.2
    for dataset in datasets:
        data = load_dataset('THUDM/LongBench', dataset, split='test', trust_remote_code=True)
        save_dir = model.save_dir
        if not os.path.exists(f"{save_dir}"):
            os.makedirs(f"{save_dir}")
        out_path = f"{save_dir}/{dataset}.jsonl"
        prompt_template = dataset2prompt_dict[dataset]
        # print(prompt_template.format_map(data[0]))
        get_pred(data, model, enc,
                 prompt_template,
                 max_gen=dataset2maxlen_dict[dataset], 
                 max_len=args.max_len,
                 max_prefill=args.max_len - dataset2maxlen_dict[dataset],
                 out_path=out_path,
                 eos_token_id=eos_token_id,
                 test_mode=args.test_mode)
        if args.test_mode:
            break
