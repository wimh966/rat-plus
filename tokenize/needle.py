from argparse import ArgumentParser
from transformers import AutoTokenizer
import os
import shutil
from datasets import concatenate_datasets
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
from transformers import LlamaTokenizer


# prefill_max_len_dict = {
#     "needle": {
#         4096: 3968,
#     }
# }


cache_dir = "/home/xwei/fake_path/dataset/needle/"


def tokenize_answer_only(dataset, tokenizer, task, num_proc, max_length, split="train"):
    org_max_length = max_length
    if tokenizer == "olmo":
        enc = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
        special_token = 100257
    elif tokenizer == "llama":
        enc = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
        special_token = 2
    if split == "validation":
        max_length = 4096 - 128 #prefill_max_len_dict.get(task).get(max_length)
    def tokenize_process(example):
        prompt = example["input"]
        ids = enc(prompt, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
        labels = [-100] * (len(ids) - 1)
        if split != "validation":
            joined = ",".join(str(n) for n in example["outputs"])
            answer_ids = enc(joined, truncation=False, padding=False, add_special_tokens=False)["input_ids"]
            ids = ids + answer_ids
            labels = labels + answer_ids
        assert len(ids) < max_length
        input_ids = ids + [special_token] * (org_max_length - len(ids))
        labels = labels + [special_token] + [-100] * (org_max_length - len(ids))
        assert len(input_ids) == len(labels)
        out = {'input_ids': input_ids, 'labels': labels}
        return out

    tokenized = dataset.map(
        tokenize_process,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )
    print(tokenized)
    # print(tokenized["input_ids"][0])
    text = enc.decode(tokenized["input_ids"][0], skip_special_tokens=True)
    print(text)
    # label = enc.decode(tokenized["labels"][0], skip_special_tokens=True)
    # print(label)
    return tokenized


def save_to_npmemmap(split, tokenizer, dset, path, max_length):
    filename = os.path.join(cache_dir, f"{path}.bin")
    arr_len = dset.num_rows
    if tokenizer == "olmo":
        dtype = np.int32
    else:
        dtype = np.int16 # (can do since enc.max_token_value == 32000 is < 2**16)
    arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len, max_length))
    total_batches = 10

    idx = 0
    for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
        # Batch together samples for faster write
        batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
        # Write into mmap
        arr_batch = np.stack(batch[split])
        arr[idx : idx + arr_batch.shape[0], :] = arr_batch
        idx += arr_batch.shape[0]
    arr.flush()


def main(args):
    global cache_dir
    cache_dir = os.path.join(cache_dir, args.task)
    dset = load_dataset("json", data_files=f"/home/xwei/transformers/RULER/scripts/data/{args.task}/{args.split}.jsonl", split="train") 
    shutil.copy(f"/home/xwei/transformers/RULER/scripts/data/{args.task}/{args.split}.jsonl", os.path.join(cache_dir, "test.jsonl"))
    print(dset)
    new_dset = tokenize_answer_only(dset, args.tokenizer, args.task, args.num_proc, args.max_length, args.split) # get the train split here
    save_to_npmemmap("input_ids", args.tokenizer, new_dset, f"{args.tokenizer}-{args.split}-{args.max_length}-inputs", args.max_length)
    save_to_npmemmap("labels", args.tokenizer, new_dset, f"{args.tokenizer}-{args.split}-{args.max_length}-labels", args.max_length)


def main_train(args):
    global cache_dir
    cache_dir = os.path.join(cache_dir, "train_niah")
    dsets = []
    for task in ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3", "niah_multiquery", "niah_multivalue"]:
        dset = load_dataset("json", data_files=f"/home/xwei/transformers/RULER/scripts/data/{task}/{args.split}.jsonl", split="train")
        # shutil.copy(f"/home/xwei/transformers/RULER/scripts/data/{args.task}/{args.split}.jsonl", os.path.join(cache_dir, "test.jsonl"))
        # print(dset)
        new_dset = tokenize_answer_only(dset, args.tokenizer, args.task, args.num_proc, args.max_length, args.split) # get the train split here
        dsets.append(new_dset)
    new_dset = concatenate_datasets(dsets).shuffle(seed=42)
    print(new_dset)
    save_to_npmemmap("input_ids", args.tokenizer, new_dset, f"{args.tokenizer}-{args.split}-{args.max_length}-inputs", args.max_length)
    save_to_npmemmap("labels", args.tokenizer, new_dset, f"{args.tokenizer}-{args.split}-{args.max_length}-labels", args.max_length)


if __name__ == "__main__":
    parser = ArgumentParser(description="Convert dataset into MDS format, optionally concatenating and tokenizing")
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--num_proc", type=int, required=True, default=None)
    parser.add_argument("--max_length", type=int, required=True, default=4096)
    parser.add_argument("--split", type=str, required=True)
    # main(parser.parse_args()) # for validation
    main_train(parser.parse_args())
