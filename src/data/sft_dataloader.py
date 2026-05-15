import torch
import numpy as np
import os
import torch.distributed as dist
from ..utils.registry import data_registry
bytes2datatype = {1: np.int8, 2: np.int16, 4: np.int32}
# because there are ignore labels in labels, which is usually -100


"""
sft trainer should be fixed length, and separate inputs and labels)
and also no updates on labels now.
"""
@data_registry.register("sft_random")
class SFTRandomDataloader:

    def __init__(
        self,
        tokenized_input_path, 
        tokenized_label_path,
        num_bytes,  # thes data type that you used to store tokens
        batch_size,
        global_batch_size,
        seq_len,
        limit_tokens=-1,
        seed=0,
        **kwargs,
    ):
        self.tokenized_input_path = tokenized_input_path
        self.tokenized_label_path = tokenized_label_path
        self.num_bytes = num_bytes
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.gpu_id = kwargs["dp_rank"] if "dp_rank" in kwargs else int(os.getenv("RANK", -1))
        ngpus = kwargs["dp_degree"] if "dp_degree" in kwargs else dist.get_world_size()
        assert self.global_batch_size % (ngpus * self.batch_size) == 0, "global batch size is for all gpus"
        arr = np.memmap(self.tokenized_input_path, dtype=bytes2datatype.get(self.num_bytes), mode="r")
        limit_tokens = len(arr) if limit_tokens == -1 else limit_tokens
        limit_tokens = int(limit_tokens / global_batch_size / self.seq_len) * self.seq_len * global_batch_size
        self.nsamples = limit_tokens // self.seq_len  # total samples
        nsamples_per_gpu = self.nsamples // ngpus
        self.start = max(self.gpu_id, 0) * nsamples_per_gpu
        self.end = self.start + nsamples_per_gpu  # [start, end) to determine the number of samples on each gpu, 
        self.gen = torch.Generator().manual_seed(self.seed + max(self.gpu_id, 0))

    def __len__(self):
        return (self.end - self.start) // self.batch_size

    def get_batch(self):
        arr_input = np.memmap(
            self.tokenized_input_path,
            dtype=bytes2datatype.get(self.num_bytes),
            mode="r",
        )
        arr_label = np.memmap(
            self.tokenized_label_path,
            dtype=bytes2datatype.get(self.num_bytes),
            mode="r",
        )
        ids = torch.randint(len(arr_input) // self.seq_len, (self.batch_size,), generator=self.gen)
        x = torch.stack([torch.from_numpy((arr_input[id * self.seq_len: (id + 1) * self.seq_len]).astype(np.int64)) for id in ids])
        y = torch.stack([torch.from_numpy((arr_label[id * self.seq_len: (id + 1) * self.seq_len]).astype(np.int64)) for id in ids])
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y

    def get_fixlen_iter(self,):
        for i in range(self.start, self.end, self.batch_size):
            self.last_iter = i
            yield self.get_batch()

    def __iter__(self,):
        return self.get_fixlen_iter()


# sft for test
@data_registry.register("sft_ordered")
class SFTOrderedDataloader:

    def __init__(
        self,
        tokenized_input_path, 
        tokenized_label_path,
        num_bytes,  # thes data type that you used to store tokens
        batch_size,
        global_batch_size,
        seq_len,
        limit_tokens=-1,
        **kwargs,
    ):
        self.tokenized_input_path = tokenized_input_path
        self.tokenized_label_path = tokenized_label_path
        self.num_bytes = num_bytes
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.gpu_id = kwargs["dp_rank"] if "dp_rank" in kwargs else int(os.getenv("RANK", -1))
        ngpus = kwargs["dp_degree"] if "dp_degree" in kwargs else dist.get_world_size()
        assert self.global_batch_size % (ngpus * self.batch_size) == 0, "global batch size is for all gpus"
        arr = np.memmap(self.tokenized_input_path, dtype=bytes2datatype.get(self.num_bytes), mode="r")
        limit_tokens = len(arr) if limit_tokens == -1 else limit_tokens
        limit_tokens = int(limit_tokens / global_batch_size / self.seq_len) * self.seq_len * global_batch_size
        self.nsamples = limit_tokens // self.seq_len  # total samples
        nsamples_per_gpu = self.nsamples // ngpus
        self.start = max(self.gpu_id, 0) * nsamples_per_gpu
        self.end = self.start + nsamples_per_gpu

    def __len__(self):
        return (self.end - self.start) // self.batch_size

    def get_batch(self, offset_row):
        arr_input = np.memmap(
            self.tokenized_input_path,
            dtype=bytes2datatype.get(self.num_bytes),
            mode="r",
            offset=offset_row * self.seq_len * self.num_bytes,
            shape=(self.batch_size, self.seq_len)
        )
        arr_label = np.memmap(
            self.tokenized_label_path,
            dtype=bytes2datatype.get(self.num_bytes),
            mode="r",
            offset=offset_row * self.seq_len * self.num_bytes,
            shape=(self.batch_size, self.seq_len)
        )
        x = torch.from_numpy(arr_input.astype(np.int64))
        y = torch.from_numpy(arr_label.astype(np.int64))
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y  # return input and labels in torch format

    def get_fixlen_iter(self,):
        for i in range(self.start, self.end, self.batch_size):
            self.last_iter = i
            yield self.get_batch(i)

    def __iter__(self,):
        return self.get_fixlen_iter()
