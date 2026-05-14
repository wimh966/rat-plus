# the tokenized_data has already concat all the samples together
# we use np.memmap to deal with lm tasks because it suits large files well
import torch
import numpy as np
import os
import torch.distributed as dist
from ..utils.registry import data_registry


bytes2datatype = {1: np.uint8, 2: np.uint16, 4: np.uint32}


# TODO: support drop_last=False
@data_registry.register("lm_ordered")
class LMOrderedDataloader:  # ensure that we only iterate it just once. Otherwise, I think for LM tasks, it's better to use LMRandomDataloader.
    # based on samples (* seq_len = tokens)
    # ddp used as many gpu as we have here
    # TODO: to ensure all the things are same and no repetition here, current lmordered requires user to always to same number of gpu during training.
    def __init__(
        self,
        tokenized_file_path,  # always passed tokenized_file_path
        num_bytes,  # the data type that you used to store tokens
        batch_size,
        global_batch_size,
        seq_len,  # seq_len of the LM tasks
        limit_tokens=-1,
        drop_last=True,
        **kwargs,
    ):
        self.tokenized_file_path = tokenized_file_path
        self.num_bytes = num_bytes
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.drop_last = drop_last
        # split samples here
        assert self.drop_last is True, "drop_last=False has not been implemented"
        self.gpu_id = kwargs["dp_rank"] if "dp_rank" in kwargs else int(os.getenv("RANK", -1))
        ngpus = kwargs["dp_degree"] if "dp_degree" in kwargs else dist.get_world_size()
        assert self.global_batch_size % (ngpus * self.batch_size) == 0, "global batch size is for all gpus"
        arr = np.memmap(self.tokenized_file_path, dtype=bytes2datatype.get(self.num_bytes), mode="r")
        limit_tokens = len(arr) - 1 if limit_tokens == -1 else limit_tokens
        limit_tokens = int(limit_tokens / global_batch_size / self.seq_len) * self.seq_len * global_batch_size
        assert limit_tokens <= len(arr) - 1, "It's better to use LMRandomDataloader if you want to iterate datasets once more"
        self.nsamples = limit_tokens // self.seq_len  # total samples
        nsamples_per_gpu = self.nsamples // ngpus
        self.start = max(self.gpu_id, 0) * nsamples_per_gpu
        self.end = self.start + nsamples_per_gpu  # [start, end) # to split the dataset
        self.ignore_input_index = kwargs.get("ignore_input_index", -1)

    def __len__(self):
        return (self.end - self.start) // self.batch_size

    def get_batch(self, offset_row):
        # get batch starting at offset_row
        arr = np.memmap(
            self.tokenized_file_path,
            dtype=bytes2datatype.get(self.num_bytes),
            mode="r",
            offset=offset_row * self.seq_len * self.num_bytes,
            shape=(self.batch_size * self.seq_len + 1),
        )
        x = torch.from_numpy(arr[:-1].astype(np.int64)).reshape(self.batch_size, self.seq_len)
        y = torch.from_numpy(arr[1:].astype(np.int64)).reshape(self.batch_size, self.seq_len)
        y = torch.where(x == self.ignore_input_index, -100, y)
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y  # return input and labels in torch format

    def get_fixlen_iter(self,):
        for i in range(self.start, self.end, self.batch_size):
            self.last_iter = i
            yield self.get_batch(i)

    def __iter__(self,):
        return self.get_fixlen_iter()


@data_registry.register("lm_random")
class LMRandomDataloader:
    # Random sampling all the data, limit_tokens restrict the number of sampling
    def __init__(
        self,
        tokenized_file_path,  # always passed tokenized_file_path
        num_bytes,  # the data type that you used to store tokens
        batch_size,
        global_batch_size,
        seq_len,  # seq_len of the LM tasks
        limit_tokens=-1,
        drop_last=True,
        seed=0,
        **kwargs,
    ):
        self.tokenized_file_path = tokenized_file_path
        self.num_bytes = num_bytes
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        self.seq_len = seq_len
        self.seed = seed
        self.drop_last = drop_last
        assert self.drop_last is True, "drop_last=False has not been implemented"
        self.gpu_id = kwargs["dp_rank"] if "dp_rank" in kwargs else int(os.getenv("RANK", -1))
        ngpus = kwargs["dp_degree"] if "dp_degree" in kwargs else dist.get_world_size()
        assert self.global_batch_size % (ngpus * self.batch_size) == 0, "global batch size is for all gpus"

        arr = np.memmap(self.tokenized_file_path, dtype=bytes2datatype.get(self.num_bytes), mode="r")
        limit_tokens = len(arr) - 1 if limit_tokens == -1 else limit_tokens
        limit_tokens = int(limit_tokens / global_batch_size / self.seq_len) * self.seq_len * global_batch_size
        self.nsamples = limit_tokens // self.seq_len  # total samples
        nsamples_per_gpu = self.nsamples // ngpus
        self.start = max(self.gpu_id, 0) * nsamples_per_gpu
        self.end = self.start + nsamples_per_gpu  # [start, end) # this is not to split datasets but determine the number of samples for each gpu
        self.gen = torch.Generator().manual_seed(self.seed + max(self.gpu_id, 0))
        self.ignore_input_index = kwargs.get("ignore_input_index", -1)

    def __len__(self):
        return (self.end - self.start) // self.batch_size

    def get_batch(self):
        arr = np.memmap(
            self.tokenized_file_path,
            dtype=bytes2datatype.get(self.num_bytes),
            mode="r",
        )
        ids = torch.randint(len(arr) - self.seq_len - 1, (self.batch_size,), generator=self.gen)
        x = torch.stack([torch.from_numpy((arr[id: id + self.seq_len]).astype(np.int64)) for id in ids])
        y = torch.stack([torch.from_numpy((arr[id + 1 : id + 1 + self.seq_len]).astype(np.int64)) for id in ids])
        y = torch.where(x == self.ignore_input_index, -100, y)
        x, y = x.pin_memory().to("cuda", non_blocking=True), y.pin_memory().to("cuda", non_blocking=True)
        return x, y

    def get_fixlen_iter(self,):
        for i in range(self.start, self.end, self.batch_size):
            self.last_iter = i
            yield self.get_batch()

    def __iter__(self,):
        return self.get_fixlen_iter()
