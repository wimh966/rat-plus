# Copyright (c) 2026 Xiuying Wei, EPFL CLAIRE lab
# All rights reserved.
# Licensed under the MIT License.


import torch

class RATPlusSingleLayerCache:
    # initial + local + dilated
    def __init__(self,
                 layer_id,
                 max_bs,
                 max_seq_len,
                 initial_size,
                 local_size,
                 dilation_size,
                 num_head,
                 d_head,
                 d_model,
                 dtype=torch.bfloat16,
                 device="cuda"):
        self.layer_id = layer_id
        self.max_bs = max_bs
        self.initial_size = initial_size
        self.local_size = local_size
        self.bound = self.initial_size + self.local_size
        self.dilation_size = dilation_size
        assert self.local_size != 0 or self.dilation_size != -1 or self.initial_size != 0 # either local attention or dilated attention
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_head
        self.bs_start = 0
        self.seq_start = 0 # the position to put the new token
        self.seq_end = 0 # the final token ) unclosed pos,  also to indicate the position where we temporarily put the current token
        sz = self.bound + 1 if self.dilation_size == -1 else self.bound + 1 + (self.max_seq_len // self.dilation_size)
        self.kcache = torch.empty(self.max_bs, self.num_head, sz, self.d_head, dtype=dtype, device=device)
        self.vcache = torch.empty(self.max_bs, self.num_head, sz, self.d_head, dtype=dtype, device=device)
        self.lastkcache = torch.zeros(self.max_bs, self.num_head, 1, self.d_head, dtype=dtype, device=device)
        self.lastvcache = torch.zeros(self.max_bs, self.num_head, 1, self.d_head, dtype=dtype, device=device)
        self.d_st = self.initial_size + ((self.dilation_size - 1 - self.initial_size) % self.dilation_size)

    def reset_cache(self, ):
        self.bs_start = 0
        self.seq_start = 0
        self.seq_end = 0
        self.lastkcache.zero_()
        self.lastvcache.zero_()

    def update_kv_prefill_dilation(self, d_ed, kcache, vcache, gated_k, gated_x):
        chunk_gated_k, chunk_gated_x = gated_k[:, :, self.d_st: d_ed: self.dilation_size], gated_x[:, :, self.d_st: d_ed: self.dilation_size]
        num_dilated = chunk_gated_k.shape[-2]
        kcache[:, :, self.bound: self.bound + num_dilated].copy_(chunk_gated_k)
        vcache[:, :, self.bound: self.bound + num_dilated].copy_(chunk_gated_x)
        self.seq_end = self.bound + num_dilated

    def update_kv_prefill(self, seq_pos, bs, gated_k, gated_x):
        kcache, vcache = self.kcache[self.bs_start: self.bs_start + bs], self.vcache[self.bs_start: self.bs_start + bs]
        if self.local_size == 0:
            ed = seq_pos + 1
            if seq_pos < self.bound:
                kcache[:, :, :ed].copy_(gated_k[:, :, :ed])
                vcache[:, :, :ed].copy_(gated_x[:, :, :ed])
                self.seq_start = self.seq_end = ed
            else:
                kcache[:, :, :self.bound].copy_(gated_k[:, :, :self.bound])
                vcache[:, :, :self.bound].copy_(gated_x[:, :, :self.bound])
                if self.dilation_size != -1:
                    self.update_kv_prefill_dilation(ed, kcache, vcache, gated_k, gated_x)
                    self.seq_start = self.seq_end
                else:
                    self.seq_start = self.seq_end = self.bound
        else:
            ed = seq_pos + 1
            if seq_pos < self.bound - 1:
                kcache[:, :, :ed].copy_(gated_k[:, :, :ed])
                vcache[:, :, :ed].copy_(gated_x[:, :, :ed])
                self.seq_end = self.seq_start = ed
            else:
                kcache[:, :, :self.initial_size].copy_(gated_k[:, :, :self.initial_size])
                vcache[:, :, :self.initial_size].copy_(gated_x[:, :, :self.initial_size])
                kcache[:, :, self.initial_size: self.bound].copy_(gated_k[:, :, ed - self.local_size: ed])
                vcache[:, :, self.initial_size: self.bound].copy_(gated_x[:, :, ed - self.local_size: ed])
                # dilation
                self.seq_start = self.initial_size
                self.seq_end = self.bound
                if self.dilation_size != -1:
                    self.update_kv_prefill_dilation(ed - self.local_size, kcache, vcache, gated_k, gated_x)

    def update_kv_step(self, seq_pos, bs, gated_k, gated_x):
        kcache, vcache = self.kcache[self.bs_start: self.bs_start + bs], self.vcache[self.bs_start: self.bs_start + bs]
        if self.local_size == 0:
            # we store either initial or dilated tokens
            if seq_pos < self.bound or (self.dilation_size != -1 and (seq_pos + 1) % self.dilation_size == 0):
                kcache[:, :, self.seq_start: self.seq_start + 1].copy_(gated_k)
                vcache[:, :, self.seq_start: self.seq_start + 1].copy_(gated_x)
                self.seq_start += 1
                self.seq_end += 1
        else:
            if seq_pos >= self.bound:
                remove_pos = (seq_pos - self.local_size)
                if self.dilation_size != -1 and (remove_pos + 1) % self.dilation_size == 0:
                    kcache[:, :, self.seq_end].copy_(kcache[:, :, self.seq_start])
                    vcache[:, :, self.seq_end].copy_(vcache[:, :, self.seq_start])
                    self.seq_end += 1
            else:
                self.seq_end += 1
            kcache[:, :, self.seq_start: self.seq_start + 1].copy_(gated_k)
            vcache[:, :, self.seq_start: self.seq_start + 1].copy_(gated_x)
            self.seq_start += 1
            if self.seq_start >= self.bound:
                self.seq_start -= self.local_size

    def get_kv_step(self, seq_pos, bs, gated_k, gated_x):
        kcache, vcache = self.kcache[self.bs_start: self.bs_start + bs], self.vcache[self.bs_start: self.bs_start + bs]
        kcache[:, :, self.seq_end: self.seq_end + 1].copy_(gated_k)
        vcache[:, :, self.seq_end: self.seq_end + 1].copy_(gated_x)
        return kcache[:, :, :self.seq_end + 1], vcache[:, :, :self.seq_end + 1]

    def __repr__(self):
        return f"dilation_size={self.dilation_size}, initial_size={self.initial_size}, local_size={self.local_size}"


class RATPlusFullSingleLayerCache:
    # store the full one, take out the corresponding one
    def __init__(self,
                 layer_id,
                 max_bs,
                 max_seq_len,
                 initial_size,
                 local_size,
                 dilation_size,
                 num_head,
                 d_head,
                 d_model,
                 dtype=torch.bfloat16,
                 device="cuda"):
        self.layer_id = layer_id
        self.max_bs = max_bs
        self.initial_size = initial_size
        self.local_size = local_size
        self.bound = self.initial_size + self.local_size
        self.dilation_size = dilation_size
        assert self.local_size != 0 or self.dilation_size != -1 or self.initial_size != 0  # either local attention or dilated attention
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_head = num_head
        self.d_head = d_head
        self.bs_start = 0
        self.seq_start = 0 # the pos to put the new token
        self.seq_end = 0 # the final token ) unclosed pos
        self.kcache = torch.empty(self.max_bs, self.num_head, self.max_seq_len, self.d_head, dtype=dtype, device=device)
        self.vcache = torch.empty(self.max_bs, self.num_head, self.max_seq_len, self.d_head, dtype=dtype, device=device)
        self.lastkcache = torch.zeros(self.max_bs, self.num_head, 1, self.d_head, dtype=dtype, device=device)
        self.lastvcache = torch.zeros(self.max_bs, self.num_head, 1, self.d_head, dtype=dtype, device=device)
        self.d_st = self.initial_size + ((self.dilation_size - 1 - self.initial_size) % self.dilation_size)

    def update_kv_prefill(self, seq_pos, bs, gated_k, gated_x):
        self.kcache[self.bs_start: self.bs_start + bs, :, :seq_pos + 1].copy_(gated_k[:, :, :seq_pos + 1])
        self.vcache[self.bs_start: self.bs_start + bs, :, :seq_pos + 1].copy_(gated_x[:, :, :seq_pos + 1])
        self.seq_start = self.seq_end = seq_pos + 1

    def update_kv_step(self, seq_pos, bs, gated_k, gated_x):
        self.kcache[self.bs_start: self.bs_start + bs, :, self.seq_start: self.seq_start + 1].copy_(gated_k)
        self.vcache[self.bs_start: self.bs_start + bs, :, self.seq_start: self.seq_start + 1].copy_(gated_x)
        self.seq_start += 1
        self.seq_end += 1

    def get_kv_step(self, seq_pos, bs, gated_k, gated_x):
        assert seq_pos == self.seq_start and seq_pos == self.seq_end
        if seq_pos <= self.bound:
            self.kcache[self.bs_start: self.bs_start + bs, :, seq_pos: seq_pos + 1].copy_(gated_k)
            self.vcache[self.bs_start: self.bs_start + bs, :, seq_pos: seq_pos + 1].copy_(gated_x)
            return self.kcache[self.bs_start: self.bs_start + bs, :, :seq_pos + 1], self.vcache[self.bs_start: self.bs_start + bs, :, :seq_pos + 1]
        kcache, vcache = self.kcache[self.bs_start: self.bs_start + bs, :, :self.seq_end], self.vcache[self.bs_start: self.bs_start + bs, :, :self.seq_end]
        prefill_kcache, prefill_vcache = kcache[:, :, :self.initial_size], vcache[:, :, :self.initial_size]
        local_kcache, local_vcache = kcache[:, :, self.seq_end - self.local_size: ], vcache[:, :, self.seq_end - self.local_size: ]
        if self.dilation_size != -1:
            d_ed = self.seq_end - self.local_size # (] area
            chunk_kcache, chunk_vcache = kcache[:, :, self.d_st: d_ed: self.dilation_size], vcache[:, :, self.d_st: d_ed: self.dilation_size]
            return torch.cat([prefill_kcache, local_kcache, chunk_kcache, gated_k], dim=-2), torch.cat([prefill_vcache, local_vcache, chunk_vcache, gated_x], dim=-2)
        return torch.cat([prefill_kcache, local_kcache, gated_k], dim=-2), torch.cat([prefill_vcache, local_vcache, gated_x], dim=-2)

    def reset_cache(self, ):
        self.bs_start = 0
        self.seq_start = 0
        self.seq_end = 0
        self.lastkcache.zero_()
        self.lastvcache.zero_()

    def __repr__(self):
        return f"dilation_size={self.dilation_size}, initial_size={self.initial_size}, local_size={self.local_size}"
