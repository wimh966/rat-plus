# Copyright (c) 2026 Xiuying Wei, EPFL CLAIRE lab
# All rights reserved.
# Licensed under the MIT License.


from torch._higher_order_ops.associative_scan import associative_scan
import torch
import torch.nn.functional as F


# apply parallel scan in the last second dimension
class AScan(torch.autograd.Function): # (b c l p)
    @staticmethod
    def scan_op(i, j):
        g_i, x_i = i
        g_j, x_j = j
        return g_j * g_i, g_j * x_i + x_j

    @torch.compile
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
    def forward(ctx, g, x):
        _, x_scan = associative_scan(AScan.scan_op, (g, x), dim=-2)
        ctx.save_for_backward(g, x_scan)
        return x_scan

    @torch.compile
    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda") # (b c l p)
    def backward(ctx, grad):
        g, x_scan = ctx.saved_tensors
        g = F.pad(g, (0, 0, -1, 1))
        _, x_grad = associative_scan(AScan.scan_op, (g, grad), dim=-2, reverse=True)
        g_grad = torch.zeros_like(x_scan)
        g_grad[..., 1:, :].add_(x_scan[..., :-1, :] * x_grad[..., 1:, :])
        return g_grad, x_grad


ascan = AScan.apply
