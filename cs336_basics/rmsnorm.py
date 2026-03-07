from einops import einsum
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module

from typing import Optional

class RMSNorm(Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor):
        input_dtype = x.dtype
        rms_x = self.compute_rms_x(x)
        result = einsum(x / rms_x, self.weight, "... d_model, d_model -> ... d_model")
        return result.to(dtype=input_dtype)

    def compute_rms_x(self, x: Tensor):
        return x.float().square().mean(-1, keepdim=True).add(self.eps).sqrt()