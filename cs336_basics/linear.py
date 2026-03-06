from einops import einsum
import math
import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import init

from typing import Optional

class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        W = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std: float = math.sqrt(2.0 / (out_features + in_features))
        limit: float = 3*std
        init.trunc_normal_(W, mean=0, std=std, a=-limit, b=limit)
        self.weight = nn.Parameter(W)

    def forward(self, x: Tensor) -> Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
