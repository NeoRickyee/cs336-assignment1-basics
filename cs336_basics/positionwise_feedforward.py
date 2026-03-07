import torch
from torch import Tensor
from torch.nn import Module

from typing import Optional

from cs336_basics.linear import Linear

class SiLU(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.sigmoid(x)


class SwiGLU(Module):
    def __init__(
        self, d_model: int, d_ff: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        if d_ff is None:
            d_ff = round(d_model * 8 / (3 * 64)) * 64
        if d_ff == 0:
            d_ff = 64

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.silu = SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.w2(self.silu(self.w1(x)) * self.w3(x))
