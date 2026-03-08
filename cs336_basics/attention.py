from einops import rearrange
from jaxtyping import Float
import torch
from torch import Tensor
from torch.nn import Module

from typing import Optional
from cs336_basics import functions, linear, rope

class MultiHeadSelfAttention(Module):
    def __init__(
        self, d_model: int, num_heads: int,
        apply_rope: bool = False,
        theta: Optional[Float] = None,
        max_seq_len: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} not divisible by num_heads {num_heads} in MultiHeadSelfAttention!"
            )
        if apply_rope and (max_seq_len is None or theta is None):
            raise ValueError(
                f"apply_rope but max_seq_len or theta is not specified in MultiHeadSelfAttention!"
            )
        self.rope = None
        if apply_rope:
            self.rope = rope.RoPE(theta, d_model / num_heads, max_seq_len, device)

        self.d_model = d_model
        self.num_heads = num_heads

        self.wq = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.wk = linear.Linear(d_model, d_model, device=device, dtype=dtype)
        self.wv = linear.Linear(d_model, d_model, device=device, dtype=dtype)

        self.wo = linear.Linear(d_model, d_model, device=device, dtype=dtype)
    
    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        seq_len: int = x.shape[-2]
        if self.d_model != x.shape[-1]:
            raise ValueError(
                f"input Tensor x does not have the same d_model dimension as initialization, {x.shape[-1]} vs {self.d_model} in MultiHeadSelfAttention!"
            )

        XQ: Tensor[Float, "... n d_model"] = self.wq(x)
        XK: Tensor[Float, "... n d_model"] = self.wk(x)
        XQ = rearrange(XQ, "... n (num_heads d_per_head) -> ... num_heads n d_per_head", num_heads = self.num_heads)
        XK = rearrange(XK, "... n (num_heads d_per_head) -> ... num_heads n d_per_head", num_heads = self.num_heads)
        if self.rope is not None:
            XQ = self.rope(XQ, token_positions)
            XK = self.rope(XK, token_positions)
        
        XV: Tensor[Float, "... n d_model"] = self.wv(x)
        XV = rearrange(XV, "... n (num_heads d_per_head) -> ... num_heads n d_per_head", num_heads = self.num_heads)

        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))

        alpha: Tensor[Float, "... num_heads n d_per_head"] = functions.scaled_dot_product_attention(XQ, XK, XV, mask)
        alpha = rearrange(alpha, "... num_heads n d_per_head -> ... n (num_heads d_per_head)", num_heads = self.num_heads)
        return self.wo(alpha)

