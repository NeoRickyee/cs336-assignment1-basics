
import torch
from torch import Tensor
from torch.nn import Module

from typing import Optional

from cs336_basics import rmsnorm, attention, positionwise_feedforward

class Transformer(Module):
    def __init__(
        self, d_model: int, num_attention_heads: int, 
        rope_theta: float, max_seq_len: int,
        d_ff: Optional[int] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()

        self.pre_attention_norm = rmsnorm.RMSNorm(d_model, device=device, dtype=dtype)
        self.attention = attention.MultiHeadSelfAttention(
            d_model, num_heads=num_attention_heads, apply_rope=True, theta=rope_theta,
            max_seq_len=max_seq_len, device=device, dtype=dtype
        )
        self.pre_ff_norm = rmsnorm.RMSNorm(d_model, device=device, dtype=dtype)
        self.ff = positionwise_feedforward.SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x_post_attention: Tensor = x + self.attention(self.pre_attention_norm(x), token_positions)
        return x_post_attention + self.ff(self.pre_ff_norm(x_post_attention))
        
        # post norm transformer
        # x_post_attention: Tensor = self.pre_attention_norm(x + self.attention(x, token_positions))
        # return self.pre_ff_norm(x_post_attention + self.ff(x_post_attention))
    
        # ???
        # x_post_attention: Tensor = x + self.pre_attention_norm(self.attention(x, token_positions))
        # return x_post_attention + self.pre_ff_norm(self.ff(x_post_attention))

        # no norm transformer
        # x_post_attention: Tensor = x + self.attention(x, token_positions)
        # return x_post_attention + self.ff(x_post_attention)