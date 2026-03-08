from einops import einsum, rearrange
import torch
from torch import Tensor
from torch.nn import Module

from typing import Optional

class RoPE(Module):
    def __init__(
        self, theta: float, d_k: int, max_seq_len: int,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # Precompute the frequency tensor
        theta_power: Tensor = torch.arange(start=0, end=d_k, step=2, device=device, dtype=torch.float) / d_k
        inv_freq: Tensor = 1 / theta ** theta_power
        position_i: Tensor = torch.arange(start=0, end=max_seq_len, device=device, dtype=torch.float)

        freqs: Tensor = einsum(position_i, inv_freq, "max_seq_len, half_dk -> max_seq_len half_dk")

        freq_cis: Tensor = torch.polar(torch.ones_like(freqs), freqs)
        # creates cos(freq) + i * sin(freq)
        self.register_buffer("cis", freq_cis, persistent=False)
        # setting persistent False makes this buffer exist independently from module RoPE


    def forward(self, x: Tensor, token_positions: Optional[Tensor] = None) -> Tensor:
        x = rearrange(x, "... seq_len (d_k_split split) -> ... seq_len d_k_split split", split=2)
        x_complex = torch.view_as_complex(x)

        if token_positions is None:
            seq_len = x.shape[-3]
            token_positions = torch.arange(end = seq_len, device=x.device)

        theta_cis = self.cis[token_positions]

        q_complex = x_complex * theta_cis

        q = torch.view_as_real(q_complex)
        return rearrange(q, "... seq_len d_k_split split -> ... seq_len (d_k_split split)", split=2)
