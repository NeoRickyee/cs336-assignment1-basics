import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.nn import init

from typing import Optional

class Embedding(Module):
    def __init__(
        self, num_embeddings: int, embedding_dim: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        W = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        init.trunc_normal_(W, mean=0, std=1, a=-3, b=3)
        self.weight = nn.Parameter(W)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]