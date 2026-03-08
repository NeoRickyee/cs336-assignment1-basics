from jaxtyping import Float
import torch
from torch import Tensor, nn
from torch.nn import Module

from typing import Optional

from cs336_basics import embedding, transformer, rmsnorm, linear, functions


class TransformerLM(Module):
    def __init__(
        self, vocab_size: int, context_length: int, num_layers: int,
        d_model: int, num_attention_heads: int, d_ff: int,
        rope_theta: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        super().__init__()
        self.embedding = embedding.Embedding(vocab_size, d_model, device, dtype)
        self.transformers = nn.ModuleList([
            transformer.Transformer(
                d_model, num_attention_heads, d_ff, rope_theta, context_length, device, dtype
            ) for _ in range(num_layers)
        ])
        self.norm = rmsnorm.RMSNorm(d_model, device=device, dtype=dtype)
        self.output_embedding = linear.Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # x size (batch_size, seq_length)
        embedded_input: Tensor[Float, "batch_size seq_length d_model"] = self.embedding(x)
        
        interm_x: Tensor = embedded_input
        for tf in self.transformers:
            interm_x = tf(interm_x)
        
        interm_x = self.norm(interm_x)
        output: Tensor[Float, "batch_size seq_length vocab_size"] = self.output_embedding(interm_x)
        return output
