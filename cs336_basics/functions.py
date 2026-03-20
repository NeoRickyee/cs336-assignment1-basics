from jaxtyping import Float, Bool, Int
import math
import torch
from torch import Tensor
from typing import Optional

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x = x.transpose(dim, -1)
    C: Tensor = torch.max(x, dim=-1, keepdim=True).values
    exp_x: Tensor = (x - C).exp()
    return (exp_x / exp_x.sum(dim=-1, keepdim=True)).transpose(-1, dim)

def scaled_dot_product_attention(
    Q: Float[Tensor, "batch_size ... seq_len d_k"],
    K: Float[Tensor, "batch_size ... seq_len d_k"],
    V: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Optional[Bool[Tensor, "seq_len seq_len"]] = None
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_k: int = Q.shape[-1]
    QK: Float[Tensor, "batch_size ... seq_len seq_len"] = Q.matmul(K.transpose(-1, -2))
    if mask is not None:
        QK = QK.masked_fill(mask == 0, float('-inf'))
    return softmax(QK / math.sqrt(d_k)).matmul(V)

def cross_entropy_loss(
    inputs: Float[Tensor, "batch_size vocab_size"],
    targets: Int[Tensor, "batch_size"]
):
    # flatten inputs and targets
    inputs = inputs.view(-1, inputs.size(-1))
    max_val = inputs.max(dim=-1, keepdim=True).values
    inputs = inputs - max_val
    targets = targets.view(-1, 1)
    log_sum_exp: Tensor = inputs.exp().sum(dim=-1).log()
    true_logits: Float[Tensor, "... 1"] = inputs.gather(dim=-1, index=targets)

    return (log_sum_exp - true_logits.view(-1)).mean()