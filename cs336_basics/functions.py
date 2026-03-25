from jaxtyping import Float, Bool, Int
import math
import torch
from torch import Tensor
from typing import Optional, Iterable

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    x = x.transpose(dim, -1)
    C: Tensor = torch.max(x, dim=-1, keepdim=True).values
    exp_x: Tensor = (x - C).exp()
    return (exp_x / exp_x.sum(dim=-1, keepdim=True)).transpose(-1, dim)

def softmax_with_temp(x: Tensor, temp: float, dim: int = -1) -> Tensor:
    x = x / temp
    return softmax(x, dim)

def top_p_sampling(q: Tensor, p: float) -> Tensor:
    # sort in descending order
    sorted_q, sorted_indices = torch.sort(q, descending=True, dim=-1)
    # compute cumulative p
    cumulative_p = torch.cumsum(sorted_q, dim=-1)
    # create mask
    sorted_indices_to_remove = cumulative_p > p
    # shift mask to the right by 1 to include the crossing token
    # Example, cum top 5 is 0.45, cum top 6 is 0.55, p is 0.5
    # Want to include top 6, but since cum 6 > p, 6 is set to remove
    sorted_indices_to_remove = torch.roll(sorted_indices_to_remove, shifts=1, dims=-1)
    # It is possible that top 1 is straight up 0.99, exceeding p
    sorted_indices_to_remove[..., 0] = False

    # set all q to ignore to 0
    sorted_q.masked_fill_(sorted_indices_to_remove, 0.0)
    # normalize to sum 1
    sorted_q = sorted_q / sorted_q.sum(dim=-1, keepdim=True)
    # randomly sample a index according to probability values
    # the index here is not q's index, it's the sorted array's index
    sampled_sorted_index = torch.multinomial(sorted_q, 1)
    # get the actual index
    return torch.gather(sorted_indices, dim=-1, index=sampled_sorted_index)

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

def learning_rate_cosine_schedule(
    t: int, alpha_max: float, alpha_min: float, T_warm: int, T_c_anneal: int
):
    if t < T_warm:
        return t / T_warm * alpha_max
    if t < T_c_anneal:
        return alpha_min + 0.5 * (1 + math.cos((t - T_warm) / (T_c_anneal - T_warm) * math.pi)) * (alpha_max - alpha_min)
    return alpha_min

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter], max_norm: float
):
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    
    device = grads[0].device
    norms: Tensor = torch.stack([torch.norm(grad.detach(), 2.0).to(device) for grad in grads])
    norm: Tensor = torch.norm(norms, 2.0)

    if norm < max_norm:
        return
    eps: float = 1e-6
    clip_coef = max_norm / (norm + eps)
    for grad in grads:
        grad.detach().mul_(clip_coef)

    
