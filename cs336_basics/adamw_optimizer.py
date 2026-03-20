import math
import torch
from torch import Tensor
from typing import Optional, Callable, Tuple


class AdamW(torch.optim.Optimizer):
    def __init__(
        self, params, lr: float, weight_decay: float, betas: Tuple[float, float],
        eps: float
    ):
        defaults = {
            "alpha": lr,
            "beta1": betas[0],
            "beta2": betas[1],
            "eps": eps,
            "lamda": weight_decay
        }
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        with torch.no_grad():
            for group in self.param_groups:
                alpha = group["alpha"]
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                eps = group["eps"]
                lamda = group["lamda"]
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    grad = param.grad
                    m = state.get("m", 0)
                    v = state.get("v", 0)
                    t = state.get("t", 1)

                    m = beta1 * m + (1 - beta1) * grad
                    v = beta2 * v + (1 - beta2) * grad * grad
                    alpha_t = alpha * math.sqrt(1 - math.pow(beta2, t)) / (1 - math.pow(beta1, t))

                    param -= alpha_t * m / (v.sqrt() + eps)
                    param -= alpha * lamda * param

                    state["m"] = m
                    state["v"] = v
                    state["t"] = t + 1
        
        return loss