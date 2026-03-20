import math
import torch
from typing import Optional, Callable


class SGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, learning_rate):
        if learning_rate <= 0:
            raise ValueError(f"invalid learning_rate {learning_rate} in SGDOptimizer")
        defaults = {"lr" : learning_rate}
        super().__init__(params=params, defaults=defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            for param in group['params']:
                if param.grad is None:
                    continue
                state = self.state[param]
                t = state.get("t", 0) # get iteration number
                # grad = param.grad.data
                grad = param.grad
                # param.data -= lr/math.sqrt(t + 1) * grad
                param -= lr/math.sqrt(t + 1) * grad
                state["t"] = t + 1
        
        return loss
