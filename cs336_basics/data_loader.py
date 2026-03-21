import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor
from typing import Tuple

def data_loading(
    x: NDArray[np.int_], batch_size: int, context_length: int, device: torch.device
) -> Tuple[Tensor, Tensor]:
    # Random sampling each batch
    start_indices = np.random.randint(0, len(x) - context_length, size = batch_size)
    inputs = np.stack([x[i:i+context_length] for i in start_indices])
    targets = np.stack([x[i+1:i+context_length+1] for i in start_indices])
    input_tensor = torch.from_numpy(inputs).to(device=device)
    target_tensor = torch.from_numpy(targets).to(device=device)
    return input_tensor, target_tensor
