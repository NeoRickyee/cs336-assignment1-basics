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

class SequentialDataset(torch.utils.data.IterableDataset):
    def __init__(
        self, data: np.ndarray, batch_size: int, context_length: int,
        device: torch.device
    ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length
        self.device = device
    
    def __iter__(self):
        stream_length = len(self.data) // self.batch_size
        streams = self.data[:self.batch_size * stream_length].reshape(
            self.batch_size, stream_length
        )

        for i in range(
            0, stream_length - self.context_length, self.context_length
        ):
            chunks = streams[:, i:i+self.context_length+1].astype(np.int16)

            # Transfer as int16 to save PCIe bandwidth, then cast to int64 on the GPU
            x = torch.from_numpy(chunks[:,:-1]).to(self.device).long()
            y = torch.from_numpy(chunks[:, 1:]).to(self.device).long()

            yield x, y