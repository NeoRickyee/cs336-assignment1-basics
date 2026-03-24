import numpy as np
import torch

from torch import Tensor, LongTensor
from jaxtyping import Float, Int, Bool

from cs336_basics import functions

DEVICE = torch.device("cuda")

def decode(
    lm: torch.nn.Module, x: np.ndarray, max_gen_len: int,
    temperature: float, top_p: float, eos_id: int
):
    x_tensor = torch.from_numpy(x).to(DEVICE)
    if len(x.shape) == 1:
        # add batch_size = 1 to input x
        x_tensor: Tensor[Float, "batch_size, seq_len"] = x_tensor.view(1, -1)
    
    batch_size = x_tensor.shape[0]
    has_finished: Tensor[Bool, "batch_size"] = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    for _ in range(max_gen_len):
        output: Tensor[Float, "batch_size seq_len vocab_size"] = lm(x_tensor)
        # take the last generated token
        output_last_token: Tensor[Float, "batch_size vocab_size"] = output[:, -1, :].squeeze(1)
        # calculate the softmax
        output_last_token: Tensor[Float, "batch_size vocab_size"] = functions.softmax_with_temp(
            output_last_token, temperature
        )
        output_last_token: LongTensor[Int, "batch_size 1"] = functions.top_p_sampling(output_last_token, p=top_p)

        output_last_token.masked_fill_(has_finished.unsqueeze(-1), eos_id)
        has_finished |= (output_last_token.squeeze(-1) == eos_id)

        x_tensor = torch.cat((x_tensor, output_last_token), dim=-1)

        if has_finished.all():
            break
    return x_tensor.cpu().numpy()

        