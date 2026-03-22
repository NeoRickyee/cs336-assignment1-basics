import numpy as np
import os
import torch
import threading
import queue

from cs336_basics import transformer_lm, functions, adamw_optimizer
from cs336_basics import check_pointing, data_loader
from util import constants
from util.constants import DATASETS, DATASETS_VALID, BPE_SAVE_DIR, VOCAB_SIZE
from typing import Optional, Literal

DATASET_NAME = "openwebtext"
BATCH_SIZE = 8
CONTEXT_LENGTH = 100
NUM_LAYERS = 12
D_MODEL = 32
NUM_ATTENTION_HEAD = 4
D_FF = None # default 8/3 D_MODEL
THETA = 10000
D_TYPE = torch.float32
DEVICE = torch.device("cuda")

LEARNING_RATE = 5e-4
WEIGHT_DECAY=0.1
BETAS=(0.9,0.95)
EPS=1e-8

WARMUP_ITERS = 1000
COSINE_CYCLE_ITERS = 100000
MIN_LEARNING_RATE = 1e-5
MAX_NORM = 1.0


def load_dataset(dataset_key: str, split: Literal["train", "valid"] = "train"):
    input_path = constants.get_encoded_dataset_path(dataset_key=dataset_key, split=split)
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return
    
    return np.load(input_path, mmap_mode='r')

def background_worker(q):
    while True:
        task = q.get()
        if task is None:
            break
        step, loss_tensor = task

        loss_val = loss_tensor.item()
        print(f"Step {step} | Loss: {loss_val:.4f}")
        q.task_done()

constants.init_directories(DATASET_NAME)

lm = transformer_lm.TransformerLM(
    vocab_size=constants.VOCAB_SIZE.get(DATASET_NAME),
    context_length=CONTEXT_LENGTH,
    num_layers=NUM_LAYERS,
    d_model=D_MODEL,
    num_attention_heads=NUM_ATTENTION_HEAD,
    rope_theta=THETA,
    d_ff=D_FF,
    device=DEVICE,
    dtype=D_TYPE
)
lm.to(DEVICE) # safty check
lm = torch.compile(lm)

optimizer = adamw_optimizer.AdamW(
    params=lm.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=BETAS,
    eps=EPS
)

data = load_dataset(DATASET_NAME)

if data is None:
    print("Data missing. Exiting")
    exit(1)

logging_queue = queue.Queue()
logging_thread = threading.Thread(target=background_worker, args=(logging_queue,))
logging_thread.start()

dataset = data_loader.SequentialDataset(
    data, BATCH_SIZE, CONTEXT_LENGTH, torch.device("cpu")
)
dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=None,
    num_workers=1,
    pin_memory=True
)

for step, (x, y) in enumerate(dataloader):
    x = x.to(DEVICE, non_blocking=True)
    y = y.to(DEVICE, non_blocking=True)

    # update learning rate
    lr = functions.learning_rate_cosine_schedule(
        step, alpha_max=LEARNING_RATE, alpha_min=MIN_LEARNING_RATE,
        T_warm=WARMUP_ITERS, T_c_anneal=COSINE_CYCLE_ITERS
    )
    for group in optimizer.param_groups:
        group["alpha"] = lr

    # reset gradient
    optimizer.zero_grad(set_to_none=True)

    # forward pass
    logits = lm(x)

    loss = functions.cross_entropy_loss(logits, y)
    loss.backward()
    functions.gradient_clipping(lm.parameters(), MAX_NORM)

    optimizer.step()

    if step % 100 == 0:
        logging_queue.put((step, loss.detach().cpu()))
    
    if step > 0 and step % 3000 == 0:
        checkpoint_out_path = constants.get_checkpoint_output_path(DATASET_NAME, step)
        check_pointing.save_checkpoint(lm, optimizer, step, checkpoint_out_path)

model_out_path = constants.get_fundamental_model_save_path(DATASET_NAME)
check_pointing.save_checkpoint(lm, optimizer, step, model_out_path)

logging_queue.put(None)
logging_thread.join()
