import argparse
import numpy as np
import os
import torch
import threading
import queue
import wandb

from argparse import Namespace
from cs336_basics import transformer_lm, functions, adamw_optimizer
from cs336_basics import check_pointing, data_loader
from util import constants
from util.constants import DATASETS, DATASETS_VALID, BPE_SAVE_DIR, VOCAB_SIZE
from typing import Optional, Literal

D_TYPE = torch.float32
DEVICE = torch.device("cuda")

def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")
    
    # Experiment Management
    parser.add_argument("--dataset_name", type=str, default="openwebtext")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--wandb_name", type=str, default=None, help="Optional specific run name")
    
    # Model Architecture
    parser.add_argument("--context_length", type=int, default=100)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=None, help="Default is 8/3 * d_model")
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_norm", type=float, default=1.0)
    
    # Learning Rate Schedule
    parser.add_argument("--warmup_iters", type=int, default=1000)
    parser.add_argument("--cosine_cycle_iters", type=int, default=100000)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    
    # Checkpointing & Limits
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    parser.add_argument("--checkpoint_interval", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    
    return parser.parse_args()

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
        step, loss_tensor, lr = task

        loss_val = loss_tensor.item()
        print(f"Step {step} | Loss: {loss_val:.4f} | LR: {lr: .2e}")

        wandb.log({
                "train/loss": loss_val,
                "train/learning_rate": lr
            },
            step=step
        )
        q.task_done()

def training_loop(args: Namespace):
    constants.init_directories(args.dataset_name)

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )

    lm = transformer_lm.TransformerLM(
        vocab_size=constants.VOCAB_SIZE.get(args.dataset_name),
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        rope_theta=args.rope_theta,
        d_ff=args.d_ff,
        device=DEVICE,
        dtype=D_TYPE
    )
    lm.to(DEVICE) # safty check
    lm = torch.compile(lm)

    def train_step(x, y):
    # reset gradient
        optimizer.zero_grad(set_to_none=True)

        # forward pass
        logits = lm(x)

        loss = functions.cross_entropy_loss(logits, y)
        loss.backward()
        functions.gradient_clipping(lm.parameters(), args.max_norm)
        return loss

    compiled_train_step = torch.compile(train_step)

    optimizer = adamw_optimizer.AdamW(
        params=lm.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
        eps=args.eps
    )

    data = load_dataset(args.dataset_name)

    if data is None:
        print("Data missing. Exiting")
        exit(1)

    logging_queue = queue.Queue()
    logging_thread = threading.Thread(target=background_worker, args=(logging_queue,))
    logging_thread.start()

    dataset = data_loader.SequentialDataset(
        data, args.batch_size, args.context_length, torch.device("cpu")
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
            step, alpha_max=args.learning_rate,
            alpha_min=args.min_learning_rate,
            T_warm=args.warmup_iters,
            T_c_anneal=args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["alpha"] = lr

        loss = compiled_train_step(x, y)

        optimizer.step()

        if step % args.log_interval == 0:
            logging_queue.put((step, loss.detach().cpu()))
        
        if step > 0 and step % args.checkpoint_interval == 0:
            checkpoint_out_path = constants.get_checkpoint_output_path(args.dataset_name, step)
            check_pointing.save_checkpoint(lm, optimizer, step, checkpoint_out_path)

    model_out_path = constants.get_fundamental_model_save_path(args.dataset_name)
    check_pointing.save_checkpoint(lm, optimizer, step, model_out_path)

    logging_queue.put(None)
    logging_thread.join()

if __name__ == "__main__":
    args = get_args()
    training_loop(args)
