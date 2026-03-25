import argparse
import numpy as np
import os
import torch
import threading
import queue
import wandb
from dotenv import load_dotenv

from argparse import Namespace
from cs336_basics import transformer_lm, functions, adamw_optimizer, decoding, tokenizer
from cs336_basics import check_pointing, data_loader
from bpe_util import constants
from bpe_util.constants import DATASETS, DATASETS_VALID, BPE_SAVE_DIR, VOCAB_SIZE
from typing import Optional, Literal

D_TYPE = torch.float32
DEVICE = torch.device("cuda")


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")
    
    # Experiment Management
    parser.add_argument("--dataset_name", type=str, default="tinystory")
    parser.add_argument("--wandb_project", type=str, default="cs336-assignment1")
    parser.add_argument("--wandb_name", type=str, default=None, help="Optional specific run name")
    
    # Model Architecture
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_attention_heads", type=int, default=16) # maybe bit much, try 8
    parser.add_argument("--d_ff", type=int, default=1344, help="Default is 8/3 * d_model")
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    
    # Training Hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
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
    
    # Generation & Evaluation
    parser.add_argument("--print_sample_gen_at_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sample_prompt", type=str, default="Once upon a time", help="Prompt for sample generation")
    parser.add_argument("--max_gen_len", type=int, default=50, help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for decoding")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p for decoding")
    
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
        step, metrics = task

        loss_val = metrics.get("train/loss", 0.0)
        lr_val = metrics.get("train/learning_rate", 0.0)
        
        # Keep the terminal output clean and precise
        print(f"Step {step} | Loss: {loss_val:.4f} | LR: {lr_val:.2e}")

        wandb.log(metrics, step=step)
        q.task_done()

class Trainer:
    def __init__(self, args: Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        constants.init_directories(args.dataset_name)

        self.vocab_path = constants.get_vocab_path(args.dataset_name)
        self.merges_path = os.path.join(constants.BPE_SAVE_DIR.get(args.dataset_name, "unknown"), "merges.json")
        self.tok = None
        self.eos_id = 0
        
        if os.path.exists(self.vocab_path) and os.path.exists(self.merges_path):
            self.tok = tokenizer.Tokenizer.from_files(self.vocab_path, self.merges_path, special_tokens=["<|endoftext|>"])
            self.eos_id = self.tok.vocab_to_id.get(b"<|endoftext|>", 0)

        self.model = transformer_lm.TransformerLM(
            vocab_size=constants.VOCAB_SIZE.get(args.dataset_name),
            context_length=args.context_length,
            num_layers=args.num_layers,
            d_model=args.d_model,
            num_attention_heads=args.num_attention_heads,
            rope_theta=args.rope_theta,
            d_ff=args.d_ff,
            device=self.device,
            dtype=D_TYPE
        )
        self.model.to(self.device)

        self.activation_norms = {}
        self._register_hooks()

        self.compiled_model = torch.compile(self.model)

        self.optimizer = adamw_optimizer.AdamW(
            params=self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps
        )
        
        self.compiled_train_step = torch.compile(self._train_step)

        self._setup_logging()
        self._setup_dataloader()

    def _setup_logging(self):
        self.logging_queue = queue.Queue()
        self.logging_thread = threading.Thread(target=background_worker, args=(self.logging_queue,))

    def _setup_dataloader(self):
        data = load_dataset(self.args.dataset_name)
        if data is None:
            print("Data missing. Exiting")
            exit(1)

        self.dataset = data_loader.SequentialDataset(
            data, self.args.batch_size, self.args.context_length, torch.device("cpu")
        )
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=None,
            num_workers=1,
            pin_memory=True
        )

    def _register_hooks(self):
        def get_activation_norm_hook(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    output = output[0]
                self.activation_norms[name] = output.detach().float().norm(2.0).item()
            return hook

        for i, layer in enumerate(self.model.transformers):
            layer.register_forward_hook(get_activation_norm_hook(f"layer_{i}"))
        self.model.output_embedding.register_forward_hook(get_activation_norm_hook("output_embedding"))

    def _train_step(self, x, y):
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.compiled_model(x)
        loss = functions.cross_entropy_loss(logits, y)
        loss.backward()

        param_norm = torch.norm(torch.stack([torch.norm(p.detach(), 2.0) for p in self.model.parameters()]), 2.0)
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        grad_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2.0) for g in grads]), 2.0) if grads else torch.tensor(0.0, device=x.device)

        functions.gradient_clipping(self.model.parameters(), self.args.max_norm)
        return loss, param_norm, grad_norm

    def generate_sample(self, prompt: str, step: Optional[int] = None):
        if self.tok is None or not prompt:
            return
        
        self.model.eval()
        with torch.no_grad():
            input_ids = np.array(self.tok.encode(prompt), dtype=np.int64)
            if len(input_ids) > 0:
                output_ids = decoding.decode(
                    self.model, input_ids, self.args.max_gen_len, self.args.temperature, self.args.top_p, self.eos_id
                )
                gen_text = self.tok.decode(output_ids[0].tolist())
                step_info = f" (Step {step})" if step is not None else ""
                print(f"\n--- Sample Generation{step_info} ---\n{gen_text}\n" + "-" * 35 + "\n")
        self.model.train()

    def train(self):
        load_dotenv()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key:
            wandb.login(key=wandb_api_key)

        wandb.init(
            project=self.args.wandb_project,
            name=self.args.wandb_name,
            config=vars(self.args)
        )
        self.logging_thread.start()

        step = 0
        for step, (x, y) in enumerate(self.dataloader):
            if step >= self.args.max_steps:
                break
                
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            lr = functions.learning_rate_cosine_schedule(
                step, alpha_max=self.args.learning_rate,
                alpha_min=self.args.min_learning_rate,
                T_warm=self.args.warmup_iters,
                T_c_anneal=self.args.cosine_cycle_iters
            )
            for group in self.optimizer.param_groups:
                group["alpha"] = lr

            loss, param_norm, grad_norm = self.compiled_train_step(x, y)
            self.optimizer.step()

            if step % self.args.log_interval == 0:
                metrics = {
                    "train/loss": loss.detach().cpu().item(),
                    "train/learning_rate": lr,
                    "norms/weights_l2": param_norm.detach().cpu().item(),
                    "norms/gradients_l2": grad_norm.detach().cpu().item(),
                }
                
                for name, norm_val in self.activation_norms.items():
                    metrics[f"norms/activations_{name}"] = norm_val
                    
                self.logging_queue.put((step, metrics))
            
            if step > 0 and step % self.args.checkpoint_interval == 0:
                checkpoint_out_path = constants.get_checkpoint_output_path(self.args.dataset_name, step)
                check_pointing.save_checkpoint(self.model, self.optimizer, step, checkpoint_out_path)

                if self.args.print_sample_gen_at_checkpoint:
                    self.generate_sample(self.args.sample_prompt, step)

        # Save final model
        model_out_path = constants.get_fundamental_model_save_path(self.args.dataset_name)
        check_pointing.save_checkpoint(self.model, self.optimizer, step, model_out_path)

        # Cleanup logging
        self.logging_queue.put(None)
        self.logging_thread.join()
        
        print("\nTraining Complete!")
        self.generate_sample(self.args.sample_prompt, step)

    def interactive_prompt(self):
        print("\nEntering interactive generation mode. Type 'quit' or 'exit' to stop.")
        while True:
            try:
                user_prompt = input("Prompt> ")
                if user_prompt.strip().lower() in ['quit', 'exit']:
                    break
                if user_prompt.strip():
                    self.generate_sample(user_prompt)
            except (KeyboardInterrupt, EOFError):
                print()
                break


if __name__ == "__main__":
    args = get_args()
    trainer = Trainer(args)
    trainer.train()
    trainer.interactive_prompt()
