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
    parser.add_argument("--dataset_name", type=str, default="openwebtext")
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
    parser.add_argument("--batch_size", type=int, default=32)
    # 64 for tinystory
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    # 5e-3 for tinystory
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max_norm", type=float, default=1.0)
    
    # Learning Rate Schedule
    parser.add_argument("--warmup_iters", type=int, default=4000)
    # 1000 for tinystory
    parser.add_argument("--cosine_cycle_iters", type=int, default=100000)
    parser.add_argument("--min_learning_rate", type=float, default=1e-5)
    
    # Checkpointing & Limits
    parser.add_argument("--max_steps", type=int, default=100000, help="Maximum number of training steps")
    parser.add_argument("--max_tokens", type=int, default=327680000, help="Maximum total tokens to train before exiting")
    parser.add_argument("--checkpoint_interval", type=int, default=3000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=200, help="Number of batches to use for validation")
    
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

        if "train/loss" in metrics:
            print(f"Step {step} | Loss: {metrics['train/loss']:.4f} | LR: {metrics.get('train/learning_rate', 0.0):.2e}")
            
            # TODO: remove! print activation norms to confirm hooks are firing
            activation_keys = [k for k in metrics.keys() if k.startswith("norms/activations_")]
            if activation_keys:
                avg_norm = sum(metrics[k] for k in activation_keys) / len(activation_keys)
                print(f"          | Avg Activation Norm: {avg_norm:.4f}")
            
        if "valid/loss" in metrics:
            print(f"Step {step} | Validation Loss: {metrics['valid/loss']:.4f}")

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

        # self.compiled_model = torch.compile(self.model)
        self.compiled_model = self.model
        # TODO: change back if no need to monitor norms

        self.optimizer = adamw_optimizer.AdamW(
            params=self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
            eps=args.eps
        )

        # self.compiled_train_step = torch.compile(self._train_step)
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

        valid_data = load_dataset(self.args.dataset_name, split="valid")
        if valid_data is not None:
            self.valid_dataset = data_loader.SequentialDataset(
                valid_data, self.args.batch_size, self.args.context_length, torch.device("cpu")
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                dataset=self.valid_dataset,
                batch_size=None,
                num_workers=1,
                pin_memory=True
            )
        else:
            self.valid_dataloader = None

    @torch.no_grad()
    def evaluate_validation_loss(self) -> Optional[float]:
        if getattr(self, "valid_dataloader", None) is None:
            return None
            
        self.model.eval()
        total_loss = 0.0
        batches_evaluated = 0
        
        iterator = iter(self.valid_dataloader)
        
        for _ in range(self.args.eval_batches):
            try:
                x, y = next(iterator)
            except StopIteration:
                break
                
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            
            logits = self.compiled_model(x)
            loss = functions.cross_entropy_loss(logits, y)
            
            total_loss += loss.item()
            batches_evaluated += 1
            
        self.model.train()
        
        if batches_evaluated == 0:
            return None
            
        return total_loss / batches_evaluated

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
        total_tokens = 0

        tokens_per_step = self.args.batch_size * self.args.context_length
        if self.args.max_tokens is not None:
            calculated_max_steps = (self.args.max_tokens + tokens_per_step - 1) // tokens_per_step
        else:
            calculated_max_steps = self.args.max_steps
        if calculated_max_steps < self.args.cosine_cycle_iters:
            cosine_cycle_iters = calculated_max_steps
        else:
            cosine_cycle_iters = self.args.cosine_cycle_iters

        for step, (x, y) in enumerate(self.dataloader):
            if step >= self.args.max_steps:
                break
                
            if self.args.max_tokens is not None and total_tokens >= self.args.max_tokens:
                break

            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            total_tokens += x.numel()

            lr = functions.learning_rate_cosine_schedule(
                step, alpha_max=self.args.learning_rate,
                alpha_min=self.args.min_learning_rate,
                T_warm=self.args.warmup_iters,
                T_c_anneal=cosine_cycle_iters
            )
            for group in self.optimizer.param_groups:
                group["alpha"] = lr

            # loss, param_norm, grad_norm = self.compiled_train_step(x, y)
            loss, param_norm, grad_norm = self._train_step(x, y)
            self.optimizer.step()

            if step % self.args.log_interval == 0:
                metrics = {
                    "train/loss": loss.detach().cpu().item(),
                    "train/learning_rate": lr,
                    "norms/weights_l2": param_norm.detach().cpu().item(),
                    "norms/gradients_l2": grad_norm.detach().cpu().item(),
                    "train/total_tokens": total_tokens,
                }
                
                for name, norm_val in self.activation_norms.items():
                    metrics[f"norms/activations_{name}"] = norm_val
                    
                self.logging_queue.put((step, metrics))
            
            if step > 0 and step % self.args.checkpoint_interval == 0:
                val_loss = self.evaluate_validation_loss()
                if val_loss is not None:
                    self.logging_queue.put((step, {"valid/loss": val_loss}))

                checkpoint_out_path = constants.get_checkpoint_output_path(self.args.dataset_name, step)
                check_pointing.save_checkpoint(self.model, self.optimizer, step, checkpoint_out_path)

                if self.args.print_sample_gen_at_checkpoint:
                    self.generate_sample(self.args.sample_prompt, step)

        # Save final model
        model_out_path = constants.get_fundamental_model_save_path(self.args.dataset_name)
        check_pointing.save_checkpoint(self.model, self.optimizer, step, model_out_path)
        val_loss = self.evaluate_validation_loss()
        self.logging_queue.put((step, {"valid/loss": val_loss}))

        # Cleanup logging
        self.logging_queue.put(None)
        self.logging_thread.join()
        wandb.finish()
        
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
