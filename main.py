#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import List

from cs336_basics.bpe import BPE

# Define available datasets and their paths
DATASETS = {
    "tinystory": "data/TinyStoriesV2-GPT4-train.txt",
    "openwebtext": "data/owt_train.txt",
}

BPE_SAVE_DIR = {
    "tinystory": "data/tinystory",
    "openwebtext": "data/openwebtext",
}

VOCAB_SIZE = {
    "tinystory": 10000,
    "openwebtext": 32000,
}

def save_tokenizer_vocab(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str):
    """Saves the vocab and merges to the output directory as JSON."""
    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "vocab.json")
    # Convert bytes to hex strings for JSON serialization
    vocab_hex = {i: b.hex() for i, b in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(vocab_hex, f, indent=2)

    merges_path = os.path.join(output_dir, "merges.json")
    # Convert bytes to hex strings for JSON serialization
    merges_hex = [(p[0].hex(), p[1].hex()) for p in merges]
    with open(merges_path, "w") as f:
        json.dump(merges_hex, f, indent=2)

    print(f"Saved vocab to {vocab_path}")
    print(f"Saved merges to {merges_path}")


def train_bpe(dataset_key: str, vocab_size: int, special_tokens: List[str], output_dir: str):
    """
    Runs the BPE training process for the given dataset.
    """
    input_path = DATASETS[dataset_key]
    if not output_dir:
        output_dir = BPE_SAVE_DIR[dataset_key]
    if not vocab_size:
        vocab_size = VOCAB_SIZE[dataset_key]

    if not os.path.exists(input_path):
        print(f"Error: Data file not found at {input_path}")
        print("Please download the data first as described in README.md")
        return

    print(f"Starting BPE training on {dataset_key}...")
    print(f"Input path: {input_path}")
    print(f"Vocab size: {vocab_size}")
    print(f"Special tokens: {special_tokens}")

    # Initialize and train BPE
    bpe = BPE(input_path, vocab_size, special_tokens)
    vocab, merges = bpe.train()

    print(f"Training complete!")
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    
    save_tokenizer_vocab(vocab, merges, output_dir)

def main():
    parser = argparse.ArgumentParser(
        description="CS336 Assignment 1: Tokenizer Utility",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Create subparsers for industry-standard command structure
    # e.g. python main.py train-bpe --dataset tinystory
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Subparser for training BPE
    train_parser = subparsers.add_parser("train-bpe", help="Train BPE tokenizer")
    train_parser.add_argument(
        "--dataset", 
        type=str, 
        choices=DATASETS.keys(),
        required=True,
        help="Dataset to train on"
    )
    train_parser.add_argument(
        "--vocab-size", 
        type=int,
        default=None, 
        help="Target vocabulary size"
    )
    train_parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save tokenizer files"
    )

    # Subparser for encoding (placeholder for future)
    encode_parser = subparsers.add_parser("encode", help="Encode a dataset")
    encode_parser.add_argument(
        "--dataset", 
        type=str, 
        choices=DATASETS.keys(), 
        required=True,
        help="Dataset to encode"
    )

    args = parser.parse_args()
    
    # Default configuration
    default_special_tokens = ["<|endoftext|>"]

    # Handle commands
    if args.command == "train-bpe":
        train_bpe(args.dataset, args.vocab_size, default_special_tokens, args.output_dir)
    elif args.command == "encode":
        print(f"Encoding functionality for {args.dataset} is not yet implemented.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()