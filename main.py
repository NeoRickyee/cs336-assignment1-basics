#!/usr/bin/env python3
import argparse

from util.constants import DATASETS, DATASETS_VALID, BPE_SAVE_DIR, VOCAB_SIZE
from util.bpe_util import train_bpe, encode_dataset

# python main.py train-bpe --dataset tinystory
# python main.py encode --dataset tinystory --num-docs 10
# python main.py encode --dataset openwebtext --num-docs 10

# tinystory 7.39kb -> 1818 tokens
# openwebtext 30.88kb -> 6722 tokens

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
    encode_parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Path to vocab.json (defaults to saved dir)"
    )
    encode_parser.add_argument(
        "--merges-file",
        type=str,
        default=None,
        help="Path to merges.json (defaults to saved dir)"
    )
    encode_parser.add_argument(
        "--num-docs",
        type=int,
        default=None,
        help="Number of documents to sample"
    )
    encode_parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (JSON)"
    )

    args = parser.parse_args()
    
    # Default configuration
    default_special_tokens = ["<|endoftext|>"]

    # Handle commands
    if args.command == "train-bpe":
        train_bpe(args.dataset, args.vocab_size, default_special_tokens, args.output_dir)
    elif args.command == "encode":
        encode_dataset(
            args.dataset, 
            args.vocab_file, 
            args.merges_file, 
            args.num_docs, 
            args.output_file, 
            default_special_tokens
        )
    else:
        parser.print_help()

if __name__ == "__main__":
    main()