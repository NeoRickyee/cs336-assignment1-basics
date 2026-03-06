import json
import os
from typing import List

from util.constants import DATASETS, DATASETS_VALID, BPE_SAVE_DIR, VOCAB_SIZE
from cs336_basics.bpe import BPE
from cs336_basics.tokenizer import Tokenizer


def format_size(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.2f} KB"
    else:
        return f"{size / (1024 * 1024):.2f} MB"

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


def encode_dataset(
    dataset_key: str, vocab_path: str, merges_path: str,
    num_docs: int, output_path: str, special_tokens: List[str]
):
    """
    Encodes a sample of documents from the dataset using the provided BPE tokenizer.
    """
    input_path = DATASETS[dataset_key]
    
    if not vocab_path:
        vocab_path = os.path.join(BPE_SAVE_DIR[dataset_key], "vocab.json")
    if not merges_path:
        merges_path = os.path.join(BPE_SAVE_DIR[dataset_key], "merges.json")

    if not os.path.exists(input_path):
        print(f"Error: Data file not found at {input_path}")
        print("Please download the data first as described in README.md")
        return

    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"Error: Tokenizer files not found at {vocab_path} or {merges_path}")
        print(f"Please train the tokenizer first using 'train-bpe' command.")
        return

    if not num_docs:
        output_file_name = "encoded.json"
    else:
        output_file_name = f"encoded_{num_docs}docs.json"

    if not output_path:
        output_path = os.path.join(BPE_SAVE_DIR[dataset_key], output_file_name)

    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    print(f"Sampling and encoding {num_docs if num_docs else 'all'} documents from {input_path}...")

    encoded_docs = []
    docs_count = 0
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    delimiter = "<|endoftext|>"
    buffer = ""

    

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        while not num_docs or docs_count < num_docs:
            chunk = f.read(1024 * 1024) # Read 1MB at a time
            if not chunk:
                break
            
            buffer += chunk
            while delimiter in buffer and (not num_docs or docs_count < num_docs):
                split_index = buffer.find(delimiter)
                doc_text = buffer[:split_index + len(delimiter)]
                buffer = buffer[split_index + len(delimiter):]
                
                encoded_ids = tokenizer.encode(doc_text)
                encoded_docs.append(encoded_ids)
                total_chars += len(doc_text)
                total_bytes += len(doc_text.encode("utf-8"))
                total_tokens += len(encoded_ids)
                docs_count += 1
                if docs_count % 100 == 0:
                    if num_docs:
                        print(f"Encoded {docs_count}/{num_docs} documents")
                    else:
                        print(f"Encoded {docs_count} documents")

    # Handle case where we requested more docs than available or just finished
    if (not num_docs or docs_count < num_docs) and buffer.strip():
        encoded_ids = tokenizer.encode(buffer)
        encoded_docs.append(encoded_ids)
        total_chars += len(buffer)
        total_bytes += len(buffer.encode("utf-8"))
        total_tokens += len(encoded_ids)
        docs_count += 1

    print(f"Encoded {len(encoded_docs)} documents.")
    print(f"Total characters: {total_chars}")
    print(f"Input document size: {format_size(total_bytes)}")
    print(f"Total tokens: {total_tokens}")
    
    # Save to output
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(encoded_docs, f)
        output_size = os.path.getsize(output_path)
        print(f"Saved encoded documents to {output_path}")
        print(f"Encoded output size: {format_size(output_size)}")
    else:
        print("No output path provided, skipping save.")