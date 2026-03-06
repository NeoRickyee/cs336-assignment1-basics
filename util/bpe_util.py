import json
import os
import time
from typing import List, Optional, Tuple
import numpy as np
import shutil
import mmap

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

def save_tokenizer_vocab(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], output_dir: str) -> None:
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


def train_bpe(dataset_key: str, vocab_size: int, special_tokens: List[str], output_dir: str) -> None:
    """
    Runs the BPE training process for the given dataset.
    """
    input_path = DATASETS.get(dataset_key)
    if not output_dir:
        output_dir = BPE_SAVE_DIR.get(dataset_key)
    if not vocab_size:
        vocab_size = VOCAB_SIZE.get(dataset_key)

    if not input_path or not os.path.exists(input_path):
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


def _get_file_limit(input_path: str, num_docs: Optional[int], delimiter_bytes: bytes) -> int:
    """Finds the byte offset limit for reading num_docs documents."""
    file_size = os.path.getsize(input_path)
    if not num_docs:
        return file_size
    
    count = 0
    limit = 0
    
    if file_size == 0:
        return 0

    with open(input_path, "rb") as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            start = 0
            while count < num_docs:
                idx = mm.find(delimiter_bytes, start)
                if idx == -1:
                    return file_size
                count += 1
                start = idx + len(delimiter_bytes)
            limit = start
            
    return limit


def _finalize_npy_file(temp_path: str, output_path: str, total_tokens: int) -> None:
    """Prepends NPY header to the raw binary data."""
    with open(output_path, "wb") as out_f:
        # Write .npy header manually
        header_data = {
            'descr': np.dtype(np.uint16).str,
            'fortran_order': False,
            'shape': (total_tokens,),
        }
        np.lib.format.write_array_header_1_0(out_f, header_data)
        
        # Stream copy the raw binary data
        with open(temp_path, "rb") as in_f:
            shutil.copyfileobj(in_f, out_f)


def encode_dataset(
    dataset_key: str, vocab_path: str, merges_path: str,
    num_docs: int, output_path: str, special_tokens: List[str],
    split: str = "train"
) -> None:
    """
    Encodes a sample of documents from the dataset using the provided BPE tokenizer.
    """
    if split == "train":
        input_path = DATASETS.get(dataset_key)
    else:
        input_path = DATASETS_VALID.get(dataset_key)
    
    if not vocab_path:
        vocab_path = os.path.join(BPE_SAVE_DIR.get(dataset_key, ""), "vocab.json")
    if not merges_path:
        merges_path = os.path.join(BPE_SAVE_DIR.get(dataset_key, ""), "merges.json")

    if not input_path or not os.path.exists(input_path):
        print(f"Error: Data file not found at {input_path}")
        print("Please download the data first as described in README.md")
        return

    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"Error: Tokenizer files not found at {vocab_path} or {merges_path}")
        print(f"Please train the tokenizer first using 'train-bpe' command.")
        return

    if not num_docs:
        output_file_name = f"{split}_encoded.npy"
    else:
        output_file_name = f"{split}_encoded_{num_docs}docs.npy"

    if not output_path:
        output_path = os.path.join(BPE_SAVE_DIR.get(dataset_key, "."), output_file_name)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    print(f"Sampling and encoding {num_docs if num_docs else 'all'} documents from {input_path}...")

    delimiter = "<|endoftext|>"
    delimiter_bytes = delimiter.encode("utf-8")
    
    limit_bytes = _get_file_limit(input_path, num_docs, delimiter_bytes)

    stats = {
        "docs": 0,
        "chars": 0,
        "bytes": 0,
        "tokens": 0
    }

    BATCH_THRESHOLD = 1_000_000  # Buffer ~2MB of tokens before writing
    CHUNK_SIZE = 8 * 1024 * 1024 # 8MB

    start_time = time.perf_counter()

    temp_output_path = output_path + ".tmp"
    try:
        with open(temp_output_path, "wb") as temp_f:
            with open(input_path, "rb") as f:
                pos = 0
                remainder = b""
                batch_ids = []
                
                while pos < limit_bytes:
                    read_amount = min(CHUNK_SIZE, limit_bytes - pos)
                    chunk = f.read(read_amount)
                    pos += len(chunk)
                    
                    current_bytes = remainder + chunk
                    to_encode_bytes = None

                    if pos >= limit_bytes:
                        to_encode_bytes = current_bytes
                        remainder = b""
                    else:
                        last_idx = current_bytes.rfind(delimiter_bytes)
                        if last_idx != -1:
                            split_idx = last_idx + len(delimiter_bytes)
                            to_encode_bytes = current_bytes[:split_idx]
                            remainder = current_bytes[split_idx:]
                        else:
                            remainder = current_bytes
                            continue
                    
                    if to_encode_bytes:
                        text = to_encode_bytes.decode("utf-8", errors="ignore")
                        encoded_ids = tokenizer.encode(text)
                        
                        batch_docs_count = text.count(delimiter)
                        if pos >= limit_bytes and len(text) > 0 and not text.endswith(delimiter):
                            batch_docs_count += 1
                        
                        stats["docs"] += batch_docs_count
                        stats["chars"] += len(text)
                        stats["bytes"] += len(to_encode_bytes)
                        stats["tokens"] += len(encoded_ids)

                        batch_ids.extend(encoded_ids)
                        
                        if len(batch_ids) >= BATCH_THRESHOLD:
                            np.array(batch_ids, dtype=np.uint16).tofile(temp_f)
                            batch_ids = []

            # Flush any remaining tokens in the batch
            if batch_ids:
                np.array(batch_ids, dtype=np.uint16).tofile(temp_f)

        # Save to output
        if output_path:
            _finalize_npy_file(temp_output_path, output_path, stats["tokens"])
            
            output_size = os.path.getsize(output_path)
            print(f"Saved encoded documents to {output_path}")
            print(f"Encoded output size: {format_size(output_size)}")
        else:
            print("No output path provided, skipping save.")

    finally:
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

    elapsed_time = time.perf_counter() - start_time
    print(f"Encoded {stats['docs']} documents.")
    print(f"Total characters: {stats['chars']}")
    print(f"Input document size: {format_size(stats['bytes'])}")
    print(f"Total tokens: {stats['tokens']}")
    print(f"Elapsed time: {elapsed_time:.4f}")


def decode_dataset(
    dataset_key: str, vocab_path: str, merges_path: str,
    input_path: str, output_path: str, special_tokens: List[str]
) -> None:
    """
    Decodes a .npy file containing token IDs back to text.
    """
    if not vocab_path:
        vocab_path = os.path.join(BPE_SAVE_DIR.get(dataset_key, ""), "vocab.json")
    if not merges_path:
        merges_path = os.path.join(BPE_SAVE_DIR.get(dataset_key, ""), "merges.json")
    if not input_path:
        input_file_name = f"train_encoded.npy"
        input_path = os.path.join(BPE_SAVE_DIR.get(dataset_key, "."), input_file_name)

    if not os.path.exists(vocab_path) or not os.path.exists(merges_path):
        print(f"Error: Tokenizer files not found at {vocab_path} or {merges_path}")
        return

    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    if not output_path:
        base, _ = os.path.splitext(input_path)
        output_path = base + "_decoded.txt"

    print(f"Loading tokenizer from {vocab_path} and {merges_path}...")
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)

    print(f"Decoding {input_path}...")
    
    data = np.load(input_path, mmap_mode='r')
    CHUNK_SIZE = 8 * 1024 * 1024 # 1M tokens

    print(f"Writing decoded text to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(0, len(data), CHUNK_SIZE):
            chunk = data[i:i+CHUNK_SIZE]
            text = tokenizer.decode(chunk.tolist())
            f.write(text)
            
    print(f"Done. Decoded file saved to {output_path}")