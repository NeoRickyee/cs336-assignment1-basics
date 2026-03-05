import json
import multiprocessing
import os
import regex as re

from typing import BinaryIO, Dict, List, Set, Tuple, Optional, Iterator, Iterable

class Tokenizer:
    def __init__(
        self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[list[str]] = None
    ):
        self.id_to_vocab: Dict[int, bytes] = vocab
        self.vocab_to_id: Dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.vocab_size: int = len(vocab)
        self.merges: List[Tuple[bytes, bytes]] = merges
        self.merges_ranked: Dict[Tuple[bytes, bytes], int] = {v : k for k, v in enumerate(merges)}
        self.special_tokens: Set[str] = set(special_tokens) if special_tokens else set()
        self.encoded_word: Dict[bytes, List[int]] = {}

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.pretokenize_pat = re.compile(PAT)

        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        self.special_token_split_pattern = "(" + "|".join(re.escape(t) for t in sorted_special_tokens) + ")"

    @classmethod
    def from_files(
        cls, vocab_filepath: str, merges_filepath: str,
        special_tokens: Optional[List[str]]=None
    ) -> "Tokenizer":
        with open(vocab_filepath, "r") as f:
            vocab = {int(k): bytes.fromhex(v) for k, v in json.load(f).items()}

        with open(merges_filepath, "r") as f:
            merges = [(bytes.fromhex(p[0]), bytes.fromhex(p[1])) for p in json.load(f)]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            parts = re.split(self.special_token_split_pattern, text)
        else:
            parts = [text]
        
        ids = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                b_part: bytes = part.encode("utf-8")
                assert b_part in self.vocab_to_id
                ids.append(self.vocab_to_id[b_part])
                continue
            for word in re.finditer(self.pretokenize_pat, part):
                ids.extend(self.encode_word(word.group()))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def encode_word(self, word: str) -> list[int]:
        utf_word: bytes = word.encode("utf-8")
        # check if word is already cached
        encoded = self.encoded_word.get(utf_word)
        if encoded:
            return encoded

        # encode word
        b_word: list[bytes] = [b'%c' % b for b in utf_word]

        while len(b_word) > 1:
            min_rank = self.vocab_size
            min_pair = None
            min_pair_index = None

            for i in range(len(b_word) - 1):
                curr_tuple = (b_word[i], b_word[i+1])
                curr_tuple_rank = self.merges_ranked.get(curr_tuple)
                if curr_tuple_rank is None:
                    continue
                if curr_tuple_rank >= min_rank:
                    continue
                min_rank = curr_tuple_rank
                min_pair = curr_tuple
                min_pair_index = i
            
            if not min_pair:
                break

            b_word[min_pair_index] += b_word[min_pair_index+1]
            del b_word[min_pair_index+1]
        
        ids = [self.vocab_to_id[i] for i in b_word]
        self.encoded_word[utf_word] = ids
        return ids

    def decode(self, ids: list[int]) -> str:
        result: bytearray = bytearray()
        for id in ids:
            vocab = self.id_to_vocab.get(id)
            if not vocab:
                assert False
                continue
            result.extend(vocab)
        return result.decode("utf-8", errors="replace")
