import multiprocessing
import os
import regex as re

from collections import Counter
from typing import BinaryIO, Dict, List, Tuple

from cs336_basics.util.priority_dict import PriorityDict


class Tokenizer:
    def __init__(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        try:
            # On Linux, this respects taskset/cgroups limits
            available_cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            available_cpus = os.cpu_count() or 17
        # Leave one core free for system responsiveness
        self.available_cpus = max(1, available_cpus - 1)

        self.input_path = input_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

        # Add basic vocabs
        vocab = {i: bytes([i]) for i in range(256)}
        for st in special_tokens:
            st_encoded = st.encode("utf-8")
            if len(st_encoded) > 1:
                vocab[len(vocab)] = st_encoded
        self.vocab = vocab
        self.merged_pairs = []

        # Parallel
        self.word_counts = self.count_words_in_corpus()
        # Not parallel (yet?)
        self.word_to_vocab = self.get_words_vocabs()
        # Count basic vocab pairs
        self.priority_dict = self.initialize_priority_dict()
    
    def count_words_in_corpus(self) -> Dict[bytes, int]:
        # Split corpus into chunks
        with open(self.input_path, "rb") as f:
            boundaries = self.find_chunk_boundaries(f, self.available_cpus, b"<|endoftext|>")

        # Count words in parallel
        with multiprocessing.Pool(self.available_cpus) as pool:
            chunk_args = [
                (self.input_path, start, end, self.special_tokens)
                for start, end in zip(boundaries[:-1], boundaries[1:])
            ]
            
            word_counts: Dict[bytes, int] = Counter()
            for chunk_word_counts in pool.imap_unordered(self._get_word_and_bp_count_wrapper, chunk_args):
                word_counts.update(chunk_word_counts)

        return word_counts
    
    def get_words_vocabs(self) -> Dict[bytes, List[bytes]]:
        # Add map of word to current vocab
        word_to_vocab: Dict[bytes, List[bytes]] = dict()
        for word, _ in self.word_counts.items():
            word_vocab = [word[i:i+1] for i in range(len(word))]
            word_to_vocab[word] = word_vocab
        return word_to_vocab

    def tokenize(self) -> Tuple[Dict[int, bytes], list[Tuple[bytes, bytes]]]:
        """
            Returns:
            Tuple[Dict[int, bytes], list[Tuple[bytes, bytes]]]:
                vocab:
                    The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                    to bytes (token bytes)
                merges:
                    BPE merges. Tuples of bytes (<token1>, <token2>),
                    representing that <token1> was merged with <token2>.
                    Merges are ordered by order of creation.
        """
    
        # BPE loop to generate vocab until vocab_size
        while len(self.vocab) < self.vocab_size:
            # find top token pair
            top_pair, frequency = self.priority_dict.pop()
            if not top_pair and not frequency:
                print("Tokenizer: No more vocab pairs can be found. STOP")
                break
            # update vocab
            self.vocab[len(self.vocab)] = top_pair[0] + top_pair[1]
            self.merged_pairs.append(top_pair)

            reduced_vocab_pairs_and_counts, added_vocab_pairs_and_counts = self.update_word_to_vocab_and_get_change(top_pair)

            for vocab_pairs, count in reduced_vocab_pairs_and_counts.items():
                if vocab_pairs == top_pair:
                    continue
                self.priority_dict.reduce(vocab_pairs, count)
            for vocab_pairs, count in added_vocab_pairs_and_counts.items():
                self.priority_dict.increase(vocab_pairs, count)

        return self.vocab, self.merged_pairs
        
    def update_word_to_vocab_and_get_change(
        self, top_pair: Tuple[bytes, bytes]
    ) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], int]]:
        combined_top_pair = top_pair[0] + top_pair[1]
        reduced_vocab_pairs_and_counts: Dict[Tuple[bytes, bytes], int] = Counter()
        added_vocab_pairs_and_counts: Dict[Tuple[bytes, bytes], int] = Counter()
        for word, vocabs in self.word_to_vocab.items():
            if combined_top_pair not in word:
                continue
            # combined_top_pair exist in word
            # check if top_pair can actually combine in vocabs

            updated_vocabs = []
            updated_vocabs_top_pair_indexes = []
            word_count = self.word_counts[word]
            i = 0
            last_was_merge = False
            while i < len(vocabs):
                if vocabs[i] == top_pair[0] and i + 1 < len(vocabs) and vocabs[i+1] == top_pair[1]:
                    # found top pair exist in vocabs
                    updated_vocabs_top_pair_indexes.append(len(updated_vocabs))
                    updated_vocabs.append(combined_top_pair)
                    # record the reduced vocab pairs
                    if i > 0 and not last_was_merge:
                        reduced_vocab_pairs_and_counts[(vocabs[i-1], vocabs[i])] += word_count
                    if i + 2 < len(vocabs):
                        reduced_vocab_pairs_and_counts[(vocabs[i+1], vocabs[i+2])] += word_count
                    i += 2
                    last_was_merge = True
                else:
                    updated_vocabs.append(vocabs[i])
                    i += 1
                    last_was_merge = False

            if not updated_vocabs_top_pair_indexes:
                # No top_pair combined in vocabs
                # combined_top_pair exists in word, but they combined previously in other ways
                continue
            
            for i, index in enumerate(updated_vocabs_top_pair_indexes):
                prev_index_not_a_top_pair: bool = (i - 1) < 0 or index - 1 != updated_vocabs_top_pair_indexes[i-1]
                if index > 0 and prev_index_not_a_top_pair:
                    added_vocab_pairs_and_counts[(updated_vocabs[index-1], updated_vocabs[index])] += word_count
                if index + 1 < len(updated_vocabs):
                    added_vocab_pairs_and_counts[(updated_vocabs[index], updated_vocabs[index+1])] += word_count
            self.word_to_vocab[word] = updated_vocabs
        return reduced_vocab_pairs_and_counts, added_vocab_pairs_and_counts
                


    def find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))

    @staticmethod
    def get_word_and_bp_count(
        file: BinaryIO, chunk_start: int, chunk_end: int, special_tokens: list[str]
    ) -> Dict[bytes, int]:
        """
        Given a input corpus from start index to end index, output
        * a list of words and each word's count
        special_tokens will not show up in the return values, and will work as word boundaries
        """
        file.seek(chunk_start)
        chunk = file.read(chunk_end - chunk_start)
        text = chunk.decode("utf-8", errors="ignore")

        if special_tokens:
            pattern = "|".join(re.escape(t) for t in special_tokens)
            parts = re.split(pattern, text)
        else:
            parts = [text]
        
        # GPT-2 pre-tokenization regex
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pat = re.compile(PAT)

        word_counts = Counter()
        for part in parts:
            if not part:
                continue
            for word in re.finditer(pat, part):
                word_counts[word.group().encode("utf-8")] += 1

        return word_counts

    @staticmethod
    def _get_word_and_bp_count_wrapper(args):
        input_path, chunk_start, chunk_end, special_tokens = args
        with open(input_path, "rb") as f:
            return Tokenizer.get_word_and_bp_count(f, chunk_start, chunk_end, special_tokens)

    def initialize_priority_dict(self) -> PriorityDict:
        bytes_pair_count: Dict[Tuple[bytes, bytes], int] = Counter()
        for word, freq in self.word_counts.items():
            vocabs = self.word_to_vocab[word]
            if len(vocabs) < 2:
                continue
            for pair in zip(vocabs[:-1], vocabs[1:]):
                bytes_pair_count[pair] += freq
        
        priority_dict: PriorityDict = PriorityDict()
        for bytes_tuple, count in bytes_pair_count.items():
            priority_dict.add(bytes_tuple, count)
        
        return priority_dict
