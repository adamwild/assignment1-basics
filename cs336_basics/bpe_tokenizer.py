import os
import regex as re
import multiprocessing
from collections import defaultdict
from multiprocessing import Pool
from itertools import repeat
from cs336_basics.pretokenization_example import find_chunk_boundaries
from cs336_basics.naive_merge import naive_merge_loop

def pre_tokenize(args):
    (start, end), input_path, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore").replace("\r\n", "\n")
    
    # Removing special tokens before pre-tokenization
    docs = re.split("|".join([re.escape(token) for token in special_tokens]), chunk)

    # Pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    frequency_table = {}

    for doc in docs:
        for match in re.finditer(PAT, doc):
            pre_token = match.group()
            frequency_table[pre_token] = frequency_table.get(pre_token, 0) + 1

    return frequency_table

def add_pair_neighbors(byte_pair_count, byte_pair, count):
    # Initialize or update the byte_pair entry
    if not type(byte_pair[0]) == bytes:
        raise("add_pair_neighbors!")

    if byte_pair not in byte_pair_count:
        byte_pair_count[byte_pair] = count
    else:
        byte_pair_count[byte_pair] += count

    return byte_pair_count

def build_byte_pair_count(
        s_frequency_table
):

    byte_pair_count = {}

    for s_token, count in s_frequency_table.items():
        for byte_ind in range(len(s_token)-1):
            b1, b2 = s_token[byte_ind], s_token[byte_ind+1]
            byte_pair_count = add_pair_neighbors(byte_pair_count, (b1, b2), count)

    return byte_pair_count

def split_frequency_table(frequency_table):
    """Done once, breaks the token: count into tuple[token_bytes]: count
    """
    def break_token(token):
        byte_string = token.encode("utf-8")

        return tuple(bytes([b]) for b in byte_string)

    s_frequency_table = {break_token(token): count for token, count in frequency_table.items()}

    return s_frequency_table

def init_vocab(special_tokens):
    vocab = {num_token: token.encode("utf-8") for num_token, token in enumerate(special_tokens)}

    num_special_tokens = len(special_tokens)

    for num in range(256):
        vocab[num+num_special_tokens] = bytes([num])

    return vocab

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    import time
    ref_time = time.time()

    if type(input_path) == str:
        input_path = pathlib.Path(input_path)
    
    # Chunking the file
    with open(input_path, "rb") as f:
        num_processes = multiprocessing.cpu_count()
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # Parallelizing pre-tokenization
    start_end_pairs = zip(boundaries[:-1], boundaries[1:])
    args_list = list(zip(start_end_pairs, repeat(input_path), repeat(special_tokens)))

    with Pool() as pool:
        frequency_tables = pool.map(pre_tokenize, args_list)

    # Combining all frequency_tables
    frequency_table = defaultdict(int)

    for dictionary in frequency_tables:
        for key, val in dictionary.items():
            frequency_table[key] += val
    frequency_table = dict(frequency_table)

    # Merging step
    # Initial split of the token into bytes tuples
    s_frequency_table = split_frequency_table(frequency_table)
    byte_pair_count = build_byte_pair_count(s_frequency_table)

    # Merge loop
    merges = []
    vocab = init_vocab(special_tokens)
    while len(vocab) < vocab_size:
        vocab, merges, s_frequency_table, byte_pair_count = naive_merge_loop(vocab, merges, s_frequency_table, byte_pair_count)
        
    print(time.time() - ref_time)

    return vocab, merges

if __name__ == "__main__":
    import pathlib
    data_path = (pathlib.Path(__file__).resolve().parent.parent) / "data"
    fixtures_path = (pathlib.Path(__file__).resolve().parent.parent) / 'tests' / "fixtures"
    test_file = data_path / 'basic_bpe_example.txt'
    special_tokens=["<|endoftext|>"]

    corpus_en_file = fixtures_path / 'corpus.en'

    vocab, merges = train_bpe(test_file, 263, special_tokens)
    print(merges)

