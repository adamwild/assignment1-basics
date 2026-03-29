import os
import regex as re
import multiprocessing
from collections import defaultdict
from multiprocessing import Pool
from itertools import repeat
from cs336_basics.pretokenization_example import find_chunk_boundaries

def pre_tokenize(args):
    (start, end), input_path, special_tokens = args

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
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

        return tuple([char.encode("utf-8") for char in token])

    s_frequency_table = {break_token(token): count for token, count in frequency_table.items()}

    return s_frequency_table

def apply_merge(s_frequency_table, merge_pair):
    new_s_frequency_table = {}
    new_byte_pair_count = {}
    
    b1_target, b2_target = merge_pair
    merged_token = b1_target + b2_target

    for s_token, count in s_frequency_table.items():
        # Handle single-byte tokens immediately
        if len(s_token) < 2:
            new_s_frequency_table[s_token] = new_s_frequency_table.get(s_token, 0) + count
            continue

        new_token = []
        i = 0
        while i < len(s_token):
            # Check if current pair matches the merge_pair
            if i < len(s_token) - 1 and s_token[i] == b1_target and s_token[i+1] == b2_target:
                new_token.append(merged_token)
                i += 2 # Skip both bytes since they are now one
            else:
                new_token.append(s_token[i])
                i += 1
        
        # Convert back to tuple for the table key
        new_token_tuple = tuple(new_token)
        new_s_frequency_table[new_token_tuple] = new_s_frequency_table.get(new_token_tuple, 0) + count

        # Update the pair counts for the NEXT iteration
        for j in range(len(new_token) - 1):
            pair = (new_token[j], new_token[j+1])
            new_byte_pair_count[pair] = new_byte_pair_count.get(pair, 0) + count

    return new_s_frequency_table, new_byte_pair_count

def get_next_merge(byte_pair_count):
    # Next pair to merge by finding pairs with highest counts, ties are broken lexicographically
    max_count = max(count for count in byte_pair_count.values())
    tied_first_place = [byte_pair for byte_pair, count in byte_pair_count.items() if count == max_count]
    next_pair_merge = max(tied_first_place)
    
    return next_pair_merge

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
    
    if type(input_path) == str:
        input_path = pathlib.Path(input_path)
    
    file_str = input_path.read_text(encoding="utf-8")

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
        new_merge = get_next_merge(byte_pair_count)
        merges.append(new_merge)

        s_frequency_table, byte_pair_count = apply_merge(s_frequency_table, new_merge)

        vocab[len(vocab)] = new_merge[0] + new_merge[1]

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

