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
    """Given a merge_pair, combine bytes in the split frequency table"""
    new_s_frequency_table = {}
    new_byte_pair_count = {}

    just_merged = False

    merged_pair = merge_pair[0] + merge_pair[1]

    for s_token, count in s_frequency_table.items():
        new_s_token = []
        if len(s_token) == 1:
            new_s_frequency_table[tuple(s_token)] = count
            continue
        
        elif len(s_token) == 2:
            b1, b2 = s_token
            if (b1, b2) == merge_pair:
                new_s_frequency_table[tuple(merged_pair)] = count
                continue
            new_s_frequency_table[(b1, b2)] = count
            new_byte_pair_count = add_pair_neighbors(new_byte_pair_count, (b1, b2), count)
            continue

        for ind, (b1, b2, b3) in enumerate(zip(s_token[:-2], s_token[1:-1], s_token[2:])):

            if just_merged:
                just_merged = False
                new_byte_pair_count = add_pair_neighbors(new_byte_pair_count, (merged_pair, b2), count)
                continue

            if (b1, b2) == merge_pair:
                new_s_token.append(merged_pair)
                just_merged = True

            elif (b2, b3) == merge_pair:
                new_s_token.append(b1)
                new_byte_pair_count = add_pair_neighbors(new_byte_pair_count, (b1, merged_pair), count)
                if ind == len(s_token[:-2])-1:
                    new_s_token.append(merged_pair)

            else:
                new_s_token.append(b1)
                new_byte_pair_count = add_pair_neighbors(new_byte_pair_count, (b1, b2), count)
                if ind == len(s_token[:-2])-1:
                    new_s_token.append(b2)
                    new_s_token.append(b3)
                    new_byte_pair_count = add_pair_neighbors(new_byte_pair_count, (b2, b3), count)

        new_s_frequency_table[tuple(new_s_token)] = count

    return new_s_frequency_table, new_byte_pair_count

def get_next_merge(byte_pair_count):
    # Next pair to merge by finding pairs with highest counts, ties are broken lexicographically
    max_count = max(count for count in byte_pair_count.values())
    tied_first_place = [byte_pair for byte_pair, count in byte_pair_count.items() if count == max_count]
    next_pair_merge = max(tied_first_place)
    
    return next_pair_merge

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
    for i in range(6):
        new_merge = get_next_merge(byte_pair_count)
        s_frequency_table, byte_pair_count = apply_merge(s_frequency_table, new_merge)
        print(new_merge)


if __name__ == "__main__":
    import pathlib
    data_path = (pathlib.Path(__file__).resolve().parent.parent) / "data"
    test_file = data_path / 'basic_bpe_example.txt'
    special_tokens=["<|endoftext|>"]

    train_bpe(test_file, 0, special_tokens)