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


def build_detailed_byte_pair_count(
        frequency_table
):
    def break_token(token):
        return [char.encode("utf-8") for char in token]
    
    def add_pair_neighbors(detailed_byte_pair_count, byte_pair, prev, next, count):
        # Initialize or update the byte_pair entry
        if byte_pair not in detailed_byte_pair_count:
            detailed_byte_pair_count[byte_pair] = {
                'prev': {},
                'next': {},
                'count': count
            }
        else:
            detailed_byte_pair_count[byte_pair]['count'] += count

        if prev:
            prev_dict = detailed_byte_pair_count[byte_pair]['prev']
            prev_dict[prev] = prev_dict.get(prev, 0) + count

        if next:
            next_dict = detailed_byte_pair_count[byte_pair]['next']
            next_dict[next] = next_dict.get(next, 0) + count

        return detailed_byte_pair_count

    detailed_byte_pair_count = {}

    for token, count in frequency_table.items():
        bytes_token = break_token(token)
        for byte_ind in range(len(bytes_token)-1):
            b1, b2 = bytes_token[byte_ind], bytes_token[byte_ind+1]
            prev = None
            if byte_ind != 0:
                prev = bytes_token[byte_ind-1]
            next= None
            if byte_ind <= len(bytes_token)-3:
                next = bytes_token[byte_ind+2]

            detailed_byte_pair_count = add_pair_neighbors(detailed_byte_pair_count, (b1, b2), prev, next, count)

    return detailed_byte_pair_count

def next_merge(detailed_byte_pair_count):
    # Next pair to merge by finding pairs with highest counts, ties are broken lexicographically
    max_count = max(item['count'] for item in detailed_byte_pair_count.values())
    tied_first_place = [byte_pair for byte_pair, data in detailed_byte_pair_count.items() if data['count'] == max_count]
    next_pair_merge = max(tied_first_place)
    
    b1, b2 = next_pair_merge
    c = b1 + b2
    for b3, count in detailed_byte_pair_count[next_pair_merge]['next'].items():
        detailed_byte_pair_count[(c, b3)] = {'prev': {}, 'next': {},'count': count}



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
    detailed_byte_pair_count = build_detailed_byte_pair_count(frequency_table)

    detailed_byte_pair_count = next_merge(detailed_byte_pair_count)



if __name__ == "__main__":
    import pathlib
    data_path = (pathlib.Path(__file__).resolve().parent.parent) / "data"
    test_file = data_path / 'basic_bpe_example.txt'
    special_tokens=["<|endoftext|>"]

    train_bpe(test_file, 0, special_tokens)