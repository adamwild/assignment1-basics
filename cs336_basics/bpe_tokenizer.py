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

def apply_merge(s_frequency_table, merge_pair):
    """_summary_

    Args:
        s_frequency_table (dict[tuple[bytes], int]): {(b'l', b'o', b'w'): 1, (b' ', b'l', b'o', b'w'): 4,
        merge_pair (tuple[bytes]): (b's', b't')
    """
    new_s_frequency_table = {}
    new_byte_pair_count = {}
    
    b1_target, b2_target = merge_pair
    merged_token = b1_target + b2_target

    for s_token, count in s_frequency_table.items():
        # Handle single-byte tokens immediately
        if len(s_token) == 1:
            # new_byte_pair_count[s_token] = new_byte_pair_count.get(s_token, 0) + count
            continue

        if len(s_token) == 2:
            b1, b2 = s_token

            if (b1, b2) == merge_pair:
                # new_byte_pair_count[merged_token] = new_byte_pair_count.get(merged_token, 0) + count
                # new_s_frequency_table[merged_token] = new_s_frequency_table.get(merged_token, 0) + count
                pass
            else:
                new_byte_pair_count[(b1, b2)] = new_byte_pair_count.get((b1, b2), 0) + count
                new_s_frequency_table[(b1, b2)] = new_s_frequency_table.get((b1, b2), 0) + count
            continue

        # 3 or more bytes
        new_s_token = []

        ind_byte = 0

        # If b1 belongs to a merged pair from his predecessor, consider b1 as the merged bytes
        rename_b1 = False

        while ind_byte < len(s_token)-1:
            b1, b2 = s_token[ind_byte], s_token[ind_byte+1]

            if rename_b1:
                b1 = merged_token
                rename_b1 = False

            # Last two bytes
            if ind_byte == len(s_token)-2:
                new_byte_pair_count[(b1, b2)] = new_byte_pair_count.get((b1, b2), 0) + count
                new_s_token.append(b1)
                new_s_token.append(b2)
                break

            b3 = s_token[ind_byte+2]

            # We write the possible cases, b_i means a byte that doesn't merge, c_i is a byte that belongs to a mergure
            if (b1, b2) != merge_pair:
                new_s_token.append(b1)

                # b1, b2, b3
                if (b2, b3) != merge_pair:
                    new_byte_pair_count[(b1, b2)] = new_byte_pair_count.get((b1, b2), 0) + count
                    ind_byte += 1

                # b1, c2, c3
                else:
                    new_byte_pair_count[(b1, merged_token)] = new_byte_pair_count.get((b1, merged_token), 0) + count
                    ind_byte += 2

                    # We restart the loop on c3 which belongs to a merge
                    rename_b1 = True

                    # c2, c3 is the end of the token
                    if ind_byte == len(s_token)-1:
                        new_s_token.append(merged_token)

            else:
                new_s_token.append(merged_token)

                if ind_byte + 3 <= len(s_token)-1:
                    b4 = s_token[ind_byte+3]

                    # c1, c2, b3, b4
                    if (b3, b4) != merge_pair:
                        new_byte_pair_count[(merged_token, b3)] = new_byte_pair_count.get((merged_token, b3), 0) + count
                        ind_byte += 2

                    # c1, c2, c3, c4
                    else:
                        new_byte_pair_count[(merged_token, merged_token)] = new_byte_pair_count.get((merged_token, merged_token), 0) + count
                        ind_byte += 3
                        rename_b1 = True

                # c1, c2, b3]
                else:
                    new_byte_pair_count[(merged_token, b3)] = new_byte_pair_count.get((merged_token, b3), 0) + count
                    ind_byte += 2
                    new_s_token.append(b3)

        new_s_frequency_table[tuple(new_s_token)] = new_s_frequency_table.get(tuple(new_s_token), 0) + count

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

