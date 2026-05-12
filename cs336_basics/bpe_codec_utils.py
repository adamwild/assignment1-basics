import regex as re
import pathlib

def string_to_pretoken(text, special_tokens):
    """Pure pre-tokenization process

    'the cat ate' -> ['the', ' cat', ' ate']
    """
    if special_tokens is None:
        special_tokens = []
    # Removing special tokens before pre-tokenization
    docs = re.split("|".join([re.escape(token) for token in special_tokens]), text)

    # Pre-tokenize
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    list_pretokens = []
    for doc in docs:
        for match in re.finditer(PAT, doc):
            pre_token = match.group()
            list_pretokens.append(pre_token)

    return list_pretokens

def hash_merges(merges):
    """Turn merges into a dict

    Args:
        merges (list[tuple[bytes, bytes]]): _description_
    """
    return {(b1, b2): ind for ind, (b1, b2) in enumerate(merges)}

def reverse_vocab(vocab):
    """Reverse the vocab dict

    Args:
        vocab (dict[int, bytes]): _description_
    """
    return {token: num for num, token in vocab.items()}

def get_filepaths_checkpoints(checkpoint_name):
    """Given a checkpoint (usually the dataset name that we used for training BPE)
    We return strings since it is the requirement of the Tokenizer method

    Args:
        checkpoint_name (str): 'owt' or 'TinyStories_10000'
    """
    checkpoints_path = (pathlib.Path(__file__).resolve().parent.parent) / 'cs336_basics' / 'checkpoints' / checkpoint_name

    vocab_filepath = str(checkpoints_path / 'vocab')
    merges_filepath = str(checkpoints_path / 'merges')

    return vocab_filepath, merges_filepath
