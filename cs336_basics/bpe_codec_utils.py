import regex as re

def string_to_pretoken(text, special_tokens):
    """Pure pre-tokenization process

    'the cat ate' -> ['the', ' cat', ' ate']
    """

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