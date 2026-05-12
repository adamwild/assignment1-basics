
import pickle
from typing import Iterable, Iterator
from cs336_basics.bpe_tokenizer import break_token
from cs336_basics.bpe_codec_utils import string_to_pretoken, hash_merges, reverse_vocab, get_filepaths_checkpoints

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens.

        Args:
            vocab (dict[int, bytes]): _description_
            merges (list[tuple[bytes, bytes]]): _description_
            special_tokens (list[str], optional): _description_. Defaults to None.
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

        self.merge_to_ind = hash_merges(merges)
        self.token_to_int = reverse_vocab(vocab)

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens.

        Args:
            vocab_filepath (str): _description_
            merges_filepath (str): _description_
            special_tokens (list[str], optional): _description_. Defaults to None.
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text (str): _description_

        Returns:
            list[int]: _description_
        """
        # Pre-tokenization
        pre_tokens = string_to_pretoken(text, special_tokens=self.special_tokens)

        int_sequence = []

        for pre_token in pre_tokens:
            if self.special_tokens and pre_token in self.special_tokens:
                did_find_merge = False
                pre_token = [pre_token.encode("utf-8")]

            else:
                did_find_merge = True
                pre_token = list(break_token(pre_token))

            # Loop until we can not reduce the pre-token with any available merge
            while did_find_merge:
                curr_ind_merge = None

                # Tracks the indices of the current pre-token where we will apply merge
                indexes_merge = []

                for ind in range(len(pre_token)-1):
                    b1, b2 = pre_token[ind], pre_token[ind+1]

                    ind_merge = self.merge_to_ind.get((b1, b2), None)
                    if ind_merge is not None and (curr_ind_merge is None or ind_merge < curr_ind_merge):
                        b_merge = b1 + b2
                        curr_ind_merge = ind_merge
                        indexes_merge = [ind]

                    elif ind_merge == curr_ind_merge:
                        indexes_merge.append(ind)

                did_find_merge = curr_ind_merge is not None

                # Apply the merges
                if did_find_merge:
                    for decal, ind_merge in enumerate(indexes_merge):
                        pre_token = pre_token[:ind_merge-decal] + [b_merge] + pre_token[ind_merge+2-decal:]

            # Turn each element into a sequence of integer, our final tokens
            for token in pre_token:
                int_sequence.append(self.token_to_int[token])

        return int_sequence

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.

        Args:
            iterable (Iterable[str]): _description_

        Yields:
            Iterator[int]: _description_
        """
        for string in iterable:
            int_sequence = self.encode(string)

            for token_int in int_sequence:
                yield token_int
    
    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): _description_

        Returns:
            str: _description_
        """

        decoded = b""
        for token_int in ids:
            decoded += self.vocab[token_int]

        try:
            return decoded.decode("utf-8")
        except UnicodeDecodeError:
            return decoded



if __name__ == "__main__":
    # Test with:
    # uv run pytest tests/test_tokenizer.py

    # python -m cs336_basics.bpe_codec

    import pathlib
    from tests.common import FIXTURES_PATH

    VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
    MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"
    data_path = (pathlib.Path(__file__).resolve().parent.parent) / "data"
    fixtures_path = (pathlib.Path(__file__).resolve().parent.parent) / 'tests' / "fixtures"
    checkpoints_path = (pathlib.Path(__file__).resolve().parent.parent) / 'cs336_basics' / 'checkpoints'

    # toy example
    vocab_example = {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}
    merges_example = [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]
    text_example = 'the cat ate'

    """from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH, merges_path=MERGES_PATH, special_tokens=["<|endoftext|>"]
    )"""

    # from tests.test_tokenizer import test_roundtrip_unicode_string_with_special_tokens

    # test_roundtrip_unicode_string_with_special_tokens()