

from typing import Iterable, Iterator

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        """Construct a tokenizer from a given vocabulary, list of merges,
        and (optionally) a list of special tokens.

        Args:
            vocab (dict[int, bytes]): _description_
            merges (list[tuple[bytes, bytes]]): _description_
            special_tokens (list[str], optional): _description_. Defaults to None.
        """
        print(vocab)

    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens.

        Args:
            vocab_filepath (str): _description_
            merges_filepath (str): _description_
            special_tokens (list[str], optional): _description_. Defaults to None.
        """
        pass
    
    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs.

        Args:
            text (str): _description_

        Returns:
            list[int]: _description_
        """
        pass

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.

        Args:
            iterable (Iterable[str]): _description_

        Yields:
            Iterator[int]: _description_
        """
        pass
    
    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text.

        Args:
            ids (list[int]): _description_

        Returns:
            str: _description_
        """
        pass



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


    from tests.test_tokenizer import get_tokenizer_from_vocab_merges_path

    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
    )

    