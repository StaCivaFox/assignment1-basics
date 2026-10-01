import pytest

from cs336_basics.tokenizer import Tokenizer


@pytest.mark.parametrize(
    "text, expected_ids",
    [
        ("ab", [256]),
        ("a<|endoftext|>b", [97, 257, 98]),
        ("ab<|endoftext|>ab", [256, 257, 256]),
        ("<|endoftext|>ab", [257, 256]),
        ("ab<|endoftext|>", [256, 257]),
        ("<|endoftext|>", [257]),
        ("<|endoftext|><|endoftext|>", [257, 257]),
        ("<|endoftext|><|endoftext|><PAD>", [257, 257, 258])
    ],
)
def test_special_token_encoding(text, expected_ids):
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"ab"
    vocab[257] = b"<|endoftext|>"

    merges = [(b"a", b"b")]

    tokenizer = Tokenizer(
        vocab,
        merges,
        special_tokens=["<|endoftext|>", "<PAD>"],
    )

    assert tokenizer.encode(text) == expected_ids