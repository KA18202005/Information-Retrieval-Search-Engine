"""Unit tests for Module 2: Tokenizer"""

from engine.tokenizer import Tokenizer


def test_tokenizer_basic():
    tok = Tokenizer()
    text = "Information retrieval is great!"
    tokens = tok.tokenize(text)
    assert len(tokens) == 4
    words = [t.original for t in tokens]
    assert words == ["Information", "retrieval", "is", "great"]

    # Sequential positions must be 0, 1, 2, 3
    positions = [t.position for t in tokens]
    assert positions == [0, 1, 2, 3]


def test_tokenizer_spans():
    tok = Tokenizer()
    text = "Search engines"
    tokens = tok.tokenize(text)
    assert len(tokens) == 2
    assert text[tokens[0].start:tokens[0].end] == "Search"
    assert text[tokens[1].start:tokens[1].end] == "engines"


def test_tokenizer_hyphenated():
    tok = Tokenizer()
    text = "state-of-the-art TF-IDF model"
    tokens = tok.tokenize(text)
    words = [t.original for t in tokens]
    assert "state-of-the-art" in words or "state" in words
    assert "TF-IDF" in words or "TF" in words
    # Check that positions are strictly sequential
    for i, t in enumerate(tokens):
        assert t.position == i
