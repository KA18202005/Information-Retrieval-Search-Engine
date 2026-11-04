"""Unit tests for Module 3: Normalizer and Positional Preservation"""

from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer
from engine.stemmer import PorterStemmer


def test_porter_stemmer():
    stemmer = PorterStemmer()
    assert stemmer.stem("retrieval") == "retriev"
    assert stemmer.stem("retrieves") == "retriev"
    assert stemmer.stem("retrieved") == "retriev"
    assert stemmer.stem("connecting") == "connect"
    assert stemmer.stem("generalization") == "gener"


def test_normalizer_case_folding():
    norm = Normalizer(use_stemming=False, remove_stopwords=False)
    assert norm.normalize_term("Information") == "information"


def test_normalizer_stopword_removal_preserves_position():
    """
    CRITICAL TEST: Verify that removing stop words does NOT alter
    the original sequential stream index of retained tokens.
    """
    tok = Tokenizer()
    norm = Normalizer(use_stemming=True, remove_stopwords=True)

    text = "Tokenization and Text Normalization"
    raw_tokens = tok.tokenize(text)
    # raw positions:
    # Tokenization: 0, and: 1, Text: 2, Normalization: 3
    normalized = norm.normalize_tokens(raw_tokens)

    # 'and' is a stopword and is dropped
    assert len(normalized) == 3

    assert normalized[0].original == "token"
    assert normalized[0].position == 0

    assert normalized[1].original == "text"
    assert normalized[1].position == 2  # MUST REMAIN 2, NOT 1!

    assert normalized[2].original == "normal"
    assert normalized[2].position == 3  # MUST REMAIN 3, NOT 2!
