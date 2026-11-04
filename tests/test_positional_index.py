"""Unit tests for Positional Indexing and Phrase Queries"""

from engine.document_loader import Document
from engine.inverted_index import InvertedIndex
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer
from engine.positional_index import PositionalQueryEvaluator


def test_phrase_query_exact_order():
    tok = Tokenizer()
    norm = Normalizer()
    index = InvertedIndex(tok, norm)

    doc1 = Document(doc_id="1", title="Doc 1", body="Information retrieval system is efficient.")
    doc2 = Document(doc_id="2", title="Doc 2", body="A retrieval system for information is useful.")

    index.add_document(doc1)
    index.add_document(doc2)

    evaluator = PositionalQueryEvaluator(index, tok, norm)

    # "information retrieval system" occurs in doc1 consecutively, but not in doc2
    matched = evaluator.evaluate_phrase("information retrieval system")
    assert matched == {"1"}


def test_phrase_query_with_stopwords():
    """
    CRITICAL TEST: Ensure exact phrase with stop words matches only the true document.
    """
    tok = Tokenizer()
    norm = Normalizer()
    index = InvertedIndex(tok, norm)

    doc1 = Document(doc_id="1", title="Doc 1", body="Tokenization and text normalization in Python.")
    doc2 = Document(doc_id="2", title="Doc 2", body="Tokenization text normalization in Python.")

    index.add_document(doc1)
    index.add_document(doc2)

    evaluator = PositionalQueryEvaluator(index, tok, norm)

    # "Tokenization and text" has relative offset 2 between tokenization and text
    match1 = evaluator.evaluate_phrase("Tokenization and text")
    assert match1 == {"1"}

    # "Tokenization text" has immediate adjacency (offset 1)
    match2 = evaluator.evaluate_phrase("Tokenization text")
    assert match2 == {"2"}
