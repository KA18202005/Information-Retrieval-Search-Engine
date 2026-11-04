"""Unit tests for Levenshtein Dynamic Programming Edit Distance & Spelling Correction"""

from engine.document_loader import Document
from engine.inverted_index import InvertedIndex
from engine.spelling_correction import SpellingCorrection


def test_levenshtein_distance():
    sc = SpellingCorrection(InvertedIndex())
    assert sc.levenshtein_distance("kitten", "sitting") == 3
    assert sc.levenshtein_distance("retrieval", "retrieval") == 0
    assert sc.levenshtein_distance("retrival", "retrieval") == 1
    assert sc.levenshtein_distance("infromation", "information") == 2


def test_spelling_suggestions():
    index = InvertedIndex()
    doc = Document(doc_id="1", title="IR", body="Information retrieval and search engine systems.")
    index.add_document(doc)

    sc = SpellingCorrection(index)

    # Correct single word typo: 'retrival' -> 'retrieval'
    suggestion = sc.suggest_term("retrival")
    assert suggestion == "retrieval"

    # Full query suggestion: 'infromation retrival' -> 'information retrieval'
    query_suggestion = sc.suggest_query("infromation retrival")
    assert query_suggestion == "information retrieval"
