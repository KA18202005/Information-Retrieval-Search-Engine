"""Unit tests for Module 6: Wildcard and Permuterm Processing"""

from engine.document_loader import Document
from engine.inverted_index import InvertedIndex
from engine.permuterm_index import PermutermIndex
from engine.wildcard_processor import WildcardProcessor


def test_permuterm_rotations():
    p = PermutermIndex(["retrieval", "machine", "machinery"])
    assert "retrieval$" in p.rotations
    assert "$retrieval" in p.rotations
    assert "machine$" in p.rotations


def test_permuterm_trailing_wildcard():
    p = PermutermIndex(["machine", "machinery", "retrieval"])
    res = p.query("mach*")
    assert res == ["machine", "machinery"]


def test_permuterm_leading_wildcard():
    p = PermutermIndex(["information", "tokenization", "database", "retrieval"])
    res = p.query("*tion")
    assert res == ["information", "tokenization"]


def test_permuterm_middle_wildcard():
    p = PermutermIndex(["information", "infraction", "retriever", "retrieve"])
    # in*ation should match information, but NOT infraction
    assert p.query("in*ation") == ["information"]
    assert p.query("re*ver") == ["retriever"]


def test_wildcard_processor_end_to_end():
    index = InvertedIndex()
    doc1 = Document(doc_id="1", title="Title 1", body="Advanced machinery learning.")
    doc2 = Document(doc_id="2", title="Title 2", body="Classical machine algorithms.")
    doc3 = Document(doc_id="3", title="Title 3", body="Information retrieval indexing.")

    index.add_document(doc1)
    index.add_document(doc2)
    index.add_document(doc3)

    permuterm = PermutermIndex(index.index.keys())
    wp = WildcardProcessor(index, permuterm)

    res = wp.evaluate_wildcard_query("mach*")
    assert "1" in res["matching_documents"]
    assert "2" in res["matching_documents"]
    assert "3" not in res["matching_documents"]
