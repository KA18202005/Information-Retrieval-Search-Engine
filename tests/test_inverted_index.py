"""Unit tests for Module 4: Inverted Index Construction"""

from engine.document_loader import Document
from engine.inverted_index import InvertedIndex


def test_inverted_index_basic():
    index = InvertedIndex()
    doc1 = Document(doc_id="1", title="IR System", body="Information retrieval system")
    doc2 = Document(doc_id="2", title="Search System", body="Information search engine")

    index.add_document(doc1)
    index.add_document(doc2)

    assert index.num_documents == 2
    assert "1" in index.all_doc_ids
    assert "2" in index.all_doc_ids

    # Check postings for term "information" -> stemmed to "inform"
    postings = index.get_postings("information")
    assert postings == ["1", "2"]

    # Term "retrieval" -> stemmed to "retriev" in doc 1 only
    assert index.get_postings("retrieval") == ["1"]
    assert index.get_postings("engine") == ["2"]

    # Check document frequency
    assert index.get_doc_frequency("information") == 2
    assert index.get_doc_frequency("retrieval") == 1
    assert index.get_doc_frequency("nonexistent") == 0


def test_inverted_index_sorted_doc_ids():
    index = InvertedIndex()
    for i in [5, 2, 9, 1, 7]:
        doc = Document(doc_id=f"doc_{i}", title=f"Title {i}", body="common keyword in all docs")
        index.add_document(doc)

    postings = index.get_postings("keyword")
    assert postings == sorted(postings)
    assert postings == ["doc_1", "doc_2", "doc_5", "doc_7", "doc_9"]
