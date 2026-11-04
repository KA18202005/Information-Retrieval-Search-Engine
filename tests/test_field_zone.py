"""Unit tests for Field Indexing and Zone Weighting"""

from engine.document_loader import Document
from engine.field_index import FieldIndex
from engine.zone_index import ZoneIndex


def test_field_search():
    doc1 = Document(doc_id="1", title="Database Indexing", author="Rudolf Bayer", category="Databases", body="B-trees store records.")
    doc2 = Document(doc_id="2", title="Vector Models", author="Gerard Salton", category="IR", body="Cosine similarity vectors.")

    fi = FieldIndex()
    fi.index_document(doc1)
    fi.index_document(doc2)

    # Search author:salton
    assert fi.search_field("author", "salton") == {"2"}
    assert fi.search_field("author", "bayer") == {"1"}

    # Search title:database
    assert fi.search_field("title", "database") == {"1"}
    assert fi.search_field("title", "vector") == {"2"}

    # Parametric query parsing
    assert fi.parse_and_search("author:salton") == {"2"}


def test_zone_weighted_search():
    doc1 = Document(doc_id="1", title="Information Retrieval", body="Various computer algorithms.")
    doc2 = Document(doc_id="2", title="Machine Learning", body="Information retrieval techniques.")

    # Zone weights: Title=0.6, Body=0.4
    zi = ZoneIndex(title_weight=0.6, body_weight=0.4)
    zi.index_document(doc1)
    zi.index_document(doc2)

    # get_zone_postings verification
    title_postings = zi.get_zone_postings("information", "title")
    assert "1" in title_postings
    assert "2" not in title_postings

    body_postings = zi.get_zone_postings("information", "body")
    assert "2" in body_postings
    assert "1" not in body_postings

    # doc1 has "information" in Title (score = 0.6*1.0 + 0 = 0.6)
    # doc2 has "information" in Body (score = 0 + 0.4*1.0 = 0.4)
    ranked = zi.search_weighted_zones(["information"])
    assert len(ranked) == 2
    assert ranked[0][0] == "1"
    assert ranked[0][1] > ranked[1][1]
