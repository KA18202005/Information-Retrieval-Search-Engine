"""Unit tests for Vector Space Model (TF-IDF & Cosine Similarity)"""

import math
from engine.document_loader import Document
from engine.inverted_index import InvertedIndex
from engine.ranking import RankedRetrieval


def test_ranking_formulas():
    index = InvertedIndex()
    doc1 = Document(doc_id="1", title="IR", body="Information retrieval retrieval retrieval.")
    doc2 = Document(doc_id="2", title="ML", body="Information machine learning algorithms.")

    index.add_document(doc1)
    index.add_document(doc2)

    vsm = RankedRetrieval(index)

    # Sublinear TF test: wf = 1 + ln(tf)
    assert vsm.sublinear_tf(0) == 0.0
    assert vsm.sublinear_tf(1) == 1.0
    assert round(vsm.sublinear_tf(3), 4) == round(1.0 + math.log(3), 4)

    # Smoothed IDF test: idf = ln((N+1)/(df+1)) + 1.0
    # N=2, "retrieval" appears in doc1 only -> df=1
    expected_idf = math.log((2.0 + 1.0) / (1.0 + 1.0)) + 1.0
    assert round(vsm.get_idf("retrieval"), 4) == round(expected_idf, 4)

    # Ranked retrieval test
    results = vsm.compute_cosine_similarity(["retrieval"])
    assert len(results) == 1
    assert results[0][0] == "1"
    assert results[0][1] > 0.0
