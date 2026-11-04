"""Unit tests for Search Engine Unified Coordinator and End-to-End Queries"""

from engine.search_engine import SearchEngine


def test_search_engine_auto_detection():
    se = SearchEngine()

    # Ranked free-text search
    r1 = se.search("vector space model")
    assert r1["mode"] == "ranked"
    assert r1["total_results"] > 0
    assert r1["execution_time_ms"] >= 0.0

    # Boolean search auto-detection
    r2 = se.search("retrieval AND index")
    assert r2["mode"] == "boolean"
    assert r2["total_results"] > 0

    # Phrase search auto-detection
    r3 = se.search('"inverted index"')
    assert r3["mode"] == "boolean"
    assert r3["total_results"] > 0

    # Wildcard search auto-detection
    r4 = se.search("mach*")
    assert r4["mode"] == "wildcard"
    assert r4["total_results"] > 0

    # Field search auto-detection
    r5 = se.search("author:manning")
    assert r5["mode"] == "field"
    assert r5["total_results"] > 0


def test_search_engine_snippet_generation():
    se = SearchEngine()
    res = se.search("probabilistic BM25")
    assert res["total_results"] > 0
    first_hit = res["results"][0]
    assert "<mark>" in first_hit["snippet"]
    assert "</mark>" in first_hit["snippet"]
