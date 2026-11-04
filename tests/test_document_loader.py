"""Unit tests for Module 1: Document Loader"""

import tempfile
from pathlib import Path
from engine.document_loader import DocumentLoader, Document


def test_document_loader_real_corpus():
    loader = DocumentLoader()
    docs = loader.load_all_documents()
    assert len(docs) == 14
    assert "doc_01_tokenization.txt" in docs
    doc1 = docs["doc_01_tokenization.txt"]
    assert doc1.title == "Tokenization and Text Normalization"
    assert doc1.author == "Christopher Manning"
    assert doc1.category == "NLP & Preprocessing"
    assert len(doc1.body) > 50


def test_document_loader_empty_file():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        empty_file = tmp_path / "empty.txt"
        empty_file.write_text("   \n\n  ", encoding="utf-8")

        loader = DocumentLoader(str(tmp_path))
        docs = loader.load_all_documents()
        assert len(docs) == 0


def test_document_loader_malformed_headers():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        bad_file = tmp_path / "doc_bad.txt"
        bad_file.write_text("Random line without headers\nSecond line of content", encoding="utf-8")

        loader = DocumentLoader(str(tmp_path))
        docs = loader.load_all_documents()
        assert len(docs) == 1
        assert "doc_bad.txt" in docs
        assert docs["doc_bad.txt"].title == "Random line without headers"
        assert "Second line" in docs["doc_bad.txt"].body
