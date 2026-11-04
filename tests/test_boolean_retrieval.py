"""Unit tests for Boolean 2-Pointer Merge Algorithms and Module 5 Query Processor"""

from engine.document_loader import Document
from engine.inverted_index import InvertedIndex
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer
from engine.query_processor import QueryProcessor
from engine.boolean_retrieval import (
    intersect_postings, union_postings, and_not_postings, xor_postings
)


def test_2pointer_merges():
    p1 = ["1", "3", "5", "7", "9"]
    p2 = ["2", "3", "6", "7", "10"]

    # Conjunction (AND)
    assert intersect_postings(p1, p2) == ["3", "7"]

    # Disjunction (OR)
    assert union_postings(p1, p2) == ["1", "2", "3", "5", "6", "7", "9", "10"]

    # Negation difference (AND NOT)
    assert and_not_postings(p1, p2) == ["1", "5", "9"]

    # Exclusive OR (XOR)
    assert xor_postings(p1, p2) == ["1", "2", "5", "6", "9", "10"]


def test_boolean_query_evaluator():
    tok = Tokenizer()
    norm = Normalizer()
    index = InvertedIndex(tok, norm)

    docA = Document(doc_id="A", title="Information Systems", body="Information retrieval systems find relevant documents.")
    docB = Document(doc_id="B", title="Machine Learning", body="Machine learning algorithms learn models from training data.")
    docC = Document(doc_id="C", title="Deep Learning Systems", body="Deep learning neural network systems optimize parameters.")

    index.add_document(docA)
    index.add_document(docB)
    index.add_document(docC)

    qp = QueryProcessor(index, tok, norm)

    # Conjunction
    assert qp.process_boolean_query("systems AND retrieval") == {"A"}

    # Disjunction
    assert qp.process_boolean_query("learning OR retrieval") == {"A", "B", "C"}

    # Negation
    assert qp.process_boolean_query("systems AND NOT deep") == {"A"}

    # Parentheses and Precedence
    assert qp.process_boolean_query("(learning OR deep) AND systems") == {"C"}
