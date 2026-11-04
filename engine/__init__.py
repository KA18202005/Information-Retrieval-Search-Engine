"""
Information Retrieval Engine Package
Exports all core and advanced IR components.
"""

from engine.document_loader import DocumentLoader, Document
from engine.tokenizer import Tokenizer, Token
from engine.stemmer import PorterStemmer
from engine.normalizer import Normalizer
from engine.inverted_index import InvertedIndex
from engine.boolean_retrieval import (
    intersect_postings, union_postings, difference_postings, and_not_postings, xor_postings
)
from engine.positional_index import PositionalQueryEvaluator
from engine.query_processor import QueryProcessor
from engine.permuterm_index import PermutermIndex
from engine.kgram_index import KGramIndex
from engine.wildcard_processor import WildcardProcessor
from engine.spelling_correction import SpellingCorrection
from engine.field_index import FieldIndex
from engine.zone_index import ZoneIndex
from engine.ranking import RankedRetrieval
from engine.search_engine import SearchEngine

__all__ = [
    "DocumentLoader",
    "Document",
    "Tokenizer",
    "Token",
    "PorterStemmer",
    "Normalizer",
    "InvertedIndex",
    "intersect_postings",
    "union_postings",
    "difference_postings",
    "and_not_postings",
    "xor_postings",
    "PositionalQueryEvaluator",
    "QueryProcessor",
    "PermutermIndex",
    "KGramIndex",
    "WildcardProcessor",
    "SpellingCorrection",
    "FieldIndex",
    "ZoneIndex",
    "RankedRetrieval",
    "SearchEngine",
]
