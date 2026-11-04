"""
Module 4: Inverted Index Construction
Builds and maintains a positional inverted index mapping dictionary terms
to sorted postings lists with positional offsets, term frequencies,
and document frequencies.
"""

from typing import Dict, List, Set, Optional
from engine.document_loader import Document
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer


class InvertedIndex:
    """
    Positional Inverted Index structure.
    Data format:
        index: Dict[str, Dict[str, List[int]]]
            term -> {doc_id: [pos0, pos1, pos2, ...]}
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None, normalizer: Optional[Normalizer] = None):
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.normalizer = normalizer if normalizer else Normalizer()

        # Core Index structures
        # term -> {doc_id: sorted_positions}
        self.index: Dict[str, Dict[str, List[int]]] = {}

        # Document Store: doc_id -> Document
        self.documents: Dict[str, Document] = {}

        # Length tracking: doc_id -> number of indexed terms
        self.doc_lengths: Dict[str, int] = {}
        self.total_tokens: int = 0

    def add_document(self, doc: Document):
        """
        Tokenizes, normalizes, and indexes a single Document object.
        Preserves original sequential token positions.
        """
        self.documents[doc.doc_id] = doc

        # Tokenize full document content (Title + Body)
        raw_tokens = self.tokenizer.tokenize(doc.full_text)
        normalized_tokens = self.normalizer.normalize_tokens(raw_tokens)

        doc_token_count = len(normalized_tokens)
        self.doc_lengths[doc.doc_id] = doc_token_count
        self.total_tokens += doc_token_count

        for tok in normalized_tokens:
            term = tok.original  # Contains the normalized/stemmed term
            pos = tok.position

            if term not in self.index:
                self.index[term] = {}

            if doc.doc_id not in self.index[term]:
                self.index[term][doc.doc_id] = []

            self.index[term][doc.doc_id].append(pos)

    def build_index(self, documents: Dict[str, Document]):
        """Builds index from a collection of documents, resetting previous state."""
        self.clear()
        for doc in documents.values():
            self.add_document(doc)

    def clear(self):
        """Resets the inverted index state."""
        self.index.clear()
        self.documents.clear()
        self.doc_lengths.clear()
        self.total_tokens = 0

    @property
    def num_documents(self) -> int:
        """Total number of indexed documents (N)."""
        return len(self.documents)

    @property
    def vocabulary_size(self) -> int:
        """Total number of unique terms in the vocabulary."""
        return len(self.index)

    @property
    def all_doc_ids(self) -> Set[str]:
        """Returns the set of all document IDs in the index."""
        return set(self.documents.keys())

    def get_postings(self, term: str) -> List[str]:
        """
        Returns sorted list of document IDs containing the normalized term.
        Guarantees sorted order for 2-pointer merge algorithms.
        """
        norm_term = self.normalizer.normalize_term(term)
        if norm_term not in self.index:
            return []
        return sorted(self.index[norm_term].keys())

    def get_positions(self, term: str, doc_id: str) -> List[int]:
        """Returns sorted list of positional offsets for a term within a specific document."""
        norm_term = self.normalizer.normalize_term(term)
        if norm_term in self.index and doc_id in self.index[norm_term]:
            return self.index[norm_term][doc_id]
        return []

    def get_doc_frequency(self, term: str) -> int:
        """Returns document frequency df(t): number of documents containing term."""
        norm_term = self.normalizer.normalize_term(term)
        if norm_term not in self.index:
            return 0
        return len(self.index[norm_term])

    def get_term_frequency(self, term: str, doc_id: str) -> int:
        """Returns term frequency tf(t, d): occurrence count of term in doc_id."""
        return len(self.get_positions(term, doc_id))

    def get_avg_doc_length(self) -> float:
        """Calculates average document length across the collection (avgdl)."""
        if self.num_documents == 0:
            return 0.0
        return self.total_tokens / self.num_documents

    def get_stats(self) -> Dict[str, object]:
        """Returns statistical overview of the index."""
        return {
            "total_documents": self.num_documents,
            "vocabulary_size": self.vocabulary_size,
            "total_indexed_tokens": self.total_tokens,
            "avg_doc_length": round(self.get_avg_doc_length(), 2)
        }
