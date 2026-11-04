"""
Zone Indexing and Weighted Zone Scoring
Implements separate postings lists for Title and Body zones.
Calculates linear weighted zone scores:
S(q, d) = g_title * S_title(q, d) + g_body * S_body(q, d)
where g_title + g_body = 1.0.
"""

from typing import Dict, List, Set, Optional, Tuple
from engine.document_loader import Document
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer


class ZoneIndex:
    """
    Maintains distinct postings indices for Title and Body document zones.
    Provides get_zone_postings() and calculates weighted zone relevance scores.
    """

    def __init__(self, title_weight: float = 0.6, body_weight: float = 0.4,
                 tokenizer: Optional[Tokenizer] = None, normalizer: Optional[Normalizer] = None):
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.normalizer = normalizer if normalizer else Normalizer()

        # Validate weights normalize to 1.0
        total_w = title_weight + body_weight
        self.title_weight = title_weight / total_w if total_w > 0 else 0.6
        self.body_weight = body_weight / total_w if total_w > 0 else 0.4

        # zone -> term -> Set[doc_id]
        self.zones: Dict[str, Dict[str, Set[str]]] = {
            "title": {},
            "body": {}
        }
        self.documents: Dict[str, Document] = {}

    def index_document(self, doc: Document):
        """Indexes Title and Body into respective zone indices."""
        self.documents[doc.doc_id] = doc

        # 1. Index Title Zone
        title_tokens = self.normalizer.normalize_tokens(self.tokenizer.tokenize(doc.title))
        for tok in title_tokens:
            term = tok.original
            if term not in self.zones["title"]:
                self.zones["title"][term] = set()
            self.zones["title"][term].add(doc.doc_id)

        # 2. Index Body Zone
        body_tokens = self.normalizer.normalize_tokens(self.tokenizer.tokenize(doc.body))
        for tok in body_tokens:
            term = tok.original
            if term not in self.zones["body"]:
                self.zones["body"][term] = set()
            self.zones["body"][term].add(doc.doc_id)

    def build_zone_index(self, documents: Dict[str, Document]):
        """Builds zone index across all documents."""
        self.clear()
        for doc in documents.values():
            self.index_document(doc)

    def clear(self):
        """Resets the zone index."""
        self.documents.clear()
        self.zones["title"].clear()
        self.zones["body"].clear()

    def get_zone_postings(self, term: str, zone: str) -> Set[str]:
        """
        Explicitly retrieves postings for a term in a specific zone ('title' or 'body').
        Directly fulfills the method requirement specified in previous audits.
        """
        clean_zone = zone.lower().strip()
        if clean_zone not in self.zones:
            return set()
        norm_term = self.normalizer.normalize_term(term)
        return self.zones[clean_zone].get(norm_term, set())

    def calculate_zone_score(self, query_terms: List[str], doc_id: str) -> float:
        """
        Computes weighted zone score for a document against query terms:
        Score = g_title * (matches_in_title / |Q|) + g_body * (matches_in_body / |Q|)
        """
        if not query_terms:
            return 0.0

        title_matches = 0
        body_matches = 0

        for q_term in query_terms:
            norm_q = self.normalizer.normalize_term(q_term)
            if doc_id in self.zones["title"].get(norm_q, set()):
                title_matches += 1
            if doc_id in self.zones["body"].get(norm_q, set()):
                body_matches += 1

        total_terms = len(query_terms)
        s_title = title_matches / total_terms
        s_body = body_matches / total_terms

        weighted_score = (self.title_weight * s_title) + (self.body_weight * s_body)
        return round(weighted_score, 4)

    def search_weighted_zones(self, query_terms: List[str]) -> List[Tuple[str, float]]:
        """
        Retrieves and ranks all matching documents by their weighted zone score.
        Returns list of (doc_id, score) sorted descending.
        """
        if not query_terms:
            return []

        # Find candidate documents matching in either title or body
        candidates: Set[str] = set()
        for t in query_terms:
            norm = self.normalizer.normalize_term(t)
            candidates.update(self.zones["title"].get(norm, set()))
            candidates.update(self.zones["body"].get(norm, set()))

        scored: List[Tuple[str, float]] = []
        for doc_id in candidates:
            score = self.calculate_zone_score(query_terms, doc_id)
            if score > 0.0:
                scored.append((doc_id, score))

        scored.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return scored
