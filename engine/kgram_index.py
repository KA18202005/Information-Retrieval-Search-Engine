"""
K-Gram Index Implementation
Generates character k-grams with boundary markers ($) and indexes vocabulary terms.
Calculates Jaccard similarity for candidate filtering in tolerant retrieval and spelling correction.
"""

from typing import Dict, List, Set, Iterable


class KGramIndex:
    """
    K-Gram Inverted Index.
    Maps character k-grams -> Set of matching vocabulary terms.
    """

    def __init__(self, k: int = 3, vocabulary: Iterable[str] = ()):
        self.k = k
        # kgram -> Set[term]
        self.index: Dict[str, Set[str]] = {}
        self.vocabulary: Set[str] = set()
        self.build_kgram_index(vocabulary)

    def generate_kgrams(self, term: str) -> List[str]:
        """Generates k-grams for a term wrapped with boundary markers '$'."""
        extended = f"${term.lower().strip()}$"
        if len(extended) < self.k:
            return [extended]
        return [extended[i:i + self.k] for i in range(len(extended) - self.k + 1)]

    def add_term(self, term: str):
        """Indexes all k-grams for a single term."""
        cleaned = term.lower().strip()
        if not cleaned or cleaned in self.vocabulary:
            return

        self.vocabulary.add(cleaned)
        kgrams = self.generate_kgrams(cleaned)
        for kg in kgrams:
            if kg not in self.index:
                self.index[kg] = set()
            self.index[kg].add(cleaned)

    def build_kgram_index(self, vocabulary: Iterable[str]):
        """Populates k-gram index from vocabulary terms."""
        for term in vocabulary:
            self.add_term(term)

    def clear(self):
        """Clears the k-gram index."""
        self.index.clear()
        self.vocabulary.clear()

    def get_terms_for_kgram(self, kgram: str) -> Set[str]:
        """Returns vocabulary terms containing the specified k-gram."""
        return self.index.get(kgram, set())

    @staticmethod
    def jaccard_similarity(kgrams1: Set[str], kgrams2: Set[str]) -> float:
        """
        Computes Jaccard coefficient between two k-gram sets:
        J(A, B) = |A ∩ B| / |A ∪ B|
        """
        union_len = len(kgrams1 | kgrams2)
        if union_len == 0:
            return 0.0
        return len(kgrams1 & kgrams2) / union_len

    def query_kgrams(self, query_kgrams: List[str], min_overlap: int = 1) -> Set[str]:
        """Finds candidate terms that share at least min_overlap k-grams with the query."""
        if not query_kgrams:
            return set()

        candidates: Set[str] = set()
        for kg in query_kgrams:
            terms = self.get_terms_for_kgram(kg)
            candidates |= terms

        return candidates
