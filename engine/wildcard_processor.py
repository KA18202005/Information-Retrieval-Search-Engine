"""
Module 6: Wildcard Processing
Coordinates tolerant retrieval by expanding wildcard query patterns
(leading, trailing, middle, multiple wildcards) using the Permuterm Index
and resolving their postings across the Inverted Index.
"""

from typing import List, Set, Dict
from engine.permuterm_index import PermutermIndex
from engine.inverted_index import InvertedIndex


class WildcardProcessor:
    """
    Evaluates wildcard patterns by finding all matching dictionary terms
    and unioning their postings lists.
    """

    def __init__(self, index: InvertedIndex, permuterm: PermutermIndex):
        self.index = index
        self.permuterm = permuterm

    def expand_wildcard(self, pattern: str) -> List[str]:
        """
        Expands a wildcard pattern into matching vocabulary terms
        via the Permuterm index.
        """
        return self.permuterm.query(pattern)

    def evaluate_wildcard_query(self, pattern: str) -> Dict[str, object]:
        """
        Evaluates a wildcard query, e.g. 'mach*', '*tion', 'in*ation'.
        Returns:
            - pattern: original wildcard string
            - expanded_terms: list of vocabulary matches
            - matching_documents: set of doc_ids containing any expanded term
        """
        expanded = self.expand_wildcard(pattern)
        matched_docs: Set[str] = set()

        for term in expanded:
            postings = self.index.get_postings(term)
            matched_docs.update(postings)

        return {
            "pattern": pattern,
            "expanded_terms": expanded,
            "matching_documents": sorted(matched_docs)
        }
