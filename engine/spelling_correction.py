"""
Spelling Correction and "Did You Mean?" Suggestions
Implements Levenshtein Minimum Edit Distance via 2D Dynamic Programming matrix,
fast candidate pre-filtering via K-gram overlap, and candidate ranking by
minimum edit distance and term frequency/document popularity.
"""

import re
from typing import List, Dict, Set, Optional, Tuple
from engine.inverted_index import InvertedIndex
from engine.kgram_index import KGramIndex


class SpellingCorrection:
    """
    Vocabulary-based spelling corrector and query suggestion generator.
    Maintains a vocabulary of authentic surface words extracted from documents.
    """

    def __init__(self, index: InvertedIndex, max_edit_distance: int = 2):
        self.index = index
        self.max_edit_distance = max_edit_distance
        self.kgram_index = KGramIndex(k=2)
        # surface_word -> document frequency
        self.surface_vocabulary: Dict[str, int] = {}
        self.build_vocabulary()

    def build_vocabulary(self):
        """
        Extracts authentic English surface words from all indexed documents
        so that suggestions are genuine human-readable words (e.g. 'retrieval', 'information').
        """
        self.surface_vocabulary.clear()
        self.kgram_index.clear()

        # Extract words from all documents
        for doc in self.index.documents.values():
            words = re.findall(r"\b[A-Za-z]{3,}\b", doc.full_text)
            for w in words:
                lowered = w.lower()
                self.surface_vocabulary[lowered] = self.surface_vocabulary.get(lowered, 0) + 1

        # Also include index keys as fallback
        for term in self.index.index.keys():
            if term not in self.surface_vocabulary:
                self.surface_vocabulary[term] = len(self.index.index[term])

        for word in self.surface_vocabulary.keys():
            self.kgram_index.add_term(word)

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Calculates Levenshtein Minimum Edit Distance between two strings
        using standard 2D Dynamic Programming matrix.
        Operations: Insertion (+1), Deletion (+1), Substitution (+1).
        """
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            char1 = s1[i - 1]
            for j in range(1, n + 1):
                char2 = s2[j - 1]
                cost = 0 if char1 == char2 else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # Deletion
                    dp[i][j - 1] + 1,       # Insertion
                    dp[i - 1][j - 1] + cost # Substitution
                )

        return dp[m][n]

    def get_candidates(self, word: str) -> Set[str]:
        """Retrieves candidate vocabulary terms sharing k-grams with word."""
        query_kgrams = self.kgram_index.generate_kgrams(word)
        return self.kgram_index.query_kgrams(query_kgrams)

    def suggest_term(self, word: str) -> Optional[str]:
        """
        Finds the best spelling correction for a single word.
        If word is already in the surface vocabulary, returns None (no correction needed).
        Ranks candidates by:
        1. Minimum edit distance
        2. Maximum document frequency in vocabulary (tie-breaker)
        """
        clean_word = word.lower().strip()
        if not clean_word or len(clean_word) <= 2:
            return None

        # If already known in surface vocabulary, it is correctly spelled
        if clean_word in self.surface_vocabulary and self.surface_vocabulary[clean_word] >= 1:
            return None

        candidates = self.get_candidates(clean_word)
        if not candidates:
            candidates = set(self.surface_vocabulary.keys())

        scored_candidates: List[Tuple[int, int, str]] = []
        for cand in candidates:
            # Quick length filter to prune obvious non-matches
            if abs(len(cand) - len(clean_word)) > self.max_edit_distance:
                continue

            dist = self.levenshtein_distance(clean_word, cand)
            if dist <= self.max_edit_distance:
                freq = self.surface_vocabulary.get(cand, 1)
                # Sort: (min_dist ASC, freq DESC, term ASC)
                scored_candidates.append((dist, -freq, cand))

        if not scored_candidates:
            return None

        scored_candidates.sort()
        best_candidate = scored_candidates[0][2]
        return best_candidate

    def suggest_query(self, query_str: str) -> Optional[str]:
        """
        Generates a Google-style "Did You Mean: ..." suggestion for a multi-word query.
        Returns the corrected query string if any word was corrected; otherwise None.
        Preserves Boolean operators and punctuation.
        """
        words = query_str.split()
        if not words:
            return None

        modified = False
        suggested_words: List[str] = []
        operators = {"AND", "OR", "NOT", "XOR"}

        for w in words:
            # Preserve Boolean operators and quotes/wildcards
            if w.upper() in operators or w.startswith('"') or "*" in w or ":" in w:
                suggested_words.append(w)
                continue

            cleaned = w.strip(".,!?:;\"'()[]{}")
            if not cleaned:
                suggested_words.append(w)
                continue

            correction = self.suggest_term(cleaned)
            if correction and correction.lower() != cleaned.lower():
                suggested_words.append(correction)
                modified = True
            else:
                suggested_words.append(w)

        if modified:
            return " ".join(suggested_words)
        return None
