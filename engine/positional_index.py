"""
Positional Indexing and Phrase Query Evaluation
Implements positional intersection algorithms for exact multi-term phrase
queries ("...") and word-distance validation using exact sequential token stream offsets.
"""

from typing import List, Set, Dict, Tuple
from engine.inverted_index import InvertedIndex
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer


class PositionalQueryEvaluator:
    """
    Evaluates exact phrase queries against a Positional Inverted Index.
    """

    def __init__(self, index: InvertedIndex, tokenizer: Tokenizer, normalizer: Normalizer):
        self.index = index
        self.tokenizer = tokenizer
        self.normalizer = normalizer

    def evaluate_phrase(self, phrase_str: str) -> Set[str]:
        """
        Evaluates a multi-term exact phrase query, e.g. "information retrieval system".
        Preserves relative positional offsets across all consecutive terms.
        Handles stopwords in phrase by preserving distance offsets without
        requiring stopwords in the dictionary.
        """
        raw_tokens = self.tokenizer.tokenize(phrase_str)
        if not raw_tokens:
            return set()

        if len(raw_tokens) == 1:
            term = self.normalizer.normalize_term(raw_tokens[0].original)
            if term in self.index.index:
                return set(self.index.index[term].keys())
            return set()

        # Extract normalized terms and their relative offsets from the first indexed token
        term_offsets: List[Tuple[str, int]] = []
        base_pos: Optional[int] = None

        for tok in raw_tokens:
            lowered = tok.original.lower().strip()
            # If token is a stopword and stopword removal is enabled, omit from dictionary check
            # but its relative offset to the next token is preserved because tok.position
            # increases monotonically with every word in the stream!
            if self.normalizer.remove_stopwords and lowered in self.normalizer.stopwords:
                continue

            norm = self.normalizer.normalize_term(lowered)
            if not norm:
                continue

            if base_pos is None:
                base_pos = tok.position

            relative_offset = tok.position - base_pos
            term_offsets.append((norm, relative_offset))

        if not term_offsets:
            # If all words in phrase were stopwords, fallback to checking raw terms
            for tok in raw_tokens:
                term_offsets.append((tok.original.lower(), tok.position - raw_tokens[0].position))

        first_term, _ = term_offsets[0]
        if first_term not in self.index.index:
            return set()

        candidate_docs = set(self.index.index[first_term].keys())
        for term, _ in term_offsets[1:]:
            if term not in self.index.index:
                return set()
            candidate_docs &= set(self.index.index[term].keys())

        if not candidate_docs:
            return set()

        # Verify exact relative offsets
        matched_docs: Set[str] = set()
        for doc_id in candidate_docs:
            first_positions = self.index.index[first_term][doc_id]
            for start_pos in first_positions:
                all_match = True
                for term, offset in term_offsets[1:]:
                    term_positions = self.index.index[term][doc_id]
                    expected_pos = start_pos + offset
                    if expected_pos not in term_positions:
                        all_match = False
                        break
                if all_match:
                    matched_docs.add(doc_id)
                    break

        return matched_docs
