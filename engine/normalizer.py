"""
Module 3: Text Normalization and Case Folding
Implements lowercase case-folding, punctuation normalization,
standard stop-word filtering, and Porter stemming.
Enforces the critical IR requirement: documents and queries pass through
strictly compatible normalization rules, while original token stream
positions are preserved intact without distance compression.
"""

from typing import List, Set, Optional
from engine.tokenizer import Token
from engine.stemmer import PorterStemmer


# Standard Classical IR English Stopwords
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't",
    "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have",
    "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our",
    "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's",
    "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
    "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't",
    "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't",
    "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves"
}


class Normalizer:
    """
    Modular text normalizer supporting:
    - Case folding (lowercasing)
    - Stopword filtering (with position preservation)
    - Morphological suffix stripping via PorterStemmer
    """

    def __init__(self, use_stemming: bool = True, remove_stopwords: bool = True, custom_stopwords: Optional[Set[str]] = None):
        self.use_stemming = use_stemming
        self.remove_stopwords = remove_stopwords
        self.stopwords = custom_stopwords if custom_stopwords is not None else DEFAULT_STOPWORDS
        self.stemmer = PorterStemmer()

    def normalize_term(self, word: str) -> str:
        """
        Normalizes a single word: case folds and applies stemming.
        Does not apply stopword stripping here; used for term lookups.
        """
        lowered = word.lower().strip()
        if not lowered:
            return ""
        if self.use_stemming:
            return self.stemmer.stem(lowered)
        return lowered

    def is_stopword(self, word: str) -> bool:
        """Checks if a term is in the active stopword list."""
        return word.lower().strip() in self.stopwords

    def process_query_term(self, word: str) -> Optional[str]:
        """
        Normalizes a query term. If remove_stopwords is enabled and the word
        is a stopword, returns None.
        """
        lowered = word.lower().strip()
        if not lowered:
            return None
        if self.remove_stopwords and lowered in self.stopwords:
            return None
        return self.normalize_term(lowered)

    def normalize_tokens(self, tokens: List[Token]) -> List[Token]:
        """
        Normalizes a stream of Token objects.
        CRITICAL: Each token's original 'position' index is preserved intact!
        When stopword removal is active, non-stopword tokens retain their original
        sequential index so that phrase distance semantics are not corrupted.
        """
        normalized: List[Token] = []
        for tok in tokens:
            lowered = tok.original.lower().strip()
            if not lowered:
                continue

            if self.remove_stopwords and lowered in self.stopwords:
                # Token is a stopword; skip dictionary inclusion, but its position
                # was already recorded in tok.position, preserving stream gaps!
                continue

            term = self.stemmer.stem(lowered) if self.use_stemming else lowered
            normalized.append(Token(
                original=term,            # Storing normalized term
                start=tok.start,
                end=tok.end,
                position=tok.position     # EXACT ORIGINAL STREAM POSITION RETAINED
            ))

        return normalized
