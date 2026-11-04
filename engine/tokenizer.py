"""
Module 2: Tokenization
Converts document and query text into discrete tokens.
Keeps tokenization strictly independent of linguistic normalization.
Preserves original character spans (start, end) and original sequential
stream positions (0, 1, 2, ...) to ensure phrase queries and positional
indexing remain mathematically exact.
"""

import re
from typing import List
from dataclasses import dataclass


@dataclass
class Token:
    """Represents a single token extracted from a text stream."""
    original: str       # Raw text as encountered
    start: int          # Character start offset in source text
    end: int            # Character end offset in source text
    position: int       # Exact sequential stream index (0, 1, 2, ...)


class Tokenizer:
    """
    Standard Alphanumeric Word Tokenizer.
    Splits text on whitespace and punctuation boundaries while
    preserving alphanumeric sequences and internal hyphens/apostrophes.
    """

    # Matches words, numbers, and hyphenated/apostrophe-connected terms
    WORD_PATTERN = re.compile(r"\b[A-Za-z0-9]+(?:['\-_][A-Za-z0-9]+)*\b")

    def __init__(self):
        pass

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenizes input text into a list of Token objects.
        Every identified word receives an incrementing sequential position,
        guaranteeing that positional distance corresponds to true token distance.
        """
        tokens: List[Token] = []
        if not text:
            return tokens

        position_index = 0
        for match in self.WORD_PATTERN.finditer(text):
            word = match.group(0)
            tokens.append(Token(
                original=word,
                start=match.start(),
                end=match.end(),
                position=position_index
            ))
            position_index += 1

        return tokens

    def tokenize_raw(self, text: str) -> List[str]:
        """Convenience method returning raw string tokens."""
        return [t.original for t in self.tokenize(text)]
