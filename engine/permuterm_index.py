"""
Permuterm Index Implementation for Wildcard Retrieval
Rotates terms using terminal symbol '$' to support:
- Trailing wildcards (e.g. mach*)
- Leading wildcards (e.g. *tion)
- Middle wildcards (e.g. in*ation, re*ver)
- Multiple wildcards
Includes mandatory post-filtering against the original pattern.
"""

import re
from typing import Dict, List, Set, Iterable


class PermutermIndex:
    """
    Permuterm Index structure storing all cyclic rotations of vocabulary terms.
    Maps rotation -> original term.
    """

    TERMINAL = "$"

    def __init__(self, vocabulary: Iterable[str] = ()):
        # rotation -> original_term
        self.rotations: Dict[str, str] = {}
        # original_term -> set of rotations
        self.term_to_rotations: Dict[str, List[str]] = {}
        self.build_permuterm(vocabulary)

    def add_term(self, term: str):
        """Adds a single vocabulary term and its cyclic rotations."""
        cleaned = term.strip().lower()
        if not cleaned or cleaned in self.term_to_rotations:
            return

        extended = cleaned + self.TERMINAL
        term_rotations = []
        for i in range(len(extended)):
            rot = extended[i:] + extended[:i]
            self.rotations[rot] = cleaned
            term_rotations.append(rot)

        self.term_to_rotations[cleaned] = term_rotations

    def build_permuterm(self, vocabulary: Iterable[str]):
        """Populates the permuterm index from an iterable of terms."""
        for term in vocabulary:
            self.add_term(term)

    def clear(self):
        """Clears the permuterm index."""
        self.rotations.clear()
        self.term_to_rotations.clear()

    @property
    def size(self) -> int:
        """Returns number of indexed unique terms."""
        return len(self.term_to_rotations)

    def query(self, wildcard_pattern: str) -> List[str]:
        """
        Executes wildcard query using Permuterm rotations:
        1. Translates pattern to rotation prefix lookup
        2. Retrieves candidate terms starting with the prefix
        3. Post-filters candidates against compiled wildcard regex
        """
        clean_pat = wildcard_pattern.strip().lower()
        if not clean_pat:
            return []

        # If pattern has no wildcard, exact lookup
        if "*" not in clean_pat:
            return [clean_pat] if clean_pat in self.term_to_rotations else []

        # 1. Transform wildcard query
        if clean_pat.startswith("*") and not clean_pat[1:].endswith("*") and "*" not in clean_pat[1:]:
            # Leading wildcard: *X -> X$*
            prefix = clean_pat[1:] + self.TERMINAL
        elif clean_pat.endswith("*") and "*" not in clean_pat[:-1]:
            # Trailing wildcard: X* -> $X*
            prefix = self.TERMINAL + clean_pat[:-1]
        elif not clean_pat.startswith("*") and not clean_pat.endswith("*") and clean_pat.count("*") == 1:
            # Middle wildcard: X*Y -> Y$X*
            parts = clean_pat.split("*", 1)
            prefix = parts[1] + self.TERMINAL + parts[0]
        else:
            # Multiple wildcards: e.g. X*Y*Z -> Z$X*
            parts = clean_pat.split("*")
            prefix = parts[-1] + self.TERMINAL + parts[0]

        # 2. Retrieve candidates matching prefix
        candidates: Set[str] = set()
        for rot, orig in self.rotations.items():
            if rot.startswith(prefix):
                candidates.add(orig)

        # 3. Post-filter candidates against the original wildcard pattern
        # Escape special regex characters except '*' which becomes '.*'
        escaped_pattern = re.escape(clean_pat).replace(r"\*", ".*")
        regex = re.compile(f"^{escaped_pattern}$")

        verified = [c for c in candidates if regex.match(c)]
        return sorted(verified)
