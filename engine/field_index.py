"""
Field-Based Indexing and Retrieval
Maintains dedicated inverted indices per document field:
- title
- author
- category
- body
Supports parametric queries, e.g.:
title:information
author:salton
category:nlp
body:retrieval
"""

import re
from typing import Dict, List, Set, Optional
from engine.document_loader import Document
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer


class FieldIndex:
    """
    Parametric Field Inverted Index.
    Stores separate term dictionaries for each document attribute.
    """

    SUPPORTED_FIELDS = ("title", "author", "category", "body")

    def __init__(self, tokenizer: Optional[Tokenizer] = None, normalizer: Optional[Normalizer] = None):
        self.tokenizer = tokenizer if tokenizer else Tokenizer()
        self.normalizer = normalizer if normalizer else Normalizer()

        # field_name -> term -> Set[doc_id]
        self.field_indices: Dict[str, Dict[str, Set[str]]] = {
            field: {} for field in self.SUPPORTED_FIELDS
        }
        self.documents: Dict[str, Document] = {}

    def index_document(self, doc: Document):
        """Indexes an individual document across all supported fields."""
        self.documents[doc.doc_id] = doc

        field_contents = {
            "title": doc.title,
            "author": doc.author,
            "category": doc.category,
            "body": doc.body
        }

        for field_name, content in field_contents.items():
            if not content:
                continue

            raw_tokens = self.tokenizer.tokenize(content)
            norm_tokens = self.normalizer.normalize_tokens(raw_tokens)

            for tok in norm_tokens:
                term = tok.original
                if term not in self.field_indices[field_name]:
                    self.field_indices[field_name][term] = set()
                self.field_indices[field_name][term].add(doc.doc_id)

    def build_field_index(self, documents: Dict[str, Document]):
        """Indexes all documents across fields."""
        self.clear()
        for doc in documents.values():
            self.index_document(doc)

    def clear(self):
        """Clears all field indices."""
        self.documents.clear()
        for field in self.SUPPORTED_FIELDS:
            self.field_indices[field].clear()

    def search_field(self, field_name: str, term: str) -> Set[str]:
        """
        Retrieves document IDs matching a term in a specific field.
        e.g. search_field('author', 'salton')
        """
        clean_field = field_name.lower().strip()
        if clean_field not in self.field_indices:
            return set()

        norm_term = self.normalizer.normalize_term(term)
        return self.field_indices[clean_field].get(norm_term, set())

    def parse_and_search(self, field_query: str) -> Optional[Set[str]]:
        """
        Parses queries with syntax 'field:term', e.g. 'title:retrieval'.
        Returns matching doc IDs if query contains field syntax; otherwise None.
        """
        match = re.match(r"^(\w+):(.+)$", field_query.strip())
        if not match:
            return None

        field_name, term = match.group(1).lower(), match.group(2).strip()
        if field_name in self.SUPPORTED_FIELDS:
            return self.search_field(field_name, term)
        return None
