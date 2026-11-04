"""
Module 1: Document Loading
Handles reading and parsing documents from the corpus directory.
Preserves document identifiers, metadata headers (Title, Author, Category, Date),
and document body content while gracefully handling empty or malformed files.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """Structured representation of a document in the search engine corpus."""
    doc_id: str
    title: str
    body: str
    author: str = "Unknown"
    category: str = "General"
    date: str = ""
    raw_text: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        """Combined title and body text for holistic indexing."""
        return f"{self.title}\n{self.body}".strip()


class DocumentLoader:
    """
    Loads text documents from a specified directory.
    Supports RFC-822 style headers (e.g. Title: ..., Author: ...)
    or fallback to raw text content with filename as default title.
    """

    def __init__(self, corpus_dir: Optional[str] = None):
        self.corpus_dir = Path(corpus_dir) if corpus_dir else Path(__file__).resolve().parent.parent / "data" / "documents"

    def parse_document_content(self, doc_id: str, content: str) -> Optional[Document]:
        """
        Parses raw text content into a structured Document object.
        Extracts headers if present; otherwise treats entire content as body.
        Returns None for empty documents.
        """
        stripped_content = content.strip()
        if not stripped_content:
            return None

        lines = content.splitlines()
        headers: Dict[str, str] = {}
        body_lines: List[str] = []
        in_headers = True

        for idx, line in enumerate(lines):
            stripped_line = line.strip()
            if in_headers:
                if not stripped_line:
                    # Empty line marks the boundary between headers and body
                    in_headers = False
                    continue
                # Check for "Header: Value" format
                if ":" in line and not line.startswith(" "):
                    key, val = line.split(":", 1)
                    lower_key = key.strip().lower()
                    if lower_key in ("title", "author", "category", "date"):
                        headers[lower_key] = val.strip()
                        continue
                # If first non-empty line isn't a known header, treat entire text as body
                in_headers = False
                body_lines.append(line)
            else:
                body_lines.append(line)

        body_text = "\n".join(body_lines).strip()
        title = headers.get("title") or (body_lines[0].strip() if body_lines else Path(doc_id).stem)
        author = headers.get("author", "Unknown")
        category = headers.get("category", "General")
        date_str = headers.get("date", "")

        # If body is empty after stripping headers, use entire content
        if not body_text:
            body_text = stripped_content

        return Document(
            doc_id=doc_id,
            title=title,
            body=body_text,
            author=author,
            category=category,
            date=date_str,
            raw_text=content,
            metadata=headers
        )

    def load_document(self, file_path: Path) -> Optional[Document]:
        """Loads and parses an individual document file."""
        if not file_path.is_file():
            return None
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            return self.parse_document_content(file_path.name, content)
        except Exception:
            return None

    def load_all_documents(self) -> Dict[str, Document]:
        """
        Scans corpus_dir for .txt documents, loading and indexing them by doc_id.
        Gracefully ignores empty or unreadable files.
        """
        documents: Dict[str, Document] = {}
        if not self.corpus_dir.is_dir():
            return documents

        # Sort files deterministically by filename
        for file_path in sorted(self.corpus_dir.glob("*.txt")):
            doc = self.load_document(file_path)
            if doc is not None and doc.doc_id not in documents:
                documents[doc.doc_id] = doc

        return documents
