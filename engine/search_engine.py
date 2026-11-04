"""
Unified Search Engine Coordinator
Connects all IR modules into a coherent retrieval pipeline:
- Document Loading & Indexing
- Boolean Retrieval (conjunction, disjunction, negation)
- Exact Multi-term Phrase Retrieval
- Wildcard Search via Permuterm Index
- Parametric Field Search (title:, author:, category:, body:)
- Weighted Zone Search
- Vector Space Model (TF-IDF & Cosine Similarity)
- Spelling Correction ("Did You Mean: ...")
- Dynamic KWIC Snippets with query term highlighting
"""

import time
import re
import html
from typing import Dict, List, Set, Optional, Tuple, Any
from pathlib import Path

from engine.document_loader import DocumentLoader, Document
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer
from engine.inverted_index import InvertedIndex
from engine.query_processor import QueryProcessor
from engine.permuterm_index import PermutermIndex
from engine.wildcard_processor import WildcardProcessor
from engine.spelling_correction import SpellingCorrection
from engine.field_index import FieldIndex
from engine.zone_index import ZoneIndex
from engine.ranking import RankedRetrieval


class SearchEngine:
    """
    Central search orchestrator managing document collections,
    multi-paradigm query evaluation, relevance scoring, and snippet generation.
    """

    def __init__(self, corpus_dir: Optional[str] = None):
        self.loader = DocumentLoader(corpus_dir)
        self.tokenizer = Tokenizer()
        self.normalizer = Normalizer()
        self.index = InvertedIndex(self.tokenizer, self.normalizer)
        self.query_processor = QueryProcessor(self.index, self.tokenizer, self.normalizer)
        self.permuterm = PermutermIndex()
        self.wildcard_processor = WildcardProcessor(self.index, self.permuterm)
        self.spelling_correction = SpellingCorrection(self.index)
        self.field_index = FieldIndex(self.tokenizer, self.normalizer)
        self.zone_index = ZoneIndex(tokenizer=self.tokenizer, normalizer=self.normalizer)
        self.ranked_retrieval = RankedRetrieval(self.index, self.normalizer)

        # Initialize by loading existing corpus
        self.initialize_index()

    def initialize_index(self):
        """Loads corpus documents and constructs all indices."""
        docs = self.loader.load_all_documents()
        self.index.build_index(docs)
        self.field_index.build_field_index(docs)
        self.zone_index.build_zone_index(docs)

        # Build Permuterm vocabulary
        self.permuterm.clear()
        self.permuterm.build_permuterm(self.index.index.keys())

        # Build Spelling Corrector vocabulary
        self.spelling_correction.build_vocabulary()

        # Update Ranked Retrieval vector norms
        self.ranked_retrieval.compute_document_vector_norms()

    def create_kwic_snippet(self, content: str, query_terms: Set[str], window_size: int = 30) -> str:
        """
        Generates Keyword-in-Context (KWIC) snippet.
        Finds the word window containing the highest concentration of matched query terms.
        Safely escapes HTML and marks query terms with <mark> tags.
        """
        words = content.split()
        if not words:
            return ""

        # Normalize words to compare against normalized query terms
        clean_words = [re.sub(r"[^\w]", "", w.lower()) for w in words]
        stemmed_words = [self.normalizer.normalize_term(w) for w in clean_words]

        norm_query_set = {self.normalizer.normalize_term(t) for t in query_terms if t}

        # Find window with maximum query term occurrences
        best_start = 0
        max_matches = 0

        for i in range(len(words)):
            window = stemmed_words[i:i + window_size]
            current_matches = sum(1 for w in window if w in norm_query_set)
            if current_matches > max_matches:
                max_matches = current_matches
                best_start = i

        start_idx = max(0, best_start - 3)
        end_idx = min(len(words), start_idx + window_size)
        slice_words = words[start_idx:end_idx]

        # Highlight matching words safely
        highlighted = []
        for w in slice_words:
            cleaned = re.sub(r"[^\w]", "", w.lower())
            stemmed = self.normalizer.normalize_term(cleaned)
            escaped_w = html.escape(w)
            if stemmed in norm_query_set and cleaned not in self.normalizer.stopwords:
                highlighted.append(f"<mark>{escaped_w}</mark>")
            else:
                highlighted.append(escaped_w)

        prefix = "... " if start_idx > 0 else ""
        suffix = " ..." if end_idx < len(words) else ""
        return prefix + " ".join(highlighted) + suffix

    def search(self, query_str: str, mode: str = "auto") -> Dict[str, Any]:
        """
        Unified search endpoint.
        Modes:
        - 'auto': Automatically infers Boolean, Wildcard, Field, Phrase, or Ranked retrieval
        - 'boolean': Explicit Boolean AST parsing (AND, OR, NOT, XOR, parentheses)
        - 'ranked': Pure TF-IDF Cosine Vector Space Model
        - 'zone': Weighted Zone scoring (Title + Body)
        - 'field': Parametric search (title:..., author:..., etc.)
        - 'wildcard': Permuterm wildcard expansion
        """
        start_time = time.perf_counter()
        clean_query = query_str.strip()

        if not clean_query:
            return {
                "query": "",
                "mode": mode,
                "total_results": 0,
                "execution_time_ms": 0.0,
                "did_you_mean": None,
                "results": []
            }

        # 1. Spelling Correction Suggestion
        did_you_mean = self.spelling_correction.suggest_query(clean_query)

        # 2. Query classification for 'auto' mode
        has_boolean = any(op in clean_query.split() for op in ("AND", "OR", "NOT", "XOR"))
        has_field = bool(re.search(r"\b(title|author|category|body):", clean_query, re.I))
        has_wildcard = "*" in clean_query
        has_phrase = '"' in clean_query

        active_mode = mode
        if mode == "auto":
            if has_field:
                active_mode = "field"
            elif has_wildcard:
                active_mode = "wildcard"
            elif has_boolean or has_phrase:
                active_mode = "boolean"
            else:
                active_mode = "ranked"

        matched_doc_ids: Set[str] = set()
        doc_scores: Dict[str, float] = {}
        extracted_query_terms: List[str] = []

        # 3. Mode Evaluation
        if active_mode == "field":
            # Parametric Field Search
            field_res = self.field_index.parse_and_search(clean_query)
            if field_res is not None:
                matched_doc_ids = field_res
            else:
                matched_doc_ids = set()
            # Extract search term for snippet
            term_match = re.search(r":(.+)$", clean_query)
            if term_match:
                extracted_query_terms = [term_match.group(1).strip()]

        elif active_mode == "wildcard":
            # Wildcard search via Permuterm
            wild_res = self.wildcard_processor.evaluate_wildcard_query(clean_query)
            matched_doc_ids = set(wild_res["matching_documents"])
            extracted_query_terms = wild_res["expanded_terms"]

        elif active_mode == "boolean":
            # Boolean & Phrase AST evaluation
            matched_doc_ids = self.query_processor.process_boolean_query(clean_query)
            # Extract query terms for ranking & snippet
            raw_tokens = self.query_processor.tokenize_query(clean_query)
            for ttype, val in raw_tokens:
                if ttype == "TERM":
                    extracted_query_terms.append(val)
                elif ttype == "PHRASE":
                    extracted_query_terms.extend(val.split())

            # Score boolean results via VSM Cosine so results are neatly ordered
            if matched_doc_ids and extracted_query_terms:
                cosine_ranked = self.ranked_retrieval.compute_cosine_similarity(extracted_query_terms)
                for doc_id, score in cosine_ranked:
                    if doc_id in matched_doc_ids:
                        doc_scores[doc_id] = score

        elif active_mode == "zone":
            # Weighted Zone Scoring
            terms = [w for w in clean_query.split() if w.lower() not in self.normalizer.stopwords]
            extracted_query_terms = terms
            zone_ranked = self.zone_index.search_weighted_zones(terms)
            for doc_id, score in zone_ranked:
                matched_doc_ids.add(doc_id)
                doc_scores[doc_id] = score

        else:
            # Ranked Search (Vector Space Model Cosine Similarity)
            terms = [w for w in clean_query.split() if w.lower() not in self.normalizer.stopwords]
            if not terms:
                terms = clean_query.split()
            extracted_query_terms = terms
            cosine_ranked = self.ranked_retrieval.compute_cosine_similarity(terms)
            for doc_id, score in cosine_ranked:
                matched_doc_ids.add(doc_id)
                doc_scores[doc_id] = score

        # 4. Format Structured Results
        results_list = []
        for doc_id in matched_doc_ids:
            doc = self.index.documents.get(doc_id)
            if not doc:
                continue

            score = doc_scores.get(doc_id, 1.0)
            snippet = self.create_kwic_snippet(doc.body, set(extracted_query_terms))

            # Matched query terms in this document
            doc_matched_terms = []
            for t in extracted_query_terms:
                norm_t = self.normalizer.normalize_term(t)
                if norm_t in self.index.index and doc_id in self.index.index[norm_t]:
                    doc_matched_terms.append(t)

            results_list.append({
                "doc_id": doc.doc_id,
                "title": doc.title,
                "author": doc.author,
                "category": doc.category,
                "date": doc.date,
                "score": round(score, 4),
                "snippet": snippet,
                "matched_terms": list(set(doc_matched_terms))
            })

        # Sort results: descending by score, tie-break by doc_id
        results_list.sort(key=lambda x: (x["score"], x["doc_id"]), reverse=True)

        exec_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "query": clean_query,
            "mode": active_mode,
            "total_results": len(results_list),
            "execution_time_ms": exec_time_ms,
            "did_you_mean": did_you_mean,
            "index_stats": self.index.get_stats(),
            "results": results_list
        }
