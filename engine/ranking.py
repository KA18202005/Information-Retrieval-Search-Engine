"""
Vector Space Model (VSM) and Ranked Retrieval
Implements exact classical IR scoring:
- Sublinear TF: wf(t, d) = 1 + ln(tf(t, d)) if tf > 0 else 0
- Smoothed IDF: idf(t) = ln((N + 1) / (df(t) + 1)) + 1.0
- L2-normalized Cosine Similarity:
    CosineSim(q, d) = (V(q) · V(d)) / (||V(q)|| * ||V(d)||)
"""

import math
from typing import Dict, List, Set, Tuple, Optional
from engine.inverted_index import InvertedIndex
from engine.normalizer import Normalizer


class RankedRetrieval:
    """
    Computes vector space representations and cosine similarity scores.
    """

    def __init__(self, index: InvertedIndex, normalizer: Optional[Normalizer] = None):
        self.index = index
        self.normalizer = normalizer if normalizer else Normalizer()

        # Cached Document Vector L2 Norms: doc_id -> ||V(d)||
        self.doc_vector_norms: Dict[str, float] = {}
        self.compute_document_vector_norms()

    def get_idf(self, term: str) -> float:
        """
        Computes smoothed IDF across N documents:
        idf(t) = ln((N + 1) / (df(t) + 1)) + 1.0
        """
        N = self.index.num_documents
        if N == 0:
            return 0.0

        df = self.index.get_doc_frequency(term)
        return math.log((N + 1.0) / (df + 1.0)) + 1.0

    @staticmethod
    def sublinear_tf(tf: int) -> float:
        """
        Computes logarithmic sublinear term frequency:
        wf(t, d) = 1 + ln(tf) if tf > 0 else 0.0
        """
        if tf <= 0:
            return 0.0
        return 1.0 + math.log(tf)

    def compute_document_vector_norms(self):
        """
        Pre-computes and caches the L2 Euclidean norm for every document vector:
        ||V(d)|| = sqrt(sum_{t in d} (wf(t, d) * idf(t))^2)
        """
        self.doc_vector_norms.clear()
        doc_weights_sq: Dict[str, float] = {doc_id: 0.0 for doc_id in self.index.documents}

        for term, postings_map in self.index.index.items():
            idf = self.get_idf(term)
            for doc_id, positions in postings_map.items():
                tf = len(positions)
                wf = self.sublinear_tf(tf)
                w = wf * idf
                doc_weights_sq[doc_id] += w * w

        for doc_id, sum_sq in doc_weights_sq.items():
            self.doc_vector_norms[doc_id] = math.sqrt(sum_sq) if sum_sq > 0 else 0.0

    def compute_cosine_similarity(self, query_terms: List[str]) -> List[Tuple[str, float]]:
        """
        Ranks all documents matching at least one query term using Cosine Similarity.
        Returns list of (doc_id, score) sorted descending by relevance.
        """
        if not query_terms or self.index.num_documents == 0:
            return []

        # 1. Normalize query terms and calculate query term frequencies
        norm_query_terms = [self.normalizer.normalize_term(t) for t in query_terms]
        norm_query_terms = [t for t in norm_query_terms if t]

        if not norm_query_terms:
            return []

        query_tf: Dict[str, int] = {}
        for t in norm_query_terms:
            query_tf[t] = query_tf.get(t, 0) + 1

        # 2. Compute query vector weights and query L2 norm
        query_weights: Dict[str, float] = {}
        query_len_sq = 0.0
        for t, tf in query_tf.items():
            idf = self.get_idf(t)
            wf = self.sublinear_tf(tf)
            w_q = wf * idf
            query_weights[t] = w_q
            query_len_sq += w_q * w_q

        query_norm = math.sqrt(query_len_sq) if query_len_sq > 0 else 0.0
        if query_norm == 0.0:
            return []

        # 3. Accumulate dot products (V(q) · V(d)) across matching postings
        scores: Dict[str, float] = {}
        for t, w_q in query_weights.items():
            if t not in self.index.index:
                continue
            idf = self.get_idf(t)
            postings_map = self.index.index[t]
            for doc_id, positions in postings_map.items():
                tf_d = len(positions)
                wf_d = self.sublinear_tf(tf_d)
                w_d = wf_d * idf
                scores[doc_id] = scores.get(doc_id, 0.0) + (w_q * w_d)

        # 4. Normalize by document vector norms: Cosine = Dot / (||V(q)|| * ||V(d)||)
        ranked_results: List[Tuple[str, float]] = []
        for doc_id, dot_product in scores.items():
            doc_norm = self.doc_vector_norms.get(doc_id, 0.0)
            if doc_norm > 0.0:
                cosine = dot_product / (query_norm * doc_norm)
                # Bound between 0.0 and 1.0 to avoid float precision artifacts
                cosine = max(0.0, min(1.0, cosine))
                ranked_results.append((doc_id, round(cosine, 4)))

        # 5. Sort deterministically by score descending, then doc_id ascending
        ranked_results.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return ranked_results
