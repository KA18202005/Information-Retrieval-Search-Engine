"""
Boolean Retrieval: Classical 2-Pointer Linear Merge Operations
Implements standard linear postings list merges for Boolean operators
(AND, OR, NOT, AND NOT, XOR) operating on sorted document identifier lists.
Time complexity: O(|p1| + |p2|) linear time.
"""

from typing import List, Any


def _doc_cmp_key(doc_id: str) -> Any:
    """Helper to ensure natural sorting for numeric or string document IDs."""
    try:
        return (0, int(doc_id))
    except ValueError:
        return (1, doc_id)


def intersect_postings(p1: List[str], p2: List[str]) -> List[str]:
    """
    Classical 2-pointer merge for conjunction (AND).
    Returns documents present in both sorted postings lists.
    """
    answer: List[str] = []
    i, j = 0, 0
    len1, len2 = len(p1), len(p2)

    while i < len1 and j < len2:
        k1, k2 = _doc_cmp_key(p1[i]), _doc_cmp_key(p2[j])
        if k1 == k2:
            answer.append(p1[i])
            i += 1
            j += 1
        elif k1 < k2:
            i += 1
        else:
            j += 1

    return answer


def union_postings(p1: List[str], p2: List[str]) -> List[str]:
    """
    Classical 2-pointer merge for disjunction (OR).
    Returns documents present in either sorted postings list.
    """
    answer: List[str] = []
    i, j = 0, 0
    len1, len2 = len(p1), len(p2)

    while i < len1 and j < len2:
        k1, k2 = _doc_cmp_key(p1[i]), _doc_cmp_key(p2[j])
        if k1 == k2:
            answer.append(p1[i])
            i += 1
            j += 1
        elif k1 < k2:
            answer.append(p1[i])
            i += 1
        else:
            answer.append(p2[j])
            j += 1

    while i < len1:
        answer.append(p1[i])
        i += 1

    while j < len2:
        answer.append(p2[j])
        j += 1

    return answer


def and_not_postings(p1: List[str], p2: List[str]) -> List[str]:
    """
    Classical 2-pointer difference for (p1 AND NOT p2).
    Returns documents in p1 that do not appear in p2.
    """
    answer: List[str] = []
    i, j = 0, 0
    len1, len2 = len(p1), len(p2)

    while i < len1 and j < len2:
        k1, k2 = _doc_cmp_key(p1[i]), _doc_cmp_key(p2[j])
        if k1 == k2:
            i += 1
            j += 1
        elif k1 < k2:
            answer.append(p1[i])
            i += 1
        else:
            j += 1

    while i < len1:
        answer.append(p1[i])
        i += 1

    return answer


def difference_postings(all_docs: List[str], p: List[str]) -> List[str]:
    """
    Evaluates pure negation (NOT p) relative to the full document universe.
    """
    return and_not_postings(all_docs, p)


def xor_postings(p1: List[str], p2: List[str]) -> List[str]:
    """
    Evaluates exclusive disjunction: (p1 XOR p2).
    Returns documents in p1 or p2, but not both.
    """
    u = union_postings(p1, p2)
    inter = intersect_postings(p1, p2)
    return and_not_postings(u, inter)
