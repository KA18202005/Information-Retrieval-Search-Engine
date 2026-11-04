"""
Module 5: Query Processing
Parses and evaluates user queries against the Inverted Index.
Supports:
- Boolean conjunction (AND), disjunction (OR), negation (NOT), and exclusive OR (XOR)
- Parenthesized sub-expressions with operator precedence
- Exact phrase queries ("...")
- Implicit AND insertion between adjacent terms
- Dijkstra's Shunting-Yard algorithm (Infix to Postfix RPN)
- Linear postings list evaluation
"""

import re
from typing import List, Tuple, Set, Optional
from engine.inverted_index import InvertedIndex
from engine.tokenizer import Tokenizer
from engine.normalizer import Normalizer
from engine.boolean_retrieval import (
    intersect_postings, union_postings, difference_postings, xor_postings
)
from engine.positional_index import PositionalQueryEvaluator


class QueryProcessor:
    """
    Parses and evaluates query strings supporting Boolean logic,
    parentheses, phrases, and single/multi-term searches.
    """

    OPERATOR_PRECEDENCE = {
        "NOT": 3,
        "AND": 2,
        "XOR": 1,
        "OR": 1
    }

    def __init__(self, index: InvertedIndex, tokenizer: Tokenizer, normalizer: Normalizer):
        self.index = index
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.phrase_evaluator = PositionalQueryEvaluator(index, tokenizer, normalizer)

    def tokenize_query(self, query_str: str) -> List[Tuple[str, str]]:
        """
        Tokenizes query string into structured lexical tokens:
        - Operators: ('AND', 'AND'), ('OR', 'OR'), ('NOT', 'NOT'), ('XOR', 'XOR')
        - Parentheses: ('(', '('), (')', ')')
        - Phrases: ('PHRASE', 'phrase content')
        - Terms: ('TERM', 'word')
        """
        tokens: List[Tuple[str, str]] = []
        if not query_str or not query_str.strip():
            return tokens

        pattern = r'(\bAND\b|\bOR\b|\bNOT\b|\bXOR\b|\(|\)|"[^"]*"|[\w\*\-]+)'
        matches = re.finditer(pattern, query_str, re.IGNORECASE)

        for m in matches:
            t = m.group(0).strip()
            if not t:
                continue

            upper = t.upper()
            if upper in ("AND", "OR", "NOT", "XOR", "(", ")"):
                tokens.append((upper, upper))
            elif t.startswith('"') and t.endswith('"') and len(t) >= 2:
                phrase_content = t[1:-1].strip()
                tokens.append(("PHRASE", phrase_content))
            else:
                tokens.append(("TERM", t))

        return tokens

    def insert_implicit_and(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Inserts implicit AND operators between adjacent terms, phrases,
        or closing parentheses followed by opening terms.
        """
        result: List[Tuple[str, str]] = []
        for i, token in enumerate(tokens):
            result.append(token)
            if i < len(tokens) - 1:
                curr_type, _ = token
                next_type, _ = tokens[i + 1]

                # If current token is a term/phrase/close-paren and next is a term/phrase/open-paren/NOT
                if curr_type in ("TERM", "PHRASE", ")") and next_type in ("TERM", "PHRASE", "NOT", "("):
                    result.append(("AND", "AND"))

        return result

    def shunting_yard(self, tokens: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Converts infix query tokens to postfix (Reverse Polish Notation)
        using Dijkstra's Shunting-Yard algorithm.
        """
        output: List[Tuple[str, str]] = []
        op_stack: List[Tuple[str, str]] = []

        for token in tokens:
            ttype, val = token

            if ttype in ("TERM", "PHRASE"):
                output.append(token)
            elif ttype == "(":
                op_stack.append(token)
            elif ttype == ")":
                while op_stack and op_stack[-1][0] != "(":
                    output.append(op_stack.pop())
                if op_stack and op_stack[-1][0] == "(":
                    op_stack.pop()  # Discard '('
            elif ttype in self.OPERATOR_PRECEDENCE:
                curr_prec = self.OPERATOR_PRECEDENCE[ttype]
                while (op_stack and op_stack[-1][0] != "(" and
                       self.OPERATOR_PRECEDENCE.get(op_stack[-1][0], 0) >= curr_prec):
                    output.append(op_stack.pop())
                op_stack.append(token)

        while op_stack:
            top = op_stack.pop()
            if top[0] != "(":  # Ignore unmatched opening parentheses
                output.append(top)

        return output

    def evaluate_postfix(self, postfix_tokens: List[Tuple[str, str]], all_doc_ids: Optional[Set[str]] = None) -> Set[str]:
        """
        Evaluates postfix (RPN) tokens against the inverted index using
        classical 2-pointer linear postings merges.
        Fixes historical argument mismatch by accepting explicit all_doc_ids.
        """
        universe = sorted(all_doc_ids if all_doc_ids is not None else self.index.all_doc_ids)
        stack: List[List[str]] = []

        for token in postfix_tokens:
            ttype, val = token

            if ttype == "TERM":
                postings = self.index.get_postings(val)
                stack.append(postings)

            elif ttype == "PHRASE":
                matched_set = self.phrase_evaluator.evaluate_phrase(val)
                stack.append(sorted(matched_set))

            elif ttype == "NOT":
                operand = stack.pop() if stack else []
                diff = difference_postings(universe, operand)
                stack.append(diff)

            elif ttype == "AND":
                op2 = stack.pop() if stack else []
                op1 = stack.pop() if stack else []
                stack.append(intersect_postings(op1, op2))

            elif ttype == "OR":
                op2 = stack.pop() if stack else []
                op1 = stack.pop() if stack else []
                stack.append(union_postings(op1, op2))

            elif ttype == "XOR":
                op2 = stack.pop() if stack else []
                op1 = stack.pop() if stack else []
                stack.append(xor_postings(op1, op2))

        return set(stack[-1]) if stack else set()

    def process_boolean_query(self, query_str: str) -> Set[str]:
        """Convenience method executing full query parse and Boolean evaluation."""
        tokens = self.tokenize_query(query_str)
        if not tokens:
            return set()
        normalized = self.insert_implicit_and(tokens)
        postfix = self.shunting_yard(normalized)
        return self.evaluate_postfix(postfix)
