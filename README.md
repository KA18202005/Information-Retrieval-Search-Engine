# RetrievIR — Academic Information Retrieval Search Engine
**Course:** Web Search and Mining (WSM) / Stanford CS276  
**Developer:** Senior Information Retrieval Engineer & Full-Stack Architect  
**Language & Runtime:** Python 3.13.12, Flask, HTML5, CSS3, Vanilla JavaScript  
**Local Host:** `http://127.0.0.1:5000`  

---

## 1. Project Overview & Objectives

**RetrievIR** is a complete, native Python Information Retrieval (IR) workbench and search engine built strictly from mathematical first principles without third-party black-box retrieval libraries (no Lucene, NLTK, or Scikit-learn). 

Designed to demonstrate core concepts from classical Information Retrieval (Chapters 1–4 of the WSM syllabus), the application processes documents, performs linguistic normalization, constructs positional and permuterm inverted indices, parses complex Boolean AST expressions, executes tolerant wildcard searches, computes Levenshtein DP spelling corrections, calculates Vector Space Cosine Similarity, and presents dynamic keyword-in-context (KWIC) search results through a Google-inspired web interface.

> [!NOTE]
> Per syllabus and project requirements, post-compression storage techniques (Variable Byte, Elias Gamma, Delta coding) are intentionally excluded to focus on algorithmic correctness, indexing data structures, and ranking models.

---

## 2. System Architecture

The project adheres to a modular, decoupled architecture:

```
Information Retrieval Search Engine/
├── data/
│   └── documents/                # 14 authentic academic corpus documents (.txt)
│       ├── doc_01_tokenization.txt
│       ├── doc_02_linguistic_stemming.txt
│       ├── doc_03_inverted_index.txt
│       ├── doc_04_boolean_retrieval.txt
│       ├── doc_05_bm25_ranking.txt
│       ├── doc_06_vector_space_tfidf.txt
│       ├── doc_07_python_ir_implementation.txt
│       ├── doc_08_evaluation_metrics.txt
│       ├── neural_networks.txt
│       ├── distributed_consensus.txt
│       ├── database_b_trees.txt
│       ├── compiler_ast_parsing.txt
│       ├── python_asyncio.txt
│       └── graph_algorithms.txt
├── engine/
│   ├── __init__.py               # Package public exports
│   ├── document_loader.py        # Module 1: Document Loading & Metadata Parsing
│   ├── tokenizer.py              # Module 2: Alphanumeric Tokenizer & Span Preservation
│   ├── stemmer.py                # Martin Porter (1980) Stemming Algorithm
│   ├── normalizer.py             # Module 3: Case Folding, Stopwords, Positional Preservation
│   ├── inverted_index.py         # Module 4: Positional Inverted Index & Dictionary
│   ├── boolean_retrieval.py      # Classical 2-Pointer Linear Merge (AND, OR, NOT, XOR)
│   ├── positional_index.py       # Positional Intersection for Exact Multi-Term Phrases
│   ├── query_processor.py        # Module 5: Lexer, Shunting-Yard AST, RPN Evaluator
│   ├── permuterm_index.py        # Permuterm Rotations ($) for Tolerant Retrieval
│   ├── kgram_index.py            # K-gram Generation & Jaccard Candidate Overlap
│   ├── wildcard_processor.py     # Module 6: Wildcard Expansion & Regex Post-filter
│   ├── spelling_correction.py    # Levenshtein DP Matrix Edit Distance & "Did You Mean"
│   ├── field_index.py            # Parametric Field Search (title:, author:, category:, body:)
│   ├── zone_index.py             # Weighted Dual-Zone Scoring (Title + Body)
│   ├── ranking.py                # Vector Space Model (Sublinear TF-IDF & Cosine Sim)
│   └── search_engine.py          # Unified Coordinator & KWIC Snippet Generator
├── frontend/
│   ├── app.py                    # Flask Web Server & REST API
│   ├── templates/
│   │   ├── index.html            # Google-Inspired Minimalist Homepage
│   │   └── results.html          # Clean Search Results Page with KWIC Highlighting
│   └── static/
│       ├── css/
│       │   └── style.css         # Modern, Accessible, Responsive Stylesheet
│       └── js/
│           └── script.js         # Interactive Search Controls, Pills, and Modals
├── tests/
│   ├── test_document_loader.py   # Tests document parsing & error handling
│   ├── test_tokenizer.py         # Tests tokenization & stream preservation
│   ├── test_normalizer.py        # Tests stemming, case folding, position gaps
│   ├── test_inverted_index.py    # Tests postings lists, DF, TF statistics
│   ├── test_boolean_retrieval.py # Tests 2-pointer linear merges & AST evaluation
│   ├── test_positional_index.py  # Tests exact phrase matching with stopword gaps
│   ├── test_wildcard.py          # Tests leading, trailing, middle wildcards
│   ├── test_spelling_correction.py # Tests DP edit distance & Did You Mean
│   ├── test_field_zone.py        # Tests parametric field & zone weighting
│   ├── test_ranking.py           # Tests sublinear TF, smooth IDF, Cosine Sim
│   ├── test_search_engine.py     # Tests end-to-end query coordinator
│   └── test_flask_app.py         # Tests Flask HTTP endpoints & JSON API
├── conftest.py                   # Pytest sys.path bootstrap
├── requirements.txt              # Minimal dependencies (Flask, pytest)
├── .gitignore                    # Local environment ignore rules
├── CODEBASE_AUDIT.md             # Complete codebase audit & verification report
└── README.md                     # Comprehensive system documentation
```

---

## 3. The Seven Mandatory Modules

### Module 1: Document Loading (`engine/document_loader.py`)
- Reads text documents from `data/documents/`.
- Extracts structured headers (`Title:`, `Author:`, `Category:`, `Date:`) and body text.
- Fallback parsing for documents without standard headers.
- Gracefully handles empty, missing, or malformed files without crashes.

### Module 2: Tokenization (`engine/tokenizer.py`)
- Regex-based alphanumeric token extraction (`\b[A-Za-z0-9]+(?:['\-_][A-Za-z0-9]+)*\b`).
- Extracts surface form, character span (`start`, `end`), and sequential token position.
- **Position Preservation:** Every word in the raw stream is assigned a monotonic stream position ($0, 1, 2, \dots$), ensuring that positional distance semantics remain exact.

### Module 3: Text Normalization and Case Folding (`engine/normalizer.py` & `engine/stemmer.py`)
- **Case Folding:** Converts characters to lowercase.
- **Stopword Filtering:** Filters 120+ standard classical IR grammatical stopwords.
- **Porter Stemmer (1980):** Pure Python 220-line implementation of Martin Porter's suffix-stripping rules (Steps 1a through 5).
- **Critical Stream Guarantee:** Dropping stopwords does NOT shift the sequential index of remaining tokens. For example, in `"Tokenization and Text Normalization"`, `Tokenization` is pos 0, `Text` is pos 2, and `Normalization` is pos 3.

### Module 4: Inverted Index Construction (`engine/inverted_index.py`)
- Maintains a positional dictionary: $\text{term} \rightarrow \{\text{doc\_id}: [\text{pos}_0, \text{pos}_1, \dots]\}$.
- Postings lists are sorted deterministically for linear 2-pointer merges.
- Computes Document Frequency ($df(t)$), Term Frequency ($tf(t, d)$), total indexed tokens, and collection average document length ($avgdl$).

### Module 5: Query Processing (`engine/query_processor.py` & `engine/boolean_retrieval.py`)
- Supports Boolean operators: `AND`, `OR`, `NOT`, `XOR`, and parentheses `( )`.
- Precedence: `NOT` (3) > `AND` (2) > `OR`/`XOR` (1).
- Automatic implicit `AND` insertion between adjacent terms.
- **Dijkstra's Shunting-Yard Algorithm:** Converts infix user queries into postfix Reverse Polish Notation (RPN).
- **Two-Pointer Linear Merges:** Evaluates postings lists in $O(|p_1| + |p_2|)$ linear time.

### Module 6: Wildcard Processing (`engine/wildcard_processor.py` & `engine/permuterm_index.py`)
- Rotates vocabulary terms with terminal symbol `$` (e.g. `retrieval$`).
- Supports:
  - Trailing wildcards: `mach*` $\rightarrow$ lookup `$mach*`
  - Leading wildcards: `*tion` $\rightarrow$ lookup `tion$*`
  - Middle wildcards: `in*ation` $\rightarrow$ lookup `ation$in*`
  - Multiple wildcards: `re*v*er`
- **Candidate Verification:** Post-filters Permuterm candidates against compiled regex `^pattern$` to eliminate false positives.

### Module 7: Frontend GUI (`frontend/app.py`, `templates/`, `static/`)
- Modern, minimalist Google-inspired search engine interface.
- Centered search bar with search icon, clear button, mode dropdown, and sample query pills.
- Results page featuring query term highlighting (`<mark>`), execution time in milliseconds, result count, metadata badges, and document preview modal.
- REST API endpoints for `/api/search`, `/api/suggest`, and `/api/stats`.

---

## 4. Advanced IR Modules

| Module | Implementation | Description |
| :--- | :--- | :--- |
| **Positional Phrase Search** | `engine/positional_index.py` | Exact multi-term phrase matching (`"..."`) verifying relative positional offsets ($pos_{i} - pos_0 = i$). |
| **Spelling Correction** | `engine/spelling_correction.py` | Levenshtein Dynamic Programming edit distance matrix + K-gram overlap candidate generation for "Did You Mean: ...". |
| **Parametric Field Search** | `engine/field_index.py` | Targeted attribute search (`title:`, `author:`, `category:`, `body:`). |
| **Weighted Zone Scoring** | `engine/zone_index.py` | Dual-zone linear scoring between Title ($g_{\text{title}} = 0.6$) and Body ($g_{\text{body}} = 0.4$). |
| **Vector Space Model** | `engine/ranking.py` | Classical TF-IDF ranked retrieval with sublinear TF, smoothed IDF, and $L_2$ Cosine Similarity. |

---

## 5. Mathematical Formulations

### 1. Sublinear Term Frequency Weighting
$$\text{wf}(t, d) = \begin{cases} 1 + \ln(\text{tf}(t, d)) & \text{if } \text{tf}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2. Smoothed Inverse Document Frequency (IDF)
$$\text{idf}(t) = \ln\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1.0$$

### 3. Vector Space Cosine Similarity ($L_2$ Normalization)
$$\text{CosineSim}(q, d) = \frac{\sum_{t \in q \cap d} w(t, q) \cdot w(t, d)}{\sqrt{\sum_{t \in q} w(t, q)^2} \cdot \sqrt{\sum_{t \in d} w(t, d)^2}}$$

### 4. Weighted Zone Scoring
$$S(q, d) = g_{\text{title}} \cdot S_{\text{title}}(q, d) + g_{\text{body}} \cdot S_{\text{body}}(q, d), \quad g_{\text{title}} + g_{\text{body}} = 1.0$$

### 5. Levenshtein Minimum Edit Distance (Dynamic Programming)
$$D(i, j) = \min \begin{cases} D(i-1, j) + 1 & \text{(Deletion)} \\ D(i, j-1) + 1 & \text{(Insertion)} \\ D(i-1, j-1) + [s_1[i-1] \neq s_2[j-1]] & \text{(Substitution)} \end{cases}$$

### 6. Jaccard K-Gram Coefficient
$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

---

## 6. Local Installation & Execution on Windows

All development is strictly local to your machine. No Git or external accounts required.

### 1. Open Windows PowerShell in Project Directory
```powershell
cd "C:\Users\Kavya Agarwal\OneDrive\Documents\Information Retrieval Search Engine"
```

### 2. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```
*(If script execution is disabled on your machine, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 3. Run the Test Suite
```powershell
pytest -v
```
**Current Test Status:** 33 / 33 Unit, Integration, and E2E Tests Passing.

### 4. Start the Flask Search Engine Application
```powershell
python frontend\app.py
```

### 5. Access in Web Browser
Open your browser to:
**`http://127.0.0.1:5000`**

---

## 7. Example Queries Demonstrating Functionality

| Query | Search Mode | Description & Observed Behavior |
| :--- | :--- | :--- |
| `retrieval AND index` | **Boolean** | Conjunction of postings for "retrieval" and "index" using 2-pointer linear merge. |
| `learning OR retrieval` | **Boolean** | Disjunction returning documents matching either term. |
| `systems AND NOT deep` | **Boolean** | Difference merge excluding documents containing "deep". |
| `(learning OR deep) AND systems` | **Boolean** | Parenthesized sub-expression evaluating operator precedence correctly. |
| `"inverted index"` | **Phrase** | Exact positional adjacency matching consecutive token offsets. |
| `"tokenization and text"` | **Phrase** | Multi-word phrase with stopword gap verification. |
| `mach*` | **Wildcard** | Trailing wildcard expanded via Permuterm index to `machine`, `machinery`. |
| `*tion` | **Wildcard** | Leading wildcard expanded to `information`, `tokenization`, etc. |
| `in*ation` | **Wildcard** | Middle wildcard matching `information` (correctly excluding `infraction`). |
| `title:database` | **Field** | Parametric search matching documents with "database" in the title. |
| `author:salton` | **Field** | Parametric search matching Gerard Salton's vector space document. |
| `infromation retrival` | **Auto** | Triggers Google-style *"Did you mean: information retrieval?"* banner. |
| `vector space model cosine` | **Ranked** | Free-text scored via Sublinear TF-IDF and Cosine Similarity. |

---

## 8. REST API Documentation

The local Flask backend exposes the following REST endpoints:

- **`GET /api/search?q=<query>&mode=<mode>`**  
  Returns structured search results with snippet, document metadata, score, and matched terms in JSON format.
- **`GET /api/suggest?q=<query>`**  
  Returns Levenshtein spelling suggestions and prefix completions.
- **`GET /api/stats`**  
  Returns collection statistics ($N$, vocabulary size, total tokens, $avgdl$).
- **`GET /health`**  
  Health check endpoint verifying index readiness.

---

## 9. Verification & Quality Assurance Summary

- [x] **Document Loading:** 14 authentic documents loaded with RFC headers and body separation.
- [x] **Tokenization:** Alphanumeric extraction with character spans and strict monotonic sequential positions ($0, 1, 2, \dots$).
- [x] **Normalization:** Case folding, stopword filtering, and Porter stemmer without positional index compression.
- [x] **Inverted Index:** Term dictionary, positional postings, sorted doc IDs, DF, and TF statistics.
- [x] **Query Processing:** Infix-to-postfix Shunting Yard AST parser, linear 2-pointer merges (`AND`, `OR`, `NOT`, `XOR`).
- [x] **Wildcard Processing:** Permuterm index supporting leading, trailing, and middle wildcards with regex post-filtering.
- [x] **Frontend GUI:** Modern, responsive Google-inspired interface with execution timing, snippets, and Did You Mean suggestions.
- [x] **Positional Indexing:** Multi-term exact phrase search with stopword gap preservation.
- [x] **Spelling Correction:** 2D Levenshtein DP edit distance matrix and candidate suggestion ranking.
- [x] **Field Indexing:** Parametric field search for `title:`, `author:`, `category:`, and `body:`.
- [x] **Zone Indexing:** Weighted dual-zone scoring ($0.6 \cdot S_{\text{title}} + 0.4 \cdot S_{\text{body}}$).
- [x] **Ranked Retrieval:** Vector Space Model with sublinear TF, smoothed IDF, and $L_2$ Cosine Similarity.
- [x] **Safety & Originality:** Original `ir_engine.py` in `Downloads/` remains completely untouched. No external search engine APIs or post-compression topics implemented.
