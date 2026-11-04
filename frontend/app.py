"""
Flask Web Application Backend
Serves the Google-inspired Search Engine interface and provides REST API endpoints.
Runs strictly locally at http://127.0.0.1:5000.
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify

# Ensure parent directory is in sys.path so engine can be imported directly
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from engine.search_engine import SearchEngine

app = Flask(
    __name__,
    template_folder=str(project_root / "frontend" / "templates"),
    static_folder=str(project_root / "frontend" / "static")
)

# Initialize the global Search Engine once on startup
corpus_path = project_root / "data" / "documents"
search_engine = SearchEngine(corpus_dir=str(corpus_path))


@app.route("/")
def index():
    """Renders the minimalist Google-inspired search homepage."""
    stats = search_engine.index.get_stats()
    return render_template("index.html", stats=stats)


@app.route("/search")
def search():
    """Renders the search results page."""
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "auto").strip().lower()

    if not query:
        return render_template("results.html", query="", results=[], total=0, time_ms=0, did_you_mean=None, mode=mode)

    search_result = search_engine.search(query, mode=mode)
    return render_template(
        "results.html",
        query=query,
        mode=search_result["mode"],
        results=search_result["results"],
        total=search_result["total_results"],
        time_ms=search_result["execution_time_ms"],
        did_you_mean=search_result["did_you_mean"],
        stats=search_result.get("index_stats", {})
    )


@app.route("/api/search")
def api_search():
    """REST API endpoint returning structured search results in JSON format."""
    query = request.args.get("q", "").strip()
    mode = request.args.get("mode", "auto").strip().lower()

    if not query:
        return jsonify({
            "status": "error",
            "message": "Query parameter 'q' is required.",
            "results": []
        }), 400

    result = search_engine.search(query, mode=mode)
    return jsonify({
        "status": "success",
        "data": result
    })


@app.route("/api/suggest")
def api_suggest():
    """REST API endpoint returning spelling suggestions and query completions."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"suggestion": None, "completions": []})

    suggestion = search_engine.spelling_correction.suggest_query(query)

    # Prefix autocomplete from vocabulary terms
    last_word = query.split()[-1].lower() if query.split() else ""
    completions = []
    if last_word and len(last_word) >= 2:
        for term in search_engine.index.index.keys():
            if term.startswith(last_word) and term != last_word:
                completions.append(term)
                if len(completions) >= 5:
                    break

    return jsonify({
        "query": query,
        "suggestion": suggestion,
        "completions": completions
    })


@app.route("/api/stats")
def api_stats():
    """REST API endpoint returning index metrics."""
    return jsonify({
        "status": "success",
        "stats": search_engine.index.get_stats()
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "indexed_documents": search_engine.index.num_documents,
        "vocabulary_size": search_engine.index.vocabulary_size
    })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" INFORMATION RETRIEVAL SEARCH ENGINE (WSM COURSE)")
    print(" Running locally at: http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(host="127.0.0.1", port=5000, debug=False)
