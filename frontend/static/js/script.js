/**
 * Frontend Interactive Behavior
 * Handles search bar clear button, example pills, syntax modal, and sample queries.
 */

document.addEventListener("DOMContentLoaded", () => {
    const searchInput = document.getElementById("searchInput");
    const clearBtn = document.getElementById("clearBtn");
    const searchForm = document.getElementById("searchForm") || document.getElementById("resultsSearchForm");
    const syntaxBtn = document.getElementById("syntaxBtn");
    const syntaxModal = document.getElementById("syntaxModal");
    const closeModal = document.getElementById("closeModal");
    const randomQueryBtn = document.getElementById("randomQueryBtn");
    const modeSelect = document.getElementById("modeSelect");

    // Sample academic queries for quick testing
    const sampleQueries = [
        { q: "retrieval AND index", mode: "boolean" },
        { q: '"inverted index"', mode: "boolean" },
        { q: "mach*", mode: "wildcard" },
        { q: "*tion", mode: "wildcard" },
        { q: "in*ation", mode: "wildcard" },
        { q: "title:database", mode: "field" },
        { q: "author:salton", mode: "field" },
        { q: "infromation retrival", mode: "auto" }, // Triggers spelling correction
        { q: "vector space model cosine", mode: "ranked" },
        { q: "(learning OR deep) AND systems", mode: "boolean" },
        { q: "algorithms XOR models", mode: "boolean" }
    ];

    // Toggle clear button visibility
    function updateClearButton() {
        if (!searchInput || !clearBtn) return;
        if (searchInput.value.trim().length > 0) {
            clearBtn.style.display = "block";
        } else {
            clearBtn.style.display = "none";
        }
    }

    if (searchInput && clearBtn) {
        searchInput.addEventListener("input", updateClearButton);
        updateClearButton();

        clearBtn.addEventListener("click", () => {
            searchInput.value = "";
            searchInput.focus();
            updateClearButton();
        });
    }

    // Example Pills click
    const pills = document.querySelectorAll(".pill");
    pills.forEach(pill => {
        pill.addEventListener("click", () => {
            const queryText = pill.getAttribute("data-query");
            if (searchInput && searchForm) {
                searchInput.value = queryText;
                searchForm.submit();
            }
        });
    });

    // Random / Sample Query Button
    if (randomQueryBtn && searchInput && searchForm) {
        randomQueryBtn.addEventListener("click", () => {
            const randomIndex = Math.floor(Math.random() * sampleQueries.length);
            const sample = sampleQueries[randomIndex];
            searchInput.value = sample.q;
            if (modeSelect) {
                modeSelect.value = sample.mode;
            }
            searchForm.submit();
        });
    }

    // Syntax Guide Modal
    if (syntaxBtn && syntaxModal && closeModal) {
        syntaxBtn.addEventListener("click", () => {
            syntaxModal.style.display = "flex";
        });

        closeModal.addEventListener("click", () => {
            syntaxModal.style.display = "none";
        });

        window.addEventListener("click", (e) => {
            if (e.target === syntaxModal) {
                syntaxModal.style.display = "none";
            }
        });
    }
});
