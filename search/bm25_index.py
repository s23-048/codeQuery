"""
CodeQuery — BM25 Keyword Search Index
======================================

This module provides keyword-based search over code chunks using BM25
(Best Matching 25), a ranking algorithm used by search engines since the 1990s.

Why BM25 alongside embeddings?
-------------------------------
Embeddings are great at semantic/meaning queries ("function that processes
payments") but bad at exact identifier matches ("executeCode", "REDIS_QUEUE_KEY").
Embeddings blur specific tokens into a 384-dimensional average — the individual
token identity gets lost.

BM25 is the opposite: it matches exact tokens. If you search "executeCode",
BM25 will find chunks containing that exact word (or its camelCase-split
form "execute" + "code"). But BM25 can't understand that "payment processing"
and "charge credit card" mean the same thing.

By combining both (Day 5: RRF fusion), we get the best of both worlds.

How BM25 works (the intuition)
-------------------------------
BM25 scores how relevant a document is to a query based on:

1. **Term Frequency (TF):** How often does the query term appear in this
   document? More occurrences = more relevant. But with diminishing returns —
   the 10th occurrence matters less than the 1st.

2. **Inverse Document Frequency (IDF):** How rare is this term across ALL
   documents? A term that appears in every document (like "def" or "return")
   is not informative. A term that appears in only 2 out of 1000 documents
   (like "executeCode") is very informative.

3. **Document Length Normalization:** Longer documents naturally contain more
   terms. BM25 normalizes for this so a 500-line file doesn't automatically
   score higher than a 10-line function.

The formula (simplified):
    score(term, doc) = IDF(term) × TF(term, doc) × (k1 + 1) / (TF + k1 × (1 - b + b × len(doc)/avglen))

Where k1=1.5 and b=0.75 are standard defaults. You don't need to derive this,
but understand what each part does.

Tokenization for code
----------------------
Plain `.split()` won't work for code because:
- `executeCode` should match "execute code" → need camelCase splitting
- `validate_input` should match "validate input" → need snake_case splitting
- `parser/code_parser.py` should match "code parser" → need path splitting

We apply these transformations before feeding text to BM25.
"""

import os
import pickle
import re
import time
from typing import Optional

from rank_bm25 import BM25Okapi

from parser.code_parser import CodeChunk


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_code(text: str) -> list[str]:
    """
    Tokenize code text for BM25 indexing/searching.

    This is the critical function that makes BM25 work on code. Plain
    whitespace splitting fails because code identifiers are compound words:

        executeCode     → should match "execute code"
        validate_input  → should match "validate input"
        DataProcessor   → should match "data processor"
        parser/code.py  → should match "parser" and "code"

    Steps:
    1. Split camelCase: 'executeCode' → 'execute Code'
    2. Split ALLCAPS transitions: 'HTMLParser' → 'HTML Parser'
    3. Replace all non-alphanumeric chars with spaces (handles snake_case,
       paths, dots, punctuation)
    4. Lowercase everything
    5. Remove single-character tokens (noise: 'x', 'y', 'a', etc.)

    Returns a list of lowercase tokens.
    """
    # camelCase → camel Case
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # HTMLParser → HTML Parser (uppercase run followed by uppercase+lowercase)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    # Replace non-alphanumeric with spaces
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # Lowercase, split, filter out tiny tokens
    tokens = text.lower().split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------

class BM25Index:
    """
    BM25 keyword search index over code chunks.

    Usage:
        from parser.code_parser import CodeParser
        from search.bm25_index import BM25Index

        parser = CodeParser()
        chunks = parser.parse_repository("/path/to/repo")

        bm25 = BM25Index()
        bm25.build(chunks)
        bm25.save("bm25_index.pkl")

        results = bm25.search("validate_input", top_k=5)

    The index stores:
    - A BM25Okapi model (from rank_bm25 library)
    - A mapping from internal index → chunk metadata
    - The tokenized corpus for debugging/inspection

    Persistence: saved via pickle so you don't rebuild on restart.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunks: list[CodeChunk] = []
        self._corpus: list[list[str]] = []  # tokenized documents

    def build(self, chunks: list[CodeChunk]) -> None:
        """
        Build the BM25 index from a list of CodeChunks.

        Steps:
        1. Generate search text for each chunk (same text used for embeddings)
        2. Tokenize each text with our code-aware tokenizer
        3. Build the BM25Okapi model from the tokenized corpus

        BM25Okapi is the standard BM25 variant. The "Okapi" part refers to
        the information retrieval system where it was first implemented
        (at City University of London in the 1990s).
        """
        if not chunks:
            print("⚠ No chunks to index")
            return

        self._chunks = list(chunks)

        # --- Step 1 & 2: Generate and tokenize search texts ---
        print(f"  Tokenizing {len(chunks)} chunks for BM25...")
        start = time.time()

        self._corpus = []
        for chunk in chunks:
            text = chunk.to_search_text()
            tokens = tokenize_code(text)
            self._corpus.append(tokens)

        # --- Step 3: Build BM25 model ---
        self._bm25 = BM25Okapi(self._corpus)

        elapsed = time.time() - start
        # Count total tokens and unique tokens for stats
        total_tokens = sum(len(doc) for doc in self._corpus)
        unique_tokens = len(set(t for doc in self._corpus for t in doc))

        print(f"✓ BM25 index built in {elapsed:.2f}s: "
              f"{len(chunks)} docs, {total_tokens} total tokens, "
              f"{unique_tokens} unique tokens")

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Search the index with a keyword query.

        Parameters
        ----------
        query : str
            The search query (e.g., "validate_input", "execute code").
        top_k : int
            Number of results to return.

        Returns
        -------
        list[dict]
            Ranked results with:
            - chunk_id, score, file_path, name, chunk_type, language,
              start_line, end_line, code_preview

        The query is tokenized with the same tokenizer used for indexing.
        BM25 scores are non-negative floats (not bounded to 0-1 like cosine
        similarity). Higher = more relevant.
        """
        if self._bm25 is None or not self._chunks:
            print("⚠ Index not built — call build() first")
            return []

        # Tokenize the query the same way we tokenized the documents
        query_tokens = tokenize_code(query)
        if not query_tokens:
            return []

        # Get BM25 scores for all documents
        scores = self._bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score (descending)
        # We use argsort on the negative scores to get descending order
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        # Build results
        results = []
        for idx in ranked_indices:
            score = float(scores[idx])
            if score <= 0:
                break  # No point returning zero-score results

            chunk = self._chunks[idx]
            results.append({
                "chunk_id": chunk.chunk_id,
                "score": round(score, 4),
                "file_path": chunk.file_path,
                "name": chunk.name,
                "chunk_type": chunk.chunk_type,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "code_preview": chunk.code[:1000],
            })

        return results

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """
        Save the BM25 index to a pickle file.

        Saves the BM25 model, chunks, and corpus so the index can be
        reloaded without re-parsing and re-tokenizing the entire repo.

        Why pickle? BM25Okapi stores numpy arrays internally. Pickle
        handles this natively. JSON can't serialize numpy arrays without
        conversion, and the file would be much larger.
        """
        data = {
            "bm25": self._bm25,
            "chunks": self._chunks,
            "corpus": self._corpus,
        }
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"✓ BM25 index saved to {file_path}")

    def load(self, file_path: str) -> None:
        """
        Load a BM25 index from a pickle file.

        After loading, the index is ready for search() calls immediately.
        """
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]
        self._corpus = data["corpus"]
        print(f"✓ BM25 index loaded from {file_path}: "
              f"{len(self._chunks)} chunks")

    # -------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------

    def get_stats(self) -> dict:
        """Get index statistics."""
        if not self._corpus:
            return {"chunks": 0, "total_tokens": 0, "unique_tokens": 0}

        total = sum(len(doc) for doc in self._corpus)
        unique = len(set(t for doc in self._corpus for t in doc))
        avg_len = total / len(self._corpus) if self._corpus else 0

        return {
            "chunks": len(self._chunks),
            "total_tokens": total,
            "unique_tokens": unique,
            "avg_tokens_per_chunk": round(avg_len, 1),
        }

    def debug_tokens(self, chunk_id: str) -> list[str]:
        """Show the tokens stored for a specific chunk (for debugging)."""
        for i, chunk in enumerate(self._chunks):
            if chunk.chunk_id == chunk_id:
                return self._corpus[i]
        return []


# ---------------------------------------------------------------------------
# CLI — quick test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from parser.code_parser import CodeParser

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else None

    # Step 1: Parse
    parser = CodeParser()
    if os.path.isfile(target):
        chunks = parser.parse_file(target)
    else:
        chunks = parser.parse_repository(target)

    # Step 2: Build BM25 index
    bm25 = BM25Index()
    bm25.build(chunks)

    # Print stats
    stats = bm25.get_stats()
    print(f"  Stats: {stats}")

    # Step 3: Search
    if query:
        print(f"\n{'='*60}")
        print(f"BM25 SEARCH: \"{query}\"")
        print(f"  Query tokens: {tokenize_code(query)}")
        print(f"{'='*60}")
        results = bm25.search(query, top_k=5)
        for i, r in enumerate(results):
            print(f"\n  #{i+1} [{r['score']:.4f}] {r['chunk_type']} {r['name']}")
            print(f"      File: {r['file_path']} (lines {r['start_line']}–{r['end_line']})")
            preview_lines = r["code_preview"].split("\n")[:3]
            for line in preview_lines:
                print(f"      │ {line}")
    else:
        # Run several test queries to show BM25 strengths
        test_queries = [
            "validate_input",           # exact function name
            "CodeParser",               # exact class name
            "execute code",             # camelCase query
            "data processor",           # camelCase class match
            "format output text width", # multiple keyword match
        ]
        for q in test_queries:
            print(f"\n{'='*60}")
            print(f"BM25 SEARCH: \"{q}\"")
            print(f"  Query tokens: {tokenize_code(q)}")
            print(f"{'='*60}")
            results = bm25.search(q, top_k=3)
            for i, r in enumerate(results):
                print(f"  #{i+1} [{r['score']:.4f}] {r['chunk_type']} {r['name']} ({r['file_path']})")
