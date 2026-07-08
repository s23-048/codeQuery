"""
CodeQuery — Hybrid Search with Reciprocal Rank Fusion (RRF)
============================================================

This is the key insight of the whole project: neither semantic search nor
keyword search is good enough alone, but *combining* them is better than
either.

Why not just use embeddings?
-----------------------------
Embeddings encode "meaning" into 384-dimensional vectors. Great for
conceptual queries like "function that processes payments." But embeddings
blur individual tokens — they average the meaning of ALL tokens in a chunk
into one vector. So if you search for the exact identifier "executeCode",
the embedding for that query is close to "run program", "start process",
"call function"… it has lost the *specific token* "executeCode."

Why not just use BM25?
-----------------------
BM25 matches exact tokens. Search "executeCode" and it finds chunks
containing that exact word (after camelCase splitting). Perfect. But ask
"function that processes payments" and BM25 looks for the tokens
["function", "that", "processes", "payments"] — it can't understand that
"charge_credit_card" means the same thing.

The solution: run BOTH, then merge the result lists.

How to merge? Reciprocal Rank Fusion (RRF)
-------------------------------------------
The problem with merging is that BM25 scores and cosine similarity scores
are on completely different scales:

    BM25 scores:   [14.7, 8.2, 5.1, 3.8, ...]   (unbounded positive floats)
    Cosine scores: [0.82, 0.79, 0.71, 0.65, ...]  (0 to 1)

You can't just add these — a BM25 score of 14.7 would dominate a cosine
score of 0.82. Normalizing helps, but the distributions are different
(BM25 is spiky, cosine is smooth), so normalized scores still don't combine
well.

RRF solves this elegantly: **ignore the scores entirely, use only ranks.**

    RRF_score(chunk) = Σ  1 / (k + rank_i)
                        i

Where:
- rank_i is the chunk's position in result list i (0-indexed: rank 0 = top)
- k is a constant (default 60) that dampens the rank signal

Example with k=60:
    chunk "foo" is rank 0 in semantic, rank 5 in BM25:
        score = 1/(60+0) + 1/(60+5) = 0.01667 + 0.01538 = 0.03205

    chunk "bar" is rank 2 in semantic only (not in BM25 top-k):
        score = 1/(60+2) = 0.01613

    "foo" wins because it appeared in BOTH lists.

Why k=60?
----------
The k constant controls how much rank position matters:
- Small k (e.g., 1): rank 1 gets score 1.0, rank 2 gets 0.5, rank 10 gets
  0.1 → huge gap between top ranks, noisy.
- Large k (e.g., 60): rank 1 gets 0.0164, rank 2 gets 0.0161 → small
  differences, stable. Items appearing in multiple lists get a reliable
  boost over items in just one list.

k=60 is the standard value from the original RRF paper (Cormack et al.,
2009). It works well in practice and you don't need to tune it.

Architecture
------------
    Query
      │
      ├──→ CodeEmbedder.search(query, top_k=20) → semantic results (ranked by cosine)
      │
      ├──→ BM25Index.search(query, top_k=20) → keyword results (ranked by BM25 score)
      │
      └──→ reciprocal_rank_fusion(semantic, bm25) → fused ranking
              │
              ▼
           Top 10 fused results → ready for answer generation (Day 6)
"""

import os
import time
from typing import Optional

from parser.code_parser import CodeChunk, CodeParser
from search.embedder import CodeEmbedder
from search.bm25_index import BM25Index


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    *result_lists: list[dict],
    k: int = 60,
) -> list[str]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    This is the core algorithm — simple enough to derive on a whiteboard:

        For each result list:
            For each item at position `rank`:
                score[item] += 1 / (k + rank + 1)

        Return items sorted by score, descending.

    Parameters
    ----------
    *result_lists : list[dict]
        Any number of ranked result lists. Each result must have a
        "chunk_id" key. Results should be ordered from most relevant
        (index 0) to least relevant.
    k : int
        The RRF constant. Default 60 (from the original paper).
        Higher k = more stable rankings, less sensitivity to exact
        rank position.

    Returns
    -------
    list[str]
        chunk_ids sorted by fused RRF score, descending.

    Why rank+1 in the denominator?
    Because ranks are 0-indexed. If we used just `rank`, the #1 result
    would get 1/(k+0) = 1/k, and the formula would match the original
    paper's 1-indexed formulation: 1/(k + rank) where rank starts at 1.
    Using 0-indexed + 1 is equivalent and avoids off-by-one confusion.
    """
    scores: dict[str, float] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            chunk_id = result["chunk_id"]
            # 1 / (k + rank + 1) — the RRF score contribution
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

    # Sort by score descending
    return sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)


def reciprocal_rank_fusion_detailed(
    *result_lists: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Same as reciprocal_rank_fusion, but returns detailed info for debugging.

    Returns a list of dicts with:
    - chunk_id: the chunk identifier
    - rrf_score: the fused RRF score
    - appeared_in: how many result lists this chunk appeared in
    - ranks: the rank in each result list (None if not present)

    This is useful for understanding WHY a chunk ranked where it did —
    was it because both engines agreed (high confidence) or because one
    engine ranked it #1 (single-source dominance)?
    """
    scores: dict[str, float] = {}
    rank_tracker: dict[str, list[Optional[int]]] = {}

    for list_idx, result_list in enumerate(result_lists):
        for rank, result in enumerate(result_list):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)

            # Track which list and what rank
            if chunk_id not in rank_tracker:
                rank_tracker[chunk_id] = [None] * len(result_lists)
            rank_tracker[chunk_id][list_idx] = rank

    # Sort by score descending
    sorted_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)

    detailed = []
    for chunk_id in sorted_ids:
        ranks = rank_tracker.get(chunk_id, [])
        detailed.append({
            "chunk_id": chunk_id,
            "rrf_score": round(scores[chunk_id], 6),
            "appeared_in": sum(1 for r in ranks if r is not None),
            "ranks": ranks,  # [semantic_rank, bm25_rank] — None means absent
        })

    return detailed


# ---------------------------------------------------------------------------
# Hybrid Search Engine
# ---------------------------------------------------------------------------

class HybridSearch:
    """
    Combines semantic search (embeddings) and keyword search (BM25)
    using Reciprocal Rank Fusion.

    Usage:
        from parser.code_parser import CodeParser
        from search.hybrid_search import HybridSearch

        parser = CodeParser()
        chunks = parser.parse_repository("/path/to/repo")

        hybrid = HybridSearch()
        hybrid.index(chunks)

        results = hybrid.search("function that validates user input", top_k=10)

    The index step builds BOTH the embedding index and the BM25 index.
    The search step runs BOTH searches and fuses them with RRF.
    """

    def __init__(
        self,
        chroma_persist_dir: str = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db"),
        bm25_persist_path: str = "./bm25_index.pkl",
    ) -> None:
        """
        Initialize both search engines.

        Parameters
        ----------
        chroma_persist_dir : str
            Where ChromaDB stores its data on disk.
        bm25_persist_path : str
            Where to save/load the BM25 pickle file.
        """
        self._chroma_persist_dir = chroma_persist_dir
        self._bm25_persist_path = bm25_persist_path

        # These are initialized lazily — only created when needed.
        # This avoids loading the heavy embedding model if we're just
        # loading a pre-built index from disk.
        self._embedder: Optional[CodeEmbedder] = None
        self._bm25: Optional[BM25Index] = None

        # Cache chunks for metadata lookup after fusion
        self._chunks: list[CodeChunk] = []
        self._chunk_lookup: dict[str, CodeChunk] = {}

    def _ensure_embedder(self) -> CodeEmbedder:
        """Lazily initialize the embedder (loads the model on first use)."""
        if self._embedder is None:
            print("Initializing embedding engine...")
            self._embedder = CodeEmbedder(persist_dir=self._chroma_persist_dir)
        return self._embedder

    def _ensure_bm25(self) -> BM25Index:
        """Lazily initialize the BM25 index."""
        if self._bm25 is None:
            self._bm25 = BM25Index()
        return self._bm25

    # -------------------------------------------------------------------
    # Indexing
    # -------------------------------------------------------------------

    def index(
        self,
        chunks: list[CodeChunk],
        collection_name: str = "code_chunks",
    ) -> dict:
        """
        Build both search indexes from the same set of chunks.

        Parameters
        ----------
        chunks : list[CodeChunk]
            Code chunks to index (from CodeParser).
        collection_name : str
            Name for the ChromaDB collection.

        Returns
        -------
        dict
            Stats about what was indexed:
            - total_chunks: how many chunks were indexed
            - embedding_time: seconds to build embedding index
            - bm25_time: seconds to build BM25 index
            - bm25_stats: token statistics from BM25

        Both indexes receive the SAME chunks, ensuring the chunk_ids
        match across both systems. This is critical for fusion — if
        embedder returns chunk "abc123" and BM25 returns "abc123", we
        know it's the same function and can boost its score.
        """
        if not chunks:
            print("⚠ No chunks to index")
            return {"total_chunks": 0}

        # Cache chunks for metadata lookup after fusion
        self._chunks = list(chunks)
        self._chunk_lookup = {c.chunk_id: c for c in chunks}

        print(f"\n{'='*60}")
        print(f"HYBRID INDEX: {len(chunks)} chunks")
        print(f"{'='*60}")

        # --- Build embedding index ---
        print(f"\n[1/2] Building embedding index...")
        embedder = self._ensure_embedder()
        embed_start = time.time()
        embedder.index(chunks, collection_name=collection_name)
        embed_time = time.time() - embed_start

        # --- Build BM25 index ---
        print(f"\n[2/2] Building BM25 index...")
        bm25 = self._ensure_bm25()
        bm25_start = time.time()
        bm25.build(chunks)
        bm25_time = time.time() - bm25_start

        # Save BM25 to disk
        bm25.save(self._bm25_persist_path)

        print(f"\n{'='*60}")
        print(f"✓ Hybrid index built:")
        print(f"  Chunks:     {len(chunks)}")
        print(f"  Embed time: {embed_time:.1f}s")
        print(f"  BM25 time:  {bm25_time:.1f}s")
        print(f"  Total:      {embed_time + bm25_time:.1f}s")
        print(f"{'='*60}")

        return {
            "total_chunks": len(chunks),
            "embedding_time": round(embed_time, 2),
            "bm25_time": round(bm25_time, 2),
            "bm25_stats": bm25.get_stats(),
        }

    # -------------------------------------------------------------------
    # Search
    # -------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 10,
        search_k: int = 20,
        collection_name: str = "code_chunks",
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Run hybrid search: semantic + BM25 + RRF fusion.

        Parameters
        ----------
        query : str
            The search query — can be natural language or exact identifiers.
        top_k : int
            How many fused results to return (default 10).
        search_k : int
            How many results to fetch from EACH engine before fusion
            (default 20). Must be >= top_k. Higher search_k means RRF
            has more candidates to work with, but costs more compute.
        collection_name : str
            Which ChromaDB collection to search.
        rrf_k : int
            The RRF constant (default 60).

        Returns
        -------
        list[dict]
            Fused results, each containing:
            - chunk_id, rrf_score, appeared_in, semantic_rank, bm25_rank
            - file_path, name, chunk_type, language, start_line, end_line
            - code_preview

        Pipeline:
            query → [semantic search (top search_k)]  ─┐
                  → [BM25 search (top search_k)]       ─┤
                                                        ▼
                                    [RRF fusion] → top_k results
        """
        start = time.time()

        # --- Run both searches ---
        embedder = self._ensure_embedder()
        bm25 = self._ensure_bm25()

        semantic_results = embedder.search(query, top_k=search_k, collection_name=collection_name)
        bm25_results = bm25.search(query, top_k=search_k)

        # --- Fuse with RRF ---
        fused = reciprocal_rank_fusion_detailed(
            semantic_results, bm25_results, k=rrf_k,
        )

        # --- Enrich with chunk metadata ---
        # The RRF function only returns chunk_ids and scores. We need to
        # attach the actual metadata (file_path, code, etc.) for each result.
        enriched = []
        for item in fused[:top_k]:
            chunk_id = item["chunk_id"]

            # Try to get metadata from our cached chunks first
            chunk = self._chunk_lookup.get(chunk_id)

            # If not in cache, try to get from the search results themselves
            if chunk is None:
                # Look in semantic results
                for r in semantic_results:
                    if r["chunk_id"] == chunk_id:
                        enriched.append({
                            **item,
                            "semantic_rank": item["ranks"][0],
                            "bm25_rank": item["ranks"][1] if len(item["ranks"]) > 1 else None,
                            "file_path": r.get("file_path", ""),
                            "name": r.get("name", ""),
                            "chunk_type": r.get("chunk_type", ""),
                            "language": r.get("language", ""),
                            "start_line": r.get("start_line", 0),
                            "end_line": r.get("end_line", 0),
                            "code_preview": r.get("code_preview", ""),
                        })
                        break
                else:
                    # Must be from BM25 only
                    for r in bm25_results:
                        if r["chunk_id"] == chunk_id:
                            enriched.append({
                                **item,
                                "semantic_rank": item["ranks"][0],
                                "bm25_rank": item["ranks"][1] if len(item["ranks"]) > 1 else None,
                                "file_path": r.get("file_path", ""),
                                "name": r.get("name", ""),
                                "chunk_type": r.get("chunk_type", ""),
                                "language": r.get("language", ""),
                                "start_line": r.get("start_line", 0),
                                "end_line": r.get("end_line", 0),
                                "code_preview": r.get("code_preview", ""),
                            })
                            break
            else:
                enriched.append({
                    **item,
                    "semantic_rank": item["ranks"][0],
                    "bm25_rank": item["ranks"][1] if len(item["ranks"]) > 1 else None,
                    "file_path": chunk.file_path,
                    "name": chunk.name,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "code_preview": chunk.code[:1000],
                })

        elapsed = time.time() - start

        return enriched

    def search_debug(
        self,
        query: str,
        top_k: int = 10,
        search_k: int = 20,
        collection_name: str = "code_chunks",
    ) -> dict:
        """
        Run hybrid search and return ALL intermediate results for debugging.

        Returns a dict with:
        - query: the original query
        - semantic_results: raw results from embedding search
        - bm25_results: raw results from BM25 search
        - fused_results: the RRF-fused results with full detail
        - comparison: side-by-side view showing how rankings changed

        This is the function you use to verify that fusion actually
        *improves* results — if fused order always matches one engine,
        the other engine isn't contributing anything.
        """
        embedder = self._ensure_embedder()
        bm25 = self._ensure_bm25()

        semantic = embedder.search(query, top_k=search_k, collection_name=collection_name)
        bm25_results = bm25.search(query, top_k=search_k)

        fused = reciprocal_rank_fusion_detailed(semantic, bm25_results)

        # Build comparison: for each fused result, show its rank in each engine
        comparison = []
        for item in fused[:top_k]:
            chunk_id = item["chunk_id"]

            # Find name from either result set
            name = "?"
            file_path = "?"
            for r in semantic + bm25_results:
                if r["chunk_id"] == chunk_id:
                    name = r.get("name", "?")
                    file_path = r.get("file_path", "?")
                    break

            sem_rank = item["ranks"][0]
            bm25_rank = item["ranks"][1] if len(item["ranks"]) > 1 else None

            comparison.append({
                "fused_rank": len(comparison),
                "name": name,
                "file_path": file_path,
                "rrf_score": item["rrf_score"],
                "semantic_rank": sem_rank,
                "bm25_rank": bm25_rank,
                "in_both": sem_rank is not None and bm25_rank is not None,
            })

        return {
            "query": query,
            "semantic_count": len(semantic),
            "bm25_count": len(bm25_results),
            "semantic_results": semantic[:top_k],
            "bm25_results": bm25_results[:top_k],
            "fused_results": fused[:top_k],
            "comparison": comparison,
        }

    # -------------------------------------------------------------------
    # Persistence — load pre-built indexes
    # -------------------------------------------------------------------

    def load(
        self,
        collection_name: str = "code_chunks",
        bm25_path: Optional[str] = None,
    ) -> None:
        """
        Load pre-built indexes from disk.

        This avoids re-parsing and re-indexing when restarting the server.
        The embedding index is loaded automatically by ChromaDB (it's
        persisted on disk). We just need to load the BM25 pickle.
        """
        bm25_path = bm25_path or self._bm25_persist_path

        # Embedder — ChromaDB auto-loads from persist_dir
        embedder = self._ensure_embedder()

        # Verify the collection exists and has data
        info = embedder.get_collection_info(collection_name)
        if info["count"] == 0:
            print(f"⚠ Embedding collection '{collection_name}' is empty — need to index first")
            return

        print(f"  Embedding index: {info['count']} chunks in '{collection_name}'")

        # BM25 — load from pickle
        if os.path.exists(bm25_path):
            bm25 = self._ensure_bm25()
            bm25.load(bm25_path)

            # Rebuild chunk lookup from BM25's stored chunks
            self._chunks = bm25._chunks
            self._chunk_lookup = {c.chunk_id: c for c in self._chunks}
        else:
            print(f"⚠ BM25 index not found at {bm25_path} — need to index first")

    def get_stats(self) -> dict:
        """Get combined stats from both indexes."""
        stats: dict[str, object] = {"hybrid": True}

        if self._embedder is not None:
            stats["embedding"] = self._embedder.get_collection_info()

        if self._bm25 is not None:
            stats["bm25"] = self._bm25.get_stats()

        stats["cached_chunks"] = len(self._chunk_lookup)

        return stats


# ---------------------------------------------------------------------------
# CLI — test entrypoint with side-by-side comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    query = sys.argv[2] if len(sys.argv) > 2 else None

    # Step 1: Parse
    print("Parsing repository...")
    parser = CodeParser()
    if os.path.isfile(target):
        chunks = parser.parse_file(target)
    else:
        chunks = parser.parse_repository(target)

    # Step 2: Build hybrid index
    hybrid = HybridSearch()
    hybrid.index(chunks)

    # Step 3: Search
    if query:
        queries = [query]
    else:
        # Default test queries that show different strengths:
        # - Exact identifier (BM25 should win)
        # - Conceptual/natural language (embeddings should win)
        # - Mixed (both should contribute)
        queries = [
            "tokenize_code",               # exact function name → BM25 strength
            "function that parses code",    # conceptual → embedding strength
            "walk syntax tree find nodes",  # mixed → both should help
        ]

    for q in queries:
        debug = hybrid.search_debug(q)

        print(f"\n{'='*70}")
        print(f"QUERY: \"{q}\"")
        print(f"{'='*70}")

        # Show side-by-side comparison
        print(f"\n{'Rank':<6} {'Name':<30} {'Semantic':<10} {'BM25':<10} {'RRF Score':<12} {'Both?'}")
        print(f"{'-'*6} {'-'*30} {'-'*10} {'-'*10} {'-'*12} {'-'*5}")

        for c in debug["comparison"]:
            sem = f"#{c['semantic_rank']}" if c['semantic_rank'] is not None else "—"
            bm25 = f"#{c['bm25_rank']}" if c['bm25_rank'] is not None else "—"
            both = "✓" if c['in_both'] else ""
            print(f"  {c['fused_rank']:<4} {c['name']:<30} {sem:<10} {bm25:<10} {c['rrf_score']:<12} {both}")

        # Summary
        both_count = sum(1 for c in debug["comparison"] if c["in_both"])
        print(f"\n  Summary: {both_count}/{len(debug['comparison'])} results appeared in BOTH engines")
        print(f"  This means RRF is boosting results that both engines agree on.")
