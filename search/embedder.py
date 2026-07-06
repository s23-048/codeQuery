"""
CodeQuery — Embedding Index (ChromaDB + MiniLM)
================================================

This module takes CodeChunks, embeds them using a sentence-transformer model,
and stores them in ChromaDB for semantic (meaning-based) search.

How embeddings work (the intuition)
------------------------------------
An embedding model turns text into a list of numbers (a "vector"). Similar
text gets similar numbers. For example:

    "def add(a, b): return a + b"     → [0.12, -0.34, 0.56, ...]  (384 numbers)
    "function that adds two numbers"  → [0.11, -0.33, 0.55, ...]  (very similar!)
    "class DatabaseConnection"        → [0.87, 0.22, -0.91, ...]  (very different)

When someone searches "function that adds numbers", we embed their query the
same way, then find which stored vectors are closest → those are the most
semantically similar chunks.

Model choice: all-MiniLM-L6-v2
-------------------------------
- 384-dimensional vectors (small, fast)
- Good general-purpose English text understanding
- Not code-specific, but good enough — code is partly English (function names,
  comments, docstrings) and partly syntax. For a code-specific model you'd use
  CodeBERT or StarEncoder, but MiniLM is simpler and the quality difference is
  small for this use case.

ChromaDB
--------
ChromaDB is a vector database — it stores embeddings and lets you search them
by similarity. Think of it as "a database where the WHERE clause is 'find the
most similar vectors' instead of 'find exact matches'."

Key concepts:
- Collection: like a database table. We create one per indexed repo.
- Document: the text we're indexing (chunk's search text).
- Embedding: the vector representation of that text.
- Metadata: extra fields stored alongside (file_path, name, chunk_type, etc.)
- HNSW: the algorithm ChromaDB uses internally to find similar vectors fast.
  It builds a graph of neighbors so it doesn't have to compare every vector
  to the query — O(log n) instead of O(n).

Batch encoding
--------------
Encoding one chunk at a time is the #2 perf mistake (after rebuilding parsers
per file). The model has fixed startup overhead per batch, so encoding 64
chunks at once is ~50x faster than encoding them one at a time. We use
batch_size=64 throughout.
"""

import os
import time
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from parser.code_parser import CodeChunk


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default embedding model — small, fast, 384-dimensional vectors.
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ChromaDB persistence directory
DEFAULT_PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")

# Batch size for encoding — sweet spot between memory and speed.
# 64 is good for MiniLM on CPU. On GPU you could go to 256+.
ENCODE_BATCH_SIZE = 64

# How many results to return by default from semantic search.
DEFAULT_TOP_K = 20


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class CodeEmbedder:
    """
    Embeds code chunks and stores them in ChromaDB for semantic search.

    Usage:
        from parser.code_parser import CodeParser
        from search.embedder import CodeEmbedder

        parser = CodeParser()
        chunks = parser.parse_repository("/path/to/repo")

        embedder = CodeEmbedder()
        embedder.index(chunks, collection_name="my_repo")

        results = embedder.search("function that processes payments", top_k=5)

    Architecture:
        CodeChunks → to_search_text() → SentenceTransformer → vectors → ChromaDB
                                                                            ↑
        Query text → SentenceTransformer → query vector → cosine similarity ─┘
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        persist_dir: str = DEFAULT_PERSIST_DIR,
    ) -> None:
        """
        Initialize the embedder.

        Parameters
        ----------
        model_name : str
            HuggingFace model name for the sentence transformer.
        persist_dir : str
            Directory where ChromaDB stores its data on disk.
            Set to None for in-memory only (faster for testing).
        """
        print(f"  Loading embedding model: {model_name}...")
        start = time.time()
        self._model = SentenceTransformer(model_name)
        print(f"  Model loaded in {time.time() - start:.1f}s")

        # Initialize ChromaDB client
        # PersistentClient saves to disk so we don't re-embed on restart.
        if persist_dir:
            os.makedirs(persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection: Optional[chromadb.Collection] = None

    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """
        Get or create a ChromaDB collection.

        We use cosine distance as the similarity metric. Cosine measures
        the angle between two vectors — vectors pointing in the same direction
        are similar regardless of their magnitude. This is standard for
        text embeddings because document length shouldn't affect similarity.

        Other options: "l2" (Euclidean) or "ip" (inner product).
        Cosine is the right choice here.
        """
        return self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def index(
        self,
        chunks: list[CodeChunk],
        collection_name: str = "code_chunks",
    ) -> int:
        """
        Embed and index a list of code chunks into ChromaDB.

        Parameters
        ----------
        chunks : list[CodeChunk]
            The chunks to embed and store.
        collection_name : str
            Name of the ChromaDB collection to store in.

        Returns
        -------
        int
            Number of chunks indexed.

        Steps:
        1. Build search texts from chunks (to_search_text())
        2. Batch-encode texts into embedding vectors
        3. Upsert into ChromaDB with metadata

        We use upsert (update-or-insert) instead of add so that re-indexing
        the same repo doesn't create duplicates — chunks with the same ID
        get updated, new ones get added.
        """
        if not chunks:
            print("⚠ No chunks to index")
            return 0

        self._collection = self._get_or_create_collection(collection_name)

        # --- Step 1: Build search texts ---
        texts = [chunk.to_search_text() for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]

        # Build metadata for each chunk. ChromaDB stores this alongside
        # the embedding so we can return it in search results without
        # needing a second lookup.
        metadatas = []
        for chunk in chunks:
            metadatas.append({
                "file_path": chunk.file_path,
                "name": chunk.name,
                "chunk_type": chunk.chunk_type,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                # Store first 1000 chars of code for display in results.
                # ChromaDB metadata values must be str/int/float/bool.
                "code_preview": chunk.code[:1000],
            })

        # --- Step 2: Batch-encode ---
        print(f"  Encoding {len(texts)} chunks (batch_size={ENCODE_BATCH_SIZE})...")
        start = time.time()
        embeddings = self._model.encode(
            texts,
            batch_size=ENCODE_BATCH_SIZE,
            show_progress_bar=len(texts) > 100,  # Only show bar for large sets
            normalize_embeddings=True,  # Pre-normalize for cosine similarity
        )
        encode_time = time.time() - start
        print(f"  Encoded in {encode_time:.1f}s "
              f"({len(texts)/max(encode_time, 0.001):.0f} chunks/sec)")

        # --- Step 3: Upsert into ChromaDB ---
        # ChromaDB has a max batch size for upsert, so we chunk it.
        CHROMA_BATCH = 500
        for i in range(0, len(ids), CHROMA_BATCH):
            end = min(i + CHROMA_BATCH, len(ids))
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end].tolist(),
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )

        print(f"✓ Indexed {len(chunks)} chunks into collection '{collection_name}'")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        collection_name: str = "code_chunks",
    ) -> list[dict]:
        """
        Search for chunks semantically similar to the query.

        Parameters
        ----------
        query : str
            Natural language query (e.g., "function that handles user login")
        top_k : int
            Number of results to return.
        collection_name : str
            Which collection to search.

        Returns
        -------
        list[dict]
            Ranked results, each containing:
            - chunk_id: str
            - score: float (0 to 1, higher = more similar)
            - file_path, name, chunk_type, language, start_line, end_line
            - code_preview: first 1000 chars
            - document: full search text

        How it works:
        1. Embed the query with the same model
        2. ChromaDB finds the top_k nearest vectors (cosine similarity)
        3. We reformat the results into a clean list of dicts
        """
        # Ensure we have a collection
        if self._collection is None or self._collection.name != collection_name:
            self._collection = self._get_or_create_collection(collection_name)

        # Check if collection has any data
        if self._collection.count() == 0:
            print("⚠ Collection is empty — index some chunks first")
            return []

        # Embed the query
        query_embedding = self._model.encode(
            [query],
            normalize_embeddings=True,
        ).tolist()

        # Search ChromaDB
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=min(top_k, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        # --- Reformat results ---
        # ChromaDB returns results in a nested format:
        #   {"ids": [[id1, id2]], "distances": [[d1, d2]], ...}
        # We flatten this into a list of dicts.
        formatted = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                # ChromaDB returns cosine DISTANCE (0 = identical, 2 = opposite).
                # Convert to similarity SCORE (1 = identical, 0 = orthogonal).
                distances = results["distances"]
                distance = distances[0][i] if distances else 0.0
                score = 1.0 - (distance / 2.0)

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""

                formatted.append({
                    "chunk_id": chunk_id,
                    "score": round(score, 4),
                    "file_path": metadata.get("file_path", ""),
                    "name": metadata.get("name", ""),
                    "chunk_type": metadata.get("chunk_type", ""),
                    "language": metadata.get("language", ""),
                    "start_line": metadata.get("start_line", 0),
                    "end_line": metadata.get("end_line", 0),
                    "code_preview": metadata.get("code_preview", ""),
                    "document": document,
                })

        return formatted

    def delete_collection(self, collection_name: str = "code_chunks") -> None:
        """Delete a collection and all its data."""
        try:
            self._client.delete_collection(collection_name)
            print(f"✓ Deleted collection '{collection_name}'")
        except Exception:
            print(f"  Collection '{collection_name}' doesn't exist")

    def get_collection_info(self, collection_name: str = "code_chunks") -> dict:
        """Get info about a collection (count, metadata)."""
        try:
            collection = self._client.get_collection(collection_name)
            return {
                "name": collection_name,
                "count": collection.count(),
                "metadata": collection.metadata,
            }
        except Exception:
            return {"name": collection_name, "count": 0, "metadata": {}}


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

    # Step 2: Index
    embedder = CodeEmbedder(persist_dir="./chroma_db")
    embedder.index(chunks)

    # Step 3: Search (if query provided)
    if query:
        print(f"\n{'='*60}")
        print(f"SEARCH: \"{query}\"")
        print(f"{'='*60}")
        results = embedder.search(query, top_k=5)
        for i, r in enumerate(results):
            print(f"\n  #{i+1} [{r['score']:.4f}] {r['chunk_type']} {r['name']}")
            print(f"      File: {r['file_path']} (lines {r['start_line']}–{r['end_line']})")
            # Show first 3 lines of code
            preview_lines = r["code_preview"].split("\n")[:3]
            for line in preview_lines:
                print(f"      │ {line}")
    else:
        # Interactive mode — keep asking for queries
        print(f"\n{'='*60}")
        print("INTERACTIVE SEARCH (type 'quit' to exit)")
        print(f"{'='*60}")
        while True:
            query = input("\n🔍 Query: ").strip()
            if query.lower() in ("quit", "exit", "q"):
                break
            if not query:
                continue

            results = embedder.search(query, top_k=5)
            if not results:
                print("  No results found.")
                continue

            for i, r in enumerate(results):
                print(f"\n  #{i+1} [{r['score']:.4f}] {r['chunk_type']} {r['name']}")
                print(f"      File: {r['file_path']} (lines {r['start_line']}–{r['end_line']})")
                preview_lines = r["code_preview"].split("\n")[:3]
                for line in preview_lines:
                    print(f"      │ {line}")
