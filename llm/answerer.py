"""
CodeQuery — LLM Answer Generation
===================================

This module takes the top chunks from hybrid search and uses GPT-4o to
generate a grounded answer with citations.

Why "grounded" answers matter
------------------------------
The whole point of RAG (Retrieval-Augmented Generation) is to constrain the
LLM to only use information that was actually retrieved. Without this:
- The LLM would hallucinate function names that don't exist
- It would guess at implementation details instead of citing real code
- There'd be no way to verify the answer against the source

Our approach:
1. Take the top 5 fused results from hybrid search
2. Format them as a structured context block with file paths and line numbers
3. Tell GPT-4o to ONLY use the provided context
4. Return the answer alongside the source citations

temperature=0 is deliberate — we want deterministic, factual answers, not
creative writing. For code analysis, reproducibility matters more than variety.

Latency tracking
-----------------
We measure and return `latency_ms` for every answer. Sub-2-second end-to-end
is the target. This is a claim worth being able to back up in a live demo.
"""

import os
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

# Load .env so OPENAI_API_KEY (or LLM_API_KEY) is available
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Support both OPENAI_API_KEY and LLM_API_KEY from .env
API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "gpt-4o")

# How many chunks to include in the LLM context.
# More chunks = more context = better answers, but also more tokens = more cost.
# 5 is the sweet spot: enough to cover most questions, small enough to stay
# under GPT-4o's sweet spot for grounded answers (~2000 tokens of context).
DEFAULT_CONTEXT_CHUNKS = 5

# System prompt — this is the instruction that shapes how GPT-4o behaves.
# Key constraints:
# - "ONLY the provided code context" → prevents hallucination
# - "Mention file paths and line numbers" → forces citations
# - "If the answer isn't in the context, say so" → honest about limits
SYSTEM_PROMPT = """You are an expert software engineer analyzing a codebase.
Answer the user's question using ONLY the provided code context.
Be specific: mention file paths, function names, and line numbers.
If the answer isn't in the provided context, say so — do not guess or hallucinate.
Keep answers concise but complete."""


# ---------------------------------------------------------------------------
# Context Builder
# ---------------------------------------------------------------------------

def build_context(chunks: list[dict], max_chunks: int = DEFAULT_CONTEXT_CHUNKS) -> str:
    """
    Format retrieved chunks into a structured context block for the LLM.

    Each chunk gets a clear header with file path, function name, type,
    and line range. This structure helps the LLM cite sources accurately.

    Parameters
    ----------
    chunks : list[dict]
        Search results from HybridSearch.search(). Each dict has:
        file_path, name, chunk_type, start_line, end_line, code_preview
    max_chunks : int
        Maximum number of chunks to include (default 5).

    Returns
    -------
    str
        Formatted context string ready to inject into the prompt.

    Example output:
        --- Chunk 1/5 ---
        File: parser/code_parser.py (lines 103-118)
        Type: function | Name: to_search_text
        Code:
        def to_search_text(self) -> str:
            ...
    """
    if not chunks:
        return "(No code context available)"

    selected = chunks[:max_chunks]
    context_parts = []

    for i, chunk in enumerate(selected):
        header = (
            f"--- Chunk {i+1}/{len(selected)} ---\n"
            f"File: {chunk.get('file_path', 'unknown')} "
            f"(lines {chunk.get('start_line', '?')}-{chunk.get('end_line', '?')})\n"
            f"Type: {chunk.get('chunk_type', '?')} | Name: {chunk.get('name', '?')}\n"
            f"Code:\n"
        )
        code = chunk.get("code_preview", "(no code available)")
        context_parts.append(header + code)

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Answerer
# ---------------------------------------------------------------------------

class CodeAnswerer:
    """
    Generates grounded answers from code context using GPT-4o.

    Usage:
        from llm.answerer import CodeAnswerer

        answerer = CodeAnswerer()
        result = answerer.answer(
            query="How does the parser handle JavaScript files?",
            chunks=search_results,   # from HybridSearch.search()
        )
        print(result["answer"])
        print(result["sources"])

    The answerer does NOT do any searching — it only takes pre-retrieved
    chunks and generates an answer. The search → answer pipeline is:

        HybridSearch.search(query) → top chunks → CodeAnswerer.answer(query, chunks)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = MODEL,
    ) -> None:
        """
        Initialize the answerer with an OpenAI client.

        Parameters
        ----------
        api_key : str, optional
            OpenAI API key. If not provided, reads from env vars.
        model : str
            Which model to use (default: gpt-4o).
        """
        self._api_key = api_key or API_KEY
        self._model = model
        self._client: Optional[OpenAI] = None

    def _ensure_client(self) -> OpenAI:
        """Lazily initialize the OpenAI client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    "No OpenAI API key found. Set OPENAI_API_KEY or LLM_API_KEY "
                    "in your .env file, or pass api_key to CodeAnswerer()."
                )
            self._client = OpenAI(api_key=self._api_key)
        return self._client

    def answer(
        self,
        query: str,
        chunks: list[dict],
        max_context_chunks: int = DEFAULT_CONTEXT_CHUNKS,
    ) -> dict:
        """
        Generate an answer from retrieved code chunks.

        Parameters
        ----------
        query : str
            The user's question.
        chunks : list[dict]
            Retrieved chunks from hybrid search.
        max_context_chunks : int
            How many chunks to include in the context (default 5).

        Returns
        -------
        dict
            - answer: str — the generated answer
            - sources: list[dict] — the chunks used as context (for citations)
            - model: str — which model was used
            - latency_ms: int — end-to-end generation time in milliseconds
            - context_chunks: int — how many chunks were in the context
            - query: str — the original query (for logging)
        """
        start = time.time()

        # --- Build context from top chunks ---
        selected = chunks[:max_context_chunks]
        context = build_context(selected, max_context_chunks)

        # --- Build the prompt ---
        # The user message contains both the question and the context.
        # Keeping them in one message (rather than separate system messages)
        # makes it clearer to the model what to answer vs. what to reference.
        user_message = (
            f"Question: {query}\n\n"
            f"Code Context:\n{context}"
        )

        # --- Call GPT-4o ---
        client = self._ensure_client()

        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0,       # Deterministic — we want facts, not creativity
                max_tokens=1024,     # Enough for a detailed answer, not a novel
            )
            answer_text = response.choices[0].message.content or ""
        except Exception as e:
            answer_text = f"Error generating answer: {str(e)}"

        elapsed_ms = int((time.time() - start) * 1000)

        # --- Build sources list ---
        # These are the citations that make the tool trustworthy.
        # Each source tells the user exactly where to look in the codebase.
        sources = []
        for chunk in selected:
            sources.append({
                "file_path": chunk.get("file_path", ""),
                "name": chunk.get("name", ""),
                "chunk_type": chunk.get("chunk_type", ""),
                "start_line": chunk.get("start_line", 0),
                "end_line": chunk.get("end_line", 0),
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "model": self._model,
            "latency_ms": elapsed_ms,
            "context_chunks": len(selected),
            "query": query,
        }

    def health_check(self) -> bool:
        """Check if the OpenAI API is reachable."""
        try:
            client = self._ensure_client()
            # A minimal API call to verify the key works
            client.models.list()
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CLI — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick test with fake chunks to verify the LLM call works
    print("Testing CodeAnswerer...")
    print(f"  Model: {MODEL}")
    print(f"  API Key: {'✓ set' if API_KEY else '✗ NOT SET'}")

    if not API_KEY:
        print("\n⚠ Set OPENAI_API_KEY or LLM_API_KEY in .env to test")
        exit(1)

    # Create fake chunks for testing
    test_chunks = [
        {
            "file_path": "parser/code_parser.py",
            "name": "parse_file",
            "chunk_type": "function",
            "start_line": 200,
            "end_line": 250,
            "code_preview": "def parse_file(self, file_path):\n    \"\"\"Parse a single file.\"\"\"\n    lang = self._detect_language(file_path)\n    tree = self._parser.parse(source_code)\n    return self._extract_chunks(tree, file_path)",
        },
        {
            "file_path": "search/embedder.py",
            "name": "CodeEmbedder",
            "chunk_type": "class",
            "start_line": 85,
            "end_line": 344,
            "code_preview": "class CodeEmbedder:\n    \"\"\"Embeds code chunks using MiniLM and stores in ChromaDB.\"\"\"\n    def search(self, query, top_k=20):\n        # Semantic similarity search\n        pass",
        },
    ]

    answerer = CodeAnswerer()
    result = answerer.answer(
        query="How does the parser handle files?",
        chunks=test_chunks,
    )

    print(f"\n{'='*60}")
    print(f"Query: {result['query']}")
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Context chunks: {result['context_chunks']}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s['chunk_type']} {s['name']} in {s['file_path']} "
              f"(lines {s['start_line']}-{s['end_line']})")
