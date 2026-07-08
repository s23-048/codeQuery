"""
CodeQuery — LLM Answer Generation
===================================

This module takes the top chunks from hybrid search and uses an LLM to
generate a grounded answer with citations.

Supports two providers:
- **Google Gemini** (default, free tier available) — gemini-2.0-flash
- **OpenAI GPT-4o** (fallback, requires paid API key)

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
3. Tell the LLM to ONLY use the provided context
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

# Load .env so API keys are available
load_dotenv()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Provider: "gemini" (free) or "openai" (paid)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()

# API keys — check multiple env var names for flexibility
GEMINI_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY")
    or os.environ.get("LLM_API_KEY", "")
)
OPENAI_API_KEY = (
    os.environ.get("OPENAI_API_KEY")
    or os.environ.get("LLM_API_KEY", "")
)

# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-3.5-flash",     # Free tier, fast, great for code
    "openai": "gpt-4o",               # Paid, best quality
}
MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS.get(LLM_PROVIDER, "gemini-3.5-flash"))

# How many chunks to include in the LLM context.
DEFAULT_CONTEXT_CHUNKS = 5

# System prompt — shapes how the LLM behaves.
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
# Gemini Client
# ---------------------------------------------------------------------------

class _GeminiClient:
    """
    Wrapper around Google's Gemini API using the google-genai SDK.

    Gemini 2.0 Flash is free-tier eligible, fast (~1s responses), and
    excellent for code understanding. It's our default provider.

    The google-genai SDK uses a simple generate_content() call — no
    chat sessions needed for single-turn Q&A like ours.
    """

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a response from Gemini."""
        from google.genai import types

        response = self._client.models.generate_content(
            model=self._model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0,           # Deterministic
                max_output_tokens=1024,
            ),
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# OpenAI Client
# ---------------------------------------------------------------------------

class _OpenAIClient:
    """
    Wrapper around OpenAI's GPT-4o API.

    Used as a fallback when LLM_PROVIDER=openai. Requires a paid API key.
    """

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_message: str) -> str:
        """Generate a response from OpenAI."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            max_tokens=1024,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Answerer
# ---------------------------------------------------------------------------

class CodeAnswerer:
    """
    Generates grounded answers from code context using an LLM.

    Supports both Gemini (free) and OpenAI (paid) as providers.

    Usage:
        from llm.answerer import CodeAnswerer

        answerer = CodeAnswerer()                    # Uses Gemini by default
        answerer = CodeAnswerer(provider="openai")   # Or use OpenAI

        result = answerer.answer(
            query="How does the parser handle JavaScript files?",
            chunks=search_results,
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
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """
        Initialize the answerer.

        Parameters
        ----------
        api_key : str, optional
            API key. If not provided, reads from env vars.
        model : str, optional
            Model name. Defaults based on provider.
        provider : str, optional
            "gemini" or "openai". Defaults to LLM_PROVIDER env var.
        """
        self._provider = provider or LLM_PROVIDER
        self._model = model or MODEL

        if self._provider == "gemini":
            self._api_key = api_key or GEMINI_API_KEY
        else:
            self._api_key = api_key or OPENAI_API_KEY

        self._client: object = None

    def _ensure_client(self) -> object:
        """Lazily initialize the LLM client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError(
                    f"No API key found for provider '{self._provider}'. "
                    f"Set GEMINI_API_KEY (or GOOGLE_API_KEY) in your .env file "
                    f"for Gemini, or OPENAI_API_KEY for OpenAI."
                )

            if self._provider == "gemini":
                self._client = _GeminiClient(self._api_key, self._model)
            else:
                self._client = _OpenAIClient(self._api_key, self._model)

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
            - sources: list[dict] — the chunks used as context
            - model: str — which model was used
            - provider: str — "gemini" or "openai"
            - latency_ms: int — end-to-end generation time in milliseconds
            - context_chunks: int — how many chunks were in the context
            - query: str — the original query
        """
        start = time.time()

        # --- Build context from top chunks ---
        selected = chunks[:max_context_chunks]
        context = build_context(selected, max_context_chunks)

        # --- Build the prompt ---
        user_message = (
            f"Question: {query}\n\n"
            f"Code Context:\n{context}"
        )

        # --- Call LLM ---
        client = self._ensure_client()

        try:
            answer_text = client.generate(SYSTEM_PROMPT, user_message)  # type: ignore[union-attr]
        except Exception as e:
            answer_text = f"Error generating answer: {str(e)}"

        elapsed_ms = int((time.time() - start) * 1000)

        # --- Build sources list ---
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
            "provider": self._provider,
            "latency_ms": elapsed_ms,
            "context_chunks": len(selected),
            "query": query,
        }

    def health_check(self) -> bool:
        """Check if the LLM API is reachable."""
        try:
            client = self._ensure_client()
            # Quick test with a minimal prompt
            client.generate("You are a test.", "Say OK")  # type: ignore[union-attr]
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# CLI — quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing CodeAnswerer...")
    print(f"  Provider: {LLM_PROVIDER}")
    print(f"  Model: {MODEL}")

    key = GEMINI_API_KEY if LLM_PROVIDER == "gemini" else OPENAI_API_KEY
    print(f"  API Key: {'✓ set' if key else '✗ NOT SET'}")

    if not key:
        print(f"\n⚠ Set {'GEMINI_API_KEY' if LLM_PROVIDER == 'gemini' else 'OPENAI_API_KEY'} in .env")
        exit(1)

    # Create fake chunks for testing
    test_chunks = [
        {
            "file_path": "parser/code_parser.py",
            "name": "parse_file",
            "chunk_type": "function",
            "start_line": 200,
            "end_line": 250,
            "code_preview": (
                "def parse_file(self, file_path):\n"
                '    """Parse a single file into CodeChunks."""\n'
                "    lang = self._detect_language(file_path)\n"
                "    tree = self._parser.parse(source_code)\n"
                "    return self._extract_chunks(tree, file_path)"
            ),
        },
        {
            "file_path": "search/embedder.py",
            "name": "CodeEmbedder",
            "chunk_type": "class",
            "start_line": 85,
            "end_line": 344,
            "code_preview": (
                "class CodeEmbedder:\n"
                '    """Embeds code chunks using MiniLM and stores in ChromaDB."""\n'
                "    def search(self, query, top_k=20):\n"
                "        # Semantic similarity search\n"
                "        pass"
            ),
        },
    ]

    answerer = CodeAnswerer()
    result = answerer.answer(
        query="How does the parser handle files?",
        chunks=test_chunks,
    )

    print(f"\n{'='*60}")
    print(f"Query: {result['query']}")
    print(f"Provider: {result['provider']}")
    print(f"Model: {result['model']}")
    print(f"Latency: {result['latency_ms']}ms")
    print(f"Context chunks: {result['context_chunks']}")
    print(f"{'='*60}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nSources:")
    for s in result["sources"]:
        print(f"  - {s['chunk_type']} {s['name']} in {s['file_path']} "
              f"(lines {s['start_line']}-{s['end_line']})")
