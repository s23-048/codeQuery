"""
CodeQuery — FastAPI Server
============================

Three routes are enough for a fully functional demo:

    POST /index  — Clone a repo → parse → chunk → build hybrid index
    POST /query  — Search + answer with citations
    GET  /status — Health check + index stats

Why FastAPI?
-------------
- Automatic OpenAPI docs at /docs (free Swagger UI for demos)
- Pydantic request/response validation (catches bad input before it hits code)
- Async support (though we use sync for simplicity — mention async as a scaling improvement)
- Very little boilerplate compared to Flask

State management
-----------------
We store the HybridSearch and CodeAnswerer instances in module-level variables.
This is fine for a demo/single-user app. In production, you'd move this to
Redis or a database so multiple workers share state. This is a good interview
point: "I used in-memory state for the demo, but production would use Redis
for shared state across workers."

Running the server:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Then visit http://localhost:8000/docs for the interactive API.
"""

import os
import shutil
import time
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from parser.code_parser import CodeParser
from search.hybrid_search import HybridSearch
from llm.answerer import CodeAnswerer

# Load .env
load_dotenv()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="CodeQuery",
    description=(
        "Repository-scale code search: AST chunking + hybrid search "
        "(BM25 + embeddings + RRF fusion) + GPT-4o answers with citations."
    ),
    version="1.0.0",
)

# CORS — allow Streamlit (port 8501) and any localhost dev server to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
# Module-level state — simple for a demo, mention Redis for production.

class AppState:
    """Holds the shared state for the application."""
    def __init__(self) -> None:
        self.hybrid_search: Optional[HybridSearch] = None
        self.answerer: Optional[CodeAnswerer] = None
        self.parser: Optional[CodeParser] = None
        self.indexed_repo: Optional[str] = None
        self.chunk_count: int = 0
        self.index_time: float = 0.0

state = AppState()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
# Pydantic models validate input and generate OpenAPI docs automatically.

class IndexRequest(BaseModel):
    """Request body for POST /index"""
    repo_url: str = Field(
        ...,
        description="Git repository URL to clone and index",
        examples=["https://github.com/user/repo.git"],
    )
    collection_name: str = Field(
        default="code_chunks",
        description="Name for the ChromaDB collection",
    )

class IndexResponse(BaseModel):
    """Response from POST /index"""
    status: str
    repo_url: str
    chunks_indexed: int
    index_time_seconds: float
    message: str

class QueryRequest(BaseModel):
    """Request body for POST /query"""
    query: str = Field(
        ...,
        description="Natural language question about the codebase",
        examples=["How does the parser handle JavaScript files?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of chunks to use as context (1-20)",
    )

class QueryResponse(BaseModel):
    """Response from POST /query"""
    answer: str
    sources: list[dict]
    model: str
    latency_ms: int
    context_chunks: int
    query: str

class StatusResponse(BaseModel):
    """Response from GET /status"""
    status: str
    indexed_repo: Optional[str]
    chunks_indexed: int
    search_stats: Optional[dict]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/status", response_model=StatusResponse)
async def get_status():
    """
    Health check + index status.

    Returns whether a repo is indexed, how many chunks, and search stats.
    Hit this first to verify the server is alive and check what's indexed.
    """
    search_stats = None
    if state.hybrid_search is not None:
        search_stats = state.hybrid_search.get_stats()

    return StatusResponse(
        status="ok",
        indexed_repo=state.indexed_repo,
        chunks_indexed=state.chunk_count,
        search_stats=search_stats,
    )


@app.post("/index", response_model=IndexResponse)
async def index_repo(request: IndexRequest):
    """
    Clone a repository and build the hybrid search index.

    Pipeline:
        1. Clone the repo (git clone --depth=1 for speed)
        2. Parse all supported files into CodeChunks (Tree-sitter AST)
        3. Build both embedding index (ChromaDB) and BM25 index
        4. Initialize the LLM answerer

    This is a synchronous operation — it blocks until complete.
    For production, you'd make this async with a task queue (Celery/RQ).
    """
    start = time.time()
    clone_dir = os.environ.get("CLONE_DIR", "./repos")
    repo_url = request.repo_url.strip()

    # --- Step 1: Clone ---
    # Extract repo name from URL for the local directory
    repo_name = repo_url.rstrip("/").split("/")[-1].replace(".git", "")
    repo_path = os.path.join(clone_dir, repo_name)

    try:
        # Clean up any previous clone
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)

        os.makedirs(clone_dir, exist_ok=True)

        print(f"\n[1/4] Cloning {repo_url}...")
        import git
        git.Repo.clone_from(
            repo_url,
            repo_path,
            depth=1,                    # Shallow clone — we only need latest code
            single_branch=True,
        )
        print(f"  ✓ Cloned to {repo_path}")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to clone repository: {str(e)}",
        )

    # --- Step 2: Parse ---
    print(f"\n[2/4] Parsing repository...")
    state.parser = CodeParser()
    chunks = state.parser.parse_repository(repo_path)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No supported files found in repository (looking for .py, .js, .ts, .jsx, .tsx)",
        )

    print(f"  ✓ {len(chunks)} chunks extracted")

    # --- Step 3: Build hybrid index ---
    print(f"\n[3/4] Building hybrid search index...")
    state.hybrid_search = HybridSearch()
    index_stats = state.hybrid_search.index(chunks, collection_name=request.collection_name)

    # --- Step 4: Initialize answerer ---
    print(f"\n[4/4] Initializing LLM answerer...")
    state.answerer = CodeAnswerer()
    state.indexed_repo = repo_url
    state.chunk_count = len(chunks)

    elapsed = time.time() - start
    state.index_time = elapsed

    print(f"\n{'='*60}")
    print(f"✓ Index complete: {len(chunks)} chunks in {elapsed:.1f}s")
    print(f"{'='*60}")

    return IndexResponse(
        status="success",
        repo_url=repo_url,
        chunks_indexed=len(chunks),
        index_time_seconds=round(elapsed, 2),
        message=f"Successfully indexed {len(chunks)} code chunks from {repo_name}",
    )


@app.post("/query", response_model=QueryResponse)
async def query_code(request: QueryRequest):
    """
    Search the indexed codebase and generate an answer.

    Pipeline:
        1. Hybrid search (semantic + BM25 + RRF fusion)
        2. Take top-k fused results as context
        3. Build prompt with code context
        4. Call GPT-4o for a grounded answer
        5. Return answer + source citations

    Must call /index first to have a searchable index.
    """
    # --- Guard: index must exist ---
    if state.hybrid_search is None or state.answerer is None:
        raise HTTPException(
            status_code=400,
            detail="No repository indexed yet. Call POST /index first.",
        )

    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    # --- Step 1: Hybrid search ---
    # Fetch more results than we need (search_k=20) so RRF has enough
    # candidates, then the answerer takes the top 5.
    search_results = state.hybrid_search.search(
        query=query,
        top_k=request.top_k,
        search_k=20,
    )

    if not search_results:
        return QueryResponse(
            answer="No relevant code found for your query.",
            sources=[],
            model=state.answerer._model,
            latency_ms=0,
            context_chunks=0,
            query=query,
        )

    # --- Step 2: Generate answer ---
    result = state.answerer.answer(
        query=query,
        chunks=search_results,
        max_context_chunks=request.top_k,
    )

    return QueryResponse(**result)


# ---------------------------------------------------------------------------
# Startup / shutdown events
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup():
    """Log startup info."""
    print(f"\n{'='*60}")
    print(f"CodeQuery API starting...")
    print(f"  Docs: http://localhost:8000/docs")
    print(f"  Status: http://localhost:8000/status")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Run directly (for development)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.environ.get("API_HOST", "0.0.0.0"),
        port=int(os.environ.get("API_PORT", "8000")),
        reload=True,  # Auto-reload on code changes during development
    )
