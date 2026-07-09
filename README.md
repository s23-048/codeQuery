# CodeQuery 🔍

<div align="center">
  <p><strong>Repository-scale code search: Semantic embeddings + BM25 keyword search + LLM generation</strong></p>

  ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
  ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
  ![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
  ![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
  ![Google Gemini](https://img.shields.io/badge/Google%20Gemini-8E75B2?style=for-the-badge&logo=google&logoColor=white)
  ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)
</div>

CodeQuery is an enterprise-grade **Code Search and Retrieval-Augmented Generation (RAG)** pipeline. It allows you to index any public GitHub repository and interact with the codebase using natural language. 

Instead of relying on naive chunking strategies or standard vector searches, CodeQuery implements a highly robust pipeline combining Abstract Syntax Tree (AST) parsing, Hybrid Search (Semantic + BM25), and Reciprocal Rank Fusion (RRF) to ensure the LLM generates strictly grounded, accurate answers with precise file and line-level citations.

---

## 🏗 Architecture & Features

The project was built in a modular, 7-day progression to solve specific problems in code retrieval:

### 1. AST-Based Chunking (`parser/`)
Instead of cutting files randomly every 200 lines (which breaks functions in half), CodeQuery uses **Tree-sitter** to parse the concrete syntax tree. It guarantees that every embedded chunk is a semantically complete unit (a full `function`, `class`, etc.), capturing the actual "meaning" of the block.

### 2. Import Dependency Graph (`graph/`)
Uses `networkx` to trace Python module dependencies. This ensures that the system doesn't just treat files as isolated text, but understands how they connect and depend on one another.

### 3. Hybrid Search (`search/`)
Semantic search is great for concepts, but terrible for finding exact variables or function names. CodeQuery runs **both** simultaneously:
* **Semantic Search (ChromaDB + MiniLM):** Understands the "meaning" of a query (e.g. "how are payments processed").
* **Keyword Search (BM25):** Understands exact token matching with custom camelCase and snake_case splitting (e.g. `executePayment`).

### 4. Reciprocal Rank Fusion (`search/hybrid_search.py`)
Since Vector distances and BM25 scores are mathematically incompatible, CodeQuery merges the two search results using **Reciprocal Rank Fusion (RRF)**. It completely ignores the raw scores and merges the results based purely on their relative ranks, guaranteeing the best of both worlds.

### 5. Grounded Generation (`llm/answerer.py`)
The top 5 fused chunks are structured into a strict prompt containing the file paths and line numbers. The LLM (Google Gemini or OpenAI GPT-4o) is run with `temperature=0` to ensure deterministic, hallucination-free answers.

### 6. API & UI (`api/`, `ui/`)
* **FastAPI Backend:** Provides `/index` (to clone and process a repo) and `/query` endpoints.
* **Streamlit Frontend:** A sleek chat interface where you can paste a repo URL, ask questions, and view the specific code chunks cited by the LLM in an expandable drawer.

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* A free Gemini API key (from [Google AI Studio](https://aistudio.google.com/apikey)) or an OpenAI API key.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/s23-048/codeQuery.git
   cd codeQuery
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Rename `.env.example` to `.env` (or create a `.env` file) and add your API key:
   ```env
   # LLM Provider: gemini (free) | openai (paid)
   LLM_PROVIDER=gemini
   LLM_MODEL=gemini-2.5-flash
   GEMINI_API_KEY=your_key_here
   ```

### Running the App

You need to start both the FastAPI backend and the Streamlit frontend. Open two separate terminal windows:

**Terminal 1 (Backend API):**
```bash
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 (Streamlit UI):**
```bash
source venv/bin/activate
streamlit run ui/app.py
```

The UI will open in your browser at `http://localhost:8501`. 

---

## 💡 How to use

1. Go to the Streamlit UI.
2. In the sidebar, paste any public GitHub repository URL (e.g., `https://github.com/s23-048/codeQuery.git`).
3. Click **"Index Codebase"**. The backend will clone the repo, parse the AST, embed the chunks into ChromaDB, and generate the BM25 index.
4. Once indexing is complete, ask a question in the chat interface! 

**Example Queries:**
* *"How does the reciprocal rank fusion work in this project?"*
* *"Where is the AST parsing handled for JavaScript?"*
* *"What API endpoints are available in the FastAPI server?"*

---

## 🛠 Tech Stack
* **Language:** Python
* **AST Parsing:** Tree-sitter (`tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`)
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace `sentence-transformers` (all-MiniLM-L6-v2)
* **Keyword Search:** `rank-bm25`
* **Graph/Dependencies:** `networkx`
* **LLM Integration:** `google-genai` (Gemini), `openai`
* **Backend:** FastAPI, Uvicorn
* **Frontend:** Streamlit