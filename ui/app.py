"""
CodeQuery — Streamlit UI
========================

The frontend for CodeQuery. Connects to the FastAPI backend.
Run with: streamlit run ui/app.py
"""

import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="CodeQuery",
    page_icon="🔍",
    layout="wide"
)

# ---------------------------------------------------------------------------
# State Initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "repo_indexed" not in st.session_state:
    st.session_state.repo_indexed = False


# ---------------------------------------------------------------------------
# Sidebar: Indexing
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("1. Index Repository")
    st.markdown("Paste a GitHub URL to parse and index the codebase.")
    
    repo_url = st.text_input("Repository URL", value="https://github.com/s23-048/codeQuery.git")
    
    if st.button("Index Codebase"):
        if not repo_url:
            st.warning("Please enter a valid URL.")
        else:
            with st.spinner("Cloning, chunking, and embedding... This may take a minute."):
                try:
                    res = requests.post(
                        f"{API_URL}/index",
                        json={"repo_url": repo_url, "collection_name": "code_chunks"},
                        timeout=300
                    )
                    
                    if res.status_code == 200:
                        data = res.json()
                        st.success(f"Successfully indexed {data['chunks_indexed']} chunks in {data['index_time_seconds']}s!")
                        st.session_state.repo_indexed = True
                    else:
                        st.error(f"Error: {res.text}")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")

    st.divider()
    
    # Status Check
    st.header("System Status")
    if st.button("Check Backend Status"):
        try:
            res = requests.get(f"{API_URL}/status")
            if res.status_code == 200:
                status = res.json()
                st.json(status)
                if status.get("indexed_repo"):
                    st.session_state.repo_indexed = True
            else:
                st.error("API error.")
        except Exception as e:
            st.error("Backend not running.")

# ---------------------------------------------------------------------------
# Main Chat Interface
# ---------------------------------------------------------------------------

st.title("CodeQuery 🔍")
st.markdown("Repository-scale code search: Semantic embeddings + BM25 keyword search + LLM generation")

if not st.session_state.repo_indexed:
    st.info("👈 Start by indexing a repository in the sidebar.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # If assistant message has sources, show them in an expander
            if msg.get("sources"):
                with st.expander(f"📚 Sources ({len(msg['sources'])} chunks, {msg.get('latency_ms', 0)}ms)"):
                    for s in msg["sources"]:
                        st.markdown(f"- `{s['chunk_type']} {s['name']}` in **{s['file_path']}** (lines {s['start_line']}-{s['end_line']})")

    # Chat input
    if prompt := st.chat_input("Ask a question about the codebase..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Searching and generating..."):
                try:
                    res = requests.post(
                        f"{API_URL}/query",
                        json={"query": prompt, "top_k": 5},
                        timeout=60
                    )
                    
                    if res.status_code == 200:
                        data = res.json()
                        answer = data["answer"]
                        sources = data["sources"]
                        latency = data["latency_ms"]
                        
                        st.markdown(answer)
                        if sources:
                            with st.expander(f"📚 Sources ({len(sources)} chunks, {latency}ms)"):
                                for s in sources:
                                    st.markdown(f"- `{s['chunk_type']} {s['name']}` in **{s['file_path']}** (lines {s['start_line']}-{s['end_line']})")
                        
                        # Save to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                            "latency_ms": latency
                        })
                    else:
                        st.error(f"Error generating answer: {res.text}")
                except Exception as e:
                    st.error(f"API request failed: {e}")
