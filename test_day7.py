import requests
import json
import time
import sys

API_URL = "http://localhost:8000"

print("1. Indexing codeQuery repo...")
res = requests.post(f"{API_URL}/index", json={"repo_url": "https://github.com/s23-048/codeQuery.git"})
if res.status_code != 200:
    print(f"Error indexing: {res.text}")
    sys.exit(1)
print(f"Indexed: {res.json()['chunks_indexed']} chunks")

questions = [
    "How are the code chunks generated from files?",
    "How does the dependency graph extract imports?",
    "Explain the hybrid search strategy and how scores are fused.",
    "Where is the LLM answer generated and which models are supported?",
    "What are the API endpoints provided in the FastAPI server?"
]

print("\n--- 2. Asking 5 Real Questions ---")

results = []
for i, q in enumerate(questions, 1):
    print(f"\nQ{i}: {q}")
    res = requests.post(f"{API_URL}/query", json={"query": q, "top_k": 5})
    if res.status_code == 200:
        data = res.json()
        print(f"A: {data['answer']}")
        print(f"Sources: {[s['name'] for s in data['sources']]}")
        results.append({
            "question": q,
            "answer": data["answer"],
            "sources": data["sources"]
        })
    else:
        print(f"Error: {res.text}")
    time.sleep(4)

with open("daysOfBuild/day7", "w") as f:
    f.write("# Day 7 — UI + Final Test\n\n")
    f.write("We built the Streamlit UI in `ui/app.py` and then ran 5 real questions against our own codebase to verify it works end-to-end.\n\n")
    for i, res in enumerate(results, 1):
        f.write(f"### Q{i}: {res['question']}\n\n")
        f.write(f"**Answer:**\n{res['answer']}\n\n")
        f.write("**Sources Cited:**\n")
        for s in res['sources']:
            f.write(f"- `{s['chunk_type']} {s['name']}` in `{s['file_path']}` (lines {s['start_line']}-{s['end_line']})\n")
        f.write("\n---\n\n")

print("\nTests complete and saved to daysOfBuild/day7")
