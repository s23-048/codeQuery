"""
CodeQuery — Import-based Dependency Graph
==========================================

This module builds a directed graph of dependencies between code chunks by
analyzing Python import statements.

Scope
-----
This build uses **import-level** dependencies only — we parse `import X` and
`from X import Y` statements using Python's built-in `ast` module and create
edges when the imported module/name matches another chunk in the index.

We do NOT do call-graph analysis (walking function bodies to find which
functions call which other functions). Call-graph analysis is significantly
more complex — you'd need to resolve variable types, handle dynamic dispatch,
track closures, etc. Import-level is a practical 80/20: it tells you "file A
depends on file B" which answers most dependency questions.

How it works
------------
1. For each Python chunk, parse its code with `ast.parse()`
2. Walk the AST looking for `Import` and `ImportFrom` nodes
3. Extract the module names being imported
4. If that module name matches another chunk's name or file stem, add a
   directed edge: chunk → imported_chunk, with relation="imports"
5. Serialize to JSON for persistence

Why only Python?
----------------
JavaScript/TypeScript imports use a completely different syntax
(`import { X } from './module'`, `require()`, dynamic imports, etc.) and
would need Tree-sitter to parse correctly. For this scope, Python-only
import analysis is enough to demonstrate the concept. The architecture
supports adding JS/TS import parsing later.

DiGraph implementation
----------------------
We use a custom lightweight DiGraph instead of networkx because networkx 3.6
is incompatible with Python 3.14. Our DiGraph stores:
- Nodes: dict of {node_id: {attributes}}
- Edges: dict of {source_id: {target_id: {attributes}}}

This gives us O(1) node/edge lookup and is trivially JSON-serializable.
When deploying to production (Docker with Python 3.11), you can swap this
for networkx if you need its graph algorithms (shortest path, centrality, etc).
"""

import ast
import json
import os
from pathlib import Path
from typing import Optional

# Import our chunk type from Day 1
from parser.code_parser import CodeChunk


# ---------------------------------------------------------------------------
# Lightweight Directed Graph
# ---------------------------------------------------------------------------

class DiGraph:
    """
    A minimal directed graph implementation.

    Stores nodes with attributes and directed edges with attributes.
    Provides the subset of the networkx DiGraph API that we actually use.

    Data structures:
    - _nodes: {node_id: {attr_key: attr_value, ...}}
    - _edges: {source_id: {target_id: {attr_key: attr_value, ...}}}
    - _reverse_edges: {target_id: {source_id: {attr_key: attr_value, ...}}}
      (for efficient in-edge lookups)

    Why a reverse edge index?
    Without it, "who imports chunk X?" would require scanning ALL edges.
    With it, it's an O(1) dict lookup — same idea as a database index.
    """

    def __init__(self) -> None:
        self._nodes: dict[str, dict] = {}
        self._edges: dict[str, dict[str, dict]] = {}         # source → {target → attrs}
        self._reverse_edges: dict[str, dict[str, dict]] = {}  # target → {source → attrs}

    def add_node(self, node_id: str, **attrs) -> None:
        """Add a node with optional attributes."""
        self._nodes[node_id] = attrs
        # Ensure edge dicts exist even for nodes with no edges
        self._edges.setdefault(node_id, {})
        self._reverse_edges.setdefault(node_id, {})

    def add_edge(self, source: str, target: str, **attrs) -> None:
        """Add a directed edge from source to target with optional attributes."""
        # Auto-create nodes if they don't exist
        if source not in self._nodes:
            self._nodes[source] = {}
        if target not in self._nodes:
            self._nodes[target] = {}

        # Forward edge: source → target
        self._edges.setdefault(source, {})[target] = attrs
        # Reverse edge: target ← source (for in_edges lookup)
        self._reverse_edges.setdefault(target, {})[source] = attrs

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node_attrs(self, node_id: str) -> dict:
        """Get attributes for a node."""
        return self._nodes.get(node_id, {})

    def out_edges(self, node_id: str) -> list[tuple[str, str, dict]]:
        """Get all outgoing edges from a node: [(source, target, attrs), ...]"""
        edges = []
        for target, attrs in self._edges.get(node_id, {}).items():
            edges.append((node_id, target, attrs))
        return edges

    def in_edges(self, node_id: str) -> list[tuple[str, str, dict]]:
        """Get all incoming edges to a node: [(source, target, attrs), ...]"""
        edges = []
        for source, attrs in self._reverse_edges.get(node_id, {}).items():
            edges.append((source, node_id, attrs))
        return edges

    def out_degree(self, node_id: str) -> int:
        """Number of outgoing edges from a node."""
        return len(self._edges.get(node_id, {}))

    def in_degree(self, node_id: str) -> int:
        """Number of incoming edges to a node."""
        return len(self._reverse_edges.get(node_id, {}))

    def number_of_nodes(self) -> int:
        return len(self._nodes)

    def number_of_edges(self) -> int:
        return sum(len(targets) for targets in self._edges.values())

    def all_edges(self) -> list[tuple[str, str, dict]]:
        """Get all edges in the graph: [(source, target, attrs), ...]"""
        edges = []
        for source, targets in self._edges.items():
            for target, attrs in targets.items():
                edges.append((source, target, attrs))
        return edges

    def all_nodes(self) -> list[tuple[str, dict]]:
        """Get all nodes: [(node_id, attrs), ...]"""
        return list(self._nodes.items())

    def clear(self) -> None:
        """Remove all nodes and edges."""
        self._nodes.clear()
        self._edges.clear()
        self._reverse_edges.clear()

    # --- Serialization ---

    def to_dict(self) -> dict:
        """
        Serialize to a dict format similar to networkx's node_link_data.

        Format:
        {
            "nodes": [{"id": "abc", "name": "foo", ...}, ...],
            "edges": [{"source": "abc", "target": "def", "relation": "imports"}, ...]
        }
        """
        nodes = []
        for node_id, attrs in self._nodes.items():
            node = {"id": node_id, **attrs}
            nodes.append(node)

        edges = []
        for source, targets in self._edges.items():
            for target, attrs in targets.items():
                edge = {"source": source, "target": target, **attrs}
                edges.append(edge)

        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_dict(cls, data: dict) -> "DiGraph":
        """Deserialize from a dict (inverse of to_dict)."""
        graph = cls()
        for node in data.get("nodes", []):
            node_id = node.pop("id")
            graph.add_node(node_id, **node)

        for edge in data.get("edges", []):
            source = edge.pop("source")
            target = edge.pop("target")
            graph.add_edge(source, target, **edge)

        return graph


# ---------------------------------------------------------------------------
# Import extraction
# ---------------------------------------------------------------------------

def extract_imports(code: str) -> list[dict]:
    """
    Parse a Python code string and extract all import statements.

    Returns a list of dicts, each describing one import:
        {
            "type": "import" | "from_import",
            "module": "os" | "pathlib" | "parser.code_parser" | ...,
            "names": ["Path"] | ["CodeParser", "CodeChunk"] | ...,
            "level": 0 (absolute) | 1+ (relative),
        }

    Uses Python's built-in `ast` module, which only works for Python code.
    This is intentional — we're scoping dependency analysis to Python for
    this build.

    We wrap the parse in a try/except because chunk code might be a single
    function extracted from a larger file. `ast.parse()` can fail on
    fragments that have syntax that's only valid in a specific context
    (e.g., `yield` outside a function at module level). We silently skip
    those — losing a few import edges is better than crashing.
    """
    imports = []

    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Chunk code might not be valid as standalone Python
        # (e.g., a method body referencing `self` at module level).
        # This is expected and fine — we just skip it.
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            # `import os` or `import os, sys`
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,            # "os", "sys"
                    "names": [alias.name],
                    "level": 0,
                })

        elif isinstance(node, ast.ImportFrom):
            # `from pathlib import Path` or `from . import utils`
            module = node.module or ""  # None for `from . import X`
            names = [alias.name for alias in node.names]
            imports.append({
                "type": "from_import",
                "module": module,                    # "pathlib", ""
                "names": names,                      # ["Path"]
                "level": node.level or 0,            # 0=absolute, 1+=relative
            })

    return imports


# ---------------------------------------------------------------------------
# Dependency graph builder
# ---------------------------------------------------------------------------

class DependencyGraph:
    """
    Builds and manages a directed dependency graph between code chunks.

    The graph has:
    - Nodes: one per CodeChunk, with metadata (name, file_path, chunk_type, etc.)
    - Edges: directed "imports" relationships (source → target means source
      imports something from target)

    Usage:
        from parser.code_parser import CodeParser
        from graph.dependency_graph import DependencyGraph

        parser = CodeParser()
        chunks = parser.parse_repository("/path/to/repo")

        dep_graph = DependencyGraph()
        dep_graph.build(chunks)
        dep_graph.save("graph.json")

        # Query the graph
        deps = dep_graph.get_dependencies("some_chunk_id")
        dependents = dep_graph.get_dependents("some_chunk_id")
        summary = dep_graph.get_summary()
    """

    def __init__(self) -> None:
        self.graph: DiGraph = DiGraph()
        self._chunks_by_id: dict[str, CodeChunk] = {}

    def build(self, chunks: list[CodeChunk], repo_root: str = ".") -> DiGraph:
        """
        Build the dependency graph from a list of CodeChunks.

        Parameters
        ----------
        chunks : list[CodeChunk]
            The chunks to build the graph from.
        repo_root : str
            Path to the repository root. Used to resolve relative file paths
            back to actual files on disk for reading file-level imports.

        Steps:
        1. Add every chunk as a node in the graph
        2. Build lookup tables to match import names → chunk IDs
        3. For each Python chunk, extract imports and create edges

        Returns the DiGraph for direct use.
        """
        self._repo_root = os.path.abspath(repo_root)
        self.graph.clear()
        self._chunks_by_id = {c.chunk_id: c for c in chunks}

        # --- Step 1: Add all chunks as nodes ---
        for chunk in chunks:
            self.graph.add_node(
                chunk.chunk_id,
                name=chunk.name,
                file_path=chunk.file_path,
                chunk_type=chunk.chunk_type,
                language=chunk.language,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            )

        # --- Step 2: Build lookup tables ---
        # We need to match import names to chunks. There are several ways
        # an import might reference another chunk:
        #
        #   import parser.code_parser    → module "parser.code_parser"
        #   from parser import code_parser → name "code_parser"
        #   from parser.code_parser import CodeParser → name "CodeParser"
        #
        # So we build three lookup tables:

        # chunk name → chunk_id  (matches "CodeParser", "Calculator", etc.)
        name_to_ids: dict[str, list[str]] = {}
        for chunk in chunks:
            name_to_ids.setdefault(chunk.name, []).append(chunk.chunk_id)

        # file stem → chunk_ids  (matches "code_parser" from "code_parser.py")
        stem_to_ids: dict[str, list[str]] = {}
        for chunk in chunks:
            stem = Path(chunk.file_path).stem  # "code_parser.py" → "code_parser"
            stem_to_ids.setdefault(stem, []).append(chunk.chunk_id)

        # dotted module path → chunk_ids  (matches "parser.code_parser")
        module_to_ids: dict[str, list[str]] = {}
        for chunk in chunks:
            # "parser/code_parser.py" → "parser.code_parser"
            module_path = Path(chunk.file_path).with_suffix("").as_posix().replace("/", ".")
            module_to_ids.setdefault(module_path, []).append(chunk.chunk_id)

        # --- Step 3: Extract imports and create edges ---
        # IMPORTANT: Import statements in Python live at the FILE level, not
        # inside functions or classes. When we chunk code, we extract functions
        # and classes — but `import os` at the top of the file isn't inside
        # any function/class, so it's not part of any chunk's code.
        #
        # Solution: Group chunks by file, read the FULL file source to find
        # imports, then attribute those imports to ALL chunks from that file.
        # This means "if file A imports from file B, then every function in
        # file A depends on the relevant chunks in file B."

        python_chunks = [c for c in chunks if c.language == "python"]

        # Group chunks by file path
        file_to_chunks: dict[str, list[CodeChunk]] = {}
        for chunk in python_chunks:
            file_to_chunks.setdefault(chunk.file_path, []).append(chunk)

        for file_path, file_chunks in file_to_chunks.items():
            # Collect imports from all sources for this file:
            all_imports: list[dict] = []

            # 1) Try reading the full file source for file-level imports
            #    We need to find the actual file on disk. The file_path is
            #    relative, so we check if any chunk's code can be parsed.
            #    For module chunks, the code IS the full file.
            for chunk in file_chunks:
                if chunk.chunk_type == "module":
                    # Module chunks contain the full file — parse them
                    all_imports.extend(extract_imports(chunk.code))
                else:
                    # Function/class chunks might have imports inside them
                    # (rare but possible, e.g., lazy imports inside functions)
                    chunk_imports = extract_imports(chunk.code)
                    all_imports.extend(chunk_imports)

            # 2) Read the actual file from disk for file-level imports.
            #    This catches the common case: `import X` at the top of the file,
            #    outside any function or class.
            #    We use repo_root to resolve relative paths like "service.py"
            #    back to actual files like "/path/to/repo/service.py".
            full_path = os.path.join(self._repo_root, file_path)
            if os.path.isfile(full_path):
                try:
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        file_source = f.read()
                    all_imports.extend(extract_imports(file_source))
                except (OSError, IOError):
                    pass

            # Deduplicate imports (same import might appear from chunk + file)
            seen_imports: set[tuple] = set()
            unique_imports: list[dict] = []
            for imp in all_imports:
                key = (imp["module"], tuple(imp["names"]))
                if key not in seen_imports:
                    seen_imports.add(key)
                    unique_imports.append(imp)

            # Match imports to target chunks and create edges
            for imp in unique_imports:
                target_ids = set()

                module = imp["module"]
                if module:
                    if module in module_to_ids:
                        target_ids.update(module_to_ids[module])

                    last_part = module.rsplit(".", 1)[-1]
                    if last_part in stem_to_ids:
                        target_ids.update(stem_to_ids[last_part])

                for name in imp["names"]:
                    if name in name_to_ids:
                        target_ids.update(name_to_ids[name])

                # Add edges from every chunk in this file to the targets
                for chunk in file_chunks:
                    for target_id in target_ids:
                        if target_id != chunk.chunk_id:
                            # Also skip edges to chunks in the same file
                            # (a file doesn't "depend on" itself)
                            target_file = self.graph.node_attrs(target_id).get("file_path", "")
                            if target_file != chunk.file_path:
                                self.graph.add_edge(
                                    chunk.chunk_id,
                                    target_id,
                                    relation="imports",
                                )

        # Print summary
        print(f"✓ Dependency graph: {self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")

        return self.graph

    # -------------------------------------------------------------------
    # Query methods
    # -------------------------------------------------------------------

    def get_dependencies(self, chunk_id: str) -> list[dict]:
        """
        Get all chunks that the given chunk imports (outgoing edges).

        "What does this chunk depend on?"
        """
        if not self.graph.has_node(chunk_id):
            return []

        deps = []
        for _, target_id, data in self.graph.out_edges(chunk_id):
            node_data = self.graph.node_attrs(target_id)
            deps.append({
                "chunk_id": target_id,
                "name": node_data.get("name", ""),
                "file_path": node_data.get("file_path", ""),
                "chunk_type": node_data.get("chunk_type", ""),
                "relation": data.get("relation", "imports"),
            })
        return deps

    def get_dependents(self, chunk_id: str) -> list[dict]:
        """
        Get all chunks that import the given chunk (incoming edges).

        "What depends on this chunk?" — useful for impact analysis.
        """
        if not self.graph.has_node(chunk_id):
            return []

        deps = []
        for source_id, _, data in self.graph.in_edges(chunk_id):
            node_data = self.graph.node_attrs(source_id)
            deps.append({
                "chunk_id": source_id,
                "name": node_data.get("name", ""),
                "file_path": node_data.get("file_path", ""),
                "chunk_type": node_data.get("chunk_type", ""),
                "relation": data.get("relation", "imports"),
            })
        return deps

    def get_summary(self) -> dict:
        """
        Get a high-level summary of the dependency graph.

        Returns node/edge counts, most-imported chunks (highest in-degree),
        and most-dependent chunks (highest out-degree). Useful for the
        /graph API endpoint and for understanding codebase structure at
        a glance.
        """
        if self.graph.number_of_nodes() == 0:
            return {"nodes": 0, "edges": 0, "most_imported": [], "most_dependent": []}

        # In-degree = how many other chunks import this one
        # High in-degree = widely used utility (like a "popular" module)
        in_degrees = []
        for node_id, attrs in self.graph.all_nodes():
            degree = self.graph.in_degree(node_id)
            in_degrees.append((node_id, degree))
        in_degrees.sort(key=lambda x: x[1], reverse=True)

        most_imported = []
        for node_id, degree in in_degrees[:5]:
            if degree > 0:
                node_data = self.graph.node_attrs(node_id)
                most_imported.append({
                    "chunk_id": node_id,
                    "name": node_data.get("name", ""),
                    "file_path": node_data.get("file_path", ""),
                    "imported_by_count": degree,
                })

        # Out-degree = how many other chunks this one imports
        # High out-degree = depends on many things (potentially fragile)
        out_degrees = []
        for node_id, attrs in self.graph.all_nodes():
            degree = self.graph.out_degree(node_id)
            out_degrees.append((node_id, degree))
        out_degrees.sort(key=lambda x: x[1], reverse=True)

        most_dependent = []
        for node_id, degree in out_degrees[:5]:
            if degree > 0:
                node_data = self.graph.node_attrs(node_id)
                most_dependent.append({
                    "chunk_id": node_id,
                    "name": node_data.get("name", ""),
                    "file_path": node_data.get("file_path", ""),
                    "depends_on_count": degree,
                })

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "most_imported": most_imported,
            "most_dependent": most_dependent,
        }

    # -------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------

    def save(self, file_path: str) -> None:
        """
        Save the graph to a JSON file.

        Format:
        {
            "nodes": [{"id": "abc", "name": "foo", ...}, ...],
            "edges": [{"source": "abc", "target": "def", "relation": "imports"}, ...]
        }

        Human-readable and can be loaded back with `load()`.
        """
        data = self.graph.to_dict()
        os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Graph saved to {file_path}")

    def load(self, file_path: str) -> DiGraph:
        """
        Load a graph from a JSON file previously saved with `save()`.

        This lets you reload the graph without re-parsing the entire repo.
        """
        with open(file_path, "r") as f:
            data = json.load(f)
        self.graph = DiGraph.from_dict(data)
        self._chunks_by_id = {}
        print(f"✓ Graph loaded from {file_path}: "
              f"{self.graph.number_of_nodes()} nodes, "
              f"{self.graph.number_of_edges()} edges")
        return self.graph

    def to_dict(self) -> dict:
        """
        Serialize the graph to a dict for API responses.

        Can be directly JSON-serialized by FastAPI.
        """
        return self.graph.to_dict()


# ---------------------------------------------------------------------------
# CLI — quick test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from parser.code_parser import CodeParser

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    output = sys.argv[2] if len(sys.argv) > 2 else "graph.json"

    # Step 1: Parse the repo
    parser = CodeParser()
    if os.path.isfile(target):
        chunks = parser.parse_file(target)
    else:
        chunks = parser.parse_repository(target)

    # Step 2: Build the dependency graph
    dep_graph = DependencyGraph()
    repo_root = target if os.path.isdir(target) else os.path.dirname(target) or "."
    dep_graph.build(chunks, repo_root=repo_root)

    # Step 3: Show results
    print(f"\n{'='*60}")
    print("GRAPH SUMMARY")
    print(f"{'='*60}")
    summary = dep_graph.get_summary()
    print(f"  Nodes: {summary['nodes']}")
    print(f"  Edges: {summary['edges']}")

    if summary["most_imported"]:
        print(f"\n  Most imported (depended on by others):")
        for item in summary["most_imported"]:
            print(f"    → {item['name']} ({item['file_path']}) "
                  f"— imported by {item['imported_by_count']} chunks")

    if summary["most_dependent"]:
        print(f"\n  Most dependent (imports the most):")
        for item in summary["most_dependent"]:
            print(f"    → {item['name']} ({item['file_path']}) "
                  f"— depends on {item['depends_on_count']} chunks")

    # Show all edges
    print(f"\n{'='*60}")
    print("ALL EDGES")
    print(f"{'='*60}")
    for source, target_node, data in dep_graph.graph.all_edges():
        source_data = dep_graph.graph.node_attrs(source)
        target_data = dep_graph.graph.node_attrs(target_node)
        source_name = source_data.get("name", source)
        target_name = target_data.get("name", target_node)
        source_file = source_data.get("file_path", "")
        target_file = target_data.get("file_path", "")
        print(f"  {source_name} ({source_file})")
        print(f"    └─ imports → {target_name} ({target_file})")

    if dep_graph.graph.number_of_edges() == 0:
        print("  (no import edges found — this is normal for small")
        print("   projects or when chunks don't import each other)")

    # Step 4: Save
    dep_graph.save(output)
    print(f"\nDone! Graph saved to {output}")
