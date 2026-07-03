"""
CodeQuery — AST-based Code Chunker
===================================

This module uses Tree-sitter to parse source files and extract semantically
meaningful chunks (functions, classes, arrow functions) from Python, JavaScript,
and TypeScript codebases.

WHY AST chunking instead of fixed-line chunking?
-------------------------------------------------
Fixed-line chunking (e.g. every 200 lines) can split a function in half. When
you embed that half-function, the embedding represents a fragment — not a
complete idea. Retrieval quality drops because the vector doesn't capture the
function's full meaning.

AST chunking guarantees every chunk is a complete function or class. The
embedding represents "what this function does" as a coherent unit, which is
what we actually want to search over.

Tree-sitter basics
------------------
Tree-sitter is an incremental parsing library. It builds a concrete syntax tree
(CST) from source code — every token and whitespace is represented, unlike
Python's built-in `ast` module which builds an abstract syntax tree (loses
comments, formatting, etc).

For each language, Tree-sitter uses a compiled grammar (.so/.dylib) that defines
the node types. Key node types we care about:

  Python:     function_definition, class_definition
  JavaScript: function_declaration, class_declaration, arrow_function
  TypeScript: function_declaration, class_declaration, arrow_function

Each node has:
  - .type       → string like "function_definition"
  - .text       → the raw source bytes of the entire node
  - .children   → list of child nodes
  - .start_point, .end_point → (row, col) tuples (0-indexed)
  - .child_by_field_name("name") → named child lookup
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tree_sitter
import tree_sitter_python as tspython
import tree_sitter_javascript as tsjavascript
import tree_sitter_typescript as tstypescript


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CodeChunk:
    """
    Represents one semantically complete unit of code.

    Fields
    ------
    chunk_id : str
        A stable, short identifier derived from the file path, name, and start
        line. Using md5 truncated to 12 chars — not for security, just for a
        compact, collision-unlikely ID we can use as a Chroma document ID.

    file_path : str
        Path relative to the repository root (e.g. "parser/code_parser.py").
        We store relative paths so chunks are portable across machines.

    language : str
        "python", "javascript", or "typescript".

    chunk_type : str
        "function", "class", or "module". Module is the fallback for small
        files that have no top-level functions or classes.

    name : str
        The identifier of the function/class (e.g. "parse_file", "CodeChunk").
        For module chunks, this is the filename.

    code : str
        The full source text of this chunk.

    start_line : int
        1-indexed line number where this chunk starts in the file.

    end_line : int
        1-indexed line number where this chunk ends.
    """
    chunk_id: str
    file_path: str
    language: str
    chunk_type: str   # "function" | "class" | "module"
    name: str
    code: str
    start_line: int
    end_line: int

    def to_search_text(self) -> str:
        """
        Build the text that gets embedded and BM25-indexed.

        We prepend the chunk type, name, and file path so they become part of
        the searchable text. Without this, a query like "parse_file function"
        would only match if those words appeared in the code body itself — but
        the function name is metadata, not code content. This simple prefix
        makes both the name and the file path searchable.

        Example output:
            function parse_file in parser/code_parser.py:
            def parse_file(self, file_path: str) -> list[CodeChunk]:
                ...
        """
        return f"{self.chunk_type} {self.name} in {self.file_path}:\n{self.code}"

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage / API responses."""
        return {
            "chunk_id": self.chunk_id,
            "file_path": self.file_path,
            "language": self.language,
            "chunk_type": self.chunk_type,
            "name": self.name,
            "code": self.code,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(file_path: str, name: str, start_line: int) -> str:
    """
    Generate a deterministic, short ID for a chunk.

    Using md5 of "file_path:name:start_line" truncated to 12 hex chars.
    This gives us 48 bits = ~281 trillion possible IDs. For a repo with
    10,000 chunks, collision probability is ~1 in 28 billion — fine.

    We want deterministic IDs (not UUIDs) so that re-indexing the same repo
    produces the same chunk IDs, which means Chroma can upsert cleanly
    without creating duplicates.
    """
    raw = f"{file_path}:{name}:{start_line}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# Map file extensions to (language_name, tree-sitter Language object).
# We build this once at module load time — creating Language objects is cheap,
# and this avoids doing it per-file which is the #1 perf mistake.
LANGUAGE_MAP: dict[str, tuple[str, tree_sitter.Language]] = {
    ".py":   ("python",     tree_sitter.Language(tspython.language())),
    ".js":   ("javascript", tree_sitter.Language(tsjavascript.language())),
    ".jsx":  ("javascript", tree_sitter.Language(tsjavascript.language())),
    ".ts":   ("typescript", tree_sitter.Language(tstypescript.language_typescript())),
    ".tsx":  ("typescript", tree_sitter.Language(tstypescript.language_tsx())),
}

# Node types that represent function-like constructs in each language.
# Python uses "_definition", JS/TS use "_declaration" — a naming quirk.
FUNCTION_NODE_TYPES: dict[str, set[str]] = {
    "python":     {"function_definition"},
    "javascript": {"function_declaration"},
    "typescript": {"function_declaration"},
}

# Node types for classes.
CLASS_NODE_TYPES: dict[str, set[str]] = {
    "python":     {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
}

# Directories to skip when walking a repo. These contain third-party code,
# build artifacts, or VCS internals — never useful to index.
SKIP_DIRS: set[str] = {
    ".git", "node_modules", "__pycache__", "venv", ".venv",
    "env", "dist", "build", ".next", ".nuxt", "coverage",
    ".tox", ".mypy_cache", ".pytest_cache", "egg-info",
}

# Maximum lines for a file to be treated as a "module" chunk when it has
# no extractable functions or classes. Files above this are likely generated
# or data files — not worth indexing whole.
MODULE_CHUNK_MAX_LINES = 200


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class CodeParser:
    """
    Parses source files using Tree-sitter and extracts CodeChunk objects.

    Usage:
        parser = CodeParser()

        # Single file
        chunks = parser.parse_file("path/to/file.py")

        # Entire repository
        chunks = parser.parse_repository("/path/to/repo")

    Design notes
    ------------
    - One Parser instance per language is created lazily and cached. Tree-sitter
      Parsers are cheap to create but we avoid re-creating them per file.
    - We do NOT recurse into class bodies to extract methods as separate chunks.
      A class and its methods are one chunk. This keeps context together — if
      you split methods out, you lose the class-level docstring, attributes,
      and the relationship between methods. For search, "class Foo with methods
      a, b, c" as one chunk is more useful than three orphaned method chunks.
    - Arrow functions (const foo = () => {}) in JS/TS are captured by looking
      for variable declarations that contain an arrow_function child.
    """

    def __init__(self) -> None:
        # Cache of tree-sitter Parser objects, keyed by language name.
        # We build these lazily on first use.
        self._parsers: dict[str, tree_sitter.Parser] = {}

    def _get_parser(self, language_name: str, ts_language: tree_sitter.Language) -> tree_sitter.Parser:
        """
        Get or create a tree-sitter Parser for the given language.

        We cache parsers because while creating one is cheap (~microseconds),
        doing it 10,000 times for a large repo adds up. One parser per language
        is all we need — the Parser is stateless between parse() calls.
        """
        if language_name not in self._parsers:
            self._parsers[language_name] = tree_sitter.Parser(ts_language)
        return self._parsers[language_name]

    # -------------------------------------------------------------------
    # AST walking
    # -------------------------------------------------------------------

    def _extract_name(self, node: tree_sitter.Node) -> str:
        """
        Extract the identifier (name) from a function/class AST node.

        For function_definition / function_declaration / class_definition /
        class_declaration, the name is always in a child node of type
        'identifier' (Python, JS) or 'type_identifier' (TS classes).

        For arrow functions wrapped in variable declarations, the name is
        the variable name (the 'identifier' inside 'variable_declarator').

        Returns "<anonymous>" if no name is found (shouldn't happen for
        well-formed code, but we don't want to crash on edge cases).
        """
        # Direct child lookup — works for function_def, class_def, etc.
        for child in node.children:
            if child.type in ("identifier", "type_identifier"):
                return (child.text or b"").decode("utf-8")

        # For lexical_declaration containing an arrow function, the name
        # is inside the variable_declarator child.
        if node.type == "lexical_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    for grandchild in child.children:
                        if grandchild.type == "identifier":
                            return (grandchild.text or b"").decode("utf-8")

        return "<anonymous>"

    def _has_arrow_function(self, node: tree_sitter.Node) -> bool:
        """
        Check if a lexical_declaration contains an arrow function.

        We need this because `const x = 42;` is also a lexical_declaration
        but we only want to chunk arrow functions, not plain variable
        assignments.
        """
        for child in node.children:
            if child.type == "variable_declarator":
                for grandchild in child.children:
                    if grandchild.type == "arrow_function":
                        return True
        return False

    def _find_chunks(
        self,
        node: tree_sitter.Node,
        language: str,
        source_bytes: bytes,
        file_path: str,
    ) -> list[CodeChunk]:
        """
        Recursively walk the AST and collect CodeChunk objects.

        We walk top-down. When we find a function or class node, we extract
        it as a chunk and do NOT recurse into its children — the entire
        subtree is captured in the chunk's code text. This prevents double-
        counting (e.g. a method inside a class appearing as both part of the
        class chunk and as its own chunk).

        For nodes that aren't functions/classes, we recurse into their
        children. This handles cases like:
          - export_statement wrapping a function_declaration
          - decorated functions (decorator + function_definition)
        """
        chunks: list[CodeChunk] = []

        func_types = FUNCTION_NODE_TYPES.get(language, set())
        class_types = CLASS_NODE_TYPES.get(language, set())

        # Is this node a function?
        if node.type in func_types:
            name = self._extract_name(node)
            code = (node.text or b"").decode("utf-8")
            # Tree-sitter uses 0-indexed rows; we convert to 1-indexed lines
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunk_id = _make_chunk_id(file_path, name, start_line)
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                language=language,
                chunk_type="function",
                name=name,
                code=code,
                start_line=start_line,
                end_line=end_line,
            ))
            return chunks  # Don't recurse — we've captured the whole subtree

        # Is this node a class?
        if node.type in class_types:
            name = self._extract_name(node)
            code = (node.text or b"").decode("utf-8")
            start_line = node.start_point[0] + 1
            end_line = node.end_point[0] + 1
            chunk_id = _make_chunk_id(file_path, name, start_line)
            chunks.append(CodeChunk(
                chunk_id=chunk_id,
                file_path=file_path,
                language=language,
                chunk_type="class",
                name=name,
                code=code,
                start_line=start_line,
                end_line=end_line,
            ))
            return chunks  # Don't recurse into the class body

        # Is this a JS/TS arrow function in a variable declaration?
        if node.type == "lexical_declaration" and language in ("javascript", "typescript"):
            if self._has_arrow_function(node):
                name = self._extract_name(node)
                code = (node.text or b"").decode("utf-8")
                start_line = node.start_point[0] + 1
                end_line = node.end_point[0] + 1
                chunk_id = _make_chunk_id(file_path, name, start_line)
                chunks.append(CodeChunk(
                    chunk_id=chunk_id,
                    file_path=file_path,
                    language=language,
                    chunk_type="function",
                    name=name,
                    code=code,
                    start_line=start_line,
                    end_line=end_line,
                ))
                return chunks

        # Not a chunk-worthy node — recurse into children
        for child in node.children:
            chunks.extend(self._find_chunks(child, language, source_bytes, file_path))

        return chunks

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def parse_file(self, file_path: str, repo_root: Optional[str] = None) -> list[CodeChunk]:
        """
        Parse a single source file and return its chunks.

        Parameters
        ----------
        file_path : str
            Absolute or relative path to the source file.
        repo_root : str, optional
            If provided, chunk file_paths are stored relative to this root.
            If not provided, the file_path is used as-is.

        Returns
        -------
        list[CodeChunk]
            Functions, classes, and (if no top-level constructs) a module chunk.
        """
        path = Path(file_path)

        # Determine language from extension
        ext = path.suffix.lower()
        if ext not in LANGUAGE_MAP:
            return []  # Unsupported language — skip silently

        language_name, ts_language = LANGUAGE_MAP[ext]
        parser = self._get_parser(language_name, ts_language)

        # Read source file
        try:
            source_bytes = path.read_bytes()
        except (OSError, IOError) as e:
            print(f"  ⚠ Could not read {file_path}: {e}")
            return []

        # Parse into AST
        tree = parser.parse(source_bytes)

        # Compute the relative path for storage
        if repo_root:
            try:
                relative_path = str(path.resolve().relative_to(Path(repo_root).resolve()))
            except ValueError:
                relative_path = str(path)
        else:
            relative_path = str(path)

        # Extract function and class chunks
        chunks = self._find_chunks(tree.root_node, language_name, source_bytes, relative_path)

        # Fallback: if no functions/classes were found and the file is small,
        # index the entire file as a "module" chunk. This catches config files,
        # small scripts, __init__.py files, etc. that would otherwise be lost.
        if not chunks:
            source_text = source_bytes.decode("utf-8", errors="replace")
            line_count = source_text.count("\n") + 1
            if line_count <= MODULE_CHUNK_MAX_LINES:
                name = path.stem  # filename without extension
                chunk_id = _make_chunk_id(relative_path, name, 1)
                chunks.append(CodeChunk(
                    chunk_id=chunk_id,
                    file_path=relative_path,
                    language=language_name,
                    chunk_type="module",
                    name=name,
                    code=source_text,
                    start_line=1,
                    end_line=line_count,
                ))

        return chunks

    def parse_repository(self, repo_path: str) -> list[CodeChunk]:
        """
        Walk an entire repository and parse all supported files.

        Parameters
        ----------
        repo_path : str
            Path to the root of the repository.

        Returns
        -------
        list[CodeChunk]
            All chunks from all supported files in the repo.

        Notes
        -----
        Skips directories in SKIP_DIRS (e.g. .git, node_modules, venv).
        Files that can't be read are skipped with a warning, not an exception.
        """
        repo_path = os.path.abspath(repo_path)
        all_chunks: list[CodeChunk] = []
        supported_extensions = set(LANGUAGE_MAP.keys())
        files_parsed = 0
        files_skipped = 0

        for dirpath, dirnames, filenames in os.walk(repo_path):
            # Prune directories we don't want to enter.
            # Modifying dirnames in-place tells os.walk to skip them.
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext not in supported_extensions:
                    files_skipped += 1
                    continue

                full_path = os.path.join(dirpath, filename)
                chunks = self.parse_file(full_path, repo_root=repo_path)
                all_chunks.extend(chunks)
                files_parsed += 1

        print(f"✓ Parsed {files_parsed} files → {len(all_chunks)} chunks "
              f"(skipped {files_skipped} unsupported files)")

        return all_chunks


# ---------------------------------------------------------------------------
# CLI — quick test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    target = sys.argv[1] if len(sys.argv) > 1 else "."
    parser = CodeParser()

    if os.path.isfile(target):
        chunks = parser.parse_file(target)
    else:
        chunks = parser.parse_repository(target)

    # Pretty-print results
    for chunk in chunks:
        print(f"\n{'='*60}")
        print(f"  ID:    {chunk.chunk_id}")
        print(f"  Type:  {chunk.chunk_type}")
        print(f"  Name:  {chunk.name}")
        print(f"  File:  {chunk.file_path}")
        print(f"  Lines: {chunk.start_line}–{chunk.end_line}")
        print(f"  Code:  ({len(chunk.code)} chars)")
        print(f"{'='*60}")
        # Show first 5 lines of code as preview
        preview_lines = chunk.code.split("\n")[:5]
        for line in preview_lines:
            print(f"  │ {line}")
        if len(chunk.code.split("\n")) > 5:
            print(f"  │ ... ({len(chunk.code.split(chr(10)))} total lines)")

    print(f"\nTotal: {len(chunks)} chunks")

    # Optionally dump as JSON
    if "--json" in sys.argv:
        output = [c.to_dict() for c in chunks]
        json_path = "chunks_debug.json"
        with open(json_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Saved to {json_path}")
