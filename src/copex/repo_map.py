"""Repository map and symbol index support for Copex.

Builds a concise, cacheable map of source files and key symbols across
Python, JavaScript, TypeScript, Rust, Go, and Java repositories.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

try:
    from tree_sitter_languages import get_parser as _get_ts_parser
except Exception:  # pragma: no cover - optional dependency
    _get_ts_parser = None

_CACHE_VERSION = 1
_DEFAULT_CACHE_PATH = Path(".copex") / "repo_map.json"

_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
}

_TREE_SITTER_LANGUAGE_BY_EXTENSION: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
}

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".idea",
    ".vscode",
}

_PY_IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_.,\s]+)\s*$")
_PY_FROM_IMPORT_RE = re.compile(r"^\s*from\s+([a-zA-Z0-9_\.]+)\s+import\s+")
_JS_IMPORT_RE = re.compile(r'^\s*import\s+.*?from\s+["\']([^"\']+)["\']')
_JS_SIDE_EFFECT_IMPORT_RE = re.compile(r'^\s*import\s+["\']([^"\']+)["\']')
_JS_REQUIRE_RE = re.compile(r'require\(\s*["\']([^"\']+)["\']\s*\)')
_JAVA_IMPORT_RE = re.compile(r"^\s*import\s+([a-zA-Z0-9_.*]+)\s*;")
_RUST_USE_RE = re.compile(r"^\s*use\s+([^;]+);")
_GO_IMPORT_RE = re.compile(r'"([^"]+)"')

_PY_CLASS_RE = re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_PY_DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_JS_CLASS_RE = re.compile(r"^\s*(?:export\s+)?class\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_JS_FUNCTION_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?function\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_JS_ARROW_RE = re.compile(
    r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:async\s+)?\("
)
_RUST_CLASS_RE = re.compile(r"^\s*(?:pub\s+)?(?:struct|enum|trait)\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_RUST_FN_RE = re.compile(r"^\s*(?:pub\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_GO_STRUCT_RE = re.compile(r"^\s*type\s+([A-Za-z_][A-Za-z0-9_]*)\s+struct\b")
_GO_FUNC_RE = re.compile(
    r"^\s*func\s*(?:\([^)]+\)\s*)?([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_GO_METHOD_RE = re.compile(r"^\s*func\s*\([^)]+\)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_JAVA_CLASS_RE = re.compile(
    r"^\s*(?:public|protected|private|abstract|final|sealed|non-sealed|\s)*"
    r"(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
_JAVA_METHOD_RE = re.compile(
    r"^\s*(?:public|protected|private|static|final|abstract|synchronized|native|\s)+"
    r"(?:[A-Za-z_][A-Za-z0-9_<>\[\],\s.?]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
_CALL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_TS_PARSER_CACHE: dict[str, Any] = {}

_CLASS_NODE_TYPES = {
    "class_definition",
    "class_declaration",
    "class_specifier",
}
_FUNCTION_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "function_item",
}
_METHOD_NODE_TYPES = {
    "method_definition",
    "method_declaration",
}
_CALL_NODE_TYPES = {
    "call",
    "call_expression",
    "function_call",
    "method_invocation",
}
_NAME_NODE_HINTS = {
    "identifier",
    "property_identifier",
    "field_identifier",
    "type_identifier",
    "scoped_identifier",
    "qualified_identifier",
}
_CALL_KEYWORDS = {
    "if",
    "for",
    "while",
    "switch",
    "return",
    "sizeof",
    "catch",
    "new",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass
class RepoMapFile:
    """Indexed metadata and symbols for one source file."""

    path: str
    language: str
    parser: str
    imports: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    functions: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    calls: dict[str, list[str]] = field(default_factory=dict)
    mtime_ns: int = 0
    size: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepoMapFile:
        return cls(
            path=str(data.get("path", "")),
            language=str(data.get("language", "")),
            parser=str(data.get("parser", "regex")),
            imports=list(data.get("imports", [])),
            classes=list(data.get("classes", [])),
            functions=list(data.get("functions", [])),
            methods=list(data.get("methods", [])),
            calls={
                str(symbol): list(values)
                for symbol, values in dict(data.get("calls", {})).items()
            },
            mtime_ns=int(data.get("mtime_ns", 0)),
            size=int(data.get("size", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "parser": self.parser,
            "imports": self.imports,
            "classes": self.classes,
            "functions": self.functions,
            "methods": self.methods,
            "calls": self.calls,
            "mtime_ns": self.mtime_ns,
            "size": self.size,
        }

    def key_symbols(self, limit: int = 12) -> list[str]:
        ordered = [*self.classes, *self.functions, *self.methods]
        seen: set[str] = set()
        result: list[str] = []
        for name in ordered:
            if not name or name in seen:
                continue
            seen.add(name)
            result.append(name)
            if len(result) >= limit:
                break
        return result


@dataclass
class RelevantFile:
    """Ranked relevance information for a file."""

    path: str
    score: float
    symbols: list[str] = field(default_factory=list)


@dataclass
class _ParseResult:
    imports: set[str] = field(default_factory=set)
    classes: set[str] = field(default_factory=set)
    functions: set[str] = field(default_factory=set)
    methods: set[str] = field(default_factory=set)
    calls: dict[str, set[str]] = field(default_factory=dict)

    def add_call(self, caller: str | None, callee: str | None) -> None:
        if not caller or not callee:
            return
        if callee in _CALL_KEYWORDS:
            return
        self.calls.setdefault(caller, set()).add(callee)


class RepoMap:
    """Build, cache, and query a repository map."""

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        cache_path: Path | str | None = None,
    ) -> None:
        self.root = Path(root or Path.cwd()).resolve()
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else (self.root / _DEFAULT_CACHE_PATH)
        )
        self.files: dict[str, RepoMapFile] = {}
        self.generated_at: str | None = None
        self._loaded = False

    def load_cache(self) -> None:
        """Load map data from .copex/repo_map.json if present."""
        self._loaded = True
        if not self.cache_path.is_file():
            return
        try:
            raw = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.debug("Unable to read repo map cache", exc_info=True)
            return

        if int(raw.get("version", 0)) != _CACHE_VERSION:
            return

        files_raw = raw.get("files", {})
        if not isinstance(files_raw, dict):
            return

        parsed: dict[str, RepoMapFile] = {}
        for path, entry in files_raw.items():
            if not isinstance(entry, dict):
                continue
            item = RepoMapFile.from_dict(entry)
            if item.path:
                parsed[path] = item
        self.files = parsed
        self.generated_at = raw.get("generated_at")

    def save_cache(self) -> None:
        """Persist map data to .copex/repo_map.json."""
        payload = {
            "version": _CACHE_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "root": str(self.root),
            "files": {path: entry.to_dict() for path, entry in sorted(self.files.items())},
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.generated_at = payload["generated_at"]

    def refresh(self, *, force: bool = False) -> dict[str, RepoMapFile]:
        """Refresh the map, incrementally when possible."""
        if not self._loaded:
            self.load_cache()

        source_files = self._discover_source_files()
        source_set = set(source_files)

        for stale in sorted(set(self.files) - source_set):
            self.files.pop(stale, None)

        if force or not self.files:
            targets = source_files
        else:
            changed = self._git_changed_files()
            if changed is None:
                targets = [path for path in source_files if self._metadata_changed(path)]
            else:
                targets = sorted(
                    {
                        *{path for path in source_files if path in changed},
                        *{path for path in source_files if path not in self.files},
                        *{path for path in source_files if self._metadata_changed(path)},
                    }
                )

        for rel_path in targets:
            try:
                self.files[rel_path] = self._parse_file(rel_path)
            except Exception:
                logger.debug("Failed to parse file for repo map: %s", rel_path, exc_info=True)

        self.save_cache()
        return self.files

    build = refresh

    def render_map(self, *, max_files: int = 200, max_symbols_per_file: int = 12) -> str:
        """Render a concise aider-style repository map."""
        if not self._loaded:
            self.load_cache()
        if not self.files:
            self.refresh(force=False)

        lines = [f"Repo Map ({len(self.files)} files)"]
        count = 0
        for path, entry in sorted(self.files.items()):
            if count >= max_files:
                break
            lines.append(path)
            if entry.classes:
                lines.append(f"  classes: {', '.join(entry.classes[:max_symbols_per_file])}")
            if entry.functions:
                lines.append(f"  functions: {', '.join(entry.functions[:max_symbols_per_file])}")
            if entry.methods:
                lines.append(f"  methods: {', '.join(entry.methods[:max_symbols_per_file])}")
            if entry.imports:
                lines.append(f"  imports: {', '.join(entry.imports[:6])}")
            count += 1
        if count < len(self.files):
            lines.append(f"... ({len(self.files) - count} more files)")
        return "\n".join(lines)

    def rank_relevant(
        self,
        task_description: str,
        *,
        limit: int = 10,
    ) -> list[RelevantFile]:
        """Rank files by relevance to a task description."""
        if not self._loaded:
            self.load_cache()
        if not self.files:
            self.refresh(force=False)

        query_tokens = set(_tokenize(task_description))
        if not query_tokens:
            return []

        ranked: list[RelevantFile] = []
        for path, entry in self.files.items():
            path_tokens = set(_tokenize(path))
            symbol_pool = [*entry.classes, *entry.functions, *entry.methods]
            symbol_tokens = set(_tokenize(" ".join(symbol_pool)))
            import_tokens = set(_tokenize(" ".join(entry.imports)))

            score = 0.0
            score += 5.0 * len(query_tokens & path_tokens)
            score += 8.0 * len(query_tokens & symbol_tokens)
            score += 2.0 * len(query_tokens & import_tokens)

            matched_symbols = [
                symbol
                for symbol in symbol_pool
                if any(token in symbol.lower() for token in query_tokens)
            ]
            if matched_symbols:
                score += min(6.0, float(len(matched_symbols)))

            if score <= 0:
                continue
            ranked.append(
                RelevantFile(
                    path=path,
                    score=score,
                    symbols=_dedupe_preserve_order(matched_symbols or entry.key_symbols(8)),
                )
            )

        ranked.sort(key=lambda item: (-item.score, item.path))
        return ranked[:limit]

    def render_relevant(self, task_description: str, *, limit: int = 10) -> str:
        """Render ranked relevance output."""
        ranked = self.rank_relevant(task_description, limit=limit)
        if not ranked:
            return f"No relevant files found for: {task_description}"
        lines = [f"Relevant files for: {task_description}"]
        for idx, item in enumerate(ranked, start=1):
            symbols = ", ".join(item.symbols[:8]) if item.symbols else "(no symbols)"
            lines.append(f"{idx}. {item.path} [{item.score:.1f}] — {symbols}")
        return "\n".join(lines)

    def relevant_context(
        self,
        task_description: str,
        *,
        max_files: int = 8,
        max_symbols_per_file: int = 8,
    ) -> str:
        """Build compact context block with relevant files/symbols."""
        ranked = self.rank_relevant(task_description, limit=max_files)
        if not ranked:
            return ""
        lines = ["## Relevant Repository Map"]
        for item in ranked:
            symbols = ", ".join(item.symbols[:max_symbols_per_file]) if item.symbols else "(no symbols)"
            lines.append(f"- {item.path}: {symbols}")
        return "\n".join(lines)

    def _discover_source_files(self) -> list[str]:
        files: list[str] = []
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in _LANGUAGE_BY_EXTENSION:
                continue
            rel = path.relative_to(self.root)
            if self._should_skip(rel):
                continue
            files.append(rel.as_posix())
        return sorted(files)

    def _should_skip(self, rel_path: Path) -> bool:
        parts = rel_path.parts[:-1]
        for part in parts:
            if part in _SKIP_DIRS:
                return True
            if part.startswith(".") and part not in {".github"}:
                return True
        return False

    def _metadata_changed(self, rel_path: str) -> bool:
        current = self.root / rel_path
        cached = self.files.get(rel_path)
        if cached is None:
            return True
        try:
            stat = current.stat()
        except OSError:
            return True
        return cached.mtime_ns != stat.st_mtime_ns or cached.size != stat.st_size

    def _git_changed_files(self) -> set[str] | None:
        try:
            inside = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            if inside.returncode != 0:
                return None

            changed: set[str] = set()
            diff_proc = subprocess.run(
                ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD"],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            if diff_proc.returncode == 0:
                changed.update(_normalize_git_paths(diff_proc.stdout.splitlines()))

            untracked_proc = subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=self.root,
                capture_output=True,
                text=True,
                check=False,
            )
            if untracked_proc.returncode == 0:
                changed.update(_normalize_git_paths(untracked_proc.stdout.splitlines()))

            return changed
        except OSError:
            return None

    def _parse_file(self, rel_path: str) -> RepoMapFile:
        path = self.root / rel_path
        stat = path.stat()
        text = path.read_text(encoding="utf-8", errors="replace")
        extension = path.suffix.lower()
        language = _LANGUAGE_BY_EXTENSION.get(extension, "unknown")

        parse_result = self._parse_with_tree_sitter(text, extension)
        parser_kind = "tree-sitter"
        if parse_result is None:
            parse_result = self._parse_with_regex(text, extension)
            parser_kind = "regex"

        calls = {
            symbol: sorted(values)
            for symbol, values in sorted(parse_result.calls.items())
            if values
        }
        return RepoMapFile(
            path=rel_path,
            language=language,
            parser=parser_kind,
            imports=sorted(parse_result.imports),
            classes=sorted(parse_result.classes),
            functions=sorted(parse_result.functions),
            methods=sorted(parse_result.methods),
            calls=calls,
            mtime_ns=stat.st_mtime_ns,
            size=stat.st_size,
        )

    def _parse_with_tree_sitter(self, text: str, extension: str) -> _ParseResult | None:
        parser_language = _TREE_SITTER_LANGUAGE_BY_EXTENSION.get(extension)
        if parser_language is None:
            return None

        parser = _get_tree_sitter_parser(parser_language)
        if parser is None:
            return None

        source = text.encode("utf-8")
        try:
            tree = parser.parse(source)
        except Exception:  # pragma: no cover - parser internals vary
            logger.debug("tree-sitter parse failed for %s", extension, exc_info=True)
            return None

        result = _ParseResult()
        class_stack: list[str] = []
        symbol_stack: list[str] = []

        def visit(node: Any) -> None:
            node_type = str(getattr(node, "type", "")).lower()
            entered_class = False
            entered_symbol = False

            if _is_import_node(node_type):
                import_text = _node_text(source, node)
                for token in _extract_imports(import_text):
                    result.imports.add(token)

            if node_type in _CLASS_NODE_TYPES:
                class_name = _extract_node_name(node, source)
                if class_name:
                    result.classes.add(class_name)
                    class_stack.append(class_name)
                    entered_class = True
            elif node_type in _METHOD_NODE_TYPES:
                method_name = _extract_node_name(node, source)
                if method_name:
                    result.methods.add(method_name)
                    qualified = (
                        f"{class_stack[-1]}.{method_name}" if class_stack else method_name
                    )
                    symbol_stack.append(qualified)
                    entered_symbol = True
            elif node_type in _FUNCTION_NODE_TYPES:
                function_name = _extract_node_name(node, source)
                if function_name:
                    result.functions.add(function_name)
                    symbol_stack.append(function_name)
                    entered_symbol = True

            if node_type in _CALL_NODE_TYPES and symbol_stack:
                result.add_call(symbol_stack[-1], _extract_call_name(node, source))

            for child in list(getattr(node, "children", [])):
                visit(child)

            if entered_symbol and symbol_stack:
                symbol_stack.pop()
            if entered_class and class_stack:
                class_stack.pop()

        visit(tree.root_node)
        return result

    def _parse_with_regex(self, text: str, extension: str) -> _ParseResult:
        result = _ParseResult()
        for token in _extract_imports(text):
            result.imports.add(token)

        if extension == ".py":
            _extract_python_symbols(text, result)
            return result
        if extension in {".js", ".jsx", ".ts", ".tsx"}:
            _extract_js_ts_symbols(text, result)
            return result
        if extension == ".rs":
            _extract_rust_symbols(text, result)
            return result
        if extension == ".go":
            _extract_go_symbols(text, result)
            return result
        if extension == ".java":
            _extract_java_symbols(text, result)
            return result
        return result


def _get_tree_sitter_parser(language: str) -> Any | None:
    if _get_ts_parser is None:
        return None
    if language in _TS_PARSER_CACHE:
        return _TS_PARSER_CACHE[language]
    try:
        parser = _get_ts_parser(language)
    except Exception:  # pragma: no cover - optional dependency behavior
        parser = None
    _TS_PARSER_CACHE[language] = parser
    return parser


def _normalize_git_paths(paths: list[str]) -> set[str]:
    normalized = set()
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        normalized.add(Path(path).as_posix())
    return normalized


def _is_import_node(node_type: str) -> bool:
    return "import" in node_type


def _node_text(source: bytes, node: Any) -> str:
    start = int(getattr(node, "start_byte", 0))
    end = int(getattr(node, "end_byte", 0))
    if end <= start:
        return ""
    return source[start:end].decode("utf-8", errors="replace")


def _extract_node_name(node: Any, source: bytes) -> str | None:
    for field_name in ("name", "declarator", "field", "property"):
        child = node.child_by_field_name(field_name) if hasattr(node, "child_by_field_name") else None
        if child is None:
            continue
        name = _extract_name_from_node(child, source)
        if name:
            return name

    for child in list(getattr(node, "children", [])):
        name = _extract_name_from_node(child, source)
        if name:
            return name
    return None


def _extract_name_from_node(node: Any, source: bytes) -> str | None:
    node_type = str(getattr(node, "type", "")).lower()
    text = _node_text(source, node).strip()
    if node_type in _NAME_NODE_HINTS:
        return _sanitize_symbol_name(text)

    if text:
        candidates = _CALL_NAME_RE.findall(text)
        if candidates:
            return _sanitize_symbol_name(candidates[-1])
    return None


def _extract_call_name(node: Any, source: bytes) -> str | None:
    target = None
    if hasattr(node, "child_by_field_name"):
        target = node.child_by_field_name("function") or node.child_by_field_name("name")
    if target is None:
        named = list(getattr(node, "named_children", []))
        if named:
            target = named[0]
        else:
            children = list(getattr(node, "children", []))
            target = children[0] if children else None
    if target is None:
        return None

    raw = _node_text(source, target).strip()
    if not raw:
        return None
    candidates = _CALL_NAME_RE.findall(raw)
    if not candidates:
        return None
    return _sanitize_symbol_name(candidates[-1])


def _sanitize_symbol_name(value: str) -> str | None:
    cleaned = value.strip().strip(".")
    if not cleaned:
        return None
    return cleaned


def _extract_imports(text: str) -> list[str]:
    imports: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//")):
            continue

        py_match = _PY_IMPORT_RE.match(stripped)
        if py_match:
            imports.extend(
                item.strip()
                for item in py_match.group(1).split(",")
                if item.strip()
            )
            continue

        py_from_match = _PY_FROM_IMPORT_RE.match(stripped)
        if py_from_match:
            imports.append(py_from_match.group(1).strip())
            continue

        js_match = _JS_IMPORT_RE.match(stripped)
        if js_match:
            imports.append(js_match.group(1).strip())
            continue

        js_side_effect = _JS_SIDE_EFFECT_IMPORT_RE.match(stripped)
        if js_side_effect:
            imports.append(js_side_effect.group(1).strip())
            continue

        js_requires = _JS_REQUIRE_RE.findall(stripped)
        if js_requires:
            imports.extend(req.strip() for req in js_requires if req.strip())
            continue

        java_match = _JAVA_IMPORT_RE.match(stripped)
        if java_match:
            imports.append(java_match.group(1).strip())
            continue

        rust_match = _RUST_USE_RE.match(stripped)
        if rust_match:
            imports.append(rust_match.group(1).strip())
            continue

        if stripped.startswith("import ") and '"' in stripped:
            go_matches = _GO_IMPORT_RE.findall(stripped)
            imports.extend(match.strip() for match in go_matches if match.strip())

    return _dedupe_preserve_order(imports)


def _extract_python_symbols(text: str, result: _ParseResult) -> None:
    class_stack: list[tuple[int, str]] = []
    function_stack: list[tuple[int, str]] = []

    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue

        indent = len(line) - len(line.lstrip(" "))
        while function_stack and indent <= function_stack[-1][0]:
            function_stack.pop()
        while class_stack and indent <= class_stack[-1][0]:
            class_stack.pop()

        class_match = _PY_CLASS_RE.match(line)
        if class_match:
            class_name = class_match.group(1)
            result.classes.add(class_name)
            class_stack.append((indent, class_name))
            continue

        def_match = _PY_DEF_RE.match(line)
        if def_match:
            fn = def_match.group(1)
            if class_stack and indent > class_stack[-1][0]:
                result.methods.add(fn)
                function_stack.append((indent, f"{class_stack[-1][1]}.{fn}"))
            else:
                result.functions.add(fn)
                function_stack.append((indent, fn))
            continue

        if function_stack:
            caller = function_stack[-1][1]
            for callee in _CALL_NAME_RE.findall(line):
                if callee == caller or callee in _CALL_KEYWORDS:
                    continue
                result.add_call(caller, callee)


def _extract_js_ts_symbols(text: str, result: _ParseResult) -> None:
    current_class: str | None = None
    brace_depth = 0
    class_depth = 0

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            brace_depth += line.count("{") - line.count("}")
            if current_class and brace_depth < class_depth:
                current_class = None
            continue

        class_match = _JS_CLASS_RE.match(line)
        if class_match:
            current_class = class_match.group(1)
            result.classes.add(current_class)
            class_depth = brace_depth + max(1, line.count("{"))

        function_match = _JS_FUNCTION_RE.match(line)
        if function_match:
            result.functions.add(function_match.group(1))

        arrow_match = _JS_ARROW_RE.match(line)
        if arrow_match:
            result.functions.add(arrow_match.group(1))

        method_match = re.match(
            r"^\s*(?:public|private|protected|static|async|\s)*([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*\{",
            line,
        )
        if method_match and current_class:
            method_name = method_match.group(1)
            if method_name not in {"if", "for", "while", "switch", "catch", "function"}:
                result.methods.add(method_name)

        brace_depth += line.count("{") - line.count("}")
        if current_class and brace_depth < class_depth:
            current_class = None


def _extract_rust_symbols(text: str, result: _ParseResult) -> None:
    for line in text.splitlines():
        class_match = _RUST_CLASS_RE.match(line)
        if class_match:
            result.classes.add(class_match.group(1))
        fn_match = _RUST_FN_RE.match(line)
        if fn_match:
            result.functions.add(fn_match.group(1))


def _extract_go_symbols(text: str, result: _ParseResult) -> None:
    for line in text.splitlines():
        struct_match = _GO_STRUCT_RE.match(line)
        if struct_match:
            result.classes.add(struct_match.group(1))

        method_match = _GO_METHOD_RE.match(line)
        if method_match:
            result.methods.add(method_match.group(1))
            continue

        fn_match = _GO_FUNC_RE.match(line)
        if fn_match:
            result.functions.add(fn_match.group(1))


def _extract_java_symbols(text: str, result: _ParseResult) -> None:
    for line in text.splitlines():
        class_match = _JAVA_CLASS_RE.match(line)
        if class_match:
            result.classes.add(class_match.group(1))
            continue
        method_match = _JAVA_METHOD_RE.match(line)
        if method_match:
            result.methods.add(method_match.group(1))


def _tokenize(text: str) -> list[str]:
    normalized = (
        text.replace("/", " ")
        .replace("\\", " ")
        .replace("-", " ")
        .replace(".", " ")
    )
    tokens = [token.lower() for token in _TOKEN_RE.findall(normalized)]
    return [token for token in tokens if len(token) > 1]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = ["RepoMap", "RepoMapFile", "RelevantFile"]
