"""P1-3D: FIND_DEFINITION executor — AST-based symbol/file lookup.

Strategy:
  1. FAILURE_SITE target → walk AST to find the smallest function/class
     enclosing the target line; return its full definition.
  2. FILE target → return the file's top-level definitions (heuristic).
  3. SYMBOL / TYPE_NAME target → grep the repo for the symbol with a
     strict budget; prefer the first hit whose path matches the
     failure site when available.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from recap.acquisition.schema import (
    AcquisitionHit,
    AcquisitionResult,
    AcquisitionStatus,
)
from recap.diagnosis.search_contract import SearchAction, SearchTarget, SearchTargetKind


def _safe_resolve(repo_root: Path, raw_path: str) -> Path | None:
    """Resolve `raw_path` against `repo_root`; ensure result stays inside it.

    Returns None if the resolved path is outside repo_root or doesn't exist.
    Uses `is_relative_to` for strict containment check.
    """
    if not raw_path:
        return None
    raw_path = raw_path.lstrip("/")
    try:
        candidate = (repo_root / raw_path).resolve()
        abs_repo = repo_root.resolve()
    except (OSError, ValueError):
        return None
    try:
        candidate.relative_to(abs_repo)
    except ValueError:
        return None
    if not candidate.is_relative_to(abs_repo):
        return None
    if not candidate.exists():
        return None
    return candidate


def _read_file_safe(path: Path, max_bytes: int = 200_000) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _enclosing_node(
    tree: ast.AST,
    target_line: int,
) -> ast.AST | None:
    """Walk AST to find the smallest function/class/method whose body
    contains `target_line`."""
    best: ast.AST | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not hasattr(node, "lineno") or not hasattr(node, "body"):
            continue
        # Quick line-range check
        start = getattr(node, "lineno", 0)
        end = getattr(node, "end_lineno", 0)
        if start and end and start <= target_line <= end:
            # Pick the smallest enclosing node
            if best is None or (end - start) < (
                getattr(best, "end_lineno", 0) - getattr(best, "lineno", 0)
            ):
                best = node
    return best


def _node_source(source: str, node: ast.AST) -> str:
    """Extract source lines for the node from the source text."""
    lines = source.splitlines()
    start = (getattr(node, "lineno", 1) or 1) - 1
    end = getattr(node, "end_lineno", start + 1) or start + 1
    return "\n".join(lines[start:end])


def _format_node(node: ast.AST) -> tuple[str, str, str]:
    """Return (qualified_name, header, body) for an AST node."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        qualname = node.name
        args = ast.unparse(node.args) if hasattr(ast, "unparse") else ""
        header = f"def {node.name}({args}):"
    elif isinstance(node, ast.ClassDef):
        qualname = node.name
        header = f"class {node.name}"
    else:
        qualname = "<unknown>"
        header = "<unknown>"
    return qualname, header, ""


def _find_definition_for_failure_site(
    repo_root: Path,
    target: SearchTarget,
) -> AcquisitionResult:
    """Resolve FAILURE_SITE (file.py:LINE) → enclosing function/class."""
    raw_path, _, line_str = target.value.partition(":")
    try:
        target_line = int(line_str)
    except ValueError:
        return AcquisitionResult(
            action_id="", action_type=target.kind,
            target=target, status=AcquisitionStatus.INVALID_TARGET,
            errors=[f"invalid line number in {target.value!r}"],
        )

    abs_path = _safe_resolve(repo_root, raw_path)
    if abs_path is None:
        return AcquisitionResult(
            action_id="", action_type=target.kind,
            target=target, status=AcquisitionStatus.NOT_FOUND,
            stop_reason=f"file {raw_path!r} not in repo",
        )

    source = _read_file_safe(abs_path)
    if source is None:
        return AcquisitionResult(
            action_id="", action_type=target.kind,
            target=target, status=AcquisitionStatus.NOT_FOUND,
            stop_reason=f"cannot read {abs_path.name}",
        )
    try:
        tree = ast.parse(source, filename=str(abs_path))
    except SyntaxError as e:
        return AcquisitionResult(
            action_id="", action_type=target.kind,
            target=target, status=AcquisitionStatus.ERROR,
            errors=[f"SyntaxError: {e}"],
            files_examined=1,
        )

    node = _enclosing_node(tree, target_line)
    rel_path = str(abs_path.relative_to(repo_root.resolve()))
    if node is None:
        # No enclosing function/class — fall back to ±5 lines around the target.
        lines = source.splitlines()
        idx = max(0, target_line - 1)
        lo, hi = max(0, idx - 5), min(len(lines), idx + 5)
        snippet = "\n".join(lines[lo:hi])
        return AcquisitionResult(
            action_id="", action_type=target.kind,
            target=target, status=AcquisitionStatus.FOUND,
            files_examined=1,
            budget_used=1,
            stop_reason="no enclosing AST node; returned ±5 lines",
            hits=[AcquisitionHit(
                file_path=rel_path,
                start_line=lo + 1, end_line=hi,
                symbol="",
                content=snippet,
                retrieval_method="ast_fallback_lines",
                relevance_reason="no enclosing def/class near target line",
                action_id="",
                evidence_ids=target.source_evidence_ids,
            )],
        )

    qualname, header, _ = _format_node(node)
    body = _node_source(source, node)
    start_line = getattr(node, "lineno", target_line)
    end_line = getattr(node, "end_lineno", target_line)
    return AcquisitionResult(
        action_id="", action_type=target.kind,
        target=target, status=AcquisitionStatus.FOUND,
        files_examined=1,
        budget_used=1,
        stop_reason=f"AST enclosing {type(node).__name__}",
        hits=[AcquisitionHit(
            file_path=rel_path,
            start_line=start_line, end_line=end_line,
            symbol=qualname,
            content=f"{header}\n{body}",
            retrieval_method="ast_enclosing_node",
            relevance_reason=f"line {target_line} is inside {qualname}",
            action_id="",
            evidence_ids=target.source_evidence_ids,
        )],
    )


def _find_definition_for_symbol(
    repo_root: Path,
    target: SearchTarget,
    budget: int = 3,
    max_files_examined: int = 200,
) -> AcquisitionResult:
    """Resolve SYMBOL/TYPE_NAME → grep for `def name` or `class name`."""
    sym = target.value.split(".")[-1]
    abs_repo = repo_root.resolve()
    files_examined = 0
    hits: list[AcquisitionHit] = []
    pattern_def = f"def {sym}("
    pattern_class = f"class {sym}"

    # Bounded file walk — deterministic ordering, .py files only
    sym = target.value.split(".")[-1]
    abs_repo = repo_root.resolve()
    max_files_examined = budget * 20
    files_examined = 0
    hits: list[AcquisitionHit] = []
    pattern_def = f"def {sym}("
    pattern_class = f"class {sym}"

    candidates = sorted(
        [c for c in abs_repo.rglob("*.py")
         if not any(part.startswith(".") for part in c.relative_to(abs_repo).parts)
         and not any(skip in c.parts for skip in ("__pycache__", "build", "dist", "node_modules"))],
        key=lambda c: str(c.relative_to(abs_repo)),
    )

    for path in candidates:
        if len(hits) >= budget:
            break
        if files_examined >= max_files_examined:
            break
        files_examined += 1
        text = _read_file_safe(path)
        if text is None:
            continue
        rel = str(path.relative_to(abs_repo))
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern_def in line or pattern_class in line:
                hits.append(AcquisitionHit(
                    file_path=rel, start_line=i, end_line=i,
                    symbol=sym,
                    content=line.rstrip(),
                    retrieval_method="grep_symbol",
                    relevance_reason=f"line defines {sym!r}",
                    action_id="",
                    evidence_ids=target.source_evidence_ids,
                ))
                break  # one hit per file

    return AcquisitionResult(
        action_id="", action_type=target.kind,
        target=target,
        status=AcquisitionStatus.FOUND if hits else AcquisitionStatus.NOT_FOUND,
        hits=hits, files_examined=files_examined, budget_used=len(hits),
        budget_limit=budget,
        scan_limit=max_files_examined,
        stop_reason="budget" if hits else "no matches in repo",
    )


def find_definition(
    action: SearchAction,
    repo_root: Path,
    max_files_examined: int = 200,
) -> AcquisitionResult:
    """Executor for SearchActionType.FIND_DEFINITION.

    Wraps the internal dispatch, guaranteeing action_id and action_type
    provenance are set even when the dispatcher returns early.
    """
    _result = _dispatch_definition(action, repo_root, max_files_examined=max_files_examined)
    _result.action_id = action.action_id
    _result.action_type = action.action_type
    for _hit in _result.hits:
        _hit.action_id = action.action_id
    _result.target = action.target
    return _result


def _dispatch_definition(
    action: SearchAction, repo_root: Path,
    max_files_examined: int = 200,
) -> AcquisitionResult:
    """Internal dispatch — may return early before provenance is set."""
    target = action.target
    if target.kind == SearchTargetKind.FAILURE_SITE:
        result = _find_definition_for_failure_site(repo_root, target)
    elif target.kind in (SearchTargetKind.SYMBOL, SearchTargetKind.TYPE_NAME):
        result = _find_definition_for_symbol(
            repo_root, target, budget=action.budget,
            max_files_examined=max_files_examined,
        )
    elif target.kind == SearchTargetKind.FILE:
        # Return the FILE's top-level definitions (no AST node = empty hit
        # for now; concrete FILE handling lives in v2).
        abs_path = _safe_resolve(repo_root, target.value)
        if abs_path is None:
            return AcquisitionResult(
                action_id=action.action_id, action_type=action.action_type,
                target=target, status=AcquisitionStatus.NOT_FOUND,
                stop_reason="file outside repo or missing",
            )
        return AcquisitionResult(
            action_id=action.action_id, action_type=action.action_type,
            target=target, status=AcquisitionStatus.FOUND,
            files_examined=1, budget_used=1,
            stop_reason="file target — body returned",
            hits=[AcquisitionHit(
                file_path=target.value, start_line=1,
                end_line=(_read_file_safe(abs_path) or "").count("\n") + 1,
                symbol="",
                content=(_read_file_safe(abs_path) or "")[:2000],
                retrieval_method="file_read",
                relevance_reason="target is a FILE",
                action_id=action.action_id,
                evidence_ids=target.source_evidence_ids,
            )],
        )
    else:
        return AcquisitionResult(
            action_id=action.action_id, action_type=action.action_type,
            target=target, status=AcquisitionStatus.UNSUPPORTED,
            errors=[f"FIND_DEFINITION does not accept target_kind={target.kind.value}"],
        )
    return result
