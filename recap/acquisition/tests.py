"""P1-3D: FIND_RELATED_TESTS executor — ranked test discovery.

Ranking (descending):
  1. Test body mentions the target symbol (regex)
  2. Test imports the target module
  3. Test path mirrors source path (test_foo.py ↔ foo.py)
  4. Test name contains target symbol substring
  5. Adjacent to a known-failed test (path proximity)

Marks each hit as `already_seen_in_r1: bool` if the file path was in
the agent's trajectory.viewed_files. This prevents Router from just
returning things the agent already looked at.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from recap.acquisition.schema import (
    AcquisitionHit,
    AcquisitionResult,
    AcquisitionStatus,
)
from recap.diagnosis.search_contract import SearchAction, SearchTarget, SearchTargetKind


# ── Skip dirs when walking ───────────────────────────────────────────

_SKIP_DIRS = {
    "__pycache__", "build", "dist", ".tox", ".eggs",
    "site-packages", "node_modules",
}


def _iter_py_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*.py"):
        parts = path.relative_to(repo_root).parts
        if any(p.startswith(".") for p in parts):
            continue
        if any(p in _SKIP_DIRS for p in parts):
            continue
        yield path


def _is_test_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        name.startswith("test_") or name.endswith("_test.py")
        or "/tests/" in str(path).lower() or "/test/" in str(path).lower()
    )


def _read(path: Path) -> str | None:
    try:
        if path.stat().st_size > 500_000:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _module_of(file_path: Path, repo_root: Path) -> str:
    """Heuristic: derive module path from file_path relative to repo_root."""
    rel = file_path.relative_to(repo_root.resolve())
    parts = list(rel.parts)
    # Drop the filename
    parts[-1] = parts[-1][:-3]  # strip .py
    if parts[-1].startswith("test_"):
        parts[-1] = parts[-1][5:]
    return ".".join(parts)


def _score_test(
    test_path: Path,
    test_text: str,
    target: SearchTarget,
    target_module_hint: str,
    r1_viewed: set[str],
    repo_root: Path,
) -> tuple[int, str]:
    """Return (score, reason) for one test file vs one target."""
    # For FILE-kind targets, use the file stem as the "symbol".
    # Never split a file path on '.' — that produces 'py' from '.py'.
    if target.kind.value == "FILE" and "/" in target.value:
        sym = Path(target.value).stem
    elif target.kind.value == "FILE":
        sym = target.value.replace(".py", "")
    else:
        sym = target.value.split(".")[-1]
    try:
        rel = str(test_path.relative_to(repo_root.resolve()))
    except ValueError:
        rel = str(test_path.name)
    test_module = _module_of(test_path, repo_root)

    # 1. Symbol appears in test body
    if sym and sym in test_text:
        count = test_text.count(sym)
        return (100 + count, f"symbol {sym!r} appears {count}× in test body")

    # 2. Test imports the target module
    if target_module_hint and target_module_hint in test_text:
        return (80, f"imports target module {target_module_hint!r}")

    # 3. Path mirror (test_foo.py ↔ foo.py)
    src_name = sym.replace("_test", "") if "_test" in sym else sym
    if rel.startswith(f"test_{src_name}"):
        return (60, f"test file name mirrors {src_name!r}")
    if test_module.startswith(src_name):
        return (50, f"test module starts with {src_name!r}")

    # 4. Substring match
    if sym and sym.lower() in rel.lower():
        return (30, f"symbol substring in test file name {rel!r}")

    return (0, "")


def find_related_tests(
    action: SearchAction,
    repo_root: Path,
    r1_viewed_files: Iterable[str] | None = None,
    failed_test_names: Iterable[str] | None = None,
    target_module_hint: str = "",
    max_files_examined: int = 200,
) -> AcquisitionResult:
    """Executor for SearchActionType.FIND_RELATED_TESTS.

    Args:
      r1_viewed_files: file paths the R1 agent already inspected
        (used to mark hits as already_seen).
      failed_test_names: failing test names from the original R1 episode;
        tests in the same directory get a +20 proximity bonus.
      target_module_hint: optional module path (e.g. "astropy.coordinates")
        to match test imports against.
    """
    target = action.target
    # ── Canonicalize TEST targets ──
    # "file.py::test_name" → file path for searching, test name as symbol
    test_canonical_file: str | None = None
    if target.kind == SearchTargetKind.TEST and "::" in target.value:
        parts = target.value.split("::", 1)
        test_canonical_file = parts[0]
        # Use bare test name (without [param]) as symbol for scoring
        raw_sym = parts[1]
        sym = raw_sym.split("[")[0] if "[" in raw_sym else raw_sym
    elif target.kind.value == "FILE":
        sym = Path(target.value).stem
    else:
        sym = target.value.split(".")[-1] if target.value else ""
    r1_viewed = set(r1_viewed_files or [])
    failed_set = set(failed_test_names or [])
    budget = action.budget
    max_files = max_files_examined

    # Test paths adjacent to a failing test get a +20 proximity bonus.
    failed_dirs: set[str] = set()
    for ftname in failed_set:
        parts = ftname.split("::")
        if len(parts) == 2 and parts[0]:
            failed_dirs.add(str(Path(parts[0]).parent))

    scored: list[tuple[int, str, Path, str]] = []
    files_examined = 0

    # ── Direct file resolution for canonicalized TEST targets ──
    # When target is "file.py::test_name", check if file.py exists directly.
    # This guarantees FOUND for valid nodeids regardless of scoring heuristics.
    test_canonical_path: Path | None = None
    if test_canonical_file:
        candidate = repo_root.resolve() / test_canonical_file
        if candidate.exists() and _is_test_file(candidate):
            test_canonical_path = candidate
            text = _read(candidate)
            if text is not None:
                scored.append((200, f"direct match: test file for {test_canonical_file}", candidate, text))

    # Deterministic file walk: sort candidates for cross-fs stability.
    _d_candidates = sorted(
        _iter_py_files(repo_root),
        key=lambda p: str(p.relative_to(repo_root.resolve())),
    )

    for path in _d_candidates:
        if not _is_test_file(path):
            continue
        text = _read(path)
        if text is None:
            continue
        if files_examined >= max_files:
            break
        files_examined += 1
        score, reason = _score_test(path, text, target, target_module_hint, r1_viewed, repo_root)

        # Proximity bonus for FILE targets: same directory as target
        if score > 0 and target.kind.value == "FILE":
            try:
                target_dir = Path(target.value).parent
                test_dir = str(path.relative_to(repo_root.resolve()).parent)
                if test_dir == str(target_dir):
                    score += 30
                    reason = f"{reason}; same dir as target FILE"
            except (ValueError, OSError):
                pass

        # Proximity bonus: only apply when there IS a valid score.
        if score > 0:
            try:
                rel_dir = str(path.relative_to(repo_root.resolve()).parent)
                if rel_dir in failed_dirs:
                    score += 20
                    reason = f"{reason}; proximity to failed test"
            except (ValueError, OSError):
                pass

        if score > 0:
            scored.append((score, reason, path, text))

    scored.sort(key=lambda x: -x[0])

    hits: list[AcquisitionHit] = []
    rel_repo = repo_root.resolve()
    for score, reason, path, text in scored[:budget]:
        rel = str(path.relative_to(rel_repo))
        already_seen = rel in r1_viewed
        sym_in_body = sym in text if sym else False
        hits.append(AcquisitionHit(
            file_path=rel,
            start_line=1, end_line=text.count("\n") + 1,
            symbol=sym if sym_in_body else path.stem,
            content=text[:1500],
            retrieval_method="ranked_test_search",
            relevance_reason=f"{reason}; already_seen={already_seen}",
            action_id=action.action_id,
            evidence_ids=target.source_evidence_ids,
        ))

    return AcquisitionResult(
        action_id=action.action_id,
        action_type=action.action_type,
        target=target,
        status=AcquisitionStatus.FOUND if hits else AcquisitionStatus.NOT_FOUND,
        hits=hits,
        files_examined=files_examined,
        budget_used=len(hits),
        budget_limit=budget,
        scan_limit=max_files,
        stop_reason="budget" if hits else "no ranked matches",
    )
