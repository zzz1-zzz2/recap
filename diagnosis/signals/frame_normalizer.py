"""Shared frame normalization — single source of truth for StackFrame construction.

All extractors (pytest, Django, FW fallback) should use these functions
instead of implementing their own path/repo/test-file heuristics.

Design:
  - normalize_frame(): takes raw path + metadata → StackFrame
  - is_test_path() / is_repo_path(): extracted heuristics, unit-testable
  - PurePosixPath for all path operations (works with Linux containers)
"""
from __future__ import annotations

from pathlib import PurePosixPath

from .schema import StackFrame


def is_test_path(repo_path: str) -> bool:
    """Check if a repo-relative path corresponds to a test file.

    Heuristics:
      - Directories named 'test', 'tests', 'testing' anywhere in the path
      - Filename starting with test_ or ending with _test.py
    """
    if not repo_path:
        return False
    try:
        pp = PurePosixPath(repo_path)
    except (ValueError, TypeError):
        return False
    parts = [p.lower() for p in pp.parts]
    filename = pp.name.lower()
    # Check directories excluding the filename itself
    has_test_dir = any(part in {"test", "tests", "testing"} for part in parts[:-1])
    is_test_file = filename.startswith("test_") or filename.endswith("_test.py")
    return has_test_dir or is_test_file


def is_repo_path(path: str) -> bool:
    """Determine if a file path is within the target repository.

    Returns True if:
      - Path contains /testbed/ (SWE-bench standard working dir)
      - Path is relative (doesn't start with /, <, or >)
    Returns False for:
      - System paths (/usr/, /opt/, /lib/)
      - Empty paths
      - Dist-packages / site-packages
    """
    if not path:
        return False
    if path.startswith(("<", ">", "[")):
        return False
    if "/testbed/" in path:
        return True
    if path.startswith("/opt/") or path.startswith("/usr/") or path.startswith("/lib/"):
        return False
    if "site-packages/" in path or "dist-packages/" in path:
        return False
    # Relative path (doesn't start with /)
    if not path.startswith("/"):
        return True
    return False


def normalize_frame(
    file_path: str,
    line: int = 0,
    function: str = "",
    *,
    fallback_is_repo: bool | None = None,
    fallback_is_test: bool | None = None,
) -> StackFrame:
    """Build a StackFrame with consistent heuristics.

    Args:
        file_path: Raw file path from test output or FW.
        line: Line number (0 if unknown).
        function: Function/method name (empty if unknown).
        fallback_is_repo: Override is_repo heuristic if set (e.g. from pytest extractor).
        fallback_is_test: Override is_test heuristic if set.

    Returns:
        Populated StackFrame.
    """
    # Determine repo path (strip /testbed/ prefix)
    if "/testbed/" in file_path:
        repo_file = file_path.split("/testbed/", 1)[1]
    else:
        repo_file = file_path

    is_repo = fallback_is_repo if fallback_is_repo is not None else is_repo_path(file_path)

    # Only check test path for repo files
    is_test = False
    if fallback_is_test is not None:
        is_test = fallback_is_test
    elif is_repo:
        is_test = is_test_path(repo_file)

    return StackFrame(
        file=repo_file,
        line=line,
        function=function,
        is_repo_frame=is_repo,
        is_test_file=is_test,
    )
