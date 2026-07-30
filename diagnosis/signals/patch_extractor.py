"""Patch extractor — extract structured signals from a git diff."""
from __future__ import annotations

import re
from pathlib import PurePosixPath

from .schema import PatchSignals


# ── Patterns ────────────────────────────────────────────────────────

_RE_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
_RE_ADDED_LINE = re.compile(r"^\+[^+]")
_RE_DELETED_LINE = re.compile(r"^-[^-]")

_CONFIG_DIRS = frozenset({".circleci", ".github", ".gitlab", "ci", "build"})
_CONFIG_FILES = frozenset({
    "pyproject.toml", "setup.cfg", "setup.py", "tox.ini", "makefile",
    "dockerfile", "requirements.txt", "requirements-dev.txt",
    "environment.yml", "conda.recipe", ".gitignore",
})


def extract_patch_signals(patch_text: str) -> PatchSignals:
    """Extract structured signals from a git diff patch text."""
    signals = PatchSignals()
    if not patch_text or not patch_text.strip():
        return signals

    lines = patch_text.split("\n")
    signals.patch_size_chars = len(patch_text)
    signals.patch_size_lines = len(lines)
    signals.diff_total_lines = len(lines)

    # Extract edited files (shlex-based, handles quoted filenames)
    from integrity import extract_changed_files as _extract_files
    signals.edited_files = _extract_files(patch_text)

    # Hunk-state tracker (works for both quoted and unquoted diff --git)
    in_hunk = False
    source_files: list[str] = []
    test_files: list[str] = []
    config_files: list[str] = []

    for line in lines:
        # Track diff state by line prefix pattern
        if line.startswith("diff --git ") or line.startswith("--- "):
            in_hunk = False
        elif _RE_HUNK_HEADER.match(line):
            in_hunk = True
            signals.hunk_count += 1
        elif in_hunk and line.startswith("+"):
            signals.added_lines += 1
            signals.changed_lines += 1
        elif in_hunk and line.startswith("-"):
            signals.deleted_lines += 1
            signals.changed_lines += 1

    # Classify edited files
    for f in signals.edited_files:
        pp = PurePosixPath(f)
        name = pp.name.lower()
        parts = [p.lower() for p in pp.parts]
        is_test = (
            any(p in {"test", "tests", "testing"} for p in parts[:-1])
            or name.startswith("test_")
            or name.endswith("_test.py")
        )
        is_config = (
            name in _CONFIG_FILES
            or any(p in _CONFIG_DIRS for p in parts)
        )
        if is_test:
            test_files.append(f)
        elif is_config:
            config_files.append(f)
        else:
            source_files.append(f)

    signals.introduced_config_change = bool(config_files)

    return signals
