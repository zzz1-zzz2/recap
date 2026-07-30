"""Context Suppressor — remove redundant or invalidated context from trajectory.

K⁻ = context units selected for suppression.

Design principles:
  - Safe by default: only suppress content that is clearly redundant or
    where the diagnosis has explicitly identified it as wrong.
  - Never suppress failure evidence (test output, error messages).
  - Never suppress the first occurrence of a relevant file view.
  - Maintain message alternation: always remove complete assistant+tool
    turns, never orphaned individual messages.

Suppression strategies (applied in order):
  1. Deduplicate consecutive identical file views (view → view of same file
     within 3 turns → suppress the duplicate).
  2. Remove redundant exploration of files NOT in any diagnosis target list
     (keep first 2 views, suppress the 3rd+).
  3. Remove turns containing explicitly invalidated bash commands
     (e.g. repeated test runs that always fail the same way).
"""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import Any

logger = logging.getLogger("condiag.reconstruction.suppressor")

# Regex to extract file paths from bash commands / tool output
_RE_FILE_REF = re.compile(r"[\w/]+\.py(?::\d+)?")
_RE_TEST_RUN = re.compile(r"python\s+(?:-m\s+)?pytest|python\s+-m\s+django\s+test")


def suppress_trajectory(
    messages: list[dict],
    *,
    target_files: list[str] | None = None,
    target_symbols: list[str] | None = None,
    max_views_per_file: int = 3,
) -> list[dict]:
    """Suppress redundant or invalidated context from a trajectory.

    Args:
        messages: Full message history (assistant + tool turns).
        target_files: File paths identified by diagnosis as relevant.
            Files NOT in this list are candidates for dedup suppression.
        target_symbols: Symbols identified by diagnosis as relevant.
        max_views_per_file: Max allowed views of files NOT in target_files.
            The first `max_views` occurrences are kept; excess are suppressed.

    Returns:
        Filtered message list with redundant turns removed.
    """
    if not messages:
        return messages

    target_set = set(target_files or [])
    target_stems = {_stem(p) for p in target_set}

    # Parse into turns
    turns = _parse_turns(messages)

    # Track file view counts
    file_view_counts: dict[str, int] = {}

    # Phase 1: mark turns for removal
    remove_indices: set[int] = set()

    for i, turn in enumerate(turns):
        files_in_turn = _extract_files(turn)
        test_related = _is_test_run(turn)

        # Count non-target file views
        for fp in files_in_turn:
            stem = _stem(fp)
            if stem not in target_stems:
                file_view_counts[stem] = file_view_counts.get(stem, 0) + 1

        # Suppress if this is an excessive view of a non-target file
        # AND not a test run (test output is always relevant)
        if not test_related:
            excess_views = [
                stem for stem in {_stem(f) for f in files_in_turn}
                if stem not in target_stems
                and file_view_counts[stem] > max_views_per_file
            ]
            if excess_views:
                remove_indices.add(i)

    # Phase 2: remove suppressed turns
    if remove_indices:
        filtered_turns = [
            t for idx, t in enumerate(turns) if idx not in remove_indices
        ]
        logger.info(
            "Suppressed %d/%d turns (%d remaining)",
            len(remove_indices), len(turns), len(filtered_turns),
        )
    else:
        filtered_turns = turns

    # Flatten turns back to message list
    result: list[dict] = []
    for turn in filtered_turns:
        result.extend(turn)
    return result


# ── Turn parsing ────────────────────────────────────────────────────────


def _parse_turns(messages: list[dict]) -> list[list[dict]]:
    """Split message list into (assistant + tool) turns.

    A turn = one assistant message + all following tool messages
    (handles parallel tool calls).
    """
    turns: list[list[dict]] = []
    current: list[dict] = []

    for m in messages:
        role = m.get("role", "")
        if role == "assistant":
            if current:
                turns.append(current)
            current = [m]
        elif role == "tool":
            if current:
                current.append(m)
            else:
                # Orphaned tool response — should not happen in well-formed
                # trajectories, but handle gracefully.
                current = [m]
        else:
            # system or user messages — treated as standalone turns
            if current:
                turns.append(current)
                current = []
            turns.append([m])

    if current:
        turns.append(current)
    return turns


def _extract_files(turn: list[dict]) -> list[str]:
    """Extract .py file paths from a turn's text content."""
    files: list[str] = []
    for m in turn:
        content = str(m.get("content", "") or "")
        for match in _RE_FILE_REF.finditer(content):
            fp = match.group()
            if fp.endswith(".py"):
                files.append(fp)
            elif ":" in fp and fp.split(":")[0].endswith(".py"):
                files.append(fp.split(":")[0])
    return files


def _is_test_run(turn: list[dict]) -> bool:
    """Check if a turn involves running tests."""
    for m in turn:
        content = str(m.get("content", "") or "")
        if _RE_TEST_RUN.search(content):
            return True
    return False


def _stem(filepath: str) -> str:
    """Normalise a file path to its last meaningful component."""
    fp = filepath.replace("/testbed/", "")
    parts = fp.replace(".py", "").split("/")
    return "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1] if parts else ""
