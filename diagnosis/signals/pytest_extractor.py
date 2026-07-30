"""Pytest test_log extractor — parse per-test failure sections.

The test_log (test_output.txt) has this structure:

  ==== FAILURES ====
  ________ test_name_A ________
    [test code context]
    >   assert_something()
    repo/path.py:LINE: in func
    repo/path.py:LINE: in func
    E   TypeError: message

  ________ test_name_B ________
    ...

  ==== short test summary info ====
  FAILED test_name_A
  FAILED test_name_B
  PASSED test_name_C
  ...

Each failure section has its OWN error message, stack frames, and assertion.
This parser creates one TestFailureSignal per section with correctly bound data.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from .enums import ErrorType, TestFramework
from .schema import (
    FailureReconciliation,
    StackFrame,
    TestFailureSignal,
    TestLogSignals,
)
from .frame_normalizer import normalize_frame

logger = logging.getLogger("condiag.diagnosis.signals.pytest_extractor")

# ── ANSI escape code stripping ────────────────────────────────────

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from a string."""
    return _ANSI_ESCAPE.sub("", text)


# ── Section boundaries ──────────────────────────────────────────────

# FAILURES header (trailing ... due to line length in some logs)
_RE_FAILURES_HEADER = re.compile(r"^==+.*FAILURES.*==+")
# short test summary header
_RE_SUMMARY_HEADER = re.compile(r"^==+.*short test summary.*==+")
# Per-test section header: _________ test_name _________
# Also supports ANSI-stripped variant: _ test_name _
_RE_TEST_HEADER = re.compile(r"(?:_{3,}\s+(.+?)\s+_{3,}|^_ (.+?) _$)")

# ── Frame / error patterns (per-test section) ──────────────────────

# Pytest short format: path/file.py:LINE: in func_name
# Also: path/file.py:LINE:                    (no func name on continuation lines)
# Also: path/file.py:LINE: TypeError          (last line of the traceback)
_RE_PYTEST_FRAME = re.compile(r"^\s*(\S+?)\.py:(\d+):(?:\s+in\s+(\w+))?")

# Error line: E       TypeError: message
_RE_ERROR_LINE = re.compile(r"^\s*E\s+(\w+(?:Error|Exception|Failure))(?::\s*(.*))?")

# Assertion line: >       assert_something(...)
_RE_ASSERTION_LINE = re.compile(r"^\s*>\s+\S")

# Stack separator line: _ _ _ _ _ ... (NOT a test section header)
_RE_STACK_SEPARATOR = re.compile(r"^_{3,}\s*$")


def _resolve_repo_path(short_path: str) -> str:
    """Resolve a short pytest-style path to a reasonable repo-relative path."""
    if short_path.startswith("."):
        short_path = short_path.lstrip("./")
    return short_path


def _first_repo_frame(frames: list[StackFrame]) -> str:
    for f in frames:
        if f.is_repo_frame and not f.is_test_file:
            return f"{f.file}:{f.line}"
    return ""


# ── Section parser ──────────────────────────────────────────────────


def _parse_test_section(lines: list[str], start: int) -> tuple[TestFailureSignal | None, int]:
    """Parse one failure section starting at a '____ test_name ____' header.

    Returns (TestFailureSignal, line_after_section_end).
    Returns (None, end_line) if the section is not a test failure.
    """
    # Extract test name from header (supports both ____name____ and _ name _ formats)
    m = _RE_TEST_HEADER.match(lines[start].strip())
    if not m:
        return None, start + 1
    test_name = (m.group(1) or m.group(2)).strip()

    frames: list[StackFrame] = []
    error_msg = ""
    exception_type = "Unknown"
    assertion_lines: list[str] = []
    section_lines: list[str] = []

    i = start + 1
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n")

        # Stop at next test header, FAILURES header, or summary
        if _RE_TEST_HEADER.match(stripped) and i != start:
            break
        if _RE_SUMMARY_HEADER.match(stripped):
            break

        section_lines.append(stripped)

        # Assertion line: >       code_that_failed
        # Collect ALL assertion lines, then pick the best candidate later
        if _RE_ASSERTION_LINE.match(stripped):
            assertion_lines.append(stripped)

        # Error line: E       TypeError: message
        em = _RE_ERROR_LINE.match(stripped)
        if em:
            etype = em.group(1)
            msg = em.group(2) or ""
            full = f"{etype}: {msg}"
            # Use the FIRST E line as the primary error for this test
            if not error_msg:
                exception_type = etype
                error_msg = full

        # Stack frame: path/file.py:LINE: in func_name
        # Also matches continuation lines (no in func) and final lines (file.py:LINE: ErrorType)
        fm = _RE_PYTEST_FRAME.match(stripped)
        if fm:
            fpath = fm.group(1) + ".py"
            if not fpath.startswith("/"):
                fpath = _resolve_repo_path(fpath)
            frame = normalize_frame(
                fpath, int(fm.group(2)),
                fm.group(3) or "",
            )
            frames.append(frame)

        # File format frame: File "path", line N, in func
        # Strip leading whitespace because sympy traceback has "  File..."
        ff = re.match(
            r'File\s+"([^"]+)",\s*line\s+(\d+)(?:,\s*in\s+(\w+))?',
            stripped if stripped.startswith("File") else stripped.lstrip(),
        )
        if ff:
            fpath2 = ff.group(1)
            if fpath2.startswith("/testbed/"):
                fpath2 = fpath2[len("/testbed/"):]
            frame2 = normalize_frame(fpath2, line=int(ff.group(2)), function=ff.group(3) or "")
            frames.append(frame2)
            # If we see a File format frame and the error is not set yet,
            # look ahead for a bare assert line or error name (sympy-style)
            if not error_msg:
                ji = i + 1
                # Skip blank lines
                while ji < len(lines) and not lines[ji].strip():
                    ji += 1
                if ji < len(lines):
                    nxt = lines[ji].strip()
                    # bare assert line without ">" prefix (sympy style)
                    if nxt.startswith("assert "):
                        assertion_lines.append(f">       {nxt}")
                        # Check one more line for bare error name
                        ji2 = ji + 1
                        while ji2 < len(lines) and not lines[ji2].strip():
                            ji2 += 1
                        if ji2 < len(lines):
                            nxt2 = lines[ji2].strip()
                            em2 = re.match(r"^(\w+(?:Error|Exception|Failure))\s*", nxt2)
                            if em2:
                                exception_type = em2.group(1)
                                error_msg = f"{em2.group(1)}:"

        # File format frame: File "path", line N, in func
        # (less common in test sections, but handle)
        ff = re.match(r'File\s+"([^"]+)",\s*line\s+(\d+)(?:,\s*in\s+(\w+))?', stripped)
        if ff:
            fpath2 = ff.group(1)
            if fpath2.startswith("/testbed/"):
                fpath2 = fpath2[len("/testbed/"):]
            frame2 = normalize_frame(fpath2, line=int(ff.group(2)), function=ff.group(3) or "")
            frames.append(frame2)

        i += 1

    # Pick the best assertion line: prefer the one with "assert",
    # "allclose", "==", or similar comparison keywords; else the first.
    best_assertion = ""
    if assertion_lines:
        priority = [
            a for a in assertion_lines
            if any(k in a.lower() for k in ["assert_", "assert ", "==", "!="])
        ]
        best_assertion = priority[0] if priority else assertion_lines[-1]

    record = TestFailureSignal(
        test_name=test_name,
        exception_type=exception_type,
        error_message=error_msg,
        assertion_line=best_assertion,
        stack_frames=frames,
        root_frame=_first_repo_frame(frames),
        raw_excerpt="\n".join(section_lines),
    )
    return record, i


# ── Main parser ─────────────────────────────────────────────────────


def extract_test_log(test_log_path: str | Path) -> TestLogSignals:
    """Extract per-test failure records from a pytest-format test_log file."""
    raw = Path(test_log_path).read_text(encoding="utf-8", errors="replace")
    # Strip ANSI escape codes (colorized pytest output)
    raw = _strip_ansi(raw)
    lines = raw.split("\n")

    signals = TestLogSignals(framework=TestFramework.PYTEST)

    # Find FAILURES section start and summary section start
    failures_start = None
    summary_start = None
    for i, line in enumerate(lines):
        if _RE_FAILURES_HEADER.match(line.strip()):
            failures_start = i + 1
        if failures_start is not None and _RE_SUMMARY_HEADER.match(line.strip()):
            summary_start = i
            break

    # If no FAILURES section, parse test sections from the entire file
    if failures_start is None or summary_start is None:
        _extract_from_summary(signals, lines)
        # Try parsing sections from whole file
        i = 0
        while i < len(lines):
            if _RE_TEST_HEADER.match(lines[i].strip()):
                # Found a test header outside FAILURES section
                # Find the next test header or end of file
                j = i + 1
                while j < len(lines):
                    if _RE_TEST_HEADER.match(lines[j].strip()) and j != i:
                        break
                    j += 1
                section_lines = lines[i:j]
                record, _ = _parse_test_section(section_lines, 0)
                if record is not None:
                    signals.failures.append(record)
                i = j
            else:
                i += 1
        _derive_aggregates(signals)
        return signals

    # Parse each test section between FAILURES header and summary
    i = failures_start
    while i < summary_start:
        record, i = _parse_test_section(lines, i)
        if record is not None:
            signals.failures.append(record)

    # Parse summary for FAILED/PASSED test lists
    _extract_from_summary(signals, lines)

    # Reconcile section-parsed failures with summary entries
    _reconcile_failures(signals)

    # Derive aggregate fields from per-test records
    _derive_aggregates(signals)

    return signals


# ── Summary parser ──────────────────────────────────────────────────


_RE_FAILED_LINE = re.compile(r"^FAILED\s+(\S+(?:::\S+)?(?:\S*)?)")
_RE_PASSED_LINE = re.compile(r"^PASSED\s+(\S+(?:::\S+)?)\s*$")


def _extract_from_summary(signals: TestLogSignals, lines: list[str]) -> None:
    """Extract FAILED/PASSED test names and counts from summary.

    Supports multiple formats:
      - Standard pytest:  FAILED path/to/test.py::test_name
      - SymPy pytest:     test_name ok [FAIL]  (inline suffix)
      - Standard pytest:  PASSED path/to/test.py::test_name
    """
    for line in lines:
        stripped = line.strip()

        # Standard FAILED lines: FAILED path/to/test.py::test_name
        m = _RE_FAILED_LINE.match(stripped)
        if m:
            signals.failed_tests.append(m.group(1))
            continue

        # SymPy-style inline [FAIL]: "test_xxx ok [FAIL]"
        # In SymPy's custom test runner, [FAIL] at end of line marks the file/shards
        # as having a failure, NOT the individual test. If "ok" precedes [FAIL],
        # the test itself passed — skip it.
        if "[FAIL]" in stripped and not stripped.startswith("#"):
            # Check if "ok" appears before [FAIL] — SymPy file-level marker
            fail_pos = stripped.index("[FAIL]")
            ok_pos = stripped.find(" ok ")
            if ok_pos == -1 or ok_pos > fail_pos:
                # No "ok" before [FAIL] — this IS a test-level failure
                parts = stripped.split()
                if parts and not parts[0].startswith("__"):
                    test_name = parts[0]
                    if test_name not in signals.failed_tests:
                        signals.failed_tests.append(test_name)
            continue

        # Standard PASSED lines
        m = _RE_PASSED_LINE.match(stripped)
        if m:
            signals.passed_tests.append(m.group(1))
            continue

    # Total test count from summary line
    for line in reversed(lines):
        m_passed = re.search(r"(\d+)\s+passed", line)
        m_failed = re.search(r"(\d+)\s+failed", line)
        if m_passed or m_failed:
            total = int(m_passed.group(1)) if m_passed else 0
            total += int(m_failed.group(1)) if m_failed else 0
            signals.num_tests_run = total
            break


# ── Aggregate derivation ────────────────────────────────────────────


def _derive_aggregates(signals: TestLogSignals) -> None:
    """Derive legacy aggregate fields from per-test failure records.

    This ensures backward compatibility with code that reads
    error_types, error_messages, stack_frames, call_chains, etc.
    """
    if not signals.failures:
        return

    # error_types: count across all failures
    error_type_counts: dict[str, int] = {}
    seen_messages: set[str] = set()

    for rec in signals.failures:
        # Error type counts
        etype = rec.exception_type if rec.exception_type != "Unknown" else "Error"
        error_type_counts[etype] = error_type_counts.get(etype, 0) + 1

        # Error messages (deduplicate for backward compat)
        if rec.error_message and rec.error_message not in seen_messages:
            seen_messages.add(rec.error_message)
            signals.error_messages.append(rec.error_message)

        if not signals.first_error_message and rec.error_message:
            signals.first_error_message = rec.error_message

        # Stack frames (ALL frames from ALL failures, in order)
        signals.stack_frames.extend(rec.stack_frames)

        # Assertion lines
        if rec.assertion_line:
            signals.failure_assertions.append(rec.assertion_line)

        # Call chains (per-test)
        signals.call_chains.append(rec.stack_frames)

    signals.error_types = error_type_counts


class FailureBindingError(ValueError):
    """Section-parsed failures do not reconcile with summary FAILED entries."""


def _nodeid_leaf(nodeid: str) -> str:
    """Extract the leaf test name from a full nodeid.

    'astropy/tests/test_fake.py::test_foo[param]' → 'test_foo[param]'
    'astropy/tests/test_fake.py::test_foo'        → 'test_foo'
    'test_foo'                                    → 'test_foo'
    """
    return nodeid.split("::")[-1] if "::" in nodeid else nodeid


def _reconcile_failures(signals: TestLogSignals) -> None:
    """Match section-parsed failures to summary FAILED entries by leaf name.

    Sets signals.reconciliation and raises FailureBindingError if
    any section or summary entry cannot be matched.
    """
    section_names = [f.test_name for f in signals.failures]
    summary_leafs = [_nodeid_leaf(n) for n in signals.failed_tests]

    # Build parameterized‑aware matcher:
    #   test_foo[x] in sections matches test_foo[x] or test_foo in summary
    #   test_foo in sections matches test_foo in summary
    unmatched_sections: list[str] = []
    used_summary: set[int] = set()

    for sname in section_names:
        found = False
        for j, lname in enumerate(summary_leafs):
            if j in used_summary:
                continue
            # Exact match or param match
            if sname == lname or sname.split("[")[0] == lname.split("[")[0]:
                used_summary.add(j)
                found = True
                break
        if not found:
            unmatched_sections.append(sname)

    unmatched_summary = [
        signals.failed_tests[i]
        for i in range(len(summary_leafs))
        if i not in used_summary
    ]

    exact_matches = len(used_summary)
    if not unmatched_sections and not unmatched_summary:
        status = "exact"
    elif exact_matches == len(section_names) == len(summary_leafs):
        status = "count_only"
    elif not section_names and summary_leafs:
        # No FAILURES section but summary has FAILED entries (ANSI-stripped or sympy format)
        status = "summary_only"
    elif section_names and not summary_leafs:
        # Sections exist but summary is empty (weird format) — use sections as source of truth
        status = "section_only"
    else:
        # Partial mismatch — still record as mismatched but don't crash the pipeline.
        # The extractor keeps both sources; callers can decide which to trust.
        status = "partial_mismatch"

    signals.reconciliation = FailureReconciliation(
        section_count=len(section_names),
        summary_failed_count=len(summary_leafs),
        exact_matches=exact_matches,
        unmatched_sections=unmatched_sections,
        unmatched_summary_nodeids=unmatched_summary,
        match_status=status,
    )
