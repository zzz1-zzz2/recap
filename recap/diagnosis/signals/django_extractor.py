"""Django test_log extractor — parse SWE-bench evaluation stdout for Django instances.

Format differences from pytest:
  ┌─────────────────────┬──────────────────────────┬────────────────────────────────┐
  │ Aspect              │ pytest                   │ Django                         │
  ├─────────────────────┼──────────────────────────┼────────────────────────────────┤
  │ PASS marker         │ PASSED test::name        │ test_name (Class) ... ok       │
  │ FAIL marker         │ FAILED test::name        │ test_name (Class) ... ERROR    │
  │                     │                          │ test_name (Class) ... FAIL     │
  │ Failure header      │ ____ FAILURES ____       │ ====== ERROR: test_name (Cls)  │
  │ Summary start       │ short test summary info  │ FAILED (errors=N)              │
  │ Frame path          │ astropy/x.py:42          │ /testbed/django/x.py           │
  │ Needs /testbed/ cut │ No                       │ Yes                            │
  │ Test total          │ N passed, M failed       │ Ran N tests in Xs              │
  └─────────────────────┴──────────────────────────┴────────────────────────────────┘

Detection: Use has_django_format() before calling extract_test_log().
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .enums import TestFramework
from .schema import (
    StackFrame,
    TestLogSignals,
)
from .frame_normalizer import normalize_frame

logger = logging.getLogger("condiag.diagnosis.signals.django_extractor")

# ── Patterns ────────────────────────────────────────────────────────

# Test result line: test_name (Class) ... ok/ERROR/FAIL
_RE_TEST_RESULT = re.compile(
    r"^(\w[\w_]+)\s+\(([^)]+)\)\s+\.\.\.\s+(ok|ERROR|FAIL)"
)

# Django error/failure header:
#   ======================================================================
#   ERROR: test_name (aggregation.tests.AggregateTestCase)
_RE_ERROR_HEADER = re.compile(r"^(ERROR|FAIL):\s+(\w[\w_]+)\s+\(([^)]+)\)")

# Frame: File "/testbed/django/db/backends/utils.py", line 84, in _execute
_RE_FILE_FRAME = re.compile(
    r'File\s+"([^"]+)",\s*line\s+(\d+)(?:,\s*in\s+(\w+))?'
)

# Summary: Ran 56 tests in 0.075s
_RE_RAN_TESTS = re.compile(r"Ran\s+(\d+)\s+tests?\s+in\s+[\d.]+\w?")

# Summary: FAILED (errors=1) or FAILED (failures=2) or FAILED (errors=1, failures=0)
_RE_FAILED_SUMMARY = re.compile(r"FAILED\s+\(([^)]+)\)")

# Python exception line
_RE_EXCEPTION = re.compile(
    r"(AssertionError|AttributeError|TypeError|ValueError|ImportError|"
    r"ModuleNotFoundError|KeyError|IndexError|RuntimeError|OSError|"
    r"OperationalError|StopIteration)[^:\n]*:\s*([^\n]+)"
)


def has_django_format(test_log_path: str | Path) -> bool:
    """Check if a test_log uses Django test runner format."""
    raw = Path(test_log_path).read_text(encoding="utf-8", errors="replace")
    # Detection keywords
    if "runtests.py" in raw or "Creating test database for alias" in raw:
        return True
    # Check for Django-style test results
    django_pattern = re.compile(r"\.\.\. (ok|ERROR|FAIL)")
    matches = django_pattern.findall(raw)
    return len(matches) >= 3  # At least 3 results for confidence


def extract_test_log(test_log_path: str | Path) -> TestLogSignals:
    """Extract all signals from a Django-format test_log file.

    Args:
        test_log_path: Path to test_output.txt from SWE-bench evaluation.

    Returns:
        TestLogSignals with all extractable fields populated.
    """
    raw = Path(test_log_path).read_text(encoding="utf-8", errors="replace")
    lines = raw.split("\n")

    signals = TestLogSignals(framework=TestFramework.DJANGO)

    _extract_test_results(signals, lines)
    _extract_frames_and_errors(signals, raw, lines)

    return signals


# ── Internal: test results ──────────────────────────────────────────


def _extract_test_results(signals: TestLogSignals, lines: list[str]) -> None:
    """Extract test results from Django output lines.

    Django format:
      test_count_distinct_expression (aggregation.tests.AggregateTestCase) ... ERROR
      test_count_star (aggregation.tests.AggregateTestCase) ... ok
    """
    for line in lines:
        stripped = line.strip()
        m = _RE_TEST_RESULT.match(stripped)
        if not m:
            continue

        test_name = m.group(1)
        test_class = m.group(2)
        status = m.group(3)
        full_name = f"{test_class}::{test_name}"

        if status == "ok":
            signals.passed_tests.append(full_name)
        else:
            signals.failed_tests.append(full_name)

    # Parse total test count from summary
    for line in reversed(lines):
        m = _RE_RAN_TESTS.search(line)
        if m:
            signals.num_tests_run = int(m.group(1))
            break


# ── Internal: frames and errors ─────────────────────────────────────


def _extract_frames_and_errors(
    signals: TestLogSignals, raw: str, lines: list[str]
) -> None:
    """Extract stack frames and error details from Django output.

    Django uses File "/testbed/django/x.py", line 42 format (same as File "..." format
    in pytest, but these ARE real test frames, not build frames).

    Also captures the ERROR/FAIL sections.
    """
    in_error_section = False
    current_error_name = ""
    error_message_lines: list[str] = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Check for error section header
        m = _RE_ERROR_HEADER.match(stripped)
        if m:
            in_error_section = True
            current_error_name = f"{m.group(2)} ({m.group(3)})"
            error_message_lines = []
            continue

        # End of error section
        if in_error_section and "---" in stripped and len(stripped) >= 60:
            in_error_section = False
            # Save collected error message
            if error_message_lines:
                signals.error_messages.extend(error_message_lines)
                if not signals.first_error_message and error_message_lines:
                    signals.first_error_message = error_message_lines[0]
            continue

        # Extract frames from the error section (and anywhere in the file)
        if in_error_section:
            # Error line (text after traceback)
            em = _RE_EXCEPTION.match(stripped)
            if em:
                signals.error_types[em.group(1)] = (
                    signals.error_types.get(em.group(1), 0) + 1
                )
                if not signals.first_error_message:
                    signals.first_error_message = f"{em.group(1)}: {em.group(2)[:200]}"
                error_message_lines.append(f"{em.group(1)}: {em.group(2)[:200]}")
                continue

            # Frame lines
            fm = _RE_FILE_FRAME.search(stripped)
            if fm:
                fpath = fm.group(1)
                if fpath.startswith("/testbed/"):
                    fpath = fpath[len("/testbed/"):]
                signals.stack_frames.append(
                    normalize_frame(fpath, int(fm.group(2)), fm.group(3) or "")
                )
                continue

        # Also capture frames outside the error section (in case section parsing fails)
        if not in_error_section and "Traceback" not in line:
            fm = _RE_FILE_FRAME.search(stripped)
            if fm:
                fpath = fm.group(1)
                if fpath.startswith("/testbed/"):
                    fpath = fpath[len("/testbed/"):]
                # Deduplicate with existing frames
                existing = {(f.file, f.line, f.function) for f in signals.stack_frames}
                key = (fpath, int(fm.group(2)), fm.group(3) or "")
                if key not in existing and not _is_system_path(fpath):
                    signals.stack_frames.append(
                        normalize_frame(fpath, int(fm.group(2)), fm.group(3) or "")
                    )

    # Fallback: if no error section was detected, extract error from full text
    if not signals.first_error_message and not signals.error_messages:
        for line in lines:
            em = _RE_EXCEPTION.search(line)
            if em:
                signals.error_types[em.group(1)] = (
                    signals.error_types.get(em.group(1), 0) + 1
                )
                signals.error_messages.append(f"{em.group(1)}: {em.group(2)[:200]}")
                if not signals.first_error_message:
                    signals.first_error_message = f"{em.group(1)}: {em.group(2)[:200]}"

    # Parse the FAILED summary for error/failure types
    for line in reversed(lines):
        m = _RE_FAILED_SUMMARY.search(line)
        if m:
            # e.g. "errors=1" or "errors=1, failures=0"
            parts = m.group(1).split(",")
            for part in parts:
                part = part.strip()
                if "=" in part:
                    key, val = part.split("=")
                    key = key.strip()
                    val = int(val.strip())
                    # Add as diagnostic info
                    signals._extra = getattr(signals, "_extra", {})
                    signals._extra[f"django_{key}"] = val


# ── Helper ──────────────────────────────────────────────────────────


def _is_system_path(path: str) -> bool:
    """Check if path is a system/site-packages path (not repo code)."""
    return any(
        prefix in path
        for prefix in [
            "/opt/",
            "/usr/",
            "/lib/",
            "site-packages/",
            "dist-packages/",
        ]
    )
