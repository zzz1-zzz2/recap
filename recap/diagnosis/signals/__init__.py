"""Signal extraction — unified entry point.

Usage:
    from .signals import extract_test_log

    signals = extract_test_log("path/to/test_output.txt")
    # signals.framework == "pytest" | "django"
"""
from __future__ import annotations

from pathlib import Path


def extract_test_log(test_log_path: str | Path):
    """Extract structured signals from a SWE-bench evaluation test_log.

    Auto-detects test framework (pytest vs Django) and dispatches to
    the appropriate extractor.

    Args:
        test_log_path: Path to test_output.txt from SWE-bench evaluation.

    Returns:
        TestLogSignals with framework-specific fields populated.
    """
    from .signals import django_extractor

    if django_extractor.has_django_format(test_log_path):
        return django_extractor.extract_test_log(test_log_path)

    from .signals import pytest_extractor
    return pytest_extractor.extract_test_log(test_log_path)
