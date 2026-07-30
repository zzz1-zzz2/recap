"""Signal extraction enums — classified from SWE-bench_Verified test_log analysis.

These enums define the vocabulary used across all signal extraction modules.
They reflect the actual data we can extract, not hypothetical signals.

Design principle (from data-first-methodology):
  Every enum variant must correspond to a concrete pattern found in real test logs.
  No variants added "just in case."
"""
from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    """Python exception types that appear in SWE-bench test failures.

    Coverage: covers all error types observed across 313 SWE-bench_Verified instances.
    """

    ASSERTION_ERROR = "AssertionError"
    ATTRIBUTE_ERROR = "AttributeError"
    TYPE_ERROR = "TypeError"
    VALUE_ERROR = "ValueError"
    IMPORT_ERROR = "ImportError"
    MODULE_NOT_FOUND = "ModuleNotFoundError"
    KEY_ERROR = "KeyError"
    INDEX_ERROR = "IndexError"
    RUNTIME_ERROR = "RuntimeError"
    OSError = "OSError"
    OPERATIONAL_ERROR = "OperationalError"  # Django SQL errors
    STOP_ITERATION = "StopIteration"
    FLAKY = "FlakyFailure"  # non-deterministic test (same code, different result)
    UNKNOWN = "UnknownError"


class TestFramework(str, Enum):
    """Test framework used by the instance.

    Coverage: SWE-bench_Verified uses pytest (~270) and Django test runner (~231).
    Other frameworks (unittest directly, tox) follow one of these two output formats.
    """

    PYTEST = "pytest"
    DJANGO = "django"
    UNKNOWN = "unknown"


class IterationSignal(str, Enum):
    """High-level iteration behavior signals.

    NOTE: These are NOT diagnosis outputs. They are raw behavioral indicators
    that the Diagnoser consumes as input features.
    """

    NORMAL_EXPLORING = "normal_exploring"
    SAME_FILE_LOOP = "same_file_loop"          # editing same file repeatedly
    DIFF_OUTPUT_STAGNANT = "diff_output_stagnant"  # edits produce no test progress
    EXPLORATION_DRIFT = "exploration_drift"     # jumping between unrelated files
