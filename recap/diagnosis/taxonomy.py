"""ConDiag Taxonomy — the vocabulary of context deficiencies.

This is the core classification ontology of the paper.
Each ContextDeficiencyType represents a distinct category of missing context
that the agent can suffer from.

Design principle (from data-first-methodology):
  Each type must be (a) diagnosable from observable signals,
  and (b) actionable via a different retrieval strategy.

If two types are diagnosed the same way AND retrieved the same way,
they are the same type and should be merged.
"""
from __future__ import annotations

from enum import Enum


class ContextDeficiencyType(str, Enum):
    """What kind of context is the agent missing?

    These are the OUTPUT of the Diagnoser and the INPUT of the Router.
    """

    # ── Core types (observed in SWE-bench_Verified) ──

    NO_RELIABLE_DEFICIENCY = "NO_RELIABLE_DEFICIENCY"
    """No supported context-deficiency pattern matched. Diagnoser cannot
    determine what context is missing. Fall back to feedback-only revision.
    """

    API_DEFINITION = "API_DEFINITION"
    """Class/function/attribute doesn't exist on the object, or signature mismatch.

    Signal pattern: AttributeError, TypeError ("unexpected keyword").
    Retrieval: Look up symbol definition, parent class, or interface contract.
    """

    INTERFACE_CONSTRAINT = "INTERFACE_CONSTRAINT"
    """The type contract of a function is broken — wrong type passed.

    Signal pattern: TypeError ("unsupported operand type"), value type mismatch.
    Retrieval: Read function signature, type hints, or docstring contract.
    """

    RELATED_TESTS = "RELATED_TESTS"
    """Agent missed test logic, regression tests, or edge case behavior.

    Signal pattern: AssertionError, or failed tests with similar names to passed ones.
    Retrieval: Find adjacent test files or test coverage for edited module.
    """

    CALLER_CALLEE = "CALLER_CALLEE"
    """Wrong call site — caller passes different args than callee expects.

    Signal pattern: TypeError in transform chain where args are not forwarded correctly.
    Retrieval: Trace call chain from error stack, read caller and callee simultaneously.
    """

    DEPENDENCY = "DEPENDENCY"
    """Missing import, module, or external dependency.

    Signal pattern: ImportError, ModuleNotFoundError.
    Retrieval: Find the missing module in repo or check dependency graph.
    """

    REGISTRATION_SITE = "REGISTRATION_SITE"
    """Missing registration, routing, or export point.

    Signal pattern: Test passes with direct call but fails via framework integration.
    Retrieval: Search for registration patterns (urls, __init__.py imports, decorators).
    """

    # ── Paper-only types (for ablation / oracle) ──

    LOCALIZATION_DIRECTION = "LOCALIZATION_DIRECTION"
    """Agent is editing the wrong file — symptom not root cause.

    Signal pattern: R1 edits file A, but stack trace points to file B.
    Not a retrieval target, but a RE-DIRECTION signal for the agent.
    """

    ORACLE = "ORACLE"
    """Gold context from ContextBench — used for Oracle baseline only.
    NEVER used at inference time.
    """


class ConfidenceLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
