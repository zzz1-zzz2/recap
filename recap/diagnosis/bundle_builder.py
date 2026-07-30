"""Build a RuntimeFailureFeatureBundle from available episode data.

Standard entry point for the Diagnoser's input.
Multi-source fusion: parsed test log + FailureWitness -> merged TestLogSignals.
"""
from __future__ import annotations

import logging
from typing import Any

from recap.diagnosis.signals.frame_normalizer import normalize_frame
from recap.diagnosis.signals.schema import (
    RuntimeFailureFeatureBundle,
    RuntimeInstanceSignals,
    TestLogSignals,
)

logger = logging.getLogger("condiag.diagnosis.bundle_builder")


def build_failure_feature_bundle(
    failure_witness: dict | None = None,
    evaluation_patch: str = "",
    workspace_patch: str = "",
    trajectory: dict | None = None,
    instance_spec: Any = None,
    test_log: TestLogSignals | None = None,
) -> RuntimeFailureFeatureBundle:
    """Construct a RuntimeFailureFeatureBundle from all available episode data.

    Multi-source fusion: parsed test_log (if available) is the primary source,
    FailureWitness fills in gaps and adds non-duplicate evidence.
    """
    bundle = RuntimeFailureFeatureBundle()

    # Test log signals: fusion of parsed + FW
    _merge_test_log_signals(bundle.test_log, test_log, failure_witness)

    # Patch signals
    patch_text = evaluation_patch or workspace_patch or ""
    if patch_text.strip():
        from recap.diagnosis.signals.patch_extractor import extract_patch_signals
        bundle.patch = extract_patch_signals(patch_text)

    # Trajectory signals
    if trajectory:
        from recap.diagnosis.signals.trajectory_extractor import extract_trajectory_signals
        bundle.trajectory = extract_trajectory_signals(trajectory)

    # Instance signals (runtime-safe)
    if instance_spec:
        _populate_instance_signals(bundle.instance, instance_spec)

    return bundle


def _merge_test_log_signals(
    target: TestLogSignals,
    parsed: TestLogSignals | None,
    fw: dict | None,
) -> None:
    """Merge parsed test log signals with FailureWitness data.

    Rules:
      - Parsed is primary source; its stack_frames/build_frames are kept.
      - FW fills in gaps (missing failed_tests, error messages).
      - Stack frames deduplicated using NORMALIZED repo-relative path key
        (both /testbed/pkg/x.py and pkg/x.py -> key = "pkg/x.py").
      - Error types only counted for NEW error evidence (not already in parsed).
      - Failed tests: stable union (order preserved, duplicates removed).
    """
    if parsed is not None:
        target.framework = parsed.framework
        target.failed_tests = list(parsed.failed_tests)
        target.passed_tests = list(parsed.passed_tests)
        target.num_tests_run = parsed.num_tests_run
        target.error_types = dict(parsed.error_types)
        target.error_messages = list(parsed.error_messages)
        target.first_error_message = parsed.first_error_message
        target.failure_assertions = list(parsed.failure_assertions)
        target.call_chains = list(parsed.call_chains)
        # Critical: copy parsed stack frames and build frames
        target.stack_frames = list(parsed.stack_frames)
        target.build_frames = list(parsed.build_frames)

    if fw is None:
        return

    fw_failed = fw.get("failed_tests") or []
    fw_error = fw.get("error_message") or ""
    fw_frames = fw.get("stack_frames") or []

    # Failed tests: stable union (preserve order, remove duplicates)
    seen_failed = set(str(t) for t in target.failed_tests)
    for t in fw_failed:
        s = str(t)
        if s not in seen_failed:
            seen_failed.add(s)
            target.failed_tests.append(s)

    # Error messages: only count types for genuinely NEW evidence
    if fw_error:
        if not target.first_error_message:
            target.first_error_message = fw_error
        if fw_error not in target.error_messages:
            target.error_messages.append(fw_error)
            for etype_name in ("TypeError", "AssertionError", "AttributeError", "ValueError", "ImportError"):
                if etype_name in fw_error:
                    target.error_types[etype_name] = target.error_types.get(etype_name, 0) + 1

    # Stack frames: dedup using repo-relative path (normalized)
    existing = set()
    for f in target.stack_frames:
        path = f.file
        if "/testbed/" in path:
            path = path.split("/testbed/", 1)[1]
        existing.add((path, f.line, f.function))

    for raw_frame in fw_frames:
        if not isinstance(raw_frame, dict):
            continue
        path = raw_frame.get("file", "") or ""
        func = raw_frame.get("function", raw_frame.get("func", "")) or ""
        line = raw_frame.get("line", 0)
        nf = normalize_frame(path, line, func)
        key = (nf.file, nf.line, nf.function)
        if key not in existing:
            existing.add(key)
            target.stack_frames.append(nf)


def _populate_instance_signals(
    signals: RuntimeInstanceSignals,
    instance_spec,
) -> None:
    """Extract runtime-safe instance fields (supports both dict and object)."""
    def _get(key, default=""):
        if isinstance(instance_spec, dict):
            return instance_spec.get(key, default)
        return getattr(instance_spec, key, default)
    signals.instance_id = _get("instance_id")
    signals.repo = _get("repo")
    signals.base_commit = _get("base_commit")
    signals.version = _get("version")
