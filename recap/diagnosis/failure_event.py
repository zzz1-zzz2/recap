"""P1-3A: FailureEvent — per-test normalized failure + deterministic clustering.

Each Failed test from the harness produces one FailureEvent.
Events are grouped into FailureClusters via deterministic rules:

  1. exact error_type match
  2. message fingerprint (normalized text)
  3. shared top-of-stack frame
  4. call-chain overlap ratio
  5. parameterized test family recognition

Output feeds into EvidenceAlignment (P1-3B).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from recap.diagnosis.signals.schema import (
    RuntimeFailureFeatureBundle,
    StackFrame,
)

# ── Regex patterns for normalizing messages ─────────────────────────

# "Coordinate frame ITRS got unexpected keywords: ['location']"
#   → "Coordinate frame X got unexpected keywords: ['<KW>']"
_RE_UNEXPECTED_KW = re.compile(
    r"(unexpected keywords?: \[)[^\]]+(\])"
)

# "unsupported operand type(s) for -: 'Time' and 'float'"
#   → "unsupported operand type(s) for -: '<A>' and '<B>'"
_RE_OPERAND_TYPES = re.compile(
    r"(unsupported operand type\(s\) for[^:]+:)\s*'[^']*'\s*and\s*'[^']*'"
)

# "cannot import name 'X' from 'Y'"
#   → "cannot import name '<NAME>' from '<PACKAGE>'"
_RE_IMPORT_NAME = re.compile(
    r"(cannot import name\s+)'([^']+)'\s+from\s+'([^']+)'"
)

# "module 'X' has no attribute 'Y'"
#   → "module '<MOD>' has no attribute '<ATTR>'"
_RE_MODULE_ATTR = re.compile(
    r"(module\s+)'([^']+)'\s+has no attribute\s+'([^']+)'"
)

# number literals → <N>
_RE_NUMBER = re.compile(r"\b\d+\b")

# file paths → <path>
_RE_FILEPATH = re.compile(r"/[\w/.-]+(?:\.\w+)?")

# hex values → <hex>
_RE_HEX = re.compile(r"0x[0-9a-fA-F]+")

# line numbers in "file.py:123" → ":N"
_RE_LINENO = re.compile(r"(:)(\d+)\b")


def normalize_message(msg: str) -> str:
    """Collapse variable parts of error messages into placeholders."""
    msg = _RE_UNEXPECTED_KW.sub(r"\1'<KW>'\2", msg)
    msg = _RE_OPERAND_TYPES.sub(r"\1 '<A>' and '<B>'", msg)
    msg = _RE_IMPORT_NAME.sub(r"\1'<NAME>' from '<PACKAGE>'", msg)
    msg = _RE_MODULE_ATTR.sub(r"\1'<MOD>' has no attribute '<ATTR>'", msg)
    msg = _RE_NUMBER.sub("<N>", msg)
    msg = _RE_FILEPATH.sub("<path>", msg)
    msg = _RE_HEX.sub("<hex>", msg)
    msg = _RE_LINENO.sub(r"\1<N>", msg)
    return msg.strip()


# ── Error class taxonomy ────────────────────────────────────────────

# Coarse error categories used for first-pass grouping.
# Each maps one-or-more Python exception types to a cluster key.
ERROR_CLASS: dict[str, str] = {
    "TypeError": "TYPE_ERROR",
    "AssertionError": "ASSERTION_ERROR",
    "AttributeError": "ATTRIBUTE_ERROR",
    "ImportError": "IMPORT_ERROR",
    "ModuleNotFoundError": "IMPORT_ERROR",
    "ValueError": "VALUE_ERROR",
    "KeyError": "KEY_ERROR",
    "IndexError": "INDEX_ERROR",
    "NameError": "NAME_ERROR",
    "TimeoutError": "TIMEOUT",
    "OSError": "OS_ERROR",
}


def classify_error_type(exc_type: str) -> str:
    return ERROR_CLASS.get(exc_type, "OTHER")


# ── Data structures ─────────────────────────────────────────────────


@dataclass
class FailureEvent:
    """Normalized representation of one failed test.

    Fields are designed for deterministic clustering:
      - test_name: full dotted name
      - base_test_name: test name without parameterization suffix
      - exception_type: raw Python exception (e.g. TypeError)
      - error_class: coarse group (TYPE_ERROR, ASSERTION_ERROR, …)
      - message: original error text
      - message_fingerprint: normalized/masked version (for dedup)
      - assertion_line: the source code line that failed
      - stack_frames: ALL frames for this failure
      - call_chain: first ≤N repo frames (the call path to the failure)
      - top_repo_frame: first repo frame in call_chain (for overlap)
      - is_parameterized: True if test name contains [param]
      - param_group: base_test_name for parameterized families
    """

    test_name: str = ""
    exception_type: str = ""
    error_class: str = ""
    message: str = ""
    message_fingerprint: str = ""
    assertion_line: str = ""
    stack_frames: list[StackFrame] = field(default_factory=list)
    call_chain: list[StackFrame] = field(default_factory=list)
    top_repo_frame: str = ""
    is_parameterized: bool = False
    param_group: str = ""


@dataclass
class FailureCluster:
    """A group of FailureEvents that share a common root cause.

    'root_cause' is the highest-confidence frame/function where
    the failures converge — typically the shared top of call_chain.
    """

    events: list[FailureEvent] = field(default_factory=list)
    cluster_id: str = ""
    # Deterministic cluster key (used for hashing/comparison)
    primary_error_class: str = ""
    shared_top_frame: str = ""
    message_fingerprint: str = ""
    param_group: str = ""
    # Evidence from the cluster
    count: int = 0
    error_types: dict[str, int] = field(default_factory=dict)
    exception_types_seen: list[str] = field(default_factory=list)
    test_names: list[str] = field(default_factory=list)
    call_chain_overlap: list[str] = field(default_factory=list)
    root_cause: str = ""


# ── Helpers ─────────────────────────────────────────────────────────


def _first_repo_frame(frames: list[StackFrame]) -> str:
    """Return the first repo frame path from a stack, or ''."""
    for f in frames:
        if f.is_repo_frame and not f.is_test_file:
            return f"{f.file}:{f.line}"
    if frames:
        return f"{frames[0].file}:{frames[0].line}"
    return ""


def _call_chain_file_list(frames: list[StackFrame], max_depth: int = 5) -> list[str]:
    """Return file:line strings for the first N repo frames."""
    result: list[str] = []
    for f in frames:
        if len(result) >= max_depth:
            break
        if f.is_repo_frame:
            result.append(f"{f.file}:{f.line}")
    return result


def _param_group(test_name: str) -> str:
    """Extract base test name, stripping [param] suffix."""
    idx = test_name.find("[")
    return test_name[:idx] if idx != -1 else test_name


# ── Build FailureEvents from bundle ─────────────────────────────────


def extract_failure_events(
    bundle: RuntimeFailureFeatureBundle,
) -> list[FailureEvent]:
    """Build one FailureEvent per failed test from the bundle.

    Uses the per-test TestFailureSignal records (P1-3A refactored extractor).
    Each event has its own error_message, assertion_line, and stack_frames
    correctly bound to that specific test.
    """
    tl = bundle.test_log

    # Prefer structured per-test records (P1-3A extractor)
    records = list(tl.failures) if tl.failures else []
    events: list[FailureEvent] = []

    # Fallback: if no per-test records but failed_tests exist (summary-only
    # logs without FAILURES sections, like ANSI-stripped output), build
    # minimal events from the plain test name list.
    if not records and tl.failed_tests:
        for tn in tl.failed_tests:
            events.append(FailureEvent(
                test_name=tn,
                exception_type="Unknown",
                error_class=classify_error_type("Unknown"),
                message="",
                message_fingerprint="",
                is_parameterized="[" in tn,
                param_group=_param_group(tn),
            ))
        return events

    # Require per-test records for proper analysis.
    if not records:
        raise ValueError(
            "extract_failure_events: bundle has 0 per-test failure records. "
            "Use pytest_extractor.extract_test_log() on the raw test_output.txt "
            "to produce structured TestFailureSignals before clustering."
        )

    for rec in records:
        normalized = normalize_message(rec.error_message)
        ev = FailureEvent(
            test_name=rec.test_name,
            exception_type=rec.exception_type,
            error_class=classify_error_type(rec.exception_type),
            message=rec.error_message,
            message_fingerprint=normalized,
            assertion_line=rec.assertion_line,
            stack_frames=list(rec.stack_frames),
            call_chain=_call_chain_file_list(rec.stack_frames),
            top_repo_frame=rec.root_frame,
            is_parameterized="[" in rec.test_name,
            param_group=_param_group(rec.test_name),
        )
        events.append(ev)

    return events


# ── Deterministic clustering ────────────────────────────────────────


# ── Low-information fingerprints that should never trigger merge ────

_LOW_INFO_FINGERPRINTS = {
    "AssertionError:",
    "AssertionError",
    "Error:",
    "",
}


def cluster_failures(events: list[FailureEvent]) -> list[FailureCluster]:
    """Group FailureEvents by shared root cause.

    Priority order for grouping (probed in order; first match wins):

    1. Parameterized family: tests sharing the same param_group
       AND the same error_class go together.
    2. Message fingerprint match: identical normalized message.
    3. Shared top-repo frame: failure converges on the same file:line.
    4. Same error_class + same call_chain overlap (≥2 shared frames).
    5. Fallback: by error_class alone.
    """
    unassigned = list(events)
    clusters: list[FailureCluster] = []

    # Helper to pop matched events
    def _pop_matching(match_fn) -> list[FailureEvent]:
        matched = [e for e in unassigned if match_fn(e)]
        for e in matched:
            unassigned.remove(e)
        return matched

    # 1. Parameterized families — only merge if they also share at least
    #    one more signal (message fingerprint, top frame, or call-chain frame)
    param_groups: dict[str, list[FailureEvent]] = {}
    for e in unassigned:
        if e.is_parameterized:
            key = (e.param_group, e.error_class)
            param_groups.setdefault(str(key), []).append(e)
    for key, group in param_groups.items():
        if len(group) < 2:
            continue
        # Verify second signal: shared fingerprint, top frame, or call-chain overlap
        # Each sig must be (a) non-empty AND (b) exactly one unique value across the group.
        # An empty set (all events have low-info) is NOT a signal.
        fingerprints = {e.message_fingerprint for e in group
                        if e.message_fingerprint not in _LOW_INFO_FINGERPRINTS}
        sig1 = bool(fingerprints) and len(fingerprints) == 1

        top_frames = {e.top_repo_frame for e in group if e.top_repo_frame}
        sig2 = bool(top_frames) and len(top_frames) == 1

        chain_sets = [set(e.call_chain) for e in group if e.call_chain]
        sig3 = False
        if len(chain_sets) >= 2:
            common = set.intersection(*chain_sets)
            sig3 = len(common) >= 1
        if not (sig1 or sig2 or sig3):
            continue  # insufficient evidence to merge
        for e in group:
            unassigned.remove(e)
        _finish_cluster(clusters, group)

    # 2. Message fingerprint (non-empty, non-trivial, not low-information)
    fingerprint_groups: dict[str, list[FailureEvent]] = {}
    for e in unassigned:
        fp = e.message_fingerprint
        if fp and fp not in ("", "?") and fp not in _LOW_INFO_FINGERPRINTS:
            fingerprint_groups.setdefault(fp, []).append(e)
    for fp, group in fingerprint_groups.items():
        if len(group) < 2:
            continue  # single-test clusters handled later
        for e in group:
            unassigned.remove(e)
        _finish_cluster(clusters, group)

    # 3. Shared top-repo frame
    frame_groups: dict[str, list[FailureEvent]] = {}
    for e in unassigned:
        f = e.top_repo_frame
        if f:
            frame_groups.setdefault(f, []).append(e)
    for f, group in frame_groups.items():
        if len(group) < 2:
            continue
        for e in group:
            unassigned.remove(e)
        _finish_cluster(clusters, group)

    # 4. Same error_class + call-chain overlap ≥ 2 (conservative: must share ≥2 frames)
    unused_here = list(unassigned)
    for e1 in unused_here:
        if e1 not in unassigned:
            continue
        group = [e1]
        unassigned.remove(e1)
        for e2 in list(unassigned):
            if e1.error_class == e2.error_class and e1.call_chain and e2.call_chain:
                overlap = len(set(e1.call_chain) & set(e2.call_chain))
                if overlap >= 2:
                    group.append(e2)
                    unassigned.remove(e2)
        _finish_cluster(clusters, group)

    # 5. Remaining: each event is its OWN singleton cluster
    # NEVER force-merge by error class alone — evidence is insufficient.
    for e in list(unassigned):
        unassigned.remove(e)
        _finish_cluster(clusters, [e])

    return clusters


def _finish_cluster(
    clusters: list[FailureCluster],
    events: list[FailureEvent],
) -> None:
    """Create a FailureCluster from events and append to clusters."""
    if not events:
        return

    error_types: dict[str, int] = {}
    call_chains_seen: set[str] = set()
    call_chain_common: list[str] = []

    for e in events:
        error_types[e.exception_type] = error_types.get(e.exception_type, 0) + 1
        for frame_key in e.call_chain:
            call_chains_seen.add(frame_key)

    # Shared frames: find frames present in ALL events' call chains.
    # Preserve order from the FIRST event's call chain for determinism.
    if len(events) > 1:
        chain_sets = [set(e.call_chain) for e in events]
        common = chain_sets[0]
        for cs in chain_sets[1:]:
            common &= cs
        # Order by first event's call chain (deterministic since list order is stable)
        ordered_common = [f for f in events[0].call_chain if f in common]
        call_chain_common = ordered_common
    else:
        call_chain_common = list(events[0].call_chain)

    first = events[0]
    cluster = FailureCluster(
        events=events,
        cluster_id=_cluster_id(events),
        primary_error_class=first.error_class,
        shared_top_frame=first.top_repo_frame or "",
        message_fingerprint=first.message_fingerprint,
        param_group=first.param_group,
        count=len(events),
        error_types=error_types,
        exception_types_seen=sorted(error_types.keys()),
        test_names=[e.test_name for e in events],
        call_chain_overlap=call_chain_common,
        root_cause=call_chain_common[0] if call_chain_common else first.top_repo_frame,
    )
    clusters.append(cluster)


def _cluster_id(events: list[FailureEvent]) -> str:
    """Deterministic cluster ID from sorted canonical members."""
    canonical = sorted(
        f"{e.test_name}|{e.error_class}|{e.message_fingerprint}|{e.top_repo_frame}"
        for e in events
    )
    import hashlib
    return "C" + hashlib.sha256("\n".join(canonical).encode()).hexdigest()[:7]


# ── API ─────────────────────────────────────────────────────────────


def reasoner_v2_cluster(
    bundle: RuntimeFailureFeatureBundle,
) -> list[FailureCluster]:
    """Full P1-3A pipeline: extract events → cluster → return clusters."""
    events = extract_failure_events(bundle)
    if not events:
        return []
    return cluster_failures(events)
