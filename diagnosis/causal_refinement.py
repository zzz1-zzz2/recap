"""Causal Refinement — post-retrieval diagnosis refinement and safety gate.

After Router evidence is gathered, refines the provisional diagnosis with:

  - Baseline failure tagging via three independent dimensions:
    benchmark_role (FAIL_TO_PASS / OTHER), failure_delta (persistence vs regression),
    revision_priority (primary / secondary / monitor / excluded)
  - Patch-linkage analysis (dynamically extracted from patch content)
  - Causal mechanism inference
  - Safety warnings

NO instance-specific rules (hardcoded frame names, repo symbols, etc.).
All symbol extraction is patch-driven.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CausalAttribution:
    """Attribution result for one failure cluster.

    Follows three independent semantic axes:
      1. benchmark_role — the test's role in the SWE-bench evaluation
      2. failure_delta — how the failure state changed between base and R1
      3. patch_linkage  — whether the failure connects to R1's changes

    Combined they determine revision_priority.
    """

    # ── SWE-bench evaluation role ──
    benchmark_role: str = "UNKNOWN"
    """FAIL_TO_PASS | PASS_TO_PASS | OTHER | UNKNOWN"""

    # ── Failure state transition ──
    base_status: str = "UNKNOWN"
    """PASS | FAIL | UNKNOWN"""

    r1_status: str = "UNKNOWN"
    """PASS | FAIL | UNKNOWN"""

    failure_delta: str = "UNKNOWN"
    """PERSISTENT_TARGET_FAILURE | NEW_REGRESSION | FIXED_BY_R1 | UNKNOWN
    - PERSISTENT_TARGET_FAILURE: base fail, R1 fail, in FAIL_TO_PASS
      → task-level test not yet resolved by R1
    - NEW_REGRESSION: base pass, R1 fail
      → R1 broke something
    - FIXED_BY_R1: base fail, R1 pass
      → resolved (should not appear in R1 failures)
    """

    # ── Patch causal attribution ──
    causal_status: str = "UNCERTAIN"
    """PATCH_LINKED | UNCERTAIN
    Whether a causal connection between patch and failure is established.
    """

    patch_linkage: str = "NONE"
    """DIRECT | REGISTRATION | SYMBOL | NONE
    — DIRECT: failure frame overlaps patch-edited file
    — REGISTRATION: failure involves code R1 registered/imported
    — SYMBOL: error message mentions a symbol defined in R1's patch
    — NONE: no detected linkage
    """

    mechanism: str = ""
    """How the patch produces the failure, citing specific evidence."""

    causal_evidence_ids: list[str] = field(default_factory=list)
    counterevidence_ids: list[str] = field(default_factory=list)
    alternative_hypotheses: list[str] = field(default_factory=list)

    unsafe_repair_warnings: list[str] = field(default_factory=list)

    # ── Revision priority (derived) ──
    revision_priority: str = "MONITOR"
    """PRIMARY | SECONDARY | MONITOR | EXCLUDED
    - PRIMARY:   PATCH_LINKED + (PERSISTENT_TARGET_FAILURE or NEW_REGRESSION)
                 → must be addressed in R2
    - SECONDARY: not PATCH_LINKED but benchmark_role == FAIL_TO_PASS
                 → task-level test; revisit after primary clusters
    - MONITOR:   no linkage, not a task test
                 → watch for changes, don't modify code
    - EXCLUDED:  explicitly excluded (FIXED_BY_R1, or confirmed environment)
    """


# ── Helpers ──────────────────────────────────────────────────────────────


def _get_non_test_frames(cluster) -> list[dict]:
    """Extract non-test repo frames from a cluster's events."""
    frames: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for ev in cluster.events:
        for sf in ev.stack_frames:
            key = (sf.file, sf.line)
            if key not in seen and sf.is_repo_frame and not sf.is_test_file:
                seen.add(key)
                frames.append({"file": sf.file, "line": sf.line, "function": sf.function})
    return frames


def _get_all_test_names(cluster) -> list[str]:
    """Get all test names from a cluster."""
    names: list[str] = []
    for ev in cluster.events:
        if ev.test_name and ev.test_name not in names:
            names.append(ev.test_name)
    for tn in getattr(cluster, "test_names", []):
        if tn and tn not in names:
            names.append(tn)
    return names


# ── Patch symbol extraction (fully generic, no hardcoded symbols) ─────


def _extract_patch_symbols(patch_content: str) -> dict[str, set[str]]:
    """Extract symbols defined in patch content.

    Returns:
      - function_names: set of function names from 'def foo('  lines
      - class_names: set of class names from 'class Foo' lines
      - import_names: set of imported symbol names from '+import' lines
      - registration_symbols: set of symbols mentioned in @-decorator
        registration patterns (e.g. the frame names in
        @transform_graph.transform(..., Src, Dst))
      - edited_filenames: set of filename stems from diff headers

    Pure regex — no domain knowledge (Astropy, Django, etc.).
    """
    result: dict[str, set[str]] = {
        "function_names": set(),
        "class_names": set(),
        "import_names": set(),
        "registration_symbols": set(),
        "edited_filenames": set(),
    }

    # Function definitions: def foo(...):
    for m in re.finditer(r"^def\s+([A-Za-z_]\w*)\s*\(", patch_content, re.MULTILINE):
        result["function_names"].add(m.group(1))

    # Class definitions: class Foo(...):
    for m in re.finditer(r"^class\s+([A-Za-z_]\w*)", patch_content, re.MULTILINE):
        result["class_names"].add(m.group(1))

    # Import lines: +from X import Y or +import X
    for m in re.finditer(
        r"^\+.*\bimport\s+(\w+(?:\.\w+)*)",
        patch_content, re.MULTILINE,
    ):
        result["import_names"].add(m.group(1))

    # Registration / decorator patterns: @something.transform(... A, B)
    # Extract arguments that look like identifiers
    for m in re.finditer(
        r"@\w[\w.]*\.\w+\s*\(([^)]*)\)",
        patch_content,
    ):
        args = m.group(1)
        # Extract PascalCase identifiers (likely class/frame names)
        for sym in re.finditer(r"[A-Z][A-Za-z0-9_]+", args):
            result["registration_symbols"].add(sym.group())

    # Edited filenames from diff headers: +++ b/path/to/file.py
    for m in re.finditer(
        r"^\+\+\+\s+b/(\S+)",
        patch_content, re.MULTILINE,
    ):
        fpath = m.group(1)
        stem = fpath.split("/")[-1].replace(".py", "").lower()
        result["edited_filenames"].add(stem)

    return result


# Route-comparison patterns — generic across any repo.
# These describe test structure (comparing two execution paths),
# not domain concepts. No repo-specific names allowed.
_ROUTE_COMPARISON_LOWER = [
    "bothroutes",
    "cross_check",
    "roundtrip",
    "bidirectional",
    "dualpath",
    "round_trip",
]


def _check_route_comparison(all_text_lower: str) -> bool:
    """Check if test text suggests a route-comparison pattern."""
    return any(p in all_text_lower for p in _ROUTE_COMPARISON_LOWER)


# ── Baseline tagging ──────────────────────────────────────────────────


def tag_baseline_failure(
    cluster_test_names: list[str],
    fail_to_pass: list[str],
) -> bool:
    """Check if any test in the cluster is in FAIL_TO_PASS."""
    baseline_leaves = set()
    for ftp in fail_to_pass:
        leaf = ftp.split("::")[-1] if "::" in ftp else ftp
        baseline_leaves.add(leaf)

    for tn in cluster_test_names:
        leaf = tn.split("::")[-1] if "::" in tn else tn
        bare_leaf = leaf.split("[")[0] if "[" in leaf else leaf
        if leaf in baseline_leaves or bare_leaf in baseline_leaves:
            return True
    return False


# ── Patch-linkage analysis (fully generic) ────────────────────────────


def analyze_patch_linkage(
    cluster,
    patch_edited_files: list[str],
    patch_content: str | None = None,
) -> tuple[str, str, list[str]]:
    """Determine linkage between failure and R1 patch.

    Uses ONLY dynamically-extracted symbols from patch_content
    and frame file overlap. No hardcoded domain symbols.

    When patch_content is the actual diff text, function/class
    names and registration symbols are extracted from it.
    When patch_content is None or structured data, falls back
    to file-level matching via edited-filename stems.

    Returns (linkage_strength, mechanism, evidence_ids).
    """
    non_test_frames = _get_non_test_frames(cluster)
    test_names = _get_all_test_names(cluster)

    edited_full_paths = set(patch_edited_files)
    edited_basenames = {f.split("/")[-1] for f in patch_edited_files}
    edited_stems_lower = {f.replace(".py", "").lower() for f in edited_basenames}

    # Collect all failure text
    all_text = " ".join(test_names)
    for ev in cluster.events:
        all_text += " " + (ev.message or "")
        all_text += " " + (ev.assertion_line or "")
    all_text_lower = all_text.lower()

    # ── 1. Direct frame overlap ──
    for frame in non_test_frames:
        fp = frame["file"]
        basename = fp.split("/")[-1]
        if fp in edited_full_paths or basename in edited_basenames:
            return (
                "DIRECT",
                f"Failure frame ({fp}:{frame['line']}) is in a file modified by R1",
                [f"frame:{fp}:{frame['line']}"],
            )

    # ── 2. Module-level (edited file name appears in failure text) ──
    for stem in edited_stems_lower:
        if stem in all_text_lower:
            return (
                "SYMBOL",
                f"R1 edited file '{stem}.py' is referenced in failure text",
                [f"symbol:{stem}"],
            )

    # ── 3. Dynamic symbol extraction from patch text ──
    if patch_content:
        patch_syms = _extract_patch_symbols(patch_content)

        # Merge all patch-defined symbols EXCEPT registration_symbols
        # (those are checked separately for REGISTRATION linkage)
        all_patch_symbols: set[str] = set()
        for key, sym_set in patch_syms.items():
            if key != "registration_symbols":
                all_patch_symbols.update(sym_set)

        all_text_original = all_text

        # Check registration symbols FIRST (most specific).
        # These come from @decorator(args with PascalCase identifiers).
        for sym in patch_syms.get("registration_symbols", set()):
            if sym in all_text_original or sym.lower() in all_text_lower:
                return (
                    "REGISTRATION",
                    f"R1 registered transforms involving {sym}; "
                    f"failure references this symbol",
                    [f"registration:{sym}"],
                )

        # Check function/class/import names defined in patch
        for sym in all_patch_symbols:
            if sym.lower() in all_text_lower:
                return (
                    "SYMBOL",
                    f"Error or test references '{sym}' which is defined by R1's patch",
                    [f"symbol:{sym}"],
                )

        # Check route-comparison pattern (generic)
        if _check_route_comparison(all_text_lower):
            if patch_syms.get("registration_symbols"):
                return (
                    "REGISTRATION",
                    "Route comparison pattern detected and R1 registered new transforms. "
                    "The new path likely differs from the established path.",
                    ["registration:route_comparison"],
                )

    return ("NONE", "No detected overlap between failure and patch", [])


# ── Causal refinement ────────────────────────────────────────────────


def _determine_failure_delta(
    in_baseline: bool,
) -> str:
    """Determine failure_delta from baseline status.

    NOTE: without empty-patch eval results we only know whether a test
    is in FAIL_TO_PASS. Full delta requires base_status (from empty-patch
    run) + r1_status (from R1 run).
    """
    if in_baseline:
        # The test was failing at base. R1 didn't fix it.
        # It could still be PATCH_LINKED if R1 changed the failure mechanism.
        return "PERSISTENT_TARGET_FAILURE"
    # Not in FAIL_TO_PASS, but we don't know if it passed at base.
    # Without empty-patch eval, mark as UNKNOWN.
    return "UNKNOWN"


def _determine_revision_priority(
    causal_status: str,
    failure_delta: str,
    benchmark_role: str,
    patch_linkage: str,
) -> str:
    """Derive revision priority from the three semantic axes."""
    if causal_status == "PATCH_LINKED" and failure_delta in (
        "PERSISTENT_TARGET_FAILURE", "UNKNOWN", "NEW_REGRESSION"
    ):
        return "PRIMARY"

    if benchmark_role == "FAIL_TO_PASS":
        # Task-level test that hasn't been linked to patch
        return "SECONDARY"

    return "MONITOR"


def causal_refine(
    provisional_subtype: str,
    cluster,
    patch_edited_files: list[str],
    fail_to_pass: list[str] | None = None,
    patch_content: str | None = None,
) -> CausalAttribution:
    """Full causal refinement for one cluster."""
    test_names = _get_all_test_names(cluster)

    # 1. Benchmark role
    f2p = fail_to_pass or []
    in_baseline = tag_baseline_failure(test_names, f2p)
    benchmark_role = "FAIL_TO_PASS" if in_baseline else "OTHER"

    # 2. Patch linkage (dynamically extracted symbols)
    linkage_strength, linkage_mechanism, evidence_ids = analyze_patch_linkage(
        cluster, patch_edited_files, patch_content,
    )

    # 3. Failure delta
    failure_delta = _determine_failure_delta(in_baseline)

    # 4. Causal status
    if linkage_strength in ("DIRECT", "REGISTRATION", "SYMBOL"):
        causal_status = "PATCH_LINKED"
    else:
        causal_status = "UNCERTAIN"

    # 5. Mechanism
    mechanism_parts = [linkage_mechanism] if linkage_mechanism else []
    non_test_frames = _get_non_test_frames(cluster)
    if non_test_frames:
        frame_desc = "; ".join(
            f"{f['file']}:{f['line']}" for f in non_test_frames[:3]
        )
        mechanism_parts.append(f"Non-test frames: {frame_desc}")
    if in_baseline:
        mechanism_parts.append(
            "In FAIL_TO_PASS (task-level test: fails at base commit, "
            "gold patch should resolve)"
        )
    mechanism = ". ".join(mechanism_parts)

    # 6. Alternative hypotheses
    alternatives = []
    if linkage_strength in ("REGISTRATION", "SYMBOL") and in_baseline:
        alternatives.append(
            "Failure may be a baseline issue whose mechanism is changed "
            "by the patch (same symptom, different root cause)"
        )

    # 7. Safety warnings
    warnings = _generate_warnings(
        provisional_subtype, causal_status, cluster, patch_edited_files,
        benchmark_role=benchmark_role,
    )

    # 8. Revision priority
    revision_priority = _determine_revision_priority(
        causal_status, failure_delta, benchmark_role, linkage_strength,
    )

    return CausalAttribution(
        benchmark_role=benchmark_role,
        base_status="FAIL" if in_baseline else "UNKNOWN",
        r1_status="FAIL",
        failure_delta=failure_delta,
        causal_status=causal_status,
        patch_linkage=linkage_strength,
        mechanism=mechanism,
        causal_evidence_ids=evidence_ids,
        alternative_hypotheses=alternatives,
        unsafe_repair_warnings=warnings,
        revision_priority=revision_priority,
    )


def _generate_warnings(
    subtype: str,
    causal_status: str,
    cluster,
    patch_edited_files: list[str],
    benchmark_role: str | None = None,
) -> list[str]:
    """Generate unsafe-repair warnings from generic failure patterns.

    Uses only signal patterns (unexpected keyword, unsupported operand,
    assert_allclose) — NO hardcoded frame names or symbol references.
    """
    warnings: list[str] = []

    has_unexpected_keyword = any(
        "unexpected keyword" in (ev.message or "").lower()
        for ev in cluster.events
    )
    has_unsupported_operand = any(
        "unsupported operand" in (ev.message or "").lower()
        for ev in cluster.events
    )
    has_assert_allclose = any(
        "assert_allclose" in (ev.assertion_line or "")
        for ev in cluster.events
    )

    if has_unexpected_keyword:
        warnings.append(
            "An 'unexpected keywords' error does NOT necessarily mean a frame "
            "attribute is missing. It often means a destination-frame attribute "
            "is being incorrectly forwarded to a source frame that doesn't "
            "support it. Do NOT add the missing attribute to the source frame "
            "unless you verify the source frame is supposed to have it."
        )

    if has_unsupported_operand:
        warnings.append(
            "An 'unsupported operand type' error may be a pre-existing bug "
            "unrelated to R1's changes. Verify by checking whether the error "
            "occurs at the base commit (empty patch) before attributing it "
            "to the current patch."
        )

    if has_assert_allclose:
        warnings.append(
            "Route comparison failures (assert_allclose between two paths) "
            "indicate the new path produces different numerical results. "
            "Do NOT weaken tolerances, add empirical biases, or hard-code "
            "test-specific values. The fix requires making the new path "
            "semantically equivalent to the established path."
        )

    if causal_status == "UNCERTAIN":
        warnings.append(
            "No causal link between this failure and the current patch "
            "has been established. Do not modify code solely for this "
            "failure without first verifying the linkage."
        )

    if benchmark_role == "FAIL_TO_PASS" and causal_status != "PATCH_LINKED":
        warnings.append(
            "This test is in FAIL_TO_PASS (fails at base commit, gold "
            "patch should resolve). Although it's a task-level test, "
            "no patch linkage has been established. Only address this "
            "failure after verifying a causal path from R1's changes."
        )

    return warnings


# ── Safety gate ──────────────────────────────────────────────────────


def safety_gate(
    attributions_or_diagnoses,
    subtypes: list[str],
) -> tuple[list[int], list[str]]:
    """Filter clusters for inclusion in the revision plan.

    Accepts either CausalAttribution or SubtypedDiagnosis objects.
    In the latter case, derives benchmark_role from .baseline_in_fail_to_pass.

    Returns (active_indices, deprioritized_reasons).

    PRIMARY:   causal_status=PATCH_LINKED + linkage!=NONE
    SECONDARY: benchmark_role=FAIL_TO_PASS (not patch-linked)
    MONITOR:   no linkage, not a task test
    """
    primary: list[int] = []
    secondary: list[int] = []
    deprioritized: list[str] = []

    for i, (attr, sub) in enumerate(zip(attributions_or_diagnoses, subtypes)):
        causal_status = getattr(attr, "causal_status", "UNCERTAIN")
        patch_linkage = getattr(attr, "patch_linkage", "NONE")
        benchmark_role = getattr(attr, "benchmark_role", None)
        failure_delta = getattr(attr, "failure_delta", "UNKNOWN")

        # Derive benchmark_role from baseline_in_fail_to_pass if not set
        if benchmark_role is None or benchmark_role == "UNKNOWN":
            baseline_flag = getattr(attr, "baseline_in_fail_to_pass", False)
            benchmark_role = "FAIL_TO_PASS" if baseline_flag else "OTHER"

        is_patch_linked = causal_status == "PATCH_LINKED" and patch_linkage != "NONE"
        is_task_test = benchmark_role == "FAIL_TO_PASS"

        if is_patch_linked:
            primary.append(i)
        elif is_task_test:
            secondary.append(i)
            deprioritized.append(
                f"Cluster {i} ({sub}): {failure_delta}, "
                f"benchmark_role={benchmark_role}, "
                f"patch_linkage={patch_linkage}. "
                "Task-level test — fails at base commit, "
                "gold patch should resolve. Not yet linked to R1 patch. "
                "Revisit after primary clusters."
            )
        else:
            deprioritized.append(
                f"Cluster {i} ({sub}): {failure_delta}, "
                f"patch_linkage={patch_linkage}. "
                "No patch linkage, not a task-level test. "
                "Monitor only — do not modify code for this failure."
            )

    return primary + secondary, deprioritized


def merge_attribution_into_diagnosis(
    diagnosis,
    attribution: CausalAttribution,
):
    """Merge CausalAttribution fields into a SubtypedDiagnosis.

    Covers both the original fields and the new semantic axes
    (benchmark_role, failure_delta, revision_priority).
    """
    diagnosis.causal_status = attribution.causal_status
    diagnosis.mechanism = attribution.mechanism
    diagnosis.patch_linkage = attribution.patch_linkage
    diagnosis.causal_evidence_ids = list(attribution.causal_evidence_ids)
    diagnosis.counterevidence_ids = list(attribution.counterevidence_ids)
    diagnosis.alternative_hypotheses = list(attribution.alternative_hypotheses)
    diagnosis.unsafe_repair_warnings = list(attribution.unsafe_repair_warnings)
    diagnosis.baseline_in_fail_to_pass = attribution.benchmark_role == "FAIL_TO_PASS"
    # New semantic axes
    diagnosis.benchmark_role = attribution.benchmark_role
    diagnosis.failure_delta = attribution.failure_delta
    diagnosis.revision_priority = attribution.revision_priority
    return diagnosis
