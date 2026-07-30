"""P1-3B: Evidence Alignment — cross-source evidence fusion for FailureClusters.

Takes clustered failures + patch state + trajectory state, and produces:

  - Which symbols appear in the error, in the patch, and in the trajectory
  - Which critical files the agent never viewed
  - Which assumptions the patch makes that the evidence contradicts
  - A subtyped diagnosis per cluster (not just a coarse tag)

This is the core "reasoning without LLM" step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from recap.diagnosis.failure_event import FailureCluster, FailureEvent
from recap.diagnosis.signals.schema import (
    PatchSignals,
    RuntimeFailureFeatureBundle,
    StackFrame,
    TrajectorySignals,
)
from recap.diagnosis.taxonomy import ContextDeficiencyType


# ── Symbol extraction helpers ───────────────────────────────────────


def _extract_symbols_from_stack(frame_key: str) -> list[str]:
    """Extract likely symbol names from a file:line reference."""
    syms: list[str] = []
    path = frame_key.split(":")[0] if ":" in frame_key else frame_key
    # "astropy/coordinates/baseframe.py" → "baseframe"
    parts = path.replace(".py", "").split("/")
    if parts:
        syms.append(parts[-1])
    # Also extract function-like patterns from line content
    return syms


def _extract_symbols_from_message(msg: str) -> list[str]:
    """Extract quoted identifiers from error messages."""
    # "Coordinate frame ITRS got unexpected keywords: ['location']"
    #   → ITRS, location
    quoted = re.findall(r"""['"](\w+)['"]""", msg)
    return quoted


def _extract_symbols_from_patch(patch: PatchSignals) -> list[str]:
    """Extract likely symbol names from edited files."""
    syms: list[str] = []
    for f in patch.edited_files:
        parts = f.replace(".py", "").split("/")
        syms.append(parts[-1])  # module name
    return syms


# ── EvidenceAlignment data structure ────────────────────────────────


@dataclass
class SymbolReference:
    """Where a symbol appears across evidence sources."""

    symbol: str = ""
    in_error_stack: bool = False
    in_error_message: bool = False
    in_patch_edit: bool = False
    in_trajectory_view: bool = False
    known_provider_file: str = ""


@dataclass
class EvidenceAlignment:
    """Cross-source evidence for one failure cluster.

    Designed to be fully deterministic — no LLM calls.
    """

    cluster_id: str = ""
    # ── Source breakdown ──
    error_types: dict[str, int] = field(default_factory=dict)
    error_frames: list[str] = field(default_factory=list)
    error_symbols: list[str] = field(default_factory=list)
    shared_frames: list[str] = field(default_factory=list)
    call_chain_overlap: list[str] = field(default_factory=list)
    # ── Patch alignment ──
    patch_edited_files: list[str] = field(default_factory=list)
    patch_introduced_new_file: bool = False
    patch_edited_symbols: list[str] = field(default_factory=list)
    # ── Trajectory alignment ──
    trajectory_viewed_files: list[str] = field(default_factory=list)
    trajectory_viewed_symbols: list[str] = field(default_factory=list)
    trajectory_viewed_but_not_edited: list[str] = field(default_factory=list)
    trajectory_related_tests_viewed: list[str] = field(default_factory=list)
    # ── Gap analysis ──
    symbols_in_error_not_viewed: list[str] = field(default_factory=list)
    symbols_in_error_not_viewed_provider: list[str] = field(default_factory=list)
    missing_provider_files: list[str] = field(default_factory=list)
    # ── Assumption analysis ──
    patch_assumptions: list[str] = field(default_factory=list)
    contradictory_evidence: list[str] = field(default_factory=list)


# ── Alignment logic ─────────────────────────────────────────────────


def align_evidence(
    cluster: FailureCluster,
    patch: PatchSignals,
    trajectory: TrajectorySignals,
) -> EvidenceAlignment:
    """Build cross-source evidence for one cluster.

    Each alignment records:
      - What the error stack says (the symptom)
      - What the patch changed (the attempted fix)
      - What the trajectory explored (the search behavior)
      - Gaps between them (missed information)
    """
    # Collect all stack frames across cluster events
    all_error_frames: list[str] = []
    for e in cluster.events:
        all_error_frames.extend(e.call_chain)

    # Deduplicate preserving order
    seen_frames: set[str] = set()
    deduped_frames: list[str] = []
    for f in all_error_frames:
        if f not in seen_frames:
            seen_frames.add(f)
            deduped_frames.append(f)

    # Extract symbols from error messages across cluster
    error_symbols: list[str] = []
    for e in cluster.events:
        error_symbols.extend(_extract_symbols_from_message(e.message))
    error_symbols = list(dict.fromkeys(error_symbols))  # dedup

    # Patch analysis
    patch_symbols = _extract_symbols_from_patch(patch)
    patch_new_file = any(
        f.endswith(".py") and _is_new_file(f, patch)
        for f in patch.edited_files
    )

    # Trajectory analysis
    viewed_files = list(trajectory.viewed_files or [])
    viewed_file_names = {f.split("/")[-1] for f in viewed_files}

    # Files in the error that the agent never looked at
    error_files_in_cluster: set[str] = set()
    for frame_key in deduped_frames:
        file_path = frame_key.split(":")[0] if ":" in frame_key else frame_key
        file_name = file_path.split("/")[-1]
        if file_name not in viewed_file_names:
            error_files_in_cluster.add(file_path)

    # Symbols from error not in trajectory views
    symbols_not_viewed: list[str] = []
    for sym in error_symbols:
        if sym not in viewed_file_names and not any(
            sym in file_name for file_name in viewed_file_names
        ):
            symbols_not_viewed.append(sym)

    # Files grouped by whether trajectory showed them
    # (provider-file lookup intentionally omitted — that's Router's job)
    err_files_in_view: list[str] = []
    for frame_key in deduped_frames:
        fname = frame_key.split(":")[0] if ":" in frame_key else frame_key
        if any(fname in vf for vf in viewed_files):
            err_files_in_view.append(frame_key)

    # Files viewed but never patched
    edited_dir_parts = {
        f.replace(".py", "").split("/")[-1] for f in patch.edited_files
    }
    viewed_but_not_edited = [
        f for f in viewed_files
        if f.split("/")[-1] not in edited_dir_parts
        and f.endswith(".py")
    ][:10]  # cap

    # Generic assumption analysis: if patch edits differ from error frames
    patch_assumptions: list[str] = []
    contradictory_evidence: list[str] = []
    if patch.edited_files and error_files_in_cluster:
        error_basenames = {f.split("/")[-1] for f in error_files_in_cluster if "/" in f}
        patch_basenames = {f.split("/")[-1] for f in patch.edited_files if "/" in f}
        overlap = error_basenames & patch_basenames
        if not overlap:
            patch_assumptions.append(
                "R1 edits files not directly referenced in the error stack"
            )
            contradictory_evidence.append(
                "Error stack frames point to different files than the patch edits"
            )

    return EvidenceAlignment(
        cluster_id=cluster.cluster_id,
        error_types=cluster.error_types,
        error_frames=deduped_frames[:8],
        error_symbols=error_symbols,
        shared_frames=cluster.call_chain_overlap[:5] or [],
        call_chain_overlap=list(cluster.call_chain_overlap),
        patch_edited_files=list(patch.edited_files),
        patch_introduced_new_file=patch_new_file,
        patch_edited_symbols=patch_symbols,
        trajectory_viewed_files=viewed_files[:15],
        trajectory_viewed_symbols=list(dict.fromkeys(
            s for f in viewed_files
            for s in _extract_symbols_from_stack(f)
        )),
        trajectory_viewed_but_not_edited=viewed_but_not_edited,
        trajectory_related_tests_viewed=[
            f for f in viewed_files if "test" in f.lower()
        ],
        symbols_in_error_not_viewed=symbols_not_viewed,
        symbols_in_error_not_viewed_provider=[],
        patch_assumptions=patch_assumptions,
        contradictory_evidence=contradictory_evidence,
    )


def _is_new_file(file_path: str, patch: PatchSignals) -> bool:
    """Heuristic: a file with 0 deleted lines is likely newly created."""
    if patch.added_lines > 0 and patch.deleted_lines == 0:
        return True
    return False


# ── Subtyped Diagnosis ──────────────────────────────────────────────


# Fine-grained subtypes under each ContextDeficiencyType.
# These are the "what exactly" that the diagnosis identifies.
SUBTYPE_REGISTRY: dict[str, list[str]] = {
    "API_DEFINITION": [
        "FRAME_ATTRIBUTE_PROPAGATION",
        "FUNCTION_SIGNATURE_MISMATCH",
        "CLASS_ATTRIBUTE_MISSING",
        "METHOD_NOT_FOUND",
        "MODULE_MEMBER_MISSING",
    ],
    "INTERFACE_CONSTRAINT": [
        "TYPE_CONTRACT_VIOLATION",
        "FRAME_ATTRIBUTE_CONSTRAINT",
        "ARGUMENT_TYPE_MISMATCH",
    ],
    "RELATED_TESTS": [
        "NUMERICAL_MISMATCH",
        "EDGE_CASE_MISSING",
        "REGRESSION_DETECTED",
    ],
    "LOCALIZATION_DIRECTION": [
        "WRONG_FILE_MODIFIED",
        "WRONG_FUNCTION_MODIFIED",
        "SYMPTOM_CAUSE_MISMATCH",
    ],
    "CALLER_CALLEE": [
        "FRAME_MUTATION_IN_TRANSFORM",
        "ARGUMENT_FORWARDING_MISMATCH",
    ],
    "DEPENDENCY": [
        "MISSING_IMPORT",
        "MISSING_DATA_FILE",
        "MISSING_SYSTEM_DEP",
    ],
    "REGISTRATION_SITE": [
        "TRANSFORM_NOT_REGISTERED",
        "CONFIG_NOT_UPDATED",
        "EXPORT_MISSING",
    ],
    "NO_RELIABLE_DEFICIENCY": ["UNCLASSIFIED"],
}


@dataclass
class SubtypedDiagnosis:
    """Diagnosis output with fine-grained subtype.

    Fields before `reason` are the provisional diagnosis (set by
    classify_cluster). Fields after `reason` are populated by the
    Causal Refinement stage (post-Router).
    """

    type: ContextDeficiencyType = ContextDeficiencyType.NO_RELIABLE_DEFICIENCY
    subtype: str = "UNCLASSIFIED"
    confidence: str = "low"
    target_symbols: list[str] = field(default_factory=list)
    key_location: str = ""
    evidence_alignment: EvidenceAlignment | None = None
    # Natural-language reason (for the Revision Contract)
    reason: str = ""

    # ── Actionability Floor fields (Phase A) ──
    test_names: list[str] = field(default_factory=list)
    """Test names from the cluster — always populated, even for UNCLASSIFIED."""

    candidate_edit_targets: list[str] = field(default_factory=list)
    """Grounded file-level edit targets from patch/frame overlap."""

    # ── Scored Diagnoser fields (Phase B) ──
    type_scores: dict[str, float] = field(default_factory=dict)
    """Normalized scores per coarse type, for auditability."""

    supporting_signal_ids: list[str] = field(default_factory=list)
    """Signal evidence IDs supporting the top diagnosis."""

    secondary_candidates: list[dict] = field(default_factory=list)
    """Top-2+ candidates with type, subtype, score."""

    # ── Uncertainty-Aware fields (Phase C) ──
    signal_coverage: float = 0.0
    """Fraction of expected signal types present (0-1). Low = evidence gap."""

    score_margin: float = 0.0
    """Top-1 score minus top-2 score. Low margin = ambiguous diagnosis."""

    over_localization_risk: int = 0
    """0-7 scale. Higher = retrieval should expand beyond primary target."""

    scope_mode: str = "BALANCED"
    """NARROW | BALANCED | EXPANDED. Controls retrieval breadth."""

    related_targets: list[dict] = field(default_factory=list)
    """Secondary context targets with role field for multi-target retrieval."""
    related_target_roles: list[str] = field(default_factory=list)

    # ── Causal Refinement fields (populated post-Router) ──
    causal_status: str = "UNCERTAIN"
    """PATCH_LINKED | NEW_REGRESSION | PERSISTENT_BASELINE_FAILURE | SYMPTOM_ONLY | UNCERTAIN"""

    mechanism: str = ""
    """How the patch produces the failure. Must cite specific evidence."""

    patch_linkage: str = "NONE"
    """DIRECT | REGISTRATION | SYMBOL | NONE"""

    causal_evidence_ids: list[str] = field(default_factory=list)
    """Evidence IDs supporting the causal claim."""

    counterevidence_ids: list[str] = field(default_factory=list)
    """Evidence IDs contradicting or weakening the claim."""

    alternative_hypotheses: list[str] = field(default_factory=list)
    """Other explanations considered and deprioritized."""

    unsafe_repair_warnings: list[str] = field(default_factory=list)
    """Repair directions that would address symptoms but not root cause."""

    baseline_in_fail_to_pass: bool = False
    """True if at least one test in this cluster is in FAIL_TO_PASS."""

    # ── Causal refinement semantic axes (P1-3E) ──
    benchmark_role: str = "UNKNOWN"
    """FAIL_TO_PASS | PASS_TO_PASS | OTHER | UNKNOWN"""

    failure_delta: str = "UNKNOWN"
    """PERSISTENT_TARGET_FAILURE | NEW_REGRESSION | FIXED_BY_R1 | UNKNOWN"""

    revision_priority: str = "MONITOR"
    """PRIMARY | SECONDARY | MONITOR | EXCLUDED"""


# ── Score-based Diagnoser ──────────────────────────────────────────────


@dataclass
class _ScoreEvidence:
    """A single scored signal contributing to a diagnosis candidate."""

    signal_id: str
    weight: float
    description: str
    source: str  # test_name or frame_key


@dataclass
class _DiagnosisCandidate:
    """One scored diagnosis candidate (coarse type)."""

    deficiency_type: ContextDeficiencyType
    subtype: str
    score: float = 0.0
    supporting: list[_ScoreEvidence] = field(default_factory=list)
    contradicting: list[_ScoreEvidence] = field(default_factory=list)
    candidate_edit_targets: list[str] = field(default_factory=list)
    retrieval_actions: list[str] = field(default_factory=list)


# ── Score mapping: ContextDeficiencyType → score container ──

_COARSE_TYPES: list[ContextDeficiencyType] = [
    ContextDeficiencyType.RELATED_TESTS,
    ContextDeficiencyType.API_DEFINITION,
    ContextDeficiencyType.INTERFACE_CONSTRAINT,
    ContextDeficiencyType.CALLER_CALLEE,
    ContextDeficiencyType.REGISTRATION_SITE,
    ContextDeficiencyType.LOCALIZATION_DIRECTION,
]

_SUBTYPE_MAP: dict[ContextDeficiencyType, str] = {
    ContextDeficiencyType.RELATED_TESTS: "NUMERICAL_BEHAVIOR_MISMATCH",
    ContextDeficiencyType.API_DEFINITION: "REJECTED_ARGUMENT_OR_ATTRIBUTE",
    ContextDeficiencyType.INTERFACE_CONSTRAINT: "TYPE_OR_SIGNATURE_CONTRACT",
    ContextDeficiencyType.CALLER_CALLEE: "CALL_CHAIN_DATAFLOW_MISMATCH",
    ContextDeficiencyType.REGISTRATION_SITE: "REGISTRATION_OR_EXPORT_CONTEXT",
    ContextDeficiencyType.LOCALIZATION_DIRECTION: "PATCH_FAILURE_LOCALIZATION_MISMATCH",
}


def _derive_edit_targets(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    patch: PatchSignals,
) -> list[str]:
    """Derive grounded candidate edit targets without using diagnosis.

    Priority (deterministic, no LLM):
      1. R1 edited files that appear in non-test failure frames
      2. R1 added files (.py)
      3. R1 edited files (all)
      4. Non-test repo frames from error stack
    """
    targets: list[str] = []
    seen: set[str] = set()

    # Non-test repo frame files
    repo_frames: list[str] = []
    for frame in alignment.error_frames:
        fp = frame.split(":")[0] if ":" in frame else frame
        if fp and "test" not in fp.lower() and fp not in seen:
            seen.add(fp)
            repo_frames.append(fp)

    # Overlap: edited files ∩ repo frames → strongest signal
    for ef in alignment.patch_edited_files:
        if ef in seen:
            continue
        if any(ef == rf or ef.endswith("/" + rf) or rf.endswith(ef.split("/")[-1])
               for rf in repo_frames):
            targets.append(ef)
            seen.add(ef)

    # R1 added .py files
    for ef in alignment.patch_edited_files:
        if ef in seen:
            continue
        if ef.endswith(".py") and _is_new_file_per_file(ef, patch):
            targets.append(ef)
            seen.add(ef)

    # Rest of R1 edited files
    for ef in alignment.patch_edited_files:
        if ef not in seen:
            targets.append(ef)
            seen.add(ef)

    # Non-test repo frames not yet covered
    for rf in repo_frames:
        if rf not in seen:
            targets.append(rf)
            seen.add(rf)

    return targets[:5]


def _is_new_file_per_file(file_path: str, patch: PatchSignals) -> bool:
    """Check if a specific file is newly added based on patch metadata."""
    if not patch.edited_files or file_path not in patch.edited_files:
        return False
    # Fall back to global heuristic if per-file not available
    return patch.added_lines > 0 and patch.deleted_lines == 0



def _score_related_tests(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    events: list,
) -> tuple[float, list[_ScoreEvidence]]:
    """Score RELATED_TESTS deficiency."""
    score = 0.0
    evs: list[_ScoreEvidence] = []

    # +3 assertion expression exists
    for e in events:
        if e.assertion_line:
            score += 3.0
            evs.append(_ScoreEvidence(
                signal_id="ASSERT_EXISTS", weight=3.0,
                description="Failure contains an assertion expression",
                source=e.test_name,
            ))
            break

    # +2 allclose/almost_equal
    for e in events:
        if e.assertion_line and any(kw in e.assertion_line.lower()
                                     for kw in ["allclose", "almost_equal", "assert_equal"]):
            score += 2.0
            evs.append(_ScoreEvidence(
                signal_id="ASSERT_NUMERIC", weight=2.0,
                description="Numerical comparison assertion (allclose/almost_equal)",
                source=e.test_name,
            ))
            break

    # +2 multiple parameterized tests in same cluster
    param_tests = [e.test_name for e in events if "[" in e.test_name]
    if len(set(param_tests)) >= 2:
        score += 2.0
        evs.append(_ScoreEvidence(
            signal_id="PARAM_FAMILY", weight=2.0,
            description=f"Multiple parameterized tests in same cluster ({len(set(param_tests))})",
            source=param_tests[0],
        ))

    # +1 R1 didn't view related tests
    viewed_tests = [f for f in (cluster.test_names or [])
                    if any(f in vf for vf in alignment.trajectory_viewed_files)]
    if not viewed_tests and cluster.test_names:
        score += 1.0
        evs.append(_ScoreEvidence(
            signal_id="TESTS_NOT_VIEWED", weight=1.0,
            description="R1 did not view the failing tests",
            source=cluster.test_names[0],
        ))

    return score, evs


def _score_api_definition(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    events: list,
) -> tuple[float, list[_ScoreEvidence]]:
    """Score API_DEFINITION deficiency."""
    score = 0.0
    evs: list[_ScoreEvidence] = []

    # +3 unexpected keyword / AttributeError
    for e in events:
        if any(kw in e.message for kw in ["unexpected keyword", "AttributeError", "missing attribute",
                                            "has no attribute", "unknown attribute"]):
            score += 3.0
            evs.append(_ScoreEvidence(
                signal_id="UNEXPECTED_ARG_ATTR", weight=3.0,
                description=f"Error indicates unknown attribute/keyword: {e.message[:80]}",
                source=e.test_name,
            ))
            break

    # +2 message has specific class/attribute/method symbol
    for e in events:
        quoted = re.findall(r"""['"](\w+)['"]""", e.message)
        if quoted:
            score += 2.0
            evs.append(_ScoreEvidence(
                signal_id="QUOTED_SYMBOLS", weight=2.0,
                description=f"Error message references specific symbols: {quoted[:3]}",
                source=e.test_name,
            ))
            break

    # +1 failure site at constructor/property/descriptor
    for frame in alignment.error_frames:
        if any(kw in frame.lower() for kw in ["__init__", "__new__", "property", "descriptor"]):
            score += 1.0
            evs.append(_ScoreEvidence(
                signal_id="CONSTRUCTOR_FRAME", weight=1.0,
                description=f"Failure frame at constructor or descriptor: {frame}",
                source=frame,
            ))
            break

    return score, evs


def _score_interface_constraint(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    events: list,
) -> tuple[float, list[_ScoreEvidence]]:
    """Score INTERFACE_CONSTRAINT deficiency."""
    score = 0.0
    evs: list[_ScoreEvidence] = []

    # +3 unsupported operand / wrong argument type
    for e in events:
        if any(kw in e.message for kw in ["unsupported operand", "argument type",
                                            "cannot be interpreted", "not supported",
                                            "must be", "incompatible"]):
            score += 3.0
            evs.append(_ScoreEvidence(
                signal_id="TYPE_CONTRACT_VIOLATION", weight=3.0,
                description=f"Type or operand contract violation: {e.message[:80]}",
                source=e.test_name,
            ))
            break

    # +2 signature/argument count issue
    for e in events:
        if any(kw in e.message.lower() for kw in ["takes exactly", "missing required",
                                                    "positional argument", "got an unexpected",
                                                    "no matching", "argument"]):
            score += 2.0
            evs.append(_ScoreEvidence(
                signal_id="SIGNATURE_MISMATCH", weight=2.0,
                description=f"Call signature mismatch: {e.message[:80]}",
                source=e.test_name,
            ))
            break

    # +2 multiple tests at same interface boundary
    if cluster.count >= 2:
        same_file = len(set(
            e.call_chain[0].split(":")[0] if e.call_chain else ""
            for e in events if e.call_chain
        ))
        if same_file <= 2 and cluster.count >= 2:
            score += 2.0
            evs.append(_ScoreEvidence(
                signal_id="SAME_BOUNDARY_MULTI_TEST", weight=2.0,
                description=f"Multiple tests fail at same interface boundary ({cluster.count} tests)",
                source=cluster.test_names[0] if cluster.test_names else "?",
            ))

    return score, evs


def _score_caller_callee(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    events: list,
) -> tuple[float, list[_ScoreEvidence]]:
    """Score CALLER_CALLEE deficiency."""
    score = 0.0
    evs: list[_ScoreEvidence] = []

    # +3 call chain has 2+ non-test repo frames
    repo_chains = 0
    for e in events:
        for frame in e.call_chain:
            fp = frame.split(":")[0] if ":" in frame else frame
            if fp and "test" not in fp.lower() and not fp.startswith("<"):
                repo_chains += 1
                if repo_chains >= 2:
                    break
        if repo_chains >= 2:
            break
    if repo_chains >= 2:
        score += 3.0
        evs.append(_ScoreEvidence(
            signal_id="DEEP_CALL_CHAIN", weight=3.0,
            description="Multiple non-test frames in error call chain (caller→callee pattern)",
            source=cluster.events[0].test_name if cluster.events else "?",
        ))

    # +2 R1 edits caller, failure in callee
    patch_basenames = {f.split("/")[-1] for f in alignment.patch_edited_files}
    caller_frames = [f for f in alignment.shared_frames if f not in alignment.error_frames[:1]]
    if patch_basenames and caller_frames:
        score += 2.0
        evs.append(_ScoreEvidence(
            signal_id="EDIT_CALLER_FAIL_CALLEE", weight=2.0,
            description="R1 edited files overlap with caller frames, not the failure manifestation frame",
            source=", ".join(list(patch_basenames)[:2]),
        ))

    # +1 R1 modified wrapper/adapter/transform
    for ef in alignment.patch_edited_files:
        if any(kw in ef.lower() for kw in ["transform", "adapter", "wrapper",
                                             "intermediate", "base"]):
            score += 1.0
            evs.append(_ScoreEvidence(
                signal_id="WRAPPER_EDIT", weight=1.0,
                description=f"R1 modified a wrapper/intermediate file: {ef}",
                source=ef,
            ))
            break

    return score, evs


def _score_registration_site(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    patch: PatchSignals,
    events: list,
) -> tuple[float, list[_ScoreEvidence]]:
    """Score REGISTRATION_SITE deficiency."""
    score = 0.0
    evs: list[_ScoreEvidence] = []

    # +3 R1 patch adds decorator / registry call / graph registration
    if alignment.patch_introduced_new_file:
        score += 3.0
        evs.append(_ScoreEvidence(
            signal_id="NEW_FILE_ADDED", weight=3.0,
            description="R1 added a new .py file (likely registration or new module)",
            source=",".join(alignment.patch_edited_files[:3]),
        ))

    # +2 modifies __init__.py / exports / registry
    for ef in alignment.patch_edited_files:
        if ef.endswith("__init__.py") or "registry" in ef.lower() or "export" in ef.lower():
            score += 2.0
            evs.append(_ScoreEvidence(
                signal_id="INIT_OR_REGISTRY", weight=2.0,
                description=f"R1 modified init/registry file: {ef}",
                source=ef,
            ))
            break

    # +1 new file depends on import to be discovered
    new_files = [ef for ef in alignment.patch_edited_files
                 if ef.endswith(".py") and _is_new_file_per_file(ef, patch)]
    if new_files and any(ef.endswith("__init__.py") for ef in alignment.patch_edited_files):
        score += 1.0
        evs.append(_ScoreEvidence(
            signal_id="NEW_FILE_NEEDS_IMPORT", weight=1.0,
            description="New file added but depends on __init__ import to be discovered",
            source=new_files[0],
        ))

    return score, evs


def _score_localization_direction(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    events: list,
) -> tuple[float, list[_ScoreEvidence]]:
    """Score LOCALIZATION_DIRECTION deficiency."""
    score = 0.0
    evs: list[_ScoreEvidence] = []

    # +3 R1 edited files don't overlap with non-test failure frames
    repo_frame_files = set()
    for e in events:
        for frame in e.call_chain:
            fp = frame.split(":")[0] if ":" in frame else frame
            if fp and "test" not in fp.lower():
                repo_frame_files.add(fp)
    patch_basenames = {f.split("/")[-1] for f in alignment.patch_edited_files}
    frame_basenames = {f.split("/")[-1] for f in repo_frame_files if "/" in f}
    if patch_basenames and frame_basenames and not (patch_basenames & frame_basenames):
        score += 3.0
        evs.append(_ScoreEvidence(
            signal_id="EDIT_NO_FRAME_OVERLAP", weight=3.0,
            description="R1 edited files do not overlap with any non-test error frames",
            source=", ".join(list(alignment.patch_edited_files)[:3]),
        ))

    # +2 failure site not viewed by R1
    for e in events:
        if e.call_chain:
            fail_file = e.call_chain[0].split(":")[0] if ":" in e.call_chain[0] else e.call_chain[0]
            if not any(fail_file in vf for vf in alignment.trajectory_viewed_files):
                score += 2.0
                evs.append(_ScoreEvidence(
                    signal_id="FAIL_SITE_NOT_VIEWED", weight=2.0,
                    description=f"Failure site file not viewed by R1: {fail_file}",
                    source=e.test_name,
                ))
                break

    # +1 same error persists across multiple R1 attempts
    # (proxy: cluster has multiple events from same test)
    if cluster.count >= 2:
        same_test_count = len({e.test_name for e in events})
        if same_test_count < cluster.count:
            score += 1.0
            evs.append(_ScoreEvidence(
                signal_id="PERSISTENT_ERROR", weight=1.0,
                description="Same error survives multiple attempts (same test, different events)",
                source=events[0].test_name if events else "?",
            ))

    return score, evs


def _score_to_confidence(score: float, max_possible: float = 10.0) -> str:
    """Map a raw score to confidence label."""
    if score >= 6.0:
        return "high"
    elif score >= 3.0:
        return "medium"
    else:
        return "low"


def _render_scored_reason(
    primary: _DiagnosisCandidate,
    secondary: _DiagnosisCandidate | None,
) -> str:
    """Build a natural-language reason from scored diagnosis."""
    parts = [f"Diagnosis: {primary.subtype} (score={primary.score:.1f})."]
    if primary.supporting:
        ev = primary.supporting[0]
        parts.append(f"Signal [{ev.signal_id}]: {ev.description}")
    if secondary and secondary.score >= primary.score * 0.6:
        parts.append(f"Secondary: {secondary.subtype} (score={secondary.score:.1f})")
    if primary.candidate_edit_targets:
        parts.append(f"Target: {primary.candidate_edit_targets[0]}")
    return " ".join(parts)


def classify_cluster(
    cluster: FailureCluster,
    alignment: EvidenceAlignment,
    patch: PatchSignals | None = None,
) -> SubtypedDiagnosis:
    """Score-based deterministic diagnosis.

    Instead of first-match if/elif, each coarse deficiency type independently
    accumulates signals. Returns top-1 (and optional top-2) while preserving
    the existing SubtypedDiagnosis interface for downstream consumers.
    """
    from diagnosis.failure_event import FailureEvent

    events: list[FailureEvent] = list(cluster.events)
    key_location = cluster.root_cause or ""
    if not key_location and alignment.shared_frames:
        key_location = alignment.shared_frames[0]

    # ── Score each coarse type ──
    def _no_reg(c, a, ev):
        return (0.0, [])
    reg_scorer = (
        lambda c, a, ev: _score_registration_site(c, a, patch or _no_patch(), ev)
        if patch else _no_reg(c, a, ev)
    )

    scorers = [
        ("RELATED_TESTS", ContextDeficiencyType.RELATED_TESTS, _score_related_tests),
        ("API_DEFINITION", ContextDeficiencyType.API_DEFINITION, _score_api_definition),
        ("INTERFACE_CONSTRAINT", ContextDeficiencyType.INTERFACE_CONSTRAINT, _score_interface_constraint),
        ("CALLER_CALLEE", ContextDeficiencyType.CALLER_CALLEE, _score_caller_callee),
        ("REGISTRATION_SITE", ContextDeficiencyType.REGISTRATION_SITE, reg_scorer),
        ("LOCALIZATION_DIRECTION", ContextDeficiencyType.LOCALIZATION_DIRECTION, _score_localization_direction),
    ]

    candidates: list[_DiagnosisCandidate] = []
    for name, dtype, scorer_fn in scorers:
        s, ev_list = scorer_fn(cluster, alignment, events)
        candidates.append(_DiagnosisCandidate(
            deficiency_type=dtype,
            subtype=_SUBTYPE_MAP.get(dtype, "UNCLASSIFIED"),
            score=s,
            supporting=ev_list,
        ))

    # Sort by score descending
    candidates.sort(key=lambda c: c.score, reverse=True)

    # Normalize scores to [0, 1] range (cap at 10.0)
    max_possible = 10.0
    for c in candidates:
        c.score = min(c.score / max_possible, 1.0)

    # Top-1: always exists (even if all 0, we get lowest fallback)
    primary = candidates[0]
    best_score = primary.score

    # Secondary: score >= 60% of top-1
    secondary = None
    for c in candidates[1:]:
        if c.score >= best_score * 0.6 and c.score > 0:
            secondary = c
            break

    # ── Derive edit targets (grounded, independent of diagnosis) ──
    edit_targets = _derive_edit_targets(cluster, alignment, patch or _no_patch())

    # ── Target symbols ──
    target_symbols = list(alignment.error_symbols[:5])
    if not target_symbols and key_location:
        mod = key_location.split(":")[0].split("/")[-1].replace(".py", "")
        if mod:
            target_symbols = [mod]

    # ── Test names (from cluster) ──
    test_names = list(cluster.test_names or [])

    # Determine if actionable
    actionable = best_score >= 0.35
    if not actionable:
        # Floor: still provide grounded data, just mark as low confidence / UNCLASSIFIED
        final_type = ContextDeficiencyType.NO_RELIABLE_DEFICIENCY
        final_subtype = "UNCLASSIFIED"
        confidence = "low"
        reason_parts = ["Low-confidence diagnosis (best score below actionable threshold)."]
        if primary.supporting:
            reason_parts.append(f"Weakest signal: [{primary.supporting[0].signal_id}] {primary.supporting[0].description}")
        if edit_targets:
            reason_parts.append(f"Grounded edit targets available: {edit_targets[0]}")
            if len(edit_targets) > 1:
                reason_parts.append(f"+{len(edit_targets)-1} more")
        reason_parts.append("Proceeding with Actionability Floor — grounded evidence passed through despite uncertain diagnosis.")
        reason = " ".join(reason_parts)
    else:
        final_type = primary.deficiency_type
        final_subtype = primary.subtype
        confidence = _score_to_confidence(best_score * max_possible, max_possible)
        reason = _render_scored_reason(primary, secondary)

    # Build secondary candidates list (for audit)
    secondary_list = []
    for c in candidates[1:4]:
        if c.score > 0:
            secondary_list.append({
                "type": c.deficiency_type.value,
                "subtype": c.subtype,
                "score": round(c.score, 2),
                "n_signals": len(c.supporting),
            })

    # Supporting signal IDs
    signal_ids = [ev.signal_id for ev in primary.supporting]

    # Type scores dict (for audit)
    type_scores = {c.deficiency_type.value: round(c.score, 2) for c in candidates}

    # ── Uncertainty-Aware computations ──
    # Signal coverage: fraction of events with message, assertion, repo frame
    n_events = max(len(cluster.events), 1)
    has_msg = sum(1 for e in cluster.events if getattr(e, "message", "")) / n_events
    has_assert = sum(1 for e in cluster.events if getattr(e, "assertion_line", "")) / n_events
    has_frame = sum(1 for e in cluster.events if getattr(e, "call_chain", [])) / n_events
    coverage = (has_msg + has_assert + has_frame) / 3.0 if n_events > 0 else 0.0
    signal_coverage = min(coverage, 1.0)

    # Score margin
    sorted_scores = sorted(candidates, key=lambda c: c.score, reverse=True)
    top1_score = sorted_scores[0].score if sorted_scores else 0.0
    top2_score = sorted_scores[1].score if len(sorted_scores) > 1 else 0.0
    score_margin = top1_score - top2_score

    # Over-localization risk (0-7)
    or_risk = 0
    if signal_coverage < 0.60: or_risk += 2
    if score_margin < 0.15: or_risk += 2
    if edit_targets and any("patch" in t.lower() for t in edit_targets): or_risk += 1
    # Low primary score also increases risk
    if top1_score < 0.50: or_risk += 1
    if len(alignment.error_frames) <= 1: or_risk += 1

    # Scope mode
    if or_risk <= 1 and signal_coverage >= 0.60 and score_margin >= 0.15:
        scope_mode = "NARROW"
    elif or_risk >= 4:
        scope_mode = "EXPANDED"
    else:
        scope_mode = "BALANCED"

    return SubtypedDiagnosis(
        type=final_type,
        subtype=final_subtype,
        confidence=confidence,
        target_symbols=target_symbols,
        key_location=key_location,
        evidence_alignment=alignment,
        reason=reason,
        test_names=test_names,
        candidate_edit_targets=edit_targets,
        type_scores=type_scores,
        supporting_signal_ids=signal_ids[:10],
        secondary_candidates=secondary_list,
        signal_coverage=round(signal_coverage, 2),
        score_margin=round(score_margin, 2),
        over_localization_risk=or_risk,
        scope_mode=scope_mode,
    )


def _no_patch() -> PatchSignals:
    """Return an empty PatchSignals when none is available (standalone scoring)."""
    from recap.diagnosis.signals.schema import PatchSignals
    return PatchSignals(edited_files=[])


# ── Pipeline ────────────────────────────────────────────────────────


def reasoner_v2_diagnose(
    clusters: list[FailureCluster],
    patch: PatchSignals,
    trajectory: TrajectorySignals,
) -> list[SubtypedDiagnosis]:
    """Full P1-3B pipeline: align evidence → classify each cluster."""
    diagnoses: list[SubtypedDiagnosis] = []
    for cluster in clusters:
        alignment = align_evidence(cluster, patch, trajectory)
        diagnosis = classify_cluster(cluster, alignment, patch=patch)
        diagnoses.append(diagnosis)
    return diagnoses


def build_diagnosis_plan(
    diagnoses: list[SubtypedDiagnosis],
) -> dict[str, Any]:
    """Aggregate SubtypedDiagnoses into a plan dict (fed to Router + Reshaping)."""
    if not diagnoses:
        return {
            "primary": None,
            "all": [],
            "summary": "No deficiency diagnosed.",
        }

    # Sort by confidence
    priority = {"high": 0, "medium": 1, "low": 2}
    sorted_dx = sorted(diagnoses, key=lambda d: priority.get(d.confidence, 99))

    return {
        "primary": {
            "type": sorted_dx[0].type.value,
            "subtype": sorted_dx[0].subtype,
            "confidence": sorted_dx[0].confidence,
            "target_symbols": sorted_dx[0].target_symbols,
            "key_location": sorted_dx[0].key_location,
            "reason": sorted_dx[0].reason,
        },
        "all": [
            {
                "type": d.type.value,
                "subtype": d.subtype,
                "confidence": d.confidence,
                "target_symbols": d.target_symbols,
                "key_location": d.key_location,
            }
            for d in sorted_dx
        ],
        "summary": "; ".join(
            f"{d.subtype} ({d.confidence})" for d in sorted_dx[:3]
        ),
    }
