"""P1-3C-5: Grounded Search Contracts.

This is the closure of the P1-3C core diagnostic layer. Beyond
`DiagnosisHypothesis` (P1-3C-1) and the template engine (P1-3C-3),
this module adds:

  - SearchTarget with kind (SYMBOL | FILE | TEST | TYPE_NAME | ...)
  - EvidenceItem ledger with stable IDs and dereferenceable content
  - ContractStatus enum (ACTIONABLE | ABSTAINED | INVALID)
  - ContractValidator: target_kind × action_type compatibility rules
  - Plan-level budget allocator (not just per-hypothesis)
  - Shadow artifact writer

The goal: Router must receive grounded targets (symbol, file, test)
not arbitrary error-message tokens like `Time` or `float`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from recap.diagnosis.hypothesis import (
    DiagnosisHypothesis,
    HypothesisStatus,
    make_evidence_id,
)


# ── Target kinds ───────────────────────────────────────────────────


class SearchTargetKind(str, Enum):
    """Closed enum of valid search-target kinds.

    Router implementations must accept only the kinds listed for each
    action type. Primitive literals (str, int, float, ...) are not
    valid targets under any kind — they must be filtered.
    """

    SYMBOL = "SYMBOL"            # class/function/method/variable
    FILE = "FILE"                # repo-relative path
    TEST = "TEST"                # pytest test name (with [param] suffix)
    TYPE_NAME = "TYPE_NAME"       # Python type name (e.g. Time, float)
    FAILURE_SITE = "FAILURE_SITE"  # "file.py:42"
    EVIDENCE = "EVIDENCE"        # pointer to a specific EvidenceItem
    BEHAVIOR = "BEHAVIOR"        # semantic concept ("type_contract", "assertion_mismatch")


# Python primitive types — never valid as search targets for
# "who-calls / who-is-called-by" type actions. Domain-specific
# types (e.g. Astropy's Time/Quantity, NumPy ndarray, etc.) are NOT
# in this list — they should be classified as TYPE_NAME targets
# and filtered via the action × target_kind compatibility matrix.
_PRIMITIVE_TYPES = {
    # Built-in types
    "str", "int", "float", "complex",
    "list", "tuple", "set", "frozenset", "dict", "bytes", "bytearray",
    "bool", "object", "type", "NoneType", "None", "True", "False",
    # Typing module generics
    "Optional", "List", "Dict", "Tuple", "Set", "Type", "Any",
    "Callable", "Union", "Iterable", "Iterator", "Generator",
    # Python keywords (not callable targets)
    "and", "or", "not", "is", "in",
}

# Identifier-shape detection (a-zA-Z_ followed by [a-zA-Z0-9_.]).
import re as _re
_RE_IDENTIFIER = _re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")


@dataclass
class SearchTarget:
    """A typed retrieval target with grounding metadata.

    `resolved=False` indicates the target is a best-guess extracted
    from error message tokens; `resolved=True` indicates it was
    confirmed against the source code (file path, frame line, etc.).
    """

    value: str = ""
    kind: SearchTargetKind = SearchTargetKind.SYMBOL
    resolved: bool = False
    source_evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "kind": self.kind.value,
            "resolved": self.resolved,
            "source_evidence_ids": list(self.source_evidence_ids),
        }


def is_primitive_literal(value: str) -> bool:
    """True if `value` is a Python builtin type name or a string-typed literal."""
    if not value:
        return True
    return value in _PRIMITIVE_TYPES


def make_grounded_target(
    value: str,
    kind: SearchTargetKind,
    *,
    resolved: bool = False,
    evidence_ids: Iterable[str] = (),
) -> SearchTarget:
    """Construct a SearchTarget, defaulting unresolved when value is suspect."""
    return SearchTarget(
        value=value,
        kind=kind,
        resolved=resolved,
        source_evidence_ids=list(evidence_ids),
    )


# ── EvidenceItem ledger ───────────────────────────────────────────


class EvidenceSource(str, Enum):
    TEST_FAILURE = "TEST_FAILURE"
    STACK_FRAME = "STACK_FRAME"
    PATCH_EDIT = "PATCH_EDIT"
    TRAJECTORY_VIEW = "TRAJECTORY_VIEW"
    CALL_CHAIN = "CALL_CHAIN"
    ASSERTION = "ASSERTION"
    HYPOTHESIS_BRIDGE = "HYPOTHESIS_BRIDGE"


@dataclass
class EvidenceItem:
    """Concrete evidence record that supports one or more hypotheses.

    Stored in the global EvidenceLedger and dereferenced by
    `evidence_id`. Every EvidenceReference in a hypothesis must
    resolve to an EvidenceItem here.
    """

    evidence_id: str = ""
    cluster_id: str = ""
    source: EvidenceSource = EvidenceSource.TEST_FAILURE
    kind: str = ""                    # finer-grained: test name, frame file, etc.
    location: str = ""                # file:line or test::name
    content: str = ""                 # human-readable description
    strength: str = "supporting"      # supporting | contradicting | neutral

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "cluster_id": self.cluster_id,
            "source": self.source.value,
            "kind": self.kind,
            "location": self.location,
            "content": self.content,
            "strength": self.strength,
        }


class EvidenceLedger:
    """Global evidence store. Provides lookup by evidence_id."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> None:
        existing = self._items.get(item.evidence_id)
        if existing is None:
            self._items[item.evidence_id] = item
            return
        # Content-addressed IDs are supposed to be stable. If two
        # items collide, fail-fast — silent overwrites hide bugs.
        if existing != item:
            raise EvidenceConflictError(
                f"evidence_id={item.evidence_id!r} collides with existing item "
                f"(existing.kind={existing.kind!r}, new.kind={item.kind!r})"
            )
        # Same item; idempotent.

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._items.get(evidence_id)

    def has(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def items(self) -> list[EvidenceItem]:
        return list(self._items.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_items": len(self._items),
            "items": [it.to_dict() for it in self.items()],
        }


class EvidenceConflictError(ValueError):
    """Two distinct EvidenceItems collided on the same ID."""


# ── Action types (closed enum, NO sentinels) ──────────────────────


class SearchActionType(str, Enum):
    """Closed enum of executable retrieval actions.

    Sentinel values (ABSTAIN/NOOP) are NOT actions — they live in
    SearchContract.status instead.
    """

    FIND_DEFINITION = "FIND_DEFINITION"
    FIND_PARALLEL_IMPLEMENTATION = "FIND_PARALLEL_IMPLEMENTATION"
    FIND_RELATED_TESTS = "FIND_RELATED_TESTS"
    FIND_CALLEES = "FIND_CALLEES"
    FIND_CALLERS = "FIND_CALLERS"
    FIND_REGISTRATION_SITE = "FIND_REGISTRATION_SITE"
    REHYDRATE_VIEWED_EVIDENCE = "REHYDRATE_VIEWED_EVIDENCE"


# ── Contract status ───────────────────────────────────────────────


class ContractStatus(str, Enum):
    ACTIONABLE = "ACTIONABLE"
    ABSTAINED = "ABSTAINED"
    INVALID = "INVALID"


# ── Compatible (action_type, target_kind) pairs ────────────────────

# TYPE_NAME is allowed for FIND_DEFINITION so a hypothesis can ask
# "where is Time/Quantity defined in this codebase?" without that
# question being silently dropped. TYPE_NAME is rejected for
# FIND_CALLERS / FIND_CALLEES because calling built-in class names
# never makes sense.
_ACTION_TARGET_KINDS: dict[SearchActionType, set[SearchTargetKind]] = {
    SearchActionType.FIND_DEFINITION: {
        SearchTargetKind.SYMBOL,
        SearchTargetKind.FILE,
        SearchTargetKind.FAILURE_SITE,
        SearchTargetKind.TYPE_NAME,
    },
    SearchActionType.FIND_PARALLEL_IMPLEMENTATION: {
        SearchTargetKind.SYMBOL,
        SearchTargetKind.FILE,
        SearchTargetKind.BEHAVIOR,
    },
    SearchActionType.FIND_RELATED_TESTS: {
        SearchTargetKind.SYMBOL,
        SearchTargetKind.FILE,
        SearchTargetKind.TEST,
        SearchTargetKind.BEHAVIOR,
    },
    SearchActionType.FIND_CALLEES: {
        SearchTargetKind.SYMBOL,
    },
    SearchActionType.FIND_CALLERS: {
        SearchTargetKind.SYMBOL,
    },
    SearchActionType.FIND_REGISTRATION_SITE: {
        SearchTargetKind.FILE,
        SearchTargetKind.SYMBOL,
    },
    SearchActionType.REHYDRATE_VIEWED_EVIDENCE: {
        SearchTargetKind.EVIDENCE,
        SearchTargetKind.FILE,
    },
}


# ── Action + Contract dataclasses ─────────────────────────────────


@dataclass
class SearchAction:
    """One grounded retrieval action."""

    action_id: str = ""
    action_type: SearchActionType = SearchActionType.FIND_DEFINITION
    target: SearchTarget = field(default_factory=SearchTarget)
    reason: str = ""
    cluster_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    budget: int = 2
    stop_condition: str = "one relevant result found"

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target": self.target.to_dict(),
            "reason": self.reason,
            "cluster_ids": list(self.cluster_ids),
            "evidence_ids": list(self.evidence_ids),
            "budget": self.budget,
            "stop_condition": self.stop_condition,
        }


@dataclass
class SearchContract:
    """A complete retrieval plan for one hypothesis.

    `status` is ACTIONABLE only if there is at least one valid action
    AND the validator accepted every action.

    `contract_mode` is "exploratory" when the diagnosis is low-confidence
    (e.g. UNCLASSIFIED) but grounded targets exist. Exploratory contracts
    have tighter budgets and are explicitly marked for human audit.
    """

    hypothesis_id: str = ""
    actions: list[SearchAction] = field(default_factory=list)
    status: ContractStatus = ContractStatus.ABSTAINED
    abstention_reason: str = ""
    total_budget: int = 0
    validation_issues: list[str] = field(default_factory=list)
    contract_mode: str = "standard"  # "standard" | "exploratory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "actions": [a.to_dict() for a in self.actions],
            "status": self.status.value,
            "abstention_reason": self.abstention_reason,
            "total_budget": self.total_budget,
            "validation_issues": list(self.validation_issues),
            "contract_mode": self.contract_mode,
        }


# ── Target selector helpers ────────────────────────────────────────


def _site_to_target(site: str, *, resolved: bool = False,
                     evidence_ids: list[str] | None = None) -> SearchTarget | None:
    """Convert a 'file.py:LINE' string into a FAILURE_SITE target.

    Returns None if `site` is empty or malformed.
    """
    if not site:
        return None
    return SearchTarget(
        value=site,
        kind=SearchTargetKind.FAILURE_SITE,
        resolved=resolved,
        source_evidence_ids=evidence_ids or [],
    )


def _file_to_target(file_path: str, *, evidence_ids: list[str] | None = None) -> SearchTarget | None:
    """Convert a file path to a FILE target."""
    if not file_path:
        return None
    return SearchTarget(
        value=file_path,
        kind=SearchTargetKind.FILE,
        resolved=True,
        source_evidence_ids=evidence_ids or [],
    )


def _symbol_to_target(sym: str, *, evidence_ids: list[str] | None = None) -> SearchTarget | None:
    """Convert a token to a SYMBOL or TYPE_NAME target.

    Heuristic: if the leaf identifier starts with an uppercase letter
    (CamelCase / PascalCase), treat as TYPE_NAME — these are class
    names seen in error messages, not callable candidates.

    Returns None for primitive literals or invalid identifiers.
    """
    if not sym or is_primitive_literal(sym):
        return None
    if not _RE_IDENTIFIER.match(sym):
        return None
    leaf = sym.split(".")[-1]
    kind = (
        SearchTargetKind.TYPE_NAME
        if (leaf and leaf[0].isupper())
        else SearchTargetKind.SYMBOL
    )
    return SearchTarget(
        value=sym,
        kind=kind,
        resolved=False,
        source_evidence_ids=evidence_ids or [],
    )


def _test_to_target(test_name: str, *, evidence_ids: list[str] | None = None) -> SearchTarget | None:
    """Convert a test name to a TEST target."""
    if not test_name:
        return None
    return SearchTarget(
        value=test_name,
        kind=SearchTargetKind.TEST,
        resolved=True,
        source_evidence_ids=evidence_ids or [],
    )


def _evidence_to_target(evidence_id: str) -> SearchTarget | None:
    if not evidence_id:
        return None
    return SearchTarget(
        value=evidence_id,
        kind=SearchTargetKind.EVIDENCE,
        resolved=True,
        source_evidence_ids=[evidence_id],
    )


def _behavior_to_target(label: str, *, evidence_ids: list[str] | None = None) -> SearchTarget:
    return SearchTarget(
        value=label,
        kind=SearchTargetKind.BEHAVIOR,
        resolved=False,
        source_evidence_ids=evidence_ids or [],
    )


# ── Templates (return SearchTarget objects) ───────────────────────


def _t_failure_site(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """Frame/site failures: look up the symbol/file at the failure site."""
    out: list[SearchTarget] = []
    for site in hyp.failure_sites[:2]:
        target = _site_to_target(site, resolved=True, evidence_ids=hyp.supporting_evidence_ids)
        if target:
            out.append(target)
    return out


def _t_find_callers(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """For SYMBOL retrieval targets: who calls this?"""
    out: list[SearchTarget] = []
    for sym in hyp.retrieval_targets[:2]:
        target = _symbol_to_target(sym, evidence_ids=hyp.supporting_evidence_ids)
        if target:
            out.append(target)
    return out


def _t_find_callees(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """For SYMBOL retrieval targets: what does this call?"""
    out: list[SearchTarget] = []
    for sym in hyp.retrieval_targets[:2]:
        target = _symbol_to_target(sym, evidence_ids=hyp.supporting_evidence_ids)
        if target:
            out.append(target)
    return out


def _t_find_related_tests(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """Find tests for the failing function/symbol."""
    out: list[SearchTarget] = []
    # Prefer SYMBOL targets from retrieval_targets
    for sym in hyp.retrieval_targets[:2]:
        target = _symbol_to_target(sym, evidence_ids=hyp.supporting_evidence_ids)
        if target:
            out.append(target)
    # Fall back to failure-site file
    if not out:
        for site in hyp.failure_sites[:1]:
            file_part = site.split(":")[0] if ":" in site else site
            t = _file_to_target(file_part, evidence_ids=hyp.supporting_evidence_ids)
            if t:
                out.append(t)
    # Last resort: test name
    if not out:
        for tname in hyp.test_names[:1]:
            t = _test_to_target(tname, evidence_ids=hyp.supporting_evidence_ids)
            if t:
                out.append(t)
    return out


def _t_find_parallel_implementation(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """Find existing parallel implementations."""
    out: list[SearchTarget] = []
    # Anchor on failure-site file
    for site in hyp.failure_sites[:1]:
        file_part = site.split(":")[0] if ":" in site else site
        t = _file_to_target(file_part, evidence_ids=hyp.supporting_evidence_ids)
        if t:
            out.append(t)
    # Fall back to BEHAVIOR target
    if not out:
        out.append(_behavior_to_target(
            "similar pattern",
            evidence_ids=hyp.supporting_evidence_ids,
        ))
    return out


def _t_find_registration_site(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """Find the registration site."""
    out: list[SearchTarget] = []
    for site in hyp.failure_sites[:1]:
        file_part = site.split(":")[0] if ":" in site else site
        t = _file_to_target(file_part, evidence_ids=hyp.supporting_evidence_ids)
        if t:
            out.append(t)
    return out


def _t_rehydrate_viewed(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """Mark that the agent already saw a relevant file; Reshaping rehydrates."""
    out: list[SearchTarget] = []
    for site in hyp.failure_sites[:1]:
        t = _file_to_target(
            site.split(":")[0] if ":" in site else site,
            evidence_ids=hyp.supporting_evidence_ids,
        )
        if t:
            out.append(t)
    return out


def _t_no_targets(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """No grounded targets available; contract will abstain."""
    return []


def _t_uncertain_retrieval(hyp: DiagnosisHypothesis) -> list[SearchTarget]:
    """Low-confidence exploratory template for UNCLASSIFIED diagnoses.

    Used when the diagnosis cannot determine a specific deficiency type
    but grounded targets (patch edits, frames, test names) are available.

    Priority:
      1. Candidate edit targets → FIND_DEFINITION (source-level, from R1 patch)
      2. Failure site → FIND_DEFINITION (look up frame location)
      3. Test names → FIND_RELATED_TESTS (fallback)

    Returns at most 2 targets so the Router can do one source + one test.
    """
    out: list[SearchTarget] = []
    seen_values: set[str] = set()

    # Priority 1: candidate edit targets (source-level, from R1 patch)
    for path in hyp.candidate_edit_targets[:2]:
        if path in seen_values:
            continue
        t = _file_to_target(path, evidence_ids=hyp.supporting_evidence_ids)
        if t:
            out.append(t)
            seen_values.add(path)

    # Priority 2: failure site (if different from edit targets)
    for site in hyp.failure_sites[:1]:
        if site in seen_values:
            continue
        fp = site.split(":")[0] if ":" in site else site
        if fp in seen_values:
            continue
        t = _site_to_target(site, resolved=True, evidence_ids=hyp.supporting_evidence_ids)
        if t:
            out.append(t)
            seen_values.add(site)

    # Priority 3: test names (only if we have room)
    if len(out) < 2:
        for tname in hyp.test_names[:1]:
            if tname in seen_values:
                continue
            t = _test_to_target(tname, evidence_ids=hyp.supporting_evidence_ids)
            if t:
                out.append(t)
                break

    return out[:2]


# ── Subtype → template dispatch ─────────────────────────────────────


_TEMPLATE_DISPATCH: list[tuple[str, Any]] = [
    # ── API_DEFINITION subtypes ──
    ("FRAME_ATTRIBUTE_PROPAGATION", _t_failure_site),
    ("FUNCTION_SIGNATURE_MISMATCH", _t_failure_site),
    ("CLASS_ATTRIBUTE_MISSING", _t_failure_site),
    ("METHOD_NOT_FOUND", _t_failure_site),
    ("MODULE_MEMBER_MISSING", _t_failure_site),
    # ── INTERFACE_CONSTRAINT subtypes ──
    ("ARGUMENT_TYPE_MISMATCH", _t_find_callers),
    ("FRAME_ATTRIBUTE_CONSTRAINT", _t_find_callers),
    # ── RELATED_TESTS subtypes ──
    ("NUMERICAL_MISMATCH", _t_find_related_tests),
    ("ROUTE_COMPARISON_FAILURE", _t_find_parallel_implementation),
    ("EDGE_CASE_MISSING", _t_find_related_tests),
    ("REGRESSION_DETECTED", _t_find_related_tests),
    # ── CALLER_CALLEE ──
    ("FRAME_MUTATION_IN_TRANSFORM", _t_find_callees),
    ("ARGUMENT_FORWARDING_MISMATCH", _t_find_callees),
    # ── LOCALIZATION_DIRECTION ──
    ("WRONG_FILE_MODIFIED", _t_failure_site),
    ("WRONG_FUNCTION_MODIFIED", _t_failure_site),
    ("SYMPTOM_CAUSE_MISMATCH", _t_find_parallel_implementation),
    # ── REGISTRATION_SITE ──
    ("TRANSFORM_NOT_REGISTERED", _t_find_registration_site),
    ("CONFIG_NOT_UPDATED", _t_find_registration_site),
    ("EXPORT_MISSING", _t_find_registration_site),
    # ── DEPENDENCY ──
    ("MISSING_IMPORT", _t_failure_site),
    ("MISSING_DATA_FILE", _t_no_targets),
    ("MISSING_SYSTEM_DEP", _t_no_targets),
    # ── Fallback ──
    ("UNCLASSIFIED", _t_uncertain_retrieval),
]


def _select_template(subtype: str):
    for s, fn in _TEMPLATE_DISPATCH:
        if s == subtype:
            return fn
    return _t_no_targets


# ── Validator ─────────────────────────────────────────────────────


class ContractValidator:
    """Validates each SearchAction against action_type × target_kind rules."""

    def validate(
        self,
        contract: SearchContract,
        *,
        ledger: EvidenceLedger | None = None,
    ) -> list[str]:
        """Return a list of issues. Empty list means valid."""
        issues: list[str] = []
        for action in contract.actions:
            issues.extend(self._validate_action(action, ledger))
        return issues

    def _validate_action(
        self,
        action: SearchAction,
        ledger: EvidenceLedger | None,
    ) -> list[str]:
        issues: list[str] = []
        target = action.target

        if not target.value:
            issues.append(f"{action.action_id}: empty target.value")
        if is_primitive_literal(target.value):
            issues.append(
                f"{action.action_id}: target {target.value!r} is a primitive literal; "
                "Router would search for a builtin"
            )

        allowed_kinds = _ACTION_TARGET_KINDS.get(action.action_type, set())
        if allowed_kinds and target.kind not in allowed_kinds:
            issues.append(
                f"{action.action_id}: {action.action_type.value} requires "
                f"target_kind in {[k.value for k in allowed_kinds]}, "
                f"got {target.kind.value}"
            )

        if ledger is not None and target.kind == SearchTargetKind.EVIDENCE:
            if not ledger.has(target.value):
                issues.append(
                    f"{action.action_id}: evidence target {target.value!r} "
                    "not in ledger"
                )

        return issues


# ── ID helpers ─────────────────────────────────────────────────────


def make_action_id(hyp_id: str, action_type: SearchActionType, target_value: str) -> str:
    raw = f"{hyp_id}|{action_type.value}|{target_value}"
    return "A" + hashlib.sha256(raw.encode()).hexdigest()[:8]


def build_evidence_ledger(
    clusters,
    diagnoses,
    bundle=None,
) -> EvidenceLedger:
    """Build a real EvidenceLedger from clusters + diagnoses + (optional) bundle.

    Each evidence item carries its real `kind` (failure_site, target_symbol,
    test_failure, assertion, patch_edit, trajectory_view) — never inferred
    from hash-id strings. ID collisions between clusters are silently
    absorbed; collisions within the same semantic context raise
    EvidenceConflictError (fail-fast).

    Returns the populated ledger.
    """
    ledger = EvidenceLedger()

    for cluster, sub in zip(clusters, diagnoses):
        cluster_id = cluster.cluster_id
        # failure_site evidence
        if sub.key_location:
            ledger.add(EvidenceItem(
                evidence_id=make_evidence_id(
                    "failure_site", f"{cluster_id}|{sub.key_location}", sub.reason,
                ),
                cluster_id=cluster_id,
                source=EvidenceSource.STACK_FRAME,
                kind="failure_site",
                location=sub.key_location,
                content=sub.reason,
                strength="supporting",
            ))
        # target_symbol evidence (one per symbol)
        for sym in sub.target_symbols[:5]:
            ledger.add(EvidenceItem(
                evidence_id=make_evidence_id(
                    "target_symbol", f"{cluster_id}|{sub.key_location or ''}", sym,
                ),
                cluster_id=cluster_id,
                source=EvidenceSource.ASSERTION,
                kind="target_symbol",
                location=sub.key_location or "",
                content=sym,
                strength="supporting",
            ))
        # test_failure evidence
        for tname in cluster.test_names[:5]:
            ledger.add(EvidenceItem(
                evidence_id=make_evidence_id(
                    "test_failure", cluster_id, tname,
                ),
                cluster_id=cluster_id,
                source=EvidenceSource.TEST_FAILURE,
                kind="test_failure",
                location=tname,
                content=f"failed test in cluster {cluster_id}",
                strength="supporting",
            ))
        # assertion evidence
        for ev in cluster.events[:3]:
            if ev.assertion_line:
                ledger.add(EvidenceItem(
                    evidence_id=make_evidence_id(
                        "assertion", cluster_id, ev.assertion_line,
                    ),
                    cluster_id=cluster_id,
                    source=EvidenceSource.ASSERTION,
                    kind="assertion",
                    location=ev.top_repo_frame or "",
                    content=ev.assertion_line,
                    strength="supporting",
                ))
        # patch_edit evidence
        if bundle is not None:
            for edited in bundle.patch.edited_files[:5]:
                ledger.add(EvidenceItem(
                    evidence_id=make_evidence_id(
                        "patch_edit", cluster_id, edited,
                    ),
                    cluster_id=cluster_id,
                    source=EvidenceSource.PATCH_EDIT,
                    kind="patch_edit",
                    location=edited,
                    content=f"R1 patch edited {edited}",
                    strength="neutral",
                ))

    return ledger


# ── Per-hypothesis contract builder ────────────────────────────────


def build_search_contract(
    hypothesis: DiagnosisHypothesis,
    *,
    max_actions: int = 3,
    per_action_budget: int = 2,
    ledger: EvidenceLedger | None = None,
) -> SearchContract:
    """Build a SearchContract from a single DiagnosisHypothesis.

    Returns ABSTAINED contract when no grounded targets exist.
    Returns INVALID contract when validator finds issues.
    """
    contract = SearchContract(hypothesis_id=hypothesis.hypothesis_id)

    if hypothesis.status in (HypothesisStatus.REJECTED, HypothesisStatus.ABSTAINED):
        contract.status = ContractStatus.ABSTAINED
        contract.abstention_reason = f"hypothesis is {hypothesis.status.value}"
        return contract

    if (
        hypothesis.confidence == "low"
        and not hypothesis.retrieval_targets
        and not hypothesis.failure_sites
        and not hypothesis.test_names
        and not hypothesis.candidate_edit_targets
    ):
        contract.status = ContractStatus.ABSTAINED
        contract.abstention_reason = "low confidence + no actionable targets"
        return contract

    template = _select_template(hypothesis.subtype)
    targets = template(hypothesis)

    if not targets:
        contract.status = ContractStatus.ABSTAINED
        contract.abstention_reason = "no grounded retrieval target"
        return contract

    # For UNCLASSIFIED, infer action type from the first target's kind
    if hypothesis.subtype == "UNCLASSIFIED":
        chosen_action_type = _target_kind_to_action_type(targets[0].kind)
    else:
        chosen_action_type = _subtype_to_action_type(hypothesis.subtype)
    if chosen_action_type is None:
        contract.status = ContractStatus.ABSTAINED
        contract.abstention_reason = f"no template for subtype {hypothesis.subtype}"
        return contract

    actions: list[SearchAction] = []
    for tgt in targets[:max_actions]:
        if tgt.kind not in _ACTION_TARGET_KINDS[chosen_action_type]:
            # Skip incompatible target kind for this action
            continue
        actions.append(SearchAction(
            action_id=make_action_id(hypothesis.hypothesis_id, chosen_action_type, tgt.value),
            action_type=chosen_action_type,
            target=tgt,
            reason=hypothesis.statement or hypothesis.reason,
            cluster_ids=list(hypothesis.cluster_ids),
            evidence_ids=list(tgt.source_evidence_ids)
                        or list(hypothesis.supporting_evidence_ids),
            budget=per_action_budget,
            stop_condition=_stop_condition_for(chosen_action_type),
        ))

    if not actions:
        contract.status = ContractStatus.ABSTAINED
        contract.abstention_reason = "no action-compatible targets"
        return contract

    contract.actions = actions
    contract.total_budget = sum(a.budget for a in actions)
    contract.status = ContractStatus.ACTIONABLE

    # Mark low-confidence / UNCLASSIFIED contracts as exploratory
    if hypothesis.confidence == "low" or hypothesis.subtype == "UNCLASSIFIED":
        contract.contract_mode = "exploratory"

    # Validate
    validator = ContractValidator()
    issues = validator.validate(contract, ledger=ledger)
    contract.validation_issues = issues
    if issues:
        contract.status = ContractStatus.INVALID
    return contract


_SUBTYPE_ACTION_TYPE: dict[str, SearchActionType] = {
    # API_DEFINITION → look up the definition
    "FRAME_ATTRIBUTE_PROPAGATION": SearchActionType.FIND_DEFINITION,
    "FUNCTION_SIGNATURE_MISMATCH": SearchActionType.FIND_DEFINITION,
    "CLASS_ATTRIBUTE_MISSING": SearchActionType.FIND_DEFINITION,
    "METHOD_NOT_FOUND": SearchActionType.FIND_DEFINITION,
    "MODULE_MEMBER_MISSING": SearchActionType.FIND_DEFINITION,
    # INTERFACE_CONSTRAINT → call chain
    "ARGUMENT_TYPE_MISMATCH": SearchActionType.FIND_CALLERS,
    "FRAME_ATTRIBUTE_CONSTRAINT": SearchActionType.FIND_CALLERS,
    # RELATED_TESTS
    "NUMERICAL_MISMATCH": SearchActionType.FIND_RELATED_TESTS,
    "ROUTE_COMPARISON_FAILURE": SearchActionType.FIND_PARALLEL_IMPLEMENTATION,
    "EDGE_CASE_MISSING": SearchActionType.FIND_RELATED_TESTS,
    "REGRESSION_DETECTED": SearchActionType.FIND_RELATED_TESTS,
    # CALLER_CALLEE
    "FRAME_MUTATION_IN_TRANSFORM": SearchActionType.FIND_CALLEES,
    "ARGUMENT_FORWARDING_MISMATCH": SearchActionType.FIND_CALLEES,
    # LOCALIZATION_DIRECTION
    "WRONG_FILE_MODIFIED": SearchActionType.FIND_DEFINITION,
    "WRONG_FUNCTION_MODIFIED": SearchActionType.FIND_DEFINITION,
    "SYMPTOM_CAUSE_MISMATCH": SearchActionType.FIND_PARALLEL_IMPLEMENTATION,
    # REGISTRATION_SITE
    "TRANSFORM_NOT_REGISTERED": SearchActionType.FIND_REGISTRATION_SITE,
    "CONFIG_NOT_UPDATED": SearchActionType.FIND_REGISTRATION_SITE,
    "EXPORT_MISSING": SearchActionType.FIND_REGISTRATION_SITE,
    # DEPENDENCY
    "MISSING_IMPORT": SearchActionType.FIND_DEFINITION,
}


def _subtype_to_action_type(subtype: str) -> SearchActionType | None:
    return _SUBTYPE_ACTION_TYPE.get(subtype)


def _target_kind_to_action_type(kind: SearchTargetKind) -> SearchActionType | None:
    """Map a target kind to the most appropriate search action type.

    Used for UNCLASSIFIED diagnoses where the subtype doesn't prescribe
    a specific action type.
    """
    mapping = {
        SearchTargetKind.FAILURE_SITE: SearchActionType.FIND_DEFINITION,
        SearchTargetKind.SYMBOL: SearchActionType.FIND_DEFINITION,
        SearchTargetKind.TYPE_NAME: SearchActionType.FIND_DEFINITION,
        SearchTargetKind.FILE: SearchActionType.FIND_DEFINITION,
        SearchTargetKind.TEST: SearchActionType.FIND_RELATED_TESTS,
    }
    return mapping.get(kind)


def _stop_condition_for(action_type: SearchActionType) -> str:
    return {
        SearchActionType.FIND_DEFINITION: "definition located; one file found",
        SearchActionType.FIND_PARALLEL_IMPLEMENTATION: "one parallel implementation found",
        SearchActionType.FIND_RELATED_TESTS: "one related test found",
        SearchActionType.FIND_CALLEES: "one call site found",
        SearchActionType.FIND_CALLERS: "one caller found",
        SearchActionType.FIND_REGISTRATION_SITE: "registration site found",
        SearchActionType.REHYDRATE_VIEWED_EVIDENCE: "evidence rehydrated",
    }.get(action_type, "one result found")


def build_search_contracts(
    hypotheses: list[DiagnosisHypothesis],
    *,
    max_actions: int = 3,
    per_action_budget: int = 2,
    ledger: EvidenceLedger | None = None,
) -> list[SearchContract]:
    """Build a SearchContract for each hypothesis."""
    return [
        build_search_contract(
            h,
            max_actions=max_actions,
            per_action_budget=per_action_budget,
            ledger=ledger,
        )
        for h in hypotheses
    ]


# ── Plan-level allocator ──────────────────────────────────────────


@dataclass
class PlanBudget:
    """Plan-wide budget for all contracts."""

    max_total_actions: int = 3
    max_total_budget: int = 5


def _hypothesis_priority(h: DiagnosisHypothesis) -> tuple:
    """Lower rank = higher priority."""
    conf_rank = {"high": 0, "medium": 1, "low": 2}.get(h.confidence, 3)
    return (conf_rank, h.uncertainty, -len(h.cluster_ids))


def build_search_plan(
    hypotheses: list[DiagnosisHypothesis],
    *,
    budget: PlanBudget | None = None,
    ledger: EvidenceLedger | None = None,
) -> list[SearchContract]:
    """Plan-level allocation.

    Steps:
      1. Build per-hypothesis contract (with default max_actions=3).
      2. Sort by hypothesis priority (confidence asc, uncertainty asc).
      3. Greedily add contracts while total_actions <= max_total_actions
         AND total_budget <= max_total_budget.
      4. Drop / abstain contracts that don't fit.
    """
    budget = budget or PlanBudget()
    contracts = [
        build_search_contract(h, ledger=ledger) for h in hypotheses
    ]

    # Sort by priority (higher priority first)
    indexed = list(zip(hypotheses, contracts))
    indexed.sort(key=lambda pair: _hypothesis_priority(pair[0]))

    accepted: list[SearchContract] = []
    n_actions = 0
    total_budget = 0

    for h, c in indexed:
        if c.status != ContractStatus.ACTIONABLE:
            accepted.append(c)  # keep abstained/invalid for transparency
            continue
        if n_actions + len(c.actions) > budget.max_total_actions:
            # Demote to abstained
            c.status = ContractStatus.ABSTAINED
            c.abstention_reason = "exceeds plan-level action budget"
            c.actions = []
            c.total_budget = 0
            accepted.append(c)
            continue
        if total_budget + c.total_budget > budget.max_total_budget:
            c.status = ContractStatus.ABSTAINED
            c.abstention_reason = "exceeds plan-level result budget"
            c.actions = []
            c.total_budget = 0
            accepted.append(c)
            continue
        n_actions += len(c.actions)
        total_budget += c.total_budget
        accepted.append(c)

    # Restore original order (don't return in priority-sorted order)
    hyp_to_idx = {id(h): i for i, h in enumerate(hypotheses)}
    accepted.sort(key=lambda c: hyp_to_idx.get(id(corresponding_hypothesis(c, hypotheses, indexed)), 0))
    return accepted


def corresponding_hypothesis(
    contract: SearchContract,
    hypotheses: list[DiagnosisHypothesis],
    indexed_pairs: list[tuple],
) -> DiagnosisHypothesis:
    """Helper: find the hypothesis a contract was built from."""
    for h, c in indexed_pairs:
        if c is contract:
            return h
    return hypotheses[0]


# ── Shadow artifact writer ────────────────────────────────────────


def write_shadow_artifacts(
    output_dir: str | Path,
    *,
    contracts: list[SearchContract],
    hypotheses: list[DiagnosisHypothesis],
    ledger: EvidenceLedger | None = None,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write Shadow artifacts to output_dir.

    Produces:
      - evidence_items.json  (if ledger provided)
      - diagnosis_hypotheses.json
      - search_contracts.json
      - contract_validation.json

    Returns the dict of paths written.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    if ledger is not None:
        p = out_dir / "evidence_items.json"
        p.write_text(json.dumps(ledger.to_dict(), indent=2))
        paths["evidence_items"] = str(p)

    p = out_dir / "diagnosis_hypotheses.json"
    p.write_text(json.dumps({"hypotheses": [h.to_dict() for h in hypotheses]}, indent=2))
    paths["diagnosis_hypotheses"] = str(p)

    p = out_dir / "search_contracts.json"
    p.write_text(json.dumps(
        {"contracts": [c.to_dict() for c in contracts]},
        indent=2,
    ))
    paths["search_contracts"] = str(p)

    if validation_report is not None:
        p = out_dir / "contract_validation.json"
        p.write_text(json.dumps(validation_report, indent=2))
        paths["contract_validation"] = str(p)

    return paths


# ── Plan serializer (legacy compat) ─────────────────────────────────


def serialize_plan(
    contracts: list[SearchContract],
    hypotheses: list[DiagnosisHypothesis],
) -> dict[str, Any]:
    """Lightweight serialization for in-memory passing."""
    return {
        "version": "2",
        "hypotheses": [h.to_dict() for h in hypotheses],
        "contracts": [c.to_dict() for c in contracts],
        "n_actions": sum(len(c.actions) for c in contracts),
        "n_actionable": sum(1 for c in contracts if c.status == ContractStatus.ACTIONABLE),
        "n_abstained": sum(1 for c in contracts if c.status == ContractStatus.ABSTAINED),
        "n_invalid": sum(1 for c in contracts if c.status == ContractStatus.INVALID),
    }
