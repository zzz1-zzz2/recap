"""P1-3C: DiagnosisHypothesis — structured output of the ReasonerV2.

Each cluster produces one or more hypotheses describing what context
the agent is missing. A hypothesis is *evidence-linked*: every claim
points to specific cluster_ids, test_names, frames, or symbols.

Hypotheses are designed to be:
  - audit-friendly: every field traces back to source signals
  - shape-stable: schema is JSON-serializable for Shadow artifacts
  - action-ready: candidate_edit_targets / retrieval_targets drive
                   the SearchContract template engine
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from recap.diagnosis.taxonomy import ContextDeficiencyType


class HypothesisStatus(str, Enum):
    """Lifecycle of a hypothesis."""
    PROPOSED = "PROPOSED"
    CONFIRMED = "CONFIRMED"   # multiple independent signals agree
    REJECTED = "REJECTED"     # contradicting evidence wins
    ABSTAINED = "ABSTAINED"   # insufficient evidence to commit


@dataclass
class EvidenceReference:
    """A single evidence item linked to a hypothesis.

    `ref_id` is stable across runs (sha256 of canonical content) so
    evidence can be cross-referenced between hypotheses and search actions.
    """

    ref_id: str = ""
    kind: str = ""        # test_failure | stack_frame | patch_edit | trajectory_view | call_chain
    text: str = ""        # human-readable description
    source: str = ""       # file:line or test::name


@dataclass
class DiagnosisHypothesis:
    """One structured hypothesis about a failure cluster's context deficiency.

    Required fields are populated by the Reasoner. Optional fields are
    populated by Router/Reshaping once those layers run.
    """

    hypothesis_id: str = ""
    cluster_ids: list[str] = field(default_factory=list)
    deficiency_type: ContextDeficiencyType = ContextDeficiencyType.NO_RELIABLE_DEFICIENCY
    subtype: str = "UNCLASSIFIED"
    confidence: str = "low"  # low | medium | high
    uncertainty: float = 1.0  # 0.0 = certain, 1.0 = no idea
    status: HypothesisStatus = HypothesisStatus.PROPOSED

    # Where the failure manifests
    failure_sites: list[str] = field(default_factory=list)         # file:line
    test_names: list[str] = field(default_factory=list)            # bounded cluster test names

    # Where the agent should look for missing context
    retrieval_targets: list[str] = field(default_factory=list)     # symbols, files

    # Where the agent should consider editing
    candidate_edit_targets: list[str] = field(default_factory=list)  # files

    # Evidence ledger
    supporting_evidence_ids: list[str] = field(default_factory=list)
    contradicting_evidence_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)             # from R1 trajectory
    rejected_assumptions: list[str] = field(default_factory=list)    # what R1 got wrong

    # Free-text natural language (for human audit only; not consumed by code)
    statement: str = ""
    reason: str = ""

    # Source for traceability
    raw_signals_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "cluster_ids": self.cluster_ids,
            "deficiency_type": self.deficiency_type.value,
            "subtype": self.subtype,
            "confidence": self.confidence,
            "uncertainty": self.uncertainty,
            "status": self.status.value,
            "failure_sites": self.failure_sites,
            "test_names": self.test_names,
            "retrieval_targets": self.retrieval_targets,
            "candidate_edit_targets": self.candidate_edit_targets,
            "supporting_evidence_ids": self.supporting_evidence_ids,
            "contradicting_evidence_ids": self.contradicting_evidence_ids,
            "assumptions": self.assumptions,
            "rejected_assumptions": self.rejected_assumptions,
            "statement": self.statement,
            "reason": self.reason,
            "raw_signals_sha": self.raw_signals_sha,
        }


# ── Evidence ID helpers ─────────────────────────────────────────────


def make_evidence_id(kind: str, source: str, text: str) -> str:
    """Stable, content-addressed evidence ID."""
    raw = f"{kind}|{source}|{text}"
    return "E" + hashlib.sha256(raw.encode()).hexdigest()[:10]


def make_hypothesis_id(
    cluster_ids: list[str],
    deficiency_type: ContextDeficiencyType,
    subtype: str,
) -> str:
    """Stable hypothesis ID from cluster + type."""
    raw = "|".join(sorted(cluster_ids)) + "|" + deficiency_type.value + "|" + subtype
    return "H" + hashlib.sha256(raw.encode()).hexdigest()[:10]


# ── Conversion from SubtypedDiagnosis → DiagnosisHypothesis ──────────


def from_subtyped_diagnosis(
    sub,
    cluster_id: str,
    cluster_test_names: list[str],
    raw_signals_sha: str = "",
) -> DiagnosisHypothesis:
    """Wrap a SubtypedDiagnosis (P1-3B output) into a DiagnosisHypothesis.

    This is the bridge between the existing classify_cluster() output
    and the new evidence-linked structure. Repository-specific details
    (e.g. ITRS vs SpacetimeCoordinate) flow through unchanged.
    """
    supporting_ids = []
    if sub.target_symbols:
        for sym in sub.target_symbols:
            supporting_ids.append(
                make_evidence_id("target_symbol", f"{cluster_id}|{sub.key_location}", sym)
            )
    if sub.key_location:
        supporting_ids.append(
            make_evidence_id("failure_site", f"{cluster_id}|{sub.key_location}", sub.reason)
        )

    return DiagnosisHypothesis(
        hypothesis_id=make_hypothesis_id(
            [cluster_id], sub.type, sub.subtype,
        ),
        cluster_ids=[cluster_id],
        deficiency_type=sub.type,
        subtype=sub.subtype,
        confidence=sub.confidence,
        uncertainty={"high": 0.2, "medium": 0.5, "low": 0.9}.get(sub.confidence, 0.7),
        failure_sites=[sub.key_location] if sub.key_location else [],
        test_names=cluster_test_names,
        retrieval_targets=list(sub.target_symbols),
        candidate_edit_targets=list(sub.candidate_edit_targets),
        supporting_evidence_ids=supporting_ids,
        statement=sub.reason or sub.subtype,
        reason=sub.reason,
        raw_signals_sha=raw_signals_sha,
    )
