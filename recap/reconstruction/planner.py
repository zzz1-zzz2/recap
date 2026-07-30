"""Plan — compile diagnoses into structured reconstruction operations.

Produces (K_plus, K_minus, Q) as ContextUnit lists, not bare file paths.

K_plus:  units to preserve or rehydrate
K_minus:  context to suppress (assumptions + hard-drop file refs)
Q:   grounded search contracts for acquisition
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from recap.diagnosis.search_contract import SearchContract
from recap.reconstruction.context_unit import ContextUnit, make_unit

logger = logging.getLogger("condiag.reconstruction.planner")


@dataclasses.dataclass
class ReconstructionPlan:
    """Structured output of the Plan step.

    K_plus_units: preserve/rehydrate context units (prioritised).
    K_minus_assumptions: invalidated assumptions → N_forbid.
    K_minus_files: files whose trajectory content should be hard-dropped.
    Q_contracts: search contracts for acquisition.
    """
    K_plus_units: list[ContextUnit] = dataclasses.field(default_factory=list)
    K_minus_assumptions: list[str] = dataclasses.field(default_factory=list)
    K_minus_files: list[str] = dataclasses.field(default_factory=list)
    Q_contracts: list[SearchContract] = dataclasses.field(default_factory=list)
    # Summary for logging
    primary_edit_target: str = ""
    summary: str = ""


def build_plan(
    clusters: list,
    refined_diagnoses: list,
    attributions: list,
    revision_contract: dict,
    hypotheses: list,
    search_contracts: list[SearchContract],
) -> ReconstructionPlan:
    """Compile diagnoses into structured plan with ContextUnits."""
    plan = ReconstructionPlan()
    plan.Q_contracts = list(search_contracts)
    seen_paths: set[str] = set()

    # ── MANDATORY units (always survive packing) ──
    # FailureWitness and primary edit target are mandatory
    primary = revision_contract.get("primary_edit_target", {})
    if isinstance(primary, dict):
        pp = primary.get("path", "")
    else:
        pp = str(primary)
    if pp and pp != "unknown" and pp.endswith(".py"):
        plan.primary_edit_target = pp
        plan.K_plus_units.append(make_unit(
            content=pp,
            operation="PRESERVE",
            provenance=pp,
            priority=0,
            source_type="trajectory",
            diagnosis_ids=["primary"],
        ))
        seen_paths.add(pp)

    # Candidate edit targets from refined diagnoses
    for d in refined_diagnoses:
        diag_id = getattr(d, "subtype", "UNKNOWN")
        for cet in (getattr(d, "candidate_edit_targets", []) or []):
            if cet not in seen_paths and cet.endswith(".py"):
                seen_paths.add(cet)
                plan.K_plus_units.append(make_unit(
                    content=cet,
                    operation="PRESERVE",
                    provenance=cet,
                    priority=1 if getattr(d, "causal_status", "") == "PATCH_LINKED" else 2,
                    source_type="trajectory",
                    diagnosis_ids=[diag_id],
                ))

    # Failure sites
    for d in refined_diagnoses:
        kl = getattr(d, "key_location", "") or ""
        if kl:
            fp = kl.split(":")[0] if ":" in kl else kl
            if fp not in seen_paths and fp.endswith(".py"):
                seen_paths.add(fp)
                plan.K_plus_units.append(make_unit(
                    content=fp,
                    operation="PRESERVE",
                    provenance=fp,
                    priority=2,
                    source_type="trajectory",
                    diagnosis_ids=[getattr(d, "subtype", "failure")],
                ))

    # Related targets
    for rt in revision_contract.get("related_targets", []):
        rtp = rt.get("path", "")
        if rtp not in seen_paths and rtp.endswith(".py"):
            seen_paths.add(rtp)
            plan.K_plus_units.append(make_unit(
                content=rtp,
                operation="PRESERVE",
                provenance=rtp,
                priority=3,
                source_type="trajectory",
                diagnosis_ids=[rt.get("role", "related")],
            ))

    # ── K_minus: Suppress targets ──
    for d in refined_diagnoses:
        subtype = getattr(d, "subtype", "UNCLASSIFIED")
        if subtype == "PATCH_FAILURE_LOCALIZATION_MISMATCH":
            for cet in (getattr(d, "candidate_edit_targets", []) or []):
                if cet not in seen_paths and cet.endswith(".py"):
                    plan.K_minus_files.append(cet)
                    seen_paths.add(cet)

    # K_minus: assumptions from evidence alignment
    for d in refined_diagnoses:
        ea = getattr(d, "evidence_alignment", None)
        if ea:
            for a in getattr(ea, "patch_assumptions", []) or []:
                plan.K_minus_assumptions.append(a)
            for c in getattr(ea, "contradictory_evidence", []) or []:
                plan.K_minus_assumptions.append(c)

    # K_minus: from revision contract
    for inv in revision_contract.get("invalidated_assumptions", []):
        plan.K_minus_assumptions.append(inv)
    for fb in revision_contract.get("forbidden_changes", []):
        plan.K_minus_assumptions.append(fb)

    plan.summary = (
        f"K_plus={len(plan.K_plus_units)} units, "
        f"K_minus_files={len(plan.K_minus_files)}, "
        f"K_minus_assumptions={len(plan.K_minus_assumptions)}, "
        f"Q={len(plan.Q_contracts)} contracts"
    )
    logger.info("Plan: %s", plan.summary)
    return plan
