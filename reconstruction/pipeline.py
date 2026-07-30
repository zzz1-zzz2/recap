"""ReCAP reconstruction pipeline — Plan → Acquire → Pack → Guide.

Algorithm 1 (L5-L9):
  (K_plus, K_minus, Q) ← Plan(C₁, H, B)
  K_new ← Acquire(x₁, Q, B)
  K_2 → Pack(Dedup(K_plus ∪ K_new ∪ K_reh) \ K_minus; B)
  rho_2 ← Guide(H, K_2)
  z₂ ← (K_2, rho_2)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from reconstruction.context_unit import ContextUnit
from reconstruction.planner import build_plan, ReconstructionPlan
from reconstruction.rehydrator import identify_rehydratable_units
from reconstruction.packer import pack
from reconstruction.revision_brief import (
    build_revision_brief,
    render_revision_brief,
    render_diagnosis_detail,
)

logger = logging.getLogger("condiag.reconstruction.pipeline")


def run_pipeline(
    *,
    messages: list[dict],
    clusters: list,
    refined_diagnoses: list,
    attributions: list,
    revision_contract: dict,
    hypotheses: list,
    search_contracts: list,
    failure_witness: dict | None = None,
    repo_root: str | None = None,
    pack_budget_chars: int = 50000,
    diagnosis_max_chars: int = 12000,
    output_dir: str | None = None,
) -> dict:
    """Run full Plan → Acquire → Pack → Guide pipeline.

    Returns dict with structured outputs for each step, plus audit artifacts.
    """
    artifacts: dict[str, Any] = {}

    # ── Abstention check (L2-4) ──
    abstain_reason = _should_abstain(refined_diagnoses, revision_contract)
    if abstain_reason:
        logger.warning("Abstaining: %s", abstain_reason)
        # Conservative bounded fallback: FW + last relevant turns only,
        # same budget as normal execution. No diagnosis-guided suppression
        # or acquisition.
        fallback_messages = _conservative_fallback(messages, pack_budget_chars)
        return {
            "abstained": True,
            "abstain_reason": abstain_reason,
            "K_2_messages": fallback_messages,
            "rho_2_text": None,
            "plan": None,
            "K_plus_units": [],
            "K_new_units": [],
            "K_reh_units": [],
            "K_minus_files": [],
            "pack_metrics": {"n_input_turns": 0, "n_selected_units": 0},
        }

    # ── L5: Plan (K_plus, K_minus, Q) ──
    plan = build_plan(
        clusters=clusters,
        refined_diagnoses=refined_diagnoses,
        attributions=attributions,
        revision_contract=revision_contract,
        hypotheses=hypotheses,
        search_contracts=search_contracts,
    )
    artifacts["plan"] = _plan_to_dict(plan)

    # ── L6: Acquire (K_new) ──
    K_new_units: list[ContextUnit] = []
    if repo_root and plan.Q_contracts:
        K_new_units = _run_acquisition(plan.Q_contracts, repo_root)
        artifacts["acquired_units"] = [u.to_dict() for u in K_new_units]

    # ── Rehydrate (K_reh) — complementary to Acquire ──
    K_plus_paths = {u.provenance for u in plan.K_plus_units}
    K_reh_units = identify_rehydratable_units(
        messages, K_plus_paths, hypotheses, refined_diagnoses,
    )
    artifacts["rehydrated_units"] = [u.to_dict() for u in K_reh_units]

    # ── L7: Pack (K_2) ──
    K_2_messages, pack_metrics = pack(
        messages,
        preserve_units=plan.K_plus_units,
        acquire_units=K_new_units,
        rehydrate_units=K_reh_units,
        suppress_files=plan.K_minus_files,
        max_chars=pack_budget_chars,
    )
    artifacts["pack_metrics"] = pack_metrics

    # ── L8: Guide (rho_2) ──
    brief = build_revision_brief(
        revision_contract=revision_contract,
        refined_diagnoses=refined_diagnoses,
        router_results=artifacts.get("acquired_units", []),
        failure_witness=failure_witness,
    )
    brief_text = render_revision_brief(brief, max_chars=800)
    detail_text = render_diagnosis_detail(
        failure_witness=failure_witness,
        refined_diagnoses=refined_diagnoses,
        router_results=artifacts.get("acquired_units", []),
        revision_contract=revision_contract,
        max_chars=diagnosis_max_chars - min(len(brief_text), 5000),
    )
    rho_2_text = (brief_text + "\n" + detail_text)[:diagnosis_max_chars]

    # ── Output ──
    result = {
        "abstained": False,
        "K_2_messages": K_2_messages,
        "rho_2_text": rho_2_text,
        "plan": plan,
        "K_plus_units": plan.K_plus_units,
        "K_new_units": K_new_units,
        "K_reh_units": K_reh_units,
        "K_minus_files": plan.K_minus_files,
        "pack_metrics": pack_metrics,
    }

    # Write audit artifacts
    if output_dir:
        _write_artifacts(output_dir, artifacts, plan.K_minus_assumptions, K_2_messages, rho_2_text)

    return result


# ── Abstention ──────────────────────────────────────────────────────────


def _should_abstain(
    refined_diagnoses: list,
    revision_contract: dict,
) -> str | None:
    """Check if diagnosis is too weak for reconstruction.

    Returns None if confident enough, else string reason.
    """
    if not refined_diagnoses:
        return "no diagnoses produced"

    all_weak = True
    any_linked = False
    for d in refined_diagnoses:
        dtype = getattr(d, "type", None)
        subtype = getattr(d, "subtype", "UNCLASSIFIED")
        if dtype is not None:
            from .taxonomy import ContextDeficiencyType
            if dtype != ContextDeficiencyType.NO_RELIABLE_DEFICIENCY:
                all_weak = False
        if subtype != "UNCLASSIFIED":
            all_weak = False
        if getattr(d, "causal_status", "") == "PATCH_LINKED":
            any_linked = True

    if all_weak and not any_linked:
        return "all diagnoses are NO_RELIABLE_DEFICIENCY or UNCLASSIFIED"

    any_confident = any(
        getattr(d, "confidence", "low") in ("high", "medium")
        for d in refined_diagnoses
    )
    if not any_confident and not any_linked:
        return "no high/medium confidence diagnosis and no PATCH_LINKED"

    primary = revision_contract.get("primary_edit_target", {})
    pp = primary.get("path", "") if isinstance(primary, dict) else str(primary)
    primary_failures = revision_contract.get("primary_failures", [])
    if not pp and not primary_failures:
        return "revision contract has no primary edit target or failures"

    return None


def _conservative_fallback(messages: list[dict], max_chars: int) -> list[dict]:
    """Conservative bounded fallback when diagnosis abstains.

    Keeps only: system message, PR description, FW (last error output),
    and the last N turns within budget. No diagnosis-guided operations.
    """
    if not messages:
        return messages

    # Parse turns
    turns: list[list[dict]] = []
    current: list[dict] = []
    for m in messages:
        role = m.get("role", "")
        if role == "assistant":
            if current:
                turns.append(current)
            current = [m]
        elif role == "tool":
            if current:
                current.append(m)
        else:
            if current:
                turns.append(current)
                current = []
            turns.append([m])
    if current:
        turns.append(current)

    # Keep system + first user (PR description) unchanged
    kept: list[dict] = []
    remaining_budget = max_chars

    # Always keep leading system/user messages
    for t in turns:
        is_meta = all(m.get("role") in ("system", "user") for m in t)
        if is_meta and turns.index(t) < 3:
            for m in t:
                kept.append(m)
                remaining_budget -= len(str(m.get("content", "") or ""))

    # Fill rest with last N turns within budget (keep FW visible)
    for t in reversed(turns):
        turn_chars = sum(len(str(m.get("content", "") or "")) for m in t)
        if turn_chars <= remaining_budget:
            # Insert at beginning (we're iterating reversed)
            kept = kept[:-1] if kept else kept
            keep_idx = len(kept)
            for i, m in enumerate(t):
                kept.insert(keep_idx + i, m)
            remaining_budget -= turn_chars

    # Ensure at least the last tool output (FW) survives
    return kept[:150]  # safety cap


# ── Acquisition ─────────────────────────────────────────────────────────


def _run_acquisition(
    contracts: list,
    repo_root: str,
) -> list[ContextUnit]:
    """Execute search contracts, return acquired ContextUnits."""
    from pathlib import Path
    from acquisition.router import AcquisitionRouter
    from acquisition.schema import AcquisitionStatus
    from reconstruction.context_unit import make_unit

    router = AcquisitionRouter(repo_root=repo_root)
    units: list[ContextUnit] = []
    seen_paths: set[str] = set()

    for contract in contracts:
        if not _is_actionable(contract):
            continue
        results = router.dispatch_contract(contract)
        for result in results:
            status = getattr(result, "status", None)
            if status == AcquisitionStatus.FOUND:
                for hit in getattr(result, "hits", []) or []:
                    fp = getattr(hit, "file_path", "") or ""
                    if fp in seen_paths or not fp.endswith(".py"):
                        continue
                    seen_paths.add(fp)
                    line_start = getattr(hit, "start_line", None) or 0
                    line_end = getattr(hit, "end_line", None) or 0
                    content = _read_file_snippet(str(repo_root), fp, line_start, line_end)
                    units.append(make_unit(
                        content=content or f"{fp} (not accessible)",
                        operation="ACQUIRE",
                        provenance=fp,
                        priority=1,
                        source_type="repo",
                        target_symbol=getattr(hit, "symbol", None),
                        line_start=line_start,
                        line_end=line_end,
                    ))

    logger.info("Acquire: %d units from %d contracts", len(units), len(contracts))
    return units


def _is_actionable(contract) -> bool:
    """Check if a search contract is actionable."""
    try:
        from diagnosis.search_contract import ContractStatus
        return getattr(contract, "status", None) == ContractStatus.ACTIONABLE
    except Exception:
        return bool(getattr(contract, "actions", None))


def _read_file_snippet(
    repo_root: str, file_path: str,
    start: int, end: int, max_chars: int = 3000,
) -> str:
    """Read a snippet from a repository file."""
    try:
        full_path = Path(repo_root).resolve() / file_path
        if not full_path.exists():
            return ""
        content = full_path.read_text(encoding="utf-8", errors="replace")
        if start <= 0 and end <= 0:
            return content[:max_chars]
        lines = content.split("\n")
        selected = lines[max(0, start - 1):min(len(lines), end)]
        snippet = "\n".join(selected)
        return snippet[:max_chars]
    except Exception:
        return ""


# ── Artifacts ───────────────────────────────────────────────────────────


def _plan_to_dict(plan: ReconstructionPlan) -> dict:
    return {
        "K_plus_units": [u.to_dict() for u in plan.K_plus_units],
        "K_minus_files": plan.K_minus_files,
        "K_minus_assumptions": plan.K_minus_assumptions,
        "Q_contracts": [c.to_dict() for c in plan.Q_contracts],
        "primary_edit_target": plan.primary_edit_target,
        "summary": plan.summary,
    }


def _write_artifacts(
    output_dir: str,
    artifacts: dict,
    assumptions: list[str],
    K_2_messages: list[dict],
    rho_2_text: str,
) -> None:
    """Write audit artifacts to disk."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if "plan" in artifacts:
        (out / "plan.json").write_text(json.dumps(artifacts["plan"], indent=2, default=str))
    if "pack_metrics" in artifacts:
        (out / "pack_metrics.json").write_text(json.dumps(artifacts["pack_metrics"], indent=2))
    if "acquired_units" in artifacts:
        (out / "acquired_units.json").write_text(json.dumps(artifacts["acquired_units"], indent=2))
    if "rehydrated_units" in artifacts:
        (out / "rehydrated_units.json").write_text(json.dumps(artifacts["rehydrated_units"], indent=2))
    if assumptions:
        (out / "suppressed_assumptions.json").write_text(json.dumps(assumptions, indent=2))
    if K_2_messages:
        (out / "K_2_messages.json").write_text(
            json.dumps([{"role": m.get("role"), "content": str(m.get("content", ""))[:200]}
                       for m in K_2_messages], indent=2),
        )
    if rho_2_text:
        (out / "revision_brief.md").write_text(rho_2_text)

    logger.info("Wrote %d audit artifacts to %s", len(list(out.iterdir())), output_dir)


# ── Compatibility wrapper ──────────────────────────────────────────────


def run_pipeline_from_checkpoint(
    *,
    checkpoint_dir: Path,
    messages: list[dict],
    failure_witness: dict,
    workspace_patch: str,
    repo_root: str | None = None,
    pack_budget_chars: int = 50000,
    diagnosis_max_chars: int = 12000,
) -> dict | None:
    """Build the full pipeline output from a checkpoint directory.

    Runs the v5 diagnosis pipeline then calls run_pipeline().
    Returns a flat dict for backward compatibility with ablation_runner.
    """
    try:
        from diagnosis.bundle_builder import build_failure_feature_bundle
        from diagnosis.failure_event import reasoner_v2_cluster
        from diagnosis.alignment import reasoner_v2_diagnose
        from diagnosis.hypothesis import from_subtyped_diagnosis
        from diagnosis.search_contract import (
            PlanBudget, build_evidence_ledger, build_search_plan,
        )
        from diagnosis.causal_refinement import (
            causal_refine, merge_attribution_into_diagnosis,
        )
        from diagnosis.revision_contract import build_revision_contract
        from diagnosis.signals.schema import TestLogSignals
        from .signals import extract_test_log
        from instance_registry import InstanceRegistry
        import json as _json

        r1_dir = checkpoint_dir / "round1"

        he = _json.loads((r1_dir / "harness_eval.json").read_text())
        tl_path_str = he.get("test_log_path", "")
        test_log = None
        if tl_path_str:
            tp = Path(tl_path_str)
            if tp.exists():
                try:
                    test_log = extract_test_log(tp)
                except Exception:
                    pass
        if test_log is None or not getattr(test_log, "failed_tests", []):
            fw_failed = (failure_witness or {}).get("failed_tests", [])
            test_log = TestLogSignals(
                failed_tests=list(fw_failed[:20]), failures=[],
                passed_tests=[], framework="fallback",
                num_tests_run=0, error_types={},
            )

        iid = r1_dir.parent.parent.name
        try:
            reg = InstanceRegistry()
            spec = reg.get_instance(iid)
            if spec is None:
                class _MS:
                    fail_to_pass = []
                spec = _MS()
        except Exception:
            class _MS:
                fail_to_pass = []
            spec = _MS()

        bundle = build_failure_feature_bundle(
            failure_witness=failure_witness,
            workspace_patch=workspace_patch,
            trajectory=messages, instance_spec=spec, test_log=test_log,
        )
        clusters = reasoner_v2_cluster(bundle)
        diagnoses = reasoner_v2_diagnose(clusters, bundle.patch, bundle.trajectory)
        hypotheses = [
            from_subtyped_diagnosis(d, c.cluster_id, c.test_names)
            for c, d in zip(clusters, diagnoses)
        ]
        ledg = build_evidence_ledger(clusters, diagnoses, bundle=bundle)
        pb = PlanBudget(max_total_actions=3, max_total_budget=8)
        contracts = build_search_plan(hypotheses, budget=pb, ledger=ledg)

        f2p = list(getattr(spec, "fail_to_pass", []) or [])
        refined = list(diagnoses)
        attributions = []
        for i, (c, dx) in enumerate(zip(clusters, refined)):
            attr = causal_refine(dx.subtype, c, list(bundle.patch.edited_files), f2p, workspace_patch)
            refined[i] = merge_attribution_into_diagnosis(dx, attr)
            attributions.append(attr)

        rc = build_revision_contract(clusters, refined, [], attributions)

        result = run_pipeline(
            messages=messages, clusters=clusters,
            refined_diagnoses=refined, attributions=attributions,
            revision_contract=rc, hypotheses=hypotheses,
            search_contracts=contracts,
            failure_witness=failure_witness, repo_root=repo_root,
            pack_budget_chars=pack_budget_chars,
            diagnosis_max_chars=diagnosis_max_chars,
        )

        # Flatten to flat dict for ablation_runner compatibility
        flat: dict = {
            "abstained": result.get("abstained", False),
            "diagnosis_text": result.get("rho_2_text"),
            "plan": result.get("plan"),
            "packed_messages": result.get("K_2_messages", messages),
            "pack_metrics": result.get("pack_metrics", {}),
            "preserved_files": [u.provenance for u in result.get("K_plus_units", [])],
            "suppressed_files": result.get("K_minus_files", []),
            "n_contracts": len(plan.Q_contracts) if (plan := result.get("plan")) else 0,
        }
        if result.get("abstained"):
            flat["abstain_reason"] = result.get("abstain_reason", "unknown")
        return flat

    except Exception as e:
        logger.error("run_pipeline_from_checkpoint failed: %s: %s", type(e).__name__, e)
        import traceback
        traceback.print_exc()
        return None
