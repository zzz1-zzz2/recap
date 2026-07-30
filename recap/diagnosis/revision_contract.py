"""Build revision contract from refined diagnoses + Router results.

The revision contract is a structured dict that feeds into the CD
context reshaping. It captures:

  - Primary edit target (file path)
  - Edit objective
  - Invalidated R1 assumptions
  - Unsafe repair warnings
  - Forbidden changes
  - Validation targets
  - Required evidence references
"""
from __future__ import annotations


def build_revision_contract(
    clusters: list,
    refined_diagnoses: list,
    router_results: list,
    attributions: list,
) -> dict:
    """Build revision contract dict from v5 pipeline outputs."""

    # Collect PRIMARY diagnoses
    primary_diags = [
        d for d in refined_diagnoses
        if getattr(d, "causal_status", "") == "PATCH_LINKED"
        and getattr(d, "revision_priority", "") == "PRIMARY"
    ]

    # Collect MONITOR diagnoses
    monitor_diags = [
        d for d in refined_diagnoses
        if getattr(d, "revision_priority", "") == "MONITOR"
    ]

    # Primary failures — with cluster-level fallback
    primary_failures: list[str] = []
    for d in primary_diags:
        test_names = getattr(d, "test_names", []) or []
        primary_failures.extend(test_names[:5])
    # Fallback: if diagnosis test_names empty, use cluster test_names
    if not primary_failures:
        for c in clusters:
            if getattr(c, "test_names", None):
                primary_failures.extend(c.test_names[:5])
    primary_failures = list(dict.fromkeys(primary_failures))  # dedup, preserve order

    # Collect unsafe repair warnings
    all_warnings: list[str] = []
    for d in primary_diags:
        all_warnings.extend(getattr(d, "unsafe_repair_warnings", []))

    # ── Build instance-specific invalidated assumptions ──
    invalidated: list[str] = _build_invalidated_assumptions(primary_diags, refined_diagnoses, attributions, clusters)

    # Primary edit target - fallback chain
    primary_target = _resolve_primary_target(primary_diags, router_results, refined_diagnoses, clusters)

    # Validation targets — from primary failure test names
    validation_targets: list[str] = list(primary_failures)  # copy

    # ── Build instance-specific forbidden changes ──
    forbidden_changes: list[str] = _build_forbidden_changes(primary_diags, refined_diagnoses, invalidated)

    # Evidence from Router results
    required_evidence: list[dict] = []
    for r in router_results:
        if isinstance(r, dict):
            action = r.get("action_type", "?")
            target = r.get("target", {})
            if isinstance(target, dict):
                target_val = target.get("value", "?")
            else:
                target_val = str(target)
            status = r.get("status", "?")
            required_evidence.append({
                "action": action,
                "target": target_val,
                "status": status,
            })

    # ── Universal fallback (actionability floor) ──
    # When no PRIMARY diagnoses exist (all UNCERTAIN/unlinked), still provide
    # grounded edit_targets and test names from any available diagnosis.
    if not primary_diags:
        if not primary_target:
            for d in refined_diagnoses:
                cet = getattr(d, "candidate_edit_targets", []) or []
                if cet:
                    primary_target = cet[0]
                    break
        if not primary_failures:
            for d in refined_diagnoses:
                tn = getattr(d, "test_names", []) or []
                if tn:
                    primary_failures.extend(tn[:5])
                    break
            primary_failures = list(dict.fromkeys(primary_failures))

    # ── Uncertainty-aware fields ──
    # Collect scope_mode, signal_coverage, score_margin from primary diagnoses
    scope_modes = set()
    signal_coverages = []
    score_margins = []
    related_targets = []  # list of {path, role, reason}

    for d in refined_diagnoses:
        sm = getattr(d, "scope_mode", "")
        if sm: scope_modes.add(sm)
        sc = getattr(d, "signal_coverage", 0.0) or 0.0
        sg = getattr(d, "score_margin", 0.0) or 0.0
        if sc > 0: signal_coverages.append(sc)
        if sg >= 0: score_margins.append(sg)

        # Collect related targets from candidate_edit_targets beyond the first
        cets = getattr(d, "candidate_edit_targets", []) or []
        if len(cets) > 1:
            for path in cets[1:4]:
                if path and path != primary_target and _is_valid_target(path):
                    related_targets.append({
                        "path": path,
                        "role": "RELATED_SOURCE_FILE",
                        "reason": "Co-edited file in R1 patch",
                    })

    # Determine final scope mode
    if "NARROW" in scope_modes and "EXPANDED" not in scope_modes:
        final_scope = "NARROW"
    elif "EXPANDED" in scope_modes:
        final_scope = "EXPANDED"
    else:
        final_scope = "BALANCED"

    avg_coverage = sum(signal_coverages) / len(signal_coverages) if signal_coverages else 0.0
    avg_margin = sum(score_margins) / len(score_margins) if score_margins else 0.0

    return {
        "primary_failures": primary_failures,
        "deprioritized_failures": [
            {
                "subtype": getattr(d, "subtype", "UNKNOWN"),
                "reason": (getattr(d, "mechanism", "") or
                           "No causal link to R1 patch established"),
            }
            for d in monitor_diags
        ],
        "primary_edit_target": {
            "path": primary_target or "unknown",
            "reason": "Primary patch-linked failure cluster",
        },
        "edit_objective": _build_edit_objective(primary_diags, primary_target),
        "invalidated_assumptions": invalidated,
        "unsafe_repair_warnings": all_warnings,
        "required_evidence": required_evidence,
        "forbidden_changes": forbidden_changes,
        "validation_targets": validation_targets,
        # Uncertainty-aware fields
        "scope_mode": final_scope,
        "signal_coverage": round(avg_coverage, 2),
        "score_margin": round(avg_margin, 2),
        "over_localization_risk": max(
            (getattr(d, "over_localization_risk", 0) or 0)
            for d in refined_diagnoses
        ) if refined_diagnoses else 0,
        "related_targets": related_targets[:5],
    }


# ── Helper: build instance-specific invalidated assumptions ────────────


def _build_invalidated_assumptions(
    primary_diags: list,
    refined_diagnoses: list,
    attributions: list,
    clusters: list,
) -> list[str]:
    """Build instance-specific list of invalidated R1 assumptions."""
    invalidated: list[str] = []

    # 1. From evidence_alignment.patch_assumptions (captures edit-target mismatch)
    seen_assumptions: set[str] = set()
    for d in refined_diagnoses:
        ea = getattr(d, "evidence_alignment", None)
        if ea:
            for assumption in getattr(ea, "patch_assumptions", []) or []:
                if assumption not in seen_assumptions:
                    seen_assumptions.add(assumption)
                    invalidated.append(f"R1 edit strategy: {assumption}")
            for contradiction in getattr(ea, "contradictory_evidence", []) or []:
                if contradiction not in seen_assumptions:
                    seen_assumptions.add(contradiction)
                    invalidated.append(f"Evidence contradicts R1: {contradiction}")

    # 2. From attribution alternative_hypotheses
    for attr in attributions:
        for ah in getattr(attr, "alternative_hypotheses", []) or []:
            if ah not in seen_assumptions:
                seen_assumptions.add(ah)
                invalidated.append(f"Alternative explanation: {ah}")

    # 3. From diagnosis subtype → specific R1 mistake
    for d in primary_diags:
        subtype = getattr(d, "subtype", "UNCLASSIFIED")
        key_loc = getattr(d, "key_location", "")
        edit_targets = getattr(d, "candidate_edit_targets", []) or []

        if subtype == "REJECTED_ARGUMENT_OR_ATTRIBUTE" and key_loc:
            msg = (f"R1 modified {key_loc} but the issue is that a function/class "
                   f"there ignores or rejects the provided parameter")
            if msg not in seen_assumptions:
                seen_assumptions.add(msg)
                invalidated.append(msg)

        if subtype == "TYPE_OR_SIGNATURE_CONTRACT" and key_loc:
            msg = (f"R1 modified {key_loc} but violated an interface contract "
                   f"(type or signature mismatch)")
            if msg not in seen_assumptions:
                seen_assumptions.add(msg)
                invalidated.append(msg)

        if subtype == "CALL_CHAIN_DATAFLOW_MISMATCH":
            msg = ("R1 patched a caller but the failure is in a callee — "
                   "the call chain has a dataflow or mutation mismatch")
            if msg not in seen_assumptions:
                seen_assumptions.add(msg)
                invalidated.append(msg)

        if subtype == "PATCH_FAILURE_LOCALIZATION_MISMATCH":
            edited = ", ".join(edit_targets[:3]) if edit_targets else "unknown"
            msg = (f"R1 edited {edited} but the error manifests in a different "
                   f"file — wrong localization")
            if msg not in seen_assumptions:
                seen_assumptions.add(msg)
                invalidated.append(msg)

        if subtype == "NUMERICAL_BEHAVIOR_MISMATCH":
            msg = ("R1's patch changes numerical behavior without making the "
                   "new path semantically equivalent to the established path")
            if msg not in seen_assumptions:
                seen_assumptions.add(msg)
                invalidated.append(msg)

    # 4. Generic fallback if nothing instance-specific found
    if not invalidated and primary_diags:
        invalidated.append(
            "Round-1 assumption about failure cause may be incorrect. "
            "See causal refinement for corrected direction."
        )

    return invalidated


# ── Helper: build instance-specific forbidden changes ──────────────────


def _build_forbidden_changes(
    primary_diags: list,
    refined_diagnoses: list,
    invalidated: list,
) -> list[str]:
    """Build forbidden changes list with instance-specific items."""
    forbidden: list[str] = [
        "Do not edit tests",
        "Do not weaken numerical tolerances",
        "Do not suppress or bypass failing behavior",
        "Do not modify unrelated configuration or harness code",
    ]

    # Add instance-specific prohibitions from diagnosis
    seen: set[str] = set(forbidden)

    # From invalidated assumptions, derive actionable prohibitions
    for inv in invalidated:
        if "R1 edit strategy" in inv:
            # "R1 edits files not directly referenced in the error stack"
            # → prohibit just repeating R1's file choice
            forbid = f"Do NOT repeat R1's file choice: {inv.replace('R1 edit strategy: ', '')}"
            if forbid not in seen:
                seen.add(forbid)
                forbidden.append(forbid)

    # From primary diagnosis subtypes, add specific don'ts
    for d in primary_diags:
        subtype = getattr(d, "subtype", "UNCLASSIFIED")
        if subtype == "REJECTED_ARGUMENT_OR_ATTRIBUTE":
            forbid = ("Do NOT add missing attributes or arguments to the "
                      "manifestation site without verifying the interface contract")
            if forbid not in seen:
                seen.add(forbid)
                forbidden.append(forbid)
        if subtype in ("CALL_CHAIN_DATAFLOW_MISMATCH", "PATCH_FAILURE_LOCALIZATION_MISMATCH"):
            forbid = ("Do NOT edit the same file as R1 without first inspecting "
                      "the caller and callee frames in the error call chain")
            if forbid not in seen:
                seen.add(forbid)
                forbidden.append(forbid)

    # From unsafe repair warnings
    for d in refined_diagnoses:
        for warn in (getattr(d, "unsafe_repair_warnings", []) or []):
            # Only add if it reads like a prohibition
            if any(kw in warn.lower() for kw in ["do not", "don't", "never", "avoid"]):
                if warn not in seen:
                    seen.add(warn)
                    forbidden.append(warn)

    return forbidden


# ── Helper: resolve primary edit target ────────────────────────────────


def _resolve_primary_target(
    primary_diags: list,
    router_results: list,
    refined_diagnoses: list,
    clusters: list,
) -> str:
    """Resolve primary edit target via fallback chain."""
    # Priority 1: key_location from causal refinement
    if primary_diags:
        for d in primary_diags:
            key_loc = getattr(d, "key_location", "")
            if key_loc:
                file_part = key_loc.split(":")[0] if ":" in key_loc else key_loc
                if file_part:
                    return file_part
        # Priority 2: Router FOUND hits -> first source file
        for r in router_results:
            if isinstance(r, dict) and r.get("status") == "FOUND":
                for hit in r.get("hits", []):
                    hp = hit.get("path", "") or hit.get("file", "")
                    if hp:
                        return hp
        # Priority 3: error frames from PATCH_LINKED clusters
        for d in primary_diags:
            ea = getattr(d, "evidence_alignment", None)
            if ea and hasattr(ea, "error_frames"):
                for frame in (ea.error_frames or []):
                    fp = frame.split(":")[0] if ":" in frame else frame
                    if fp and "test" not in fp.lower():
                        return fp
        # Priority 4: candidate_edit_targets from diagnosis (Actionability Floor)
        for d in primary_diags:
            cet = getattr(d, "candidate_edit_targets", []) or []
            if cet:
                return cet[0]
        # Priority 5: from clusters — use first non-test failure_site
        for c in clusters:
            fs = getattr(c, "failure_site", "") or getattr(c, "root_cause", "")
            if fs:
                fp = fs.split(":")[0] if ":" in fs else fs
                if fp and "test" not in fp.lower():
                    return fp
    return ""


# ── Helper: build edit objective ───────────────────────────────────────


def _build_edit_objective(
    primary_diags: list,
    primary_target: str,
) -> str:
    """Build a compact, instance-informed edit objective."""
    if not primary_diags:
        return ("Revise the Round-1 transformation so that primary patch-linked "
                "failures are resolved without introducing new regressions. "
                "Follow the causal refinement guidance for each cluster.")

    subtypes = [getattr(d, "subtype", "UNCLASSIFIED") for d in primary_diags]
    unique_subtypes = list(dict.fromkeys(subtypes))

    if "PATCH_FAILURE_LOCALIZATION_MISMATCH" in unique_subtypes:
        base = (f"R1 edited the wrong file. The edit target is {primary_target or 'unknown'}. "
                f"Inspect the error call chain and edit the correct layer.")
    elif "CALL_CHAIN_DATAFLOW_MISMATCH" in unique_subtypes:
        base = (f"R1 edited a caller but the failure is in a callee. "
                f"Target: {primary_target or 'unknown'}. "
                f"Follow the dataflow from entry point to failure site.")
    elif "REJECTED_ARGUMENT_OR_ATTRIBUTE" in unique_subtypes:
        base = (f"R1's edit passes arguments that a downstream function rejects. "
                f"Target: {primary_target or 'unknown'}. "
                f"Fix the receiving side, not just the call site.")
    elif "TYPE_OR_SIGNATURE_CONTRACT" in unique_subtypes:
        base = (f"R1 violated an interface contract. "
                f"Target: {primary_target or 'unknown'}. "
                f"Fix the type/signature mismatch.")
    elif "REGISTRATION_OR_EXPORT_CONTEXT" in unique_subtypes:
        base = (f"R1 added new code but did not register or export it. "
                f"Target: {primary_target or 'unknown'}. "
                f"Ensure the new path is discoverable.")
    elif "NUMERICAL_BEHAVIOR_MISMATCH" in unique_subtypes:
        base = (f"R1's new path produces different numerical results than the "
                f"established path. Target: {primary_target or 'unknown'}. "
                f"Make the new path semantically equivalent.")
    else:
        base = (f"Revise R1's patch at {primary_target or 'unknown'} so that "
                f"all patch-linked failures are resolved without regressions.")

    return base


# ── Shared helper ────────────────────────────────────────────────────────

_ARTIFACT_STEMS = {"patch.txt", "tmp", "temp", "debug", "testbed"}


def _is_valid_target(path: str) -> bool:
    """Filter out artifact files (patch.txt, temp files, etc.)."""
    stem = path.split("/")[-1].lower()
    if stem in _ARTIFACT_STEMS:
        return False
    if not path.endswith(".py"):
        return False
    return True
