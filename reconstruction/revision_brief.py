"""Revision Brief — structured, scannable repair guidance for Round 2.

ρ₂ = (c*, E_inspect, O_edit, N_forbid, V_target)

Renders the internal structured data into a compact text format placed
at the END of the R2 injection sequence. The format prioritises ACTION
items (Inspect / Edit / Forbid / Verify) so the agent can locate the
key instructions at a glance, followed by the supporting analysis.

Design principles:
  - Action-first: the 4-section brief is always at the top.
  - Compact: each item ≤ 1-2 lines where possible.
  - No information loss: all current diagnosis detail is still rendered,
    but in a predictable, scannable order.
"""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class RevisionBrief:
    """Structured revision brief ρ₂."""

    # Primary cluster identifier
    target_cluster: str = ""
    # Files/symbols the agent MUST inspect before editing
    inspect_first: list[str] = dataclasses.field(default_factory=list)
    # What to edit and how (compact, actionable)
    edit_objective: str = ""
    # Patterns / directions explicitly forbidden
    forbidden_patterns: list[str] = dataclasses.field(default_factory=list)
    # Tests to run for validation
    validation_targets: list[str] = dataclasses.field(default_factory=list)


def build_revision_brief(
    revision_contract: dict,
    refined_diagnoses: list,
    router_results: list,
    failure_witness: dict | None = None,
) -> RevisionBrief:
    """Build a RevisionBrief from the v5 pipeline outputs."""
    brief = RevisionBrief()
    brief.inspect_first = _extract_inspect_targets(
        revision_contract, refined_diagnoses, router_results,
    )
    brief.edit_objective = _extract_edit_objective(revision_contract, refined_diagnoses)
    brief.forbidden_patterns = _extract_forbidden(revision_contract, refined_diagnoses)
    brief.validation_targets = _extract_validation(revision_contract, refined_diagnoses)
    return brief


def render_revision_brief(brief: RevisionBrief, max_chars: int = 500) -> str:
    """Render ρ₂ as a compact text block for injection."""
    parts: list[str] = []
    parts.append("# Revision Brief\n")

    # Inspect
    if brief.inspect_first:
        parts.append(f"Inspect ({len(brief.inspect_first)})\n")
        for item in brief.inspect_first[:6]:
            parts.append(f"  → {item}\n")
        parts.append("\n")

    # Edit
    if brief.edit_objective:
        parts.append(f"Edit\n  {brief.edit_objective[:300]}\n\n")

    # Forbid
    if brief.forbidden_patterns:
        parts.append(f"Forbid ({len(brief.forbidden_patterns)})\n")
        for f in brief.forbidden_patterns[:5]:
            parts.append(f"  ✗ {f[:120]}\n")
        parts.append("\n")

    # Verify
    if brief.validation_targets:
        display = _collapse_test_families(brief.validation_targets)
        parts.append(f"Verify ({len(display)} families)\n")
        for family, count in display[:6]:
            label = f"{family} ({count})" if count > 1 else family
            parts.append(f"  ✓ {label}\n")
        parts.append("\n")

    return "".join(parts)[:max_chars]


def _collapse_test_families(
    test_names: list[str],
) -> list[tuple[str, int]]:
    """Collapse parameterised test nodeids into families with counts."""
    families: dict[str, int] = {}
    for tn in test_names:
        base = tn.split("[")[0] if "[" in tn else tn
        families[base] = families.get(base, 0) + 1
    return sorted(families.items(), key=lambda x: -x[1])


# ── Extractors ──────────────────────────────────────────────────────────


def _extract_inspect_targets(
    rc: dict,
    refined: list,
    router_results: list,
) -> list[str]:
    """Collect files the agent must inspect before editing.

    Priority order:
      1. Primary edit target (always first, it's the action anchor)
      2. Evidence hits from Router (found definitions/tests)
      3. Candidate edit targets from diagnoses (beyond the primary)
      4. PATCH_LINKED failure frames (where R1's patch manifests)
      5. UNCLASSIFIED failure sites (low-confidence but grounded)
      6. Target symbols from diagnoses

    Deduplication uses the base file path (strips line numbers and "frame:" prefix).
    """
    seen: set[str] = set()
    targets: list[str] = []

    def _base(fp: str) -> str:
        """Extract base file path from a file:line or frame:file:line entry."""
        s = fp.removeprefix("frame:")
        if ":" in s:
            s = s.split(":")[0]
        return s

    def _add(label: str, path: str) -> None:
        base = _base(path)
        if not base or base in seen:
            return
        seen.add(base)
        targets.append(label)

    # 1. Primary edit target path
    primary = rc.get("primary_edit_target", {})
    if isinstance(primary, dict):
        path = primary.get("path", "")
    else:
        path = str(primary)
    if path and path != "unknown":
        _add(f"{path} (edit target)", path)

    # 2. Router evidence hits (AcquisitionResult format)
    for r in router_results:
        if not isinstance(r, dict):
            continue
        # Handle ContextUnit dict format (from acquired_units)
        if r.get("operation") == "ACQUIRE":
            fp = r.get("provenance", "") or ""
            if fp and _base(fp) not in seen:
                _add(f"{fp} [acquired]", fp)
            continue
        # Handle legacy AcquisitionResult format
        if r.get("status") != "FOUND":
            continue
        for hit in r.get("hits", []):
            hit_path = hit.get("path", "") or hit.get("file", "")
            if hit_path:
                reason = hit.get("relevance_reason", "") or hit.get("method", "retrieved")
                _add(f"{hit_path} [{reason}]", hit_path)

    # 3. Candidate edit targets from diagnoses (beyond the primary)
    for d in refined:
        for cet in (getattr(d, "candidate_edit_targets", []) or []):
            if cet and _base(cet) not in seen and cet.endswith(".py"):
                _add(f"{cet} (R1 co-edited file)", cet)

    # 4. PATCH_LINKED failure frames
    for d in refined:
        if getattr(d, "causal_status", "") == "PATCH_LINKED":
            for frame in (getattr(d, "causal_evidence_ids", []) or []):
                if ":" in frame:
                    bare = _base(frame)
                    if bare not in seen:
                        _add(f"{bare} (failure site)", bare)

    # 5. Low-confidence failure sites (key_location)
    for d in refined:
        kl = getattr(d, "key_location", "") or ""
        if kl:
            bare = _base(kl)
            if bare not in seen and bare.endswith(".py"):
                _add(f"{bare} (failure manifestation)", bare)

    # 6. Target symbols from diagnoses (always appended)
    for d in refined:
        for sym in (getattr(d, "target_symbols", []) or []):
            if sym not in seen:
                seen.add(sym)
                targets.append(f"{sym} (diagnosed symbol)")

    return targets[:8]


def _extract_edit_objective(rc: dict, refined: list) -> str:
    """Build a compact edit objective.

    Uses the revision contract's edit_objective, then appends the
    PRIMARY-linked cluster mechanisms as bullet-point context.
    """
    objective = rc.get("edit_objective", "")
    if not objective:
        objective = "Revise the Round-1 patch so that diagnostic failures are resolved."

    # Collect mechanisms from PATCH_LINKED diagnoses
    mechanisms: list[str] = []
    for d in refined:
        if getattr(d, "causal_status", "") == "PATCH_LINKED":
            mech = getattr(d, "mechanism", "")
            if mech:
                mechanisms.append(mech)

    if mechanisms:
        objective += "\n\nRoot cause:\n"
        for m in mechanisms[:3]:
            objective += f"  • {m[:200]}\n"

    return objective


def _extract_forbidden(rc: dict, refined: list) -> list[str]:
    """Collect forbidden patterns.

    Combines:
      - Revision contract's forbidden_changes
      - Revision contract's invalidated_assumptions
      - Unsafe repair warnings from PATCH_LINKED diagnoses
    """
    forbidden: list[str] = []

    # From revision contract
    forbidden.extend(rc.get("forbidden_changes", []))
    for a in rc.get("invalidated_assumptions", []):
        forbidden.append(f"Invalidated: {a[:150]}")

    # Unsafe repair warnings
    seen_warnings: set[str] = set()
    for d in refined:
        if getattr(d, "causal_status", "") == "PATCH_LINKED":
            for w in (getattr(d, "unsafe_repair_warnings", []) or []):
                if w not in seen_warnings:
                    seen_warnings.add(w)
                    forbidden.append(w[:200])

    return _dedup_ordered(forbidden)


def _extract_validation(rc: dict, refined: list) -> list[str]:
    """Collect validation targets.

    Primary failures first, then remaining linked-failure test names.
    """
    targets: list[str] = []
    targets.extend(rc.get("validation_targets", []))
    for d in refined:
        if getattr(d, "causal_status", "") == "PATCH_LINKED":
            for tn in (getattr(d, "test_names", []) or []):
                if tn not in targets:
                    targets.append(tn)
    return targets


def _dedup_ordered(items: list[str]) -> list[str]:
    """Deduplicate preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ── Full diagnosis rendering ────────────────────────────────────────────


def render_diagnosis_detail(
    failure_witness: dict | None,
    refined_diagnoses: list,
    router_results: list,
    revision_contract: dict,
    max_chars: int = 12000,
) -> str:
    """Render the full diagnosis detail section.

    This is the detailed analysis that goes AFTER the Revision Brief.
    """
    parts: list[str] = []
    parts.append("# Detailed Analysis\n")

    # ── Validation Result ──
    if failure_witness:
        failed = failure_witness.get("failed_tests", [])
        error = failure_witness.get("error_message", "")
        parts.append("## Validation Result\n")
        if failed:
            parts.append(f"Failed: {', '.join(str(t) for t in failed[:5])}")
            if len(failed) > 5:
                parts.append(f" (+{len(failed) - 5} more)")
            parts.append("\n")
        if error:
            parts.append(f"First error: {error[:200]}\n")

    # ── Patch Target ──
    primary = revision_contract.get("primary_edit_target", {})
    if isinstance(primary, dict):
        target_path = primary.get("path", "unknown")
    else:
        target_path = str(primary)
    parts.append(f"\n## Patch Target\n{target_path}\n")

    # ── Linked Failures ──
    primary_diags = [
        d for d in refined_diagnoses
        if getattr(d, "causal_status", "") == "PATCH_LINKED"
        and getattr(d, "revision_priority", "") == "PRIMARY"
    ]
    monitor_diags = [
        d for d in refined_diagnoses
        if getattr(d, "revision_priority", "") == "MONITOR"
    ]

    if primary_diags:
        parts.append("\n## Linked Failures (PRIMARY)\n")
        mech_groups: dict[str, list] = {}
        for d in primary_diags:
            mech = getattr(d, "mechanism", "") or "patch-linked failure"
            mech_groups.setdefault(mech, []).append(d)

        for i, (mech, group) in enumerate(mech_groups.items()):
            if i < 4:
                subtype = getattr(group[0], "subtype", "UNKNOWN")
                test_names = getattr(group[0], "test_names", []) or []
                parts.append(f"- [{subtype}] {mech[:200]}\n")
                if test_names and len(test_names) <= 3:
                    parts.append(f"  Tests: {', '.join(str(t) for t in test_names)}\n")
                warnings = getattr(group[0], "unsafe_repair_warnings", [])
                for w in warnings[:1]:
                    parts.append(f"  Warning: {w[:150]}\n")
            else:
                parts.append(f"- {len(group)} additional patch-linked failures\n")

    # ── Evidence from Router ──
    evidence_items = _extract_top_evidence(router_results, max_items=4)
    if evidence_items:
        parts.append("\n## Retrieved Evidence\n")
        for ev in evidence_items:
            action = ev.get("action", ev.get("action_type", "search"))
            target = ev.get("target", "?")
            status = ev.get("status", "?")
            parts.append(f"- {action} → {target} ({status})\n")
            if status == "FOUND":
                for hit in ev.get("hits", [])[:2]:
                    path = hit.get("path", "") or hit.get("file", "")
                    line = hit.get("line", "")
                    if path:
                        parts.append(f"  {path}:{line}\n")

    # ── Scope ──
    scope_mode = revision_contract.get("scope_mode", "")
    if scope_mode:
        parts.append(f"\n## Scope Guidance\n")
        if scope_mode == "EXPANDED":
            parts.append("Primary edit target is a starting point, not a boundary. "
                         "Inspect related context before finalising.\n")
        elif scope_mode == "NARROW":
            parts.append("Evidence is unambiguous. Focus on the primary target.\n")
        related = revision_contract.get("related_targets", [])
        if related:
            for rt in related[:3]:
                parts.append(f"  ↔ {rt.get('path', '?')} ({rt.get('role', 'related')})\n")

    # ── Monitor-only ──
    if monitor_diags:
        parts.append(f"\n## Monitor\n{len(monitor_diags)} low-linkage failure(s) — "
                     f"do not modify code for these without new evidence.\n")

    # ── Footer ──
    parts.append("\nFirst verify the diagnosis claims by inspecting the listed files. "
                 "Trace the full call chain before editing — root cause may span "
                 "multiple layers beyond the immediate failure location.\n")

    text = "".join(parts)
    return text[:max_chars]


def _extract_top_evidence(
    router_results: list,
    max_items: int = 5,
) -> list[dict]:
    """Extract top evidence items from Router results for CD prompt."""
    items: list[dict] = []
    for r in router_results:
        if isinstance(r, dict):
            items.append(r)
        elif hasattr(r, "to_dict"):
            d = r.to_dict()
            if isinstance(d, dict):
                items.append(d)
        else:
            d = {}
            for attr in ("action_type", "status", "target", "hits"):
                val = getattr(r, attr, None)
                if val is not None:
                    d[attr] = val
            if d:
                items.append(d)

    normalized: list[dict] = []
    for item in items:
        n = dict(item)
        if "action_type" in n and "action" not in n:
            n["action"] = n["action_type"]
        if isinstance(n.get("target"), dict):
            n["target"] = n["target"].get("value", str(n["target"]))
        else:
            n["target"] = str(n.get("target", "?"))
        if "hits" not in n:
            n["hits"] = []
        normalized.append(n)

    found = [n for n in normalized if n.get("status") == "FOUND"]
    others = [n for n in normalized if n not in found]
    selected = (found + others)[:max_items]
    return selected
