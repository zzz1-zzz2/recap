"""Diagnosis-aware Context Packer — produces K_2 under strict budget.

Takes candidate ContextUnits (Preserve, Rehydrate, Acquire) + trajectory
messages, suppresses K_minus files, deduplicates, packs under token budget with
paper-specified priorities (P0-P4), and renders back to message list.

This is the Pack step in Algorithm 1 L7:
  K_2 = Pack(Dedup(K_plus ∪ K_new) \ K_minus; B)
  Tokens(K_2) ≤ B_K
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

from recap.reconstruction.context_unit import (
    ContextUnit,
    make_unit,
    total_tokens,
)

logger = logging.getLogger("condiag.reconstruction.packer")

# Priority levels (paper §4.4)
P0_MANDATORY = 0  # FailureWitness, target repo frames
P1_HIGH = 1       # High-confidence diagnosis evidence
P2_MEDIUM = 2     # Acquired definitions/tests/callers
P3_LOW = 3        # Patch-aligned first-round evidence
P4_FILL = 4       # Lower-priority history


def pack(
    messages: list[dict],
    *,
    preserve_units: list[ContextUnit] | None = None,
    acquire_units: list[ContextUnit] | None = None,
    rehydrate_units: list[ContextUnit] | None = None,
    suppress_files: list[str] | None = None,
    max_chars: int = 50000,
    min_turns: int = 5,
) -> tuple[list[dict], dict[str, Any]]:
    """Pack trajectory + ContextUnits into K_2 under budget.

    Args:
        messages: Full R1 trajectory (pre-stripped of submission turn).
        preserve_units: K_plus from Plan.
        acquire_units: K_new from Acquire.
        rehydrate_units: Rehydrated evidence.
        suppress_files: K_minus files to hard-drop.
        max_chars: Strict token budget (chars).
        min_turns: Minimum turns to retain.

    Returns:
        (packed_messages, metrics_dict)
        metrics_dict contains token accounting for audit.
    """
    preserve_units = list(preserve_units or [])
    acquire_units = list(acquire_units or [])
    rehydrate_units = list(rehydrate_units or [])
    suppress_set = set(suppress_files or [])

    # ── 1. Parse trajectory into turns ──
    turns = _parse_turns(messages)
    turn_units: list[ContextUnit] = []

    for idx, turn in enumerate(turns):
        content = " ".join(str(m.get("content", "") or "") for m in turn)

        # Check suppression: if turn focuses on suppressed files, skip
        if suppress_set:
            turn_text = content.lower()
            has_suppressed = any(
                s and s.lower().split("/")[-1].replace(".py", "") in turn_text
                for s in suppress_set
            )
            if has_suppressed:
                continue

        source = _classify_turn_source(turn, content)
        priority = _score_turn_priority(turn, preserve_units)
        turn_units.append(make_unit(
            content=content[:max_chars],
            operation="PRESERVE",
            provenance=f"turn_{idx}",
            priority=priority,
            source_type=source,
            original_turn_ids=[idx],
        ))

    # ── 2. Combine all candidate units ──
    all_units: list[ContextUnit] = list(turn_units)
    all_units.extend(preserve_units)
    all_units.extend(acquire_units)
    all_units.extend(rehydrate_units)

    # ── 3. Apply hard suppression ──
    if suppress_set:
        all_units = [
            u for u in all_units
            if not _is_suppressed(u, suppress_set)
        ]

    # ── 4. Deduplicate by provenance ──
    seen_provenance: set[str] = set()
    deduped: list[ContextUnit] = []
    for u in all_units:
        key = u.provenance
        if key and key in seen_provenance:
            continue
        if key:
            seen_provenance.add(key)
        deduped.append(u)

    # ── 5. Sort by priority, pack under budget ──
    deduped.sort(key=lambda u: u.priority)

    selected: list[ContextUnit] = []
    total = 0
    for u in deduped:
        if total + u.token_count <= max_chars:
            selected.append(u)
            total += u.token_count

    # Guarantee minimum turns
    if len([u for u in selected if u.source_type == "trajectory"]) < min_turns:
        for u in deduped:
            if u in selected:
                continue
            if len([x for x in selected if x.source_type == "trajectory"]) >= min_turns:
                break
            if total + u.token_count <= max_chars:
                selected.append(u)
                total += u.token_count

    # ── 6. Render back to message list ──
    # Restore original turn order for trajectory-derived units
    turn_indices: list[int] = []
    for u in selected:
        turn_indices.extend(u.original_turn_ids)
    turn_indices = sorted(set(turn_indices))

    result: list[dict] = []
    for idx in turn_indices:
        if idx < len(turns):
            result.extend(turns[idx])

    # Inject ACQUIRE units as synthetic user messages (code evidence)
    for u in acquire_units:
        if u in selected and u.content:
            result.append({
                "role": "user",
                "content": f"# Retrieved: {u.provenance}\n```\n{u.content}\n```",
            })

    # Inject REHYDRATE units
    for u in rehydrate_units:
        if u in selected and u.content:
            result.append({
                "role": "user",
                "content": f"# Rehydrated: {u.provenance}\n```\n{u.content}\n```",
            })

    # ── Metrics ──
    metrics = {
        "n_input_turns": len(turns),
        "n_input_chars": sum(len(str(m.get("content", ""))) for t in turns for m in t),
        "n_candidate_units": len(all_units),
        "n_selected_units": len(selected),
        "n_suppressed_files": len(suppress_set),
        "n_preserve_units": len(preserve_units),
        "n_acquire_units": len(acquire_units),
        "n_rehydrate_units": len(rehydrate_units),
        "total_chars": total,
        "budget_chars": max_chars,
        "p0_units": sum(1 for u in selected if u.priority == P0_MANDATORY),
        "p1_units": sum(1 for u in selected if u.priority == P1_HIGH),
        "p2_units": sum(1 for u in selected if u.priority == P2_MEDIUM),
        "p3_units": sum(1 for u in selected if u.priority == P3_LOW),
        "p4_units": sum(1 for u in selected if u.priority == P4_FILL),
    }

    logger.info(
        "Pack: %d→%d units, %d→%d chars (%.0f%% budget used)",
        len(all_units), len(selected),
        metrics["n_input_chars"], total,
        total / max(max_chars, 1) * 100,
    )

    return result, metrics


# ── Helpers ─────────────────────────────────────────────────────────────


def _parse_turns(messages: list[dict]) -> list[list[dict]]:
    """Split message list into (assistant + tool) turns."""
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
                current = [m]
        else:
            if current:
                turns.append(current)
                current = []
            turns.append([m])

    if current:
        turns.append(current)
    return turns


def _classify_turn_source(turn: list[dict], content: str) -> str:
    """Classify a turn's source type."""
    for m in turn:
        role = m.get("role", "")
        if role == "tool":
            return "trajectory"
    return "trajectory"


def _score_turn_priority(
    turn: list[dict], preserve_units: list[ContextUnit],
) -> int:
    """Score turn priority based on relevance to preserve targets.

    Returns P4 (fill) by default, P3 for repo-related turns,
    P2 for diagnosis target mentions, P0 for failure evidence.
    """
    content = " ".join(str(m.get("content", "") or "") for m in turn).lower()

    # P0: failure evidence
    if any(ind in content for ind in ["failed", "traceback", "error", "assertionerror"]):
        return P0_MANDATORY

    # P2: mentions preserve targets (diagnosis-relevant)
    for pu in preserve_units:
        stem = pu.provenance.split("/")[-1].replace(".py", "").lower()
        if stem and stem in content:
            return P2_MEDIUM

    # P3: mentions any .py file (code exploration)
    if ".py" in content:
        return P3_LOW

    return P4_FILL


def _is_suppressed(unit: ContextUnit, suppress_set: set[str]) -> bool:
    """Check if a ContextUnit targets a suppressed file."""
    for fp in suppress_set:
        stem = fp.split("/")[-1].replace(".py", "").lower()
        if stem and stem in unit.content.lower():
            # Don't suppress if this unit also mentions a preserve target
            return True
    return False
