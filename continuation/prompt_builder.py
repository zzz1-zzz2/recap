"""Shared branch message builder: used by both gate test and paired runner."""
from __future__ import annotations
from copy import deepcopy
from typing import Any


def build_branch_messages(
    checkpoint_messages: list[dict],
    failure_witness: dict | None,
    diagnosis: str | None = None,
    *,
    style: str = "stateful_feedback",
) -> list[dict]:
    """Build the message sequence for a forked Round 2 branch.

    Repairs ALL incomplete assistant turns by inserting synthetic tool
    responses for missing tool_call_ids AFTER all real tool responses in
    the same turn. Each assistant's order is: assistant, real tools,
    synthetic tools (for any IDs not seen).

    This avoids the previous bug where real tool responses were duplicated
    by synthetic ones in the same turn.
    """
    msgs = deepcopy(checkpoint_messages)
    # Clean exit roles
    msgs = [m for m in msgs if m.get("role") != "exit"]

    result: list[dict] = []
    pending_assistant: dict | None = None
    pending_call_ids: list[str] = []
    pending_real_tools: list[dict] = []

    for m in msgs:
        role = m.get("role", "")

        if role == "assistant":
            # Flush any previous turn (new assistant = boundary)
            if pending_assistant is not None:
                _flush_turn(
                    result, pending_assistant, pending_call_ids,
                    pending_real_tools,
                )
            pending_assistant = m
            pending_call_ids = _collect_call_ids(m)
            pending_real_tools = []

        elif role == "tool":
            # Collect tool for the current turn; do NOT flush yet
            # (otherwise we'd lose the chance to add real tools after assistant)
            pending_real_tools.append(m)

        else:
            # User/system message: flush pending turn, then append this
            if pending_assistant is not None:
                _flush_turn(
                    result, pending_assistant, pending_call_ids,
                    pending_real_tools,
                )
                pending_assistant = None
                pending_call_ids = []
                pending_real_tools = []
            result.append(m)

    # Flush final pending turn
    if pending_assistant is not None:
        _flush_turn(
            result, pending_assistant, pending_call_ids, pending_real_tools,
        )

    # Split intervention: FW always early, diagnosis at end.
    fw_text = _format_fw(failure_witness) if failure_witness else None
    diagnosis_text = diagnosis if (style == "condiag" and diagnosis) else None

    # 1) FW goes RIGHT AFTER the PR description (first user message) so the agent
    #    immediately knows the patch failed. No point reading history without context.
    if fw_text:
        insert_idx = 0
        for i, m in enumerate(result):
            if m.get("role") == "user":
                insert_idx = i + 1
                break
        result.insert(insert_idx, {"role": "user", "content": fw_text})

    # 2) Diagnosis goes at the END — after the full (compressed) history —
    #    so the agent understands what R1 tried BEFORE receiving guidance.
    #    This prevents the overconfidence trap where the diagnosis + compact
    #    history causes the agent to submit without exploring.
    if diagnosis_text:
        result.append({"role": "user", "content": diagnosis_text})

    return result

    return result


def _flush_turn(
    result: list[dict],
    assistant: dict,
    call_ids: list[str],
    real_tools: list[dict],
) -> None:
    """Append one complete Turn in order: assistant, real tool responses,
    then synthetic tool responses for any missing call_ids.
    """
    # Assistant first
    result.append(assistant)

    # Real tool responses (preserving original order)
    real_ids: set[str] = set()
    for tool in real_tools:
        result.append(tool)
        tid = tool.get("tool_call_id", "")
        if tid:
            real_ids.add(tid)

    # Synthetic responses for any missing IDs (deduped)
    for tid in call_ids:
        if tid and tid not in real_ids:
            result.append({"role": "tool", "tool_call_id": tid, "content": "(output)"})


def _collect_call_ids(msg: dict) -> list[str]:
    """Extract tool_call_ids from both top-level tool_calls and extra.actions.

    Deduplicates by ID (preserving first occurrence order). The same call
    can be represented in both tool_calls and extra.actions, so we must
    not generate duplicate synthetic responses for the same ID.
    """
    ids = []
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            ids.append(tc.get("id") or tc.get("tool_call_id") or "")
    for act in (msg.get("extra") or {}).get("actions", []) or []:
        if isinstance(act, dict):
            ids.append(act.get("tool_call_id") or "")
    # Dedupe, preserve order
    return list(dict.fromkeys(t for t in ids if t))


def _format_fw(fw: dict) -> str:
    """Build failure witness user message text."""
    failed = [str(f) if not isinstance(f, str) else f
              for f in fw.get("failed_tests", [])]
    error = fw.get("error_message", "")
    parts = ["## Validation Result\n\nYour submitted patch did not pass validation.\n"]
    if failed:
        parts.append(f"Failed tests: {', '.join(failed)}\n")
    if error:
        parts.append(f"Error:\n```\n{error}\n```")
    parts.append("\nPlease investigate and revise your patch.")
    return "".join(parts)


def _format_planning_instructions(style: str = "stateful_feedback") -> str:
    """Plan-then-Act shell. NOT USED in v2 ablation — kept for future use."""
    return ""


# ── CD context reshaping ─────────────────────────────────────────────


def build_condiag_branch_messages(
    checkpoint_messages: list[dict],
    failure_witness: dict | None,
    refined_diagnoses: list,
    router_results: list,
    revision_contract: dict,
    compressed: bool = True,
) -> list[dict]:
    """Build the message sequence for the ConDiag (CD) branch.

    Produces a single structured continuation message that includes:
      1. Validation Result (from FW)
      2. R1 patch summary
      3. Primary patch-linked failures (PATCH_LINKED / PRIMARY)
      4. Causal mechanisms + evidence
      5. Invalidated assumptions
      6. Unsafe repair warnings
      7. Retrieved evidence (max 5 items)
      8. Primary edit target
      9. Revision objective
      10. Validation targets
      11. Monitor-only failures

    Returns messages ready to pass as checkpoint_messages to run_branch()
    (with failure_witness=None, diagnosis=None since everything is embedded).
    """
    # Step 1: Turn repair (same as build_branch_messages)
    msgs = deepcopy(checkpoint_messages)
    msgs = [m for m in msgs if m.get("role") != "exit"]

    result: list[dict] = []
    pending_assistant: dict | None = None
    pending_call_ids: list[str] = []
    pending_real_tools: list[dict] = []

    for m in msgs:
        role = m.get("role", "")
        if role == "assistant":
            if pending_assistant is not None:
                _flush_turn(result, pending_assistant, pending_call_ids, pending_real_tools)
            pending_assistant = m
            pending_call_ids = _collect_call_ids(m)
            pending_real_tools = []
        elif role == "tool":
            pending_real_tools.append(m)
        else:
            if pending_assistant is not None:
                _flush_turn(result, pending_assistant, pending_call_ids, pending_real_tools)
                pending_assistant = None
                pending_call_ids = []
                pending_real_tools = []
            result.append(m)
    if pending_assistant is not None:
        _flush_turn(result, pending_assistant, pending_call_ids, pending_real_tools)

    # Step 2: Build continuation message
    continuation = _format_condiag_continuation(
        failure_witness, refined_diagnoses, router_results, revision_contract,
    )

    # Append the CD continuation at the END — after the full (compressed) history.
    # The agent sees R1's exploration first, building context about the failure,
    # then receives diagnosis-driven guidance on what to explore next.
    # This avoids the overconfidence trap (compacted context + upfront diagnosis
    # → agent submits without exploring).
    result.append({"role": "user", "content": continuation})

    return result


def _format_condiag_continuation(
    failure_witness: dict | None,
    refined_diagnoses: list,
    router_results: list,
    revision_contract: dict,
    max_chars: int = 12000,
) -> str:
    """Build the CD continuation text with budget control.

    Action-first format: Revision Brief (Inspect/Edit/Forbid/Verify)
    then Detailed Analysis (validation result, linked failures, evidence).

    The brief is always rendered; the detail is capped by max_chars.
    """
    from reconstruction.revision_brief import (
        build_revision_brief,
        render_revision_brief,
        render_diagnosis_detail,
    )
    rc = revision_contract or {}

    brief = build_revision_brief(rc, refined_diagnoses, router_results, failure_witness)
    brief_text = render_revision_brief(brief, max_chars=800)

    if len(brief_text) > max_chars - 500:
        return brief_text[:max_chars]

    detail = render_diagnosis_detail(
        failure_witness, refined_diagnoses, router_results, rc,
        max_chars=max_chars - len(brief_text),
    )

    return brief_text + "\n" + detail
