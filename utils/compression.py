"""ConDiag Context Compression — compress message history for Round 2.

Strategies (Phase 1: heuristic, no diagnosis dependency):
  1. mask_observations: truncate large tool stdout, keep command + return code
  2. consolidate_retries: merge consecutive "no tool call" retries into one
  3. truncate_window: keep only the last K turns of interaction
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any

logger = logging.getLogger("condiag.compression")

MAX_TOOL_OUTPUT_CHARS = 500  # max chars to keep per tool response
MAX_WINDOW_TURNS = 20        # max (assistant + tool) pairs to keep
MAX_TOTAL_CHARS = 80000      # target max total chars (~20K tokens)


def compress_messages(
    messages: list[dict],
    *,
    max_tool_output: int = MAX_TOOL_OUTPUT_CHARS,
    max_turns: int = MAX_WINDOW_TURNS,
    max_total_chars: int = MAX_TOTAL_CHARS,
) -> list[dict]:
    """Compress message history for injection into Round 2.

    Args:
        messages: Full message history from Round 1 checkpoint.
        max_tool_output: Max chars to keep per tool response.
        max_turns: Max (assistant+tool) pairs to retain.
        max_total_chars: Target max total chars.

    Returns:
        Compressed message list.
    """
    msgs = deepcopy(messages)

    # Phase 1: Consolidate consecutive format-error retry messages
    msgs = _consolidate_retries(msgs)

    # Phase 2: Truncate large tool outputs
    msgs = _mask_observations(msgs, max_chars=max_tool_output)

    # Phase 3: Window truncation (if still too large)
    msgs = _truncate_window(msgs, max_turns=max_turns)

    # Phase 4: Final size cap (if still over budget)
    msgs = _cap_total_size(msgs, max_chars=max_total_chars)

    before = len(messages)
    after = len(msgs)
    logger.info(
        "Compressed: %d→%d msgs, %s→%s chars (%.0f%% reduction)",
        before, after,
        _total_chars(messages), _total_chars(msgs),
        (1 - _total_chars(msgs) / max(_total_chars(messages), 1)) * 100,
    )
    return msgs


# ─── Internal strategies ───────────────────────────────────────────────


def _consolidate_retries(messages: list[dict]) -> list[dict]:
    """Strip all format-error retry messages."""
    result: list[dict] = []

    for m in messages:
        role = m.get("role", "")
        content = str(m.get("content", ""))

        if role == "user" and ("No tool calls found" in content or "Error parsing tool call" in content):
            continue

        result.append(m)

    # NOTE: we do NOT insert a summary note here because it would break
    # the assistant(tool_calls) → tool message alternation required by the API.
    # See: https://platform.openai.com/docs/guides/function-calling#managing-tool-calls

    return result


def _mask_observations(messages: list[dict], max_chars: int) -> list[dict]:
    """Truncate large tool output messages.

    Keep the command description and return code, truncate the stdout.
    """
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content", "") or ""
        if len(content) <= max_chars:
            continue

        # Extract returncode from HTML-like tags
        returncode = ""
        if "<returncode>" in content and "</returncode>" in content:
            rc_start = content.find("<returncode>") + len("<returncode>")
            rc_end = content.find("</returncode>")
            returncode = content[rc_start:rc_end]
        output_start = content.find("<output>") + len("<output>") if "<output>" in content else 0
        output_end = content.find("</output>")
        if output_end == -1:
            output_end = len(content)
        output_text = content[output_start:output_end]
        truncated = output_text[:max_chars]
        m["content"] = (
            f"<returncode>{returncode}</returncode>\n"
            f"<output>\n{truncated}\n"
            f"... (truncated, was {len(content)} chars)\n"
            f"</output>"
        )
    return messages


def _truncate_window(messages: list[dict], max_turns: int) -> list[dict]:
    """Keep first N system/user + last K turns of (assistant + tool) pairs.

    A "turn" = one assistant message + ALL following tool messages
    (handle parallel tool calls where multiple tool responses follow one assistant).
    """
    # Identify structure: leading non-assistant messages, then alternating turns
    leading: list[dict] = []
    turns: list[list[dict]] = []
    current_turn: list[dict] = []
    seen_tool_for_current = False

    for m in messages:
        role = m.get("role", "")

        if role in ("system",):
            leading.append(m)
            continue

        # Leading user messages (before any assistant)
        if role == "user" and not current_turn and not turns:
            leading.append(m)
            continue

        if role == "assistant":
            # If we have a pending turn, save it before starting new one
            if current_turn:
                turns.append(current_turn)
                current_turn = []
            current_turn.append(m)
            seen_tool_for_current = False
            continue

        if role == "tool":
            current_turn.append(m)
            continue

        # Any other role (unlikely but defensive): treat as leading
        leading.append(m)

    # If there's a partial turn at the end, keep it
    if current_turn:
        turns.append(current_turn)

    # Keep leading + last max_turns
    if len(turns) > max_turns:
        turns = turns[-max_turns:]

    result = leading[:]
    for turn in turns:
        result.extend(turn)
    return result


def _cap_total_size(messages: list[dict], max_chars: int) -> list[dict]:
    """If still over budget, drop oldest complete turns."""
    # Group into turns (same logic as _truncate_window)
    leading, turns, current = [], [], []
    for m in messages:
        r = m.get("role", "")
        if r == "system":
            leading.append(m)
        elif r == "user" and not current and not turns:
            leading.append(m)
        elif r == "assistant":
            if current:
                turns.append(current)
                current = []
            current.append(m)
        elif r == "tool":
            current.append(m)
        else:
            leading.append(m)
    if current:
        turns.append(current)

    # Drop oldest turns until under budget
    while True:
        all_msgs = leading + [m for t in turns for m in t]
        if _total_chars(all_msgs) <= max_chars or len(all_msgs) <= 5:
            break
        if not turns:
            break
        turns.pop(0)

    return leading + [m for t in turns for m in t]


def _total_chars(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages)


# ─── Utility ───────────────────────────────────────────────────────────


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate (1 token ≈ 4 chars for code/text)."""
    return _total_chars(messages) // 4


def get_message_stats(messages: list[dict]) -> dict:
    """Return stats for debugging."""
    stats: dict = {"total": len(messages), "total_chars": _total_chars(messages)}
    for role in ("system", "user", "assistant", "tool"):
        subset = [m for m in messages if m.get("role") == role]
        if subset:
            chars = [len(str(m.get("content", ""))) for m in subset]
            stats[role] = {
                "count": len(subset),
                "total_chars": sum(chars),
                "avg_chars": sum(chars) // len(subset),
                "max_chars": max(chars),
            }
    return stats
