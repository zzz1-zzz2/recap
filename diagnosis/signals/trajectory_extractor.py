"""Trajectory extractor — extract structured signals from agent trajectory."""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter

from .schema import TrajectorySignals

logger = logging.getLogger("condiag.diagnosis.signals.trajectory_extractor")

# ── Patterns ────────────────────────────────────────────────────────

_RE_BASH_COMMAND = re.compile(r"```bash\s*\n(.+?)```", re.DOTALL)
_RE_TOOL_OUTPUT_FILE = re.compile(r"/testbed/([^\s\"']+\.py)")
_RE_CAT_FILE = re.compile(r"(?:^|\s)(?:cat|head|tail|sed|nl|wc)\s+(-[a-zA-Z0-9]+\s+)?(/testbed/\S+|[^\s;|&`()]+\.py)")
_RE_VIEW_FILE = re.compile(r"(?:^|\s)(?:grep|rg|find|ack|ag)\s+(-[a-zA-Z0-9]+\s+)?['\"]?[^'\"]+['\"]?\s+(/[^\s;|&`]+\.py|[^\s;|&`]+\.py)")


def _repo_path(fp: str) -> str:
    """Strip /testbed/ prefix if present."""
    if fp.startswith("/testbed/"):
        return fp[len("/testbed/"):]
    return fp


def _tool_event_key(
    event_id: str,
    name: str,
    source: str,
    turn_index: int = 0,
    arguments: Any = None,
) -> str:
    """Generate a canonical key for a tool event across tool_calls and actions.

    If event_id is non-empty, use it (same call dedup'd across tool_calls/actions).
    Otherwise produce a content-based hash using (turn_index, source_type, tool_name,
    normalized arguments). This ensures multiple anonymous calls are distinct
    while a call replicated in both tool_calls and extra.actions dedupes.
    """
    if event_id:
        return f"evt:{event_id}"
    import json
    canonical = json.dumps(
        {
            "turn": turn_index,
            "source": source,
            "name": name,
            "args": arguments if arguments is not None else "",
        },
        sort_keys=True,
    )
    return "anon:" + hashlib.sha256(canonical.encode()).hexdigest()


def extract_trajectory_signals(trajectory: dict) -> TrajectorySignals:
    """Extract structured signals from a mini-SWE-agent trajectory dict.

    Returns:
        TrajectorySignals with populated fields.
    """
    signals = TrajectorySignals()
    messages = trajectory.get("messages", []) if isinstance(trajectory, dict) else trajectory

    tool_call_counter: Counter = Counter()
    file_view_counter: Counter = Counter()
    format_error_count = 0
    bash_command_count = 0
    test_command_count = 0
    seen_events: set[str] = set()  # canonical keys for dedup
    turn_index = 0

    for msg in messages:
        role = msg.get("role", "")

        if role == "assistant":
            tc = msg.get("tool_calls") or []
            actions = (msg.get("extra") or {}).get("actions") or []
            signals.assistant_turn_count += 1
            turn_index += 1

            # Collect command texts from arguments
            cmd_texts: list[str] = []

            # Process tool_calls
            for t in tc:
                fn = t.get("function", {}) if isinstance(t, dict) else {}
                if isinstance(fn, dict):
                    name = fn.get("name", "")
                    args_raw = fn.get("arguments", "")
                    if isinstance(args_raw, str):
                        cmd_texts.append(args_raw)
                    elif isinstance(args_raw, dict):
                        cmd_texts.append(str(args_raw))
                    eid = t.get("id", "") or t.get("tool_call_id", "")
                    key = _tool_event_key(eid, name, "tc", turn_index, args_raw)
                    if key not in seen_events:
                        seen_events.add(key)
                        tool_call_counter[name] += 1

            # Process extra.actions
            for act in actions:
                if isinstance(act, dict):
                    act_name = act.get("type", "bash")
                    act_cmd = act.get("command", act.get("arguments", ""))
                    if isinstance(act_cmd, str) and act_cmd.strip():
                        cmd_texts.append(act_cmd)
                    eid = act.get("id", "") or act.get("tool_call_id", "")
                    # Use same canonical key as tool_calls for the same call
                    key = _tool_event_key(eid, act_name, "tc", turn_index, act_cmd)
                    if key not in seen_events:
                        seen_events.add(key)
                        tool_call_counter[act_name] += 1

            # Extract viewed files from command arguments
            for txt in cmd_texts:
                if not isinstance(txt, str):
                    continue
                for m in _RE_CAT_FILE.finditer(txt):
                    fp = m.group(2) if m.group(2) else m.group(3)
                    if fp:
                        file_view_counter[_repo_path(fp)] += 1
                for m in _RE_VIEW_FILE.finditer(txt):
                    fp = m.group(2) if m.group(2) else m.group(3)
                    if fp:
                        file_view_counter[_repo_path(fp)] += 1

            # Detect bash commands from content
            content = msg.get("content", "") or ""
            cmds = _RE_BASH_COMMAND.findall(content)
            for cmd in cmds:
                bash_command_count += 1
                if any(kw in cmd for kw in ["pytest", "python -m pytest", "python -m django", "tox"]):
                    test_command_count += 1

        elif role == "tool":
            content = str(msg.get("content", "") or "")
            for m in _RE_TOOL_OUTPUT_FILE.finditer(content):
                file_view_counter[_repo_path(m.group(1))] += 1

        elif role == "user":
            content = msg.get("content", "") or ""
            if "No tool calls found" in content or "Error parsing tool call" in content:
                format_error_count += 1

    # Total unique events
    signals.total_tool_calls = len(seen_events)

    signals.format_error_count = format_error_count
    signals.tool_type_counts = dict(tool_call_counter)
    signals.viewed_files = sorted(file_view_counter.keys())
    signals.file_view_counts = dict(file_view_counter)
    signals.bash_commands_run = bash_command_count
    signals.test_commands_run = test_command_count

    # Exploration concentration: ratio of top-5 most-viewed files to total views
    total_views = sum(file_view_counter.values())
    if total_views:
        top5 = sum(c for _, c in file_view_counter.most_common(5))
        signals.exploration_concentration = top5 / total_views
    else:
        signals.exploration_concentration = 0.0

    return signals
