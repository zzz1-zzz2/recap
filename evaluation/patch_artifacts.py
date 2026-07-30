"""Patch Provenance — track every patch state through the episode.

Four distinct patch objects:
  1. agent_submitted_patch:  What the agent explicitly submitted (via patch.txt)
  2. evaluation_patch:        What's actually sent to the harness (after integrity check)
  3. workspace_snapshot:      Full workspace state at Round 1 completion (for Round 2 restore)
  4. final_evaluation_patch:  Round 2's cumulative source patch for harness

Design:
  - AgentSubmission collects from multiple candidate sources
  - PatchArtifacts holds all four patch states with SHAs
  - No silent filtering: illegal patches are BLOCKED, not auto-corrected
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def sha256_full(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_short(text: str) -> str:
    return sha256_full(text)[:16]


@dataclass
class AgentSubmission:
    """What the agent submitted — collected from multiple candidate sources.

    Sources (in priority order):
      1. exception_payload:  The Submitted exception's payload
      2. patch_file_content: Container's /testbed/patch.txt at submission time
      3. submit_command_output: Last command's stdout

    selected_patch is the chosen version.
    If sources disagree, consistency_status = SUSPICIOUS_SUBMISSION_MISMATCH.
    """

    exception_payload: str = ""
    patch_file_content: str = ""
    submit_command_output: str = ""
    selected_patch: str = ""
    selected_source: str = ""
    consistency_status: str = "unknown"  # "consistent" | "mismatch"

    @property
    def sha(self) -> str:
        return sha256_short(self.selected_patch) if self.selected_patch else ""

    def to_dict(self) -> dict:
        return {
            "sha": self.sha,
            "selected_source": self.selected_source,
            "consistency_status": self.consistency_status,
            "selected_patch_length": len(self.selected_patch),
        }


def collect_agent_submission(
    agent_messages: list[dict] | None = None,
    exception_payload: str | None = None,
    patch_file_text: str | None = None,
    command_output: str | None = None,
) -> AgentSubmission:
    """Collect agent submission from all available sources and select the best one.

    Priority: agent_messages (exit.extra.submission) > exception_payload > patch_file_content > command_output
    If the selected patch differs significantly from other sources, flag mismatch.
    """
    from .patch_artifacts import extract_submission_from_messages  # type: ignore
    from .patch_artifacts import canonicalize_patch  # type: ignore

    msg_payload, msg_source = extract_submission_from_messages(agent_messages or [])

    sub = AgentSubmission(
        exception_payload=str(exception_payload) if exception_payload else "",
        patch_file_content=patch_file_text or "",
        submit_command_output=command_output or "",
    )

    # Select
    if msg_payload.strip():
        sub.selected_patch = msg_payload.strip()
        sub.selected_source = msg_source
    elif sub.exception_payload.strip():
        sub.selected_patch = sub.exception_payload.strip()
        sub.selected_source = "exception_payload"
    elif sub.patch_file_content.strip():
        sub.selected_patch = sub.patch_file_content.strip()
        sub.selected_source = "patch_file_content"
    elif sub.submit_command_output.strip():
        sub.selected_patch = sub.submit_command_output.strip()
        sub.selected_source = "submit_command_output"
    else:
        sub.selected_patch = ""
        sub.selected_source = "none"

    # Consistency check (compare with workspace patch if provided separately)
    sources = []
    if msg_payload.strip():
        sources.append(msg_payload.strip())
    if sub.exception_payload.strip():
        sources.append(sub.exception_payload.strip())
    if sub.patch_file_content.strip():
        sources.append(sub.patch_file_content.strip())
    if sub.submit_command_output.strip():
        sources.append(sub.submit_command_output.strip())

    if len(sources) >= 2:
        # Compare each source's canonicalized SHA
        canonicals = {sha256_short(canonicalize_patch(s)) for s in sources}
        sub.consistency_status = "mismatch" if len(canonicals) > 1 else "consistent"
    elif len(sources) == 1:
        sub.consistency_status = "consistent"
    else:
        sub.consistency_status = "none"

    return sub


# ── Four-corners patch tracking ─────────────────────────────────────


@dataclass
class PatchArtifacts:
    """All patch states for one episode.

    Rules:
      - evaluation_patch must be a canonicalized version of agent_submitted_patch
        (not independently extracted from workspace)
      - agent_submitted_sha == evaluation_sha is NOT required — but the relationship
        must be auditable
      - workspace_snapshot is a superset of evaluation_patch (includes untracked files)
    """

    agent_submitted: AgentSubmission = field(default_factory=AgentSubmission)
    evaluation_patch: str = ""
    workspace_snapshot: str = ""  # Full git diff at Round 1 completion
    final_evaluation_patch: str = ""  # Round 2 cumulative source patch

    @property
    def evaluation_sha(self) -> str:
        return sha256_short(self.evaluation_patch) if self.evaluation_patch else ""

    @property
    def workspace_sha(self) -> str:
        return sha256_short(self.workspace_snapshot) if self.workspace_snapshot else ""

    def to_dict(self) -> dict:
        return {
            "agent_submitted": self.agent_submitted.to_dict(),
            "evaluation_patch_sha": self.evaluation_sha,
            "workspace_snapshot_sha": self.workspace_sha,
            "submitted_eq_evaluation": self.agent_submitted.sha == self.evaluation_sha,
        }


def canonicalize_patch(patch_text: str) -> str:
    """Normalize a patch for consistent comparison.

    Does NOT change semantic content — only whitespace normalization
    and stable sorting of diff headers if there are multiple files.
    """
    if not patch_text or not patch_text.strip():
        return ""
    return patch_text.strip() + "\n"


def extract_submission_from_messages(messages: list[dict]) -> tuple[str, str]:
    """Extract Agent's submitted patch from the final exit message.

    Returns:
        (submitted_patch, source) — submitted_patch text and the source name.

    mini-SWE-agent's default flow puts the final submission in the exit
    message's `extra.submission` field. We look for the LAST message with
    exit_status=Submitted and return its submission text.
    """
    for message in reversed(messages or []):
        if message.get("role") != "exit":
            continue
        extra = message.get("extra") or {}
        if extra.get("exit_status") == "Submitted":
            submission = extra.get("submission") or ""
            return str(submission), "exit_extra_submission"
    return "", "no_submission"


def patch_consistency_check(
    agent_submitted: str, workspace_patch: str
) -> str:
    """Compare agent_submitted and workspace_patch after canonicalization.

    Returns: "consistent" | "mismatch" | "empty"
    """
    a = canonicalize_patch(agent_submitted)
    w = canonicalize_patch(workspace_patch)
    if not a and not w:
        return "empty"
    if not a:
        return "empty"
    return "consistent" if a == w else "mismatch"
