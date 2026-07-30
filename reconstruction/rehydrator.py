"""Rehydrator — restore R1-seen evidence that was dropped from K_2.

Paper definition (§3.3):
  Rehydration restores repository observations that were available during
  the first attempt but are absent from the active continuation context.
  Such evidence may have been truncated, summarized, or displaced by later
  tool output.

Rehydrate is NOT the same as Preserve:
  - Preserve: keep evidence whose relevance survives validation
  - Rehydrate: RESTORE evidence that WAS in R1's trajectory but got
    dropped by packing/compression

Implementation:
  1. Scan R1 trajectory for files that were viewed (especially repeatedly).
  2. Check which of those files are NOT in K_plus (the preserve set).
  3. For files that are important to the diagnosis but got dropped,
     create REHYDRATE ContextUnits to restore them.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from reconstruction.context_unit import ContextUnit, make_unit, total_tokens

logger = logging.getLogger("condiag.reconstruction.rehydrator")

# Regex to extract file paths from trajectory content
_RE_FILE_REF = re.compile(r"[\w/]+\.py(?::\d+)?")


def identify_rehydratable_units(
    messages: list[dict],
    K_plus_paths: set[str],
    hypotheses: list,
    refined_diagnoses: list,
    max_units: int = 5,
) -> list[ContextUnit]:
    """Identify R1-seen evidence that should be restored.

    Scans the trajectory for files that were:
      (a) viewed multiple times (high engagement)
      (b) related to diagnosis targets (file names match hypothesis symbols)
      (c) NOT already in K_plus (dropped by packing)

    Returns REHYDRATE ContextUnits for reinsertion into K_2.
    """
    # Count file view frequency in trajectory
    file_views: dict[str, int] = {}
    file_content: dict[str, str] = {}  # last-seen content snippet

    for m in messages:
        content = str(m.get("content", "") or "")
        for match in _RE_FILE_REF.finditer(content):
            fp = match.group().split(":")[0]
            if fp.endswith(".py"):
                file_views[fp] = file_views.get(fp, 0) + 1
                # Store a content snippet from this turn
                lines = [l for l in content.split("\n") if fp in l]
                if lines:
                    file_content[fp] = lines[0][:200]

    # Collect diagnosis targets (files and symbols that matter)
    diagnosis_targets: set[str] = set()
    for h in hypotheses:
        diagnosis_targets.update(getattr(h, "retrieval_targets", []) or [])
        diagnosis_targets.update(getattr(h, "failure_sites", []) or [])
    for d in refined_diagnoses:
        for sym in (getattr(d, "target_symbols", []) or []):
            diagnosis_targets.add(sym)

    # Find rehydratable files: viewed >= 2 times, NOT in K_plus, and
    # related to diagnosis targets
    rehydratable: list[ContextUnit] = []
    seen: set[str] = set()

    for fp, count in sorted(file_views.items(), key=lambda x: -x[1]):
        if fp in K_plus_paths or fp in seen:
            continue
        if len(rehydratable) >= max_units:
            break

        # Check relevance to diagnosis
        is_relevant = False
        fp_stem = fp.split("/")[-1].replace(".py", "").lower()
        for target in diagnosis_targets:
            t_stem = target.split("/")[-1].replace(".py", "").lower()
            if fp_stem == t_stem or t_stem in fp_stem or fp_stem in t_stem:
                is_relevant = True
                break

        # Rehydrate if viewed multiple times AND diagnosis-relevant
        # OR if viewed many times (≥3) even without explicit diagnosis match
        if (count >= 2 and is_relevant) or count >= 3:
            snippet = file_content.get(fp, fp)
            seen.add(fp)
            rehydratable.append(make_unit(
                content=f"{fp}\n{_extract_snippet(messages, fp)}",
                operation="REHYDRATE",
                provenance=fp,
                priority=2,
                source_type="trajectory",
                original_turn_ids=[],
            ))

    if rehydratable:
        logger.info(
            "Rehydrate: %d units restored (from %d trajectory files)",
            len(rehydratable), len(file_views),
        )

    return rehydratable


def _extract_snippet(messages: list[dict], filepath: str, max_lines: int = 10) -> str:
    """Extract a code snippet for a rehydrated file from trajectory."""
    for m in reversed(messages):
        content = str(m.get("content", "") or "")
        if filepath in content:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if filepath in line:
                    snippet = "\n".join(lines[i:i + max_lines])
                    return snippet[:2000]
    return ""
