"""ContextUnit — unified evidence representation for reconstruction.

Every operation (Preserve, Rehydrate, Suppress, Acquire) produces or
consumes ContextUnits. This is the single data type that flows through
the Plan → Acquire → Pack → Guide pipeline.

All units carry stable IDs and provenance so every reconstruction
decision is auditable.
"""
from __future__ import annotations

import hashlib
import json
import dataclasses
from typing import Any


@dataclasses.dataclass
class ContextUnit:
    """One unit of evidence in the reconstruction pipeline.

    Fields:
        unit_id:         Stable content-addressed ID.
        content:         Actual text content (code snippet, test output, etc.).
        operation:       Which operation produced this unit.
        provenance:      Where the content came from (file path, turn index, etc.).
        token_count:     Character count (proxy for token count).
        priority:        Pack priority (0=mandatory, 1=high, 2=medium, 3=low, 4=fill).
        source_type:     'trajectory' | 'repo' | 'failure_witness' | 'synthetic'.
        diagnosis_ids:   Which diagnoses justify this unit.
        evidence_ids:    Which evidence ledger items support this unit.
        target_symbol:   Optional symbol this unit is about.
        line_start:      Optional line range start.
        line_end:        Optional line range end.
        original_turn_ids: Trajectory turn indices this unit came from.
    """
    unit_id: str = ""
    content: str = ""
    operation: str = ""          # PRESERVE | REHYDRATE | ACQUIRE | SUPPRESS | MANDATORY
    provenance: str = ""         # file path, test name, or "failure_witness"
    token_count: int = 0
    priority: int = 4
    source_type: str = "trajectory"  # trajectory | repo | failure_witness | synthetic
    diagnosis_ids: list[str] = dataclasses.field(default_factory=list)
    evidence_ids: list[str] = dataclasses.field(default_factory=list)
    target_symbol: str | None = None
    line_start: int | None = None
    line_end: int | None = None
    original_turn_ids: list[int] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not self.unit_id:
            raw = f"{self.operation}|{self.provenance}|{self.content[:200]}"
            self.unit_id = "U" + hashlib.sha256(raw.encode()).hexdigest()[:12]
        if not self.token_count and self.content:
            self.token_count = len(self.content)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ── Helpers ─────────────────────────────────────────────────────────────


def make_unit(
    content: str,
    operation: str,
    provenance: str,
    *,
    priority: int = 4,
    source_type: str = "trajectory",
    diagnosis_ids: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    target_symbol: str | None = None,
    line_start: int | None = None,
    line_end: int | None = None,
    original_turn_ids: list[int] | None = None,
) -> ContextUnit:
    """Factory: create a ContextUnit with auto-generated unit_id."""
    return ContextUnit(
        content=content,
        operation=operation,
        provenance=provenance,
        priority=priority,
        source_type=source_type,
        diagnosis_ids=diagnosis_ids or [],
        evidence_ids=evidence_ids or [],
        target_symbol=target_symbol,
        line_start=line_start,
        line_end=line_end,
        original_turn_ids=original_turn_ids or [],
    )


def serialize_units(units: list[ContextUnit]) -> list[dict]:
    """Serialize a list of ContextUnits for artifact output."""
    return [u.to_dict() for u in units]


def total_tokens(units: list[ContextUnit]) -> int:
    """Sum token counts across units."""
    return sum(u.token_count for u in units)


def units_by_operation(units: list[ContextUnit]) -> dict[str, list[ContextUnit]]:
    """Group units by operation type."""
    groups: dict[str, list[ContextUnit]] = {}
    for u in units:
        groups.setdefault(u.operation, []).append(u)
    return groups


def units_by_priority(units: list[ContextUnit]) -> dict[int, list[ContextUnit]]:
    """Group units by priority level."""
    groups: dict[int, list[ContextUnit]] = {}
    for u in units:
        groups.setdefault(u.priority, []).append(u)
    return groups
