"""Continuation state z₂ = (K₂, ρ₂)."""
from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class ContinuationState:
    """The continuation state supplied to R2.

    K₂: packed context messages (diagnosis-conditioned trajectory + evidence).
    ρ₂_text: rendered revision brief + diagnosis detail.
    plan_summary: human-readable summary of the reconstruction plan.
    """

    K2_messages: list[dict] = dataclasses.field(default_factory=list)
    rho2_text: str = ""
    plan_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_K2_messages": len(self.K2_messages),
            "rho2_length": len(self.rho2_text),
            "plan_summary": self.plan_summary,
        }
