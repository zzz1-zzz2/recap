"""Condition configuration — feature flags for 2×2 ablation protocol.

Conditions:
  SF (Stateful Feedback):   diagnosis=0, reconstruction=0
  GR (Generic Reconstruct): diagnosis=0, reconstruction=1
  SG (Structured Guidance): diagnosis=1, reconstruction=0
  ReCAP:                    diagnosis=1, reconstruction=1

reconstruction=1 means: Preserve + Rehydrate + Suppress + Acquire + Pack + Guide
reconstruction=0 means: no diagnosis-guided packing, no acquire, no targeted suppress
"""
from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class RecoveryCondition:
    """Feature flags for one recovery condition."""

    name: str
    use_diagnosis: bool = False
    use_reconstruction: bool = False
    # Sub-flags (only meaningful when reconstruction=True)
    use_preserve: bool = True
    use_rehydrate: bool = True
    use_suppress: bool = True
    use_acquire: bool = True
    use_pack: bool = True


# The four canonical conditions
SF = RecoveryCondition(
    name="sf",
    use_diagnosis=False,
    use_reconstruction=False,
)
GR = RecoveryCondition(
    name="gr",
    use_diagnosis=False,
    use_reconstruction=True,
    use_preserve=True,
    use_rehydrate=True,
    use_suppress=True,
    use_acquire=True,
    use_pack=True,
)
SG = RecoveryCondition(
    name="sg",
    use_diagnosis=True,
    use_reconstruction=False,
)
RECAP = RecoveryCondition(
    name="recap",
    use_diagnosis=True,
    use_reconstruction=True,
    use_preserve=True,
    use_rehydrate=True,
    use_suppress=True,
    use_acquire=True,
    use_pack=True,
)

ALL_CONDITIONS = [SF, GR, SG, RECAP]
