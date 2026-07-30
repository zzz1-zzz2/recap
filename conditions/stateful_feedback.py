"""Stateful Feedback — baseline condition (no diagnosis, no reconstruction).

Wraps condiag.branch_runner for R2 execution. This is the "SF" condition
in the 2×2 protocol.
"""
from branch_runner import BranchResult, RestoreResult, run_branch
from workspace import capture_workspace_fingerprint

__all__ = ["run_branch", "BranchResult", "RestoreResult", "capture_workspace_fingerprint"]
