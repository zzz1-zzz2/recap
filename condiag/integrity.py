"""Patch Integrity Gate - block invalid patches before Harness."""
from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RE_HUNK = re.compile(r"^@@\s*-?\d", re.MULTILINE)
_RE_NEW_FILE = re.compile(r"^new file mode", re.MULTILINE)
_RE_PARENT_TRAVERSAL = re.compile(r"\.\./|\.\.\\")
_RE_ABSOLUTE = re.compile(r"^/|[A-Z]:\\")
_RE_DOT_GIT = re.compile(r"(^|/)(\.git)(/|$)")


@dataclass
class PatchIntegrityReport:
    ok: bool
    status: str
    reason: str
    changed_files: list
    patch_size: int
    fallback_used: bool
    consistency: str
    submitted_sha: str
    workspace_sha: str
    evaluation_sha: str
    apply_check_status: str = "not_run"

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "reason": self.reason,
            "changed_files": self.changed_files,
            "patch_size": self.patch_size,
            "fallback_used": self.fallback_used,
            "consistency": self.consistency,
            "submitted_sha": self.submitted_sha,
            "workspace_sha": self.workspace_sha,
            "evaluation_sha": self.evaluation_sha,
            "apply_check_status": self.apply_check_status,
        }


def extract_changed_files(patch_text):
    """Extract file paths from diff --git headers using shlex (handles spaces/quotes)."""
    if not patch_text:
        return []
    files = []
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) != 4:
            continue
        if parts[0] != "diff" or parts[1] != "--git":
            continue
        if not parts[2].startswith("a/") or not parts[3].startswith("b/"):
            continue
        new_path = parts[3]
        if new_path.startswith("b/"):
            new_path = new_path[2:]
        files.append(new_path)
    return files


def has_valid_diff_header(patch_text):
    """Check if patch has at least one parseable diff --git header line."""
    if not patch_text:
        return False
    for line in patch_text.splitlines():
        if not line.startswith("diff --git "):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            continue
        if len(parts) == 4:
            return True
    return False


def is_safe_path(file_path):
    if not file_path:
        return False
    if _RE_PARENT_TRAVERSAL.search(file_path):
        return False
    if _RE_ABSOLUTE.match(file_path):
        return False
    if _RE_DOT_GIT.search(file_path):
        return False
    return True


def is_valid_unified_diff(patch_text):
    if not patch_text or not patch_text.strip():
        return False
    has_header = has_valid_diff_header(patch_text)
    has_hunk_or_new = bool(_RE_HUNK.search(patch_text) or _RE_NEW_FILE.search(patch_text))
    return has_header and has_hunk_or_new


def check_patch_integrity(
    termination_reason,
    agent_submission=None,
    workspace_patch="",
    evaluation_patch="",
    agent_workdir=None,
    run_apply_check=False,
):
    """Validate a patch before Harness submission.

    Decision rules (in order):
      1. termination_reason must be submitted
      2. evaluation_patch must be non-empty
      3. evaluation_patch must be a valid unified diff
      4. All changed file paths must be safe
      5. submitted vs workspace consistency (when explicit submission is used)
      6. If run_apply_check=True and agent_workdir provided, git apply --check
    """
    from condiag.patch_artifacts import sha256_short, canonicalize_patch

    changed_files = extract_changed_files(evaluation_patch)
    patch_size = len(evaluation_patch) if evaluation_patch else 0
    submitted = getattr(agent_submission, "selected_patch", "") if agent_submission else ""
    fallback_used = bool(
        agent_submission and getattr(agent_submission, "selected_source", "") == "workspace_diff_fallback"
    )
    submitted_sha = sha256_short(canonicalize_patch(submitted)) if submitted else ""
    workspace_sha = sha256_short(canonicalize_patch(workspace_patch)) if workspace_patch else ""
    evaluation_sha = sha256_short(canonicalize_patch(evaluation_patch)) if evaluation_patch else ""
    apply_check_status = "not_run"

    # 1. Termination reason
    if termination_reason != "submitted":
        return PatchIntegrityReport(
            ok=False, status="invalid_termination",
            reason="termination_reason={!r}".format(termination_reason),
            changed_files=changed_files, patch_size=patch_size, fallback_used=fallback_used,
            consistency="n/a", submitted_sha=submitted_sha, workspace_sha=workspace_sha,
            evaluation_sha=evaluation_sha, apply_check_status=apply_check_status,
        )

    # 2. Empty patch
    if not evaluation_patch or not evaluation_patch.strip():
        return PatchIntegrityReport(
            ok=False, status="invalid_empty", reason="evaluation_patch is empty",
            changed_files=[], patch_size=0, fallback_used=fallback_used,
            consistency="empty", submitted_sha=submitted_sha, workspace_sha=workspace_sha,
            evaluation_sha=evaluation_sha, apply_check_status=apply_check_status,
        )

    # 3. Not a unified diff
    if not is_valid_unified_diff(evaluation_patch):
        return PatchIntegrityReport(
            ok=False, status="invalid_diff_format",
            reason="Not a valid unified diff",
            changed_files=changed_files, patch_size=patch_size, fallback_used=fallback_used,
            consistency="n/a", submitted_sha=submitted_sha, workspace_sha=workspace_sha,
            evaluation_sha=evaluation_sha, apply_check_status=apply_check_status,
        )

    # 4. Unsafe paths
    unsafe = [p for p in changed_files if not is_safe_path(p)]
    if unsafe:
        return PatchIntegrityReport(
            ok=False, status="invalid_unsafe_path",
            reason="Unsafe paths: {}".format(unsafe),
            changed_files=changed_files, patch_size=patch_size, fallback_used=fallback_used,
            consistency="n/a", submitted_sha=submitted_sha, workspace_sha=workspace_sha,
            evaluation_sha=evaluation_sha, apply_check_status=apply_check_status,
        )

    # 5. Evaluation == Submission provenance (BLOCKING when explicit submission used)
    if not fallback_used and submitted and evaluation_patch:
        a = canonicalize_patch(submitted)
        e = canonicalize_patch(evaluation_patch)
        if a != e:
            return PatchIntegrityReport(
                ok=False, status="invalid_provenance",
                reason="evaluation_patch != agent submitted patch",
                changed_files=changed_files, patch_size=patch_size, fallback_used=fallback_used,
                consistency="n/a", submitted_sha=submitted_sha, workspace_sha=workspace_sha,
                evaluation_sha=evaluation_sha, apply_check_status=apply_check_status,
            )

    # 6. Workspace consistency (diagnostic only, not blocking)
    consistency = "consistent"
    if not fallback_used and submitted and workspace_patch:
        a = canonicalize_patch(submitted)
        w = canonicalize_patch(workspace_patch)
        if a != w:
            consistency = "mismatch"
    elif fallback_used:
        consistency = "fallback"
    elif not submitted and not workspace_patch:
        consistency = "empty"
    else:
        consistency = "n/a"

    # 6. git apply --check (only if explicitly requested)
    if run_apply_check and agent_workdir is not None:
        try:
            import subprocess
            (agent_workdir / "test.patch").write_text(evaluation_patch)
            r = subprocess.run(
                ["git", "apply", "--check", str(agent_workdir / "test.patch")],
                capture_output=True, text=True, cwd=agent_workdir, timeout=10,
            )
            (agent_workdir / "test.patch").unlink()
            if r.returncode != 0:
                return PatchIntegrityReport(
                    ok=False, status="invalid_unapplyable",
                    reason="git apply --check failed: {}".format(r.stderr[:200]),
                    changed_files=changed_files, patch_size=patch_size, fallback_used=fallback_used,
                    consistency=consistency, submitted_sha=submitted_sha,
                    workspace_sha=workspace_sha, evaluation_sha=evaluation_sha,
                    apply_check_status="failed",
                )
            apply_check_status = "passed"
        except Exception as e:
            return PatchIntegrityReport(
                ok=False, status="invalid_unapplyable",
                reason="git apply --check exception: {}".format(e),
                changed_files=changed_files, patch_size=patch_size, fallback_used=fallback_used,
                consistency=consistency, submitted_sha=submitted_sha,
                workspace_sha=workspace_sha, evaluation_sha=evaluation_sha,
                apply_check_status="failed",
            )

    return PatchIntegrityReport(
        ok=True, status="valid", reason="All checks passed",
        changed_files=changed_files, patch_size=patch_size, fallback_used=fallback_used,
        consistency=consistency, submitted_sha=submitted_sha,
        workspace_sha=workspace_sha, evaluation_sha=evaluation_sha,
        apply_check_status=apply_check_status,
    )


# ══════════════════════════════════════════════════════════════════════
# P0-4b: Harness Eligibility Gate
# ══════════════════════════════════════════════════════════════════════


@dataclass
class EligibilityReport:
    """Decide if R1 Harness result qualifies for entering SF/CD branches.

    Only code-level UNRESOLVED outcomes with valid FailureWitness
    should proceed. Infrastructure failures (ERROR/TIMEOUT/UNKNOWN) and
    missing/empty test logs should block.
    """

    ok: bool
    status: str
    reason: str
    harness_status: str = ""
    test_log_exists: bool = False
    test_log_size: int = 0
    witness_valid: bool = False
    failed_test_count: int = 0
    has_error_message: bool = False
    stack_frame_count: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "status": self.status,
            "reason": self.reason,
            "harness_status": self.harness_status,
            "test_log_exists": self.test_log_exists,
            "test_log_size": self.test_log_size,
            "witness_valid": self.witness_valid,
            "failed_test_count": self.failed_test_count,
            "has_error_message": self.has_error_message,
            "stack_frame_count": self.stack_frame_count,
        }


def _witness_is_valid(fw: dict | None) -> tuple[bool, int, bool, int]:
    """Check if a FailureWitness dict has at least one valid signal.

    Returns: (valid, failed_test_count, has_error_message, stack_frame_count)
    """
    if not fw:
        return False, 0, False, 0
    failed_tests = fw.get("failed_tests", []) or []
    error_message = fw.get("error_message", "") or ""
    stack_frames = fw.get("stack_frames", []) or []
    has_error = bool(error_message and error_message != "No test log")
    valid = bool(failed_tests) or has_error or bool(stack_frames)
    return valid, len(failed_tests), has_error, len(stack_frames)


def check_episode_eligibility(
    r1_eval: Any,
    fw: dict | None,
) -> EligibilityReport:
    """Decide if Episode can enter SF/CD revision phase.

    Decision rules:
      - r1_eval.status == "RESOLVED"          -> ok=True, status="r1_resolved"
      - r1_eval.status not in {"UNRESOLVED", "RESOLVED"}  -> ok=False, infrastructure failure
      - test_log missing/empty                   -> ok=False, missing test log
      - FailureWitness empty                      -> ok=False, empty witness
      - UNRESOLVED + valid witness                -> ok=True
    """
    harness_status = getattr(r1_eval, "status", "") or ""
    test_log_path = getattr(r1_eval, "test_log_path", "") or ""

    # Test log size
    test_log_exists = False
    test_log_size = 0
    if test_log_path:
        try:
            log_file = Path(test_log_path)
            if log_file.exists() and log_file.stat().st_size > 0:
                test_log_exists = True
                test_log_size = log_file.stat().st_size
        except Exception:
            pass

    witness_valid, failed_count, has_err, frame_count = _witness_is_valid(fw)

    # RESOLVED is a normal end - R1 success, no SF/CD needed
    if harness_status == "RESOLVED":
        return EligibilityReport(
            ok=True, status="r1_resolved", reason="R1 already resolved",
            harness_status=harness_status, test_log_exists=test_log_exists,
            test_log_size=test_log_size, witness_valid=witness_valid,
            failed_test_count=failed_count, has_error_message=has_err,
            stack_frame_count=frame_count,
        )

    # Anything other than UNRESOLVED is an infrastructure failure
    if harness_status != "UNRESOLVED":
        status_map = {
            "ERROR": "ineligible_harness_error",
            "TIMEOUT": "ineligible_harness_timeout",
            "UNKNOWN": "ineligible_harness_unknown",
        }
        status = status_map.get(harness_status, "ineligible_harness_unknown")
        return EligibilityReport(
            ok=False, status=status,
            reason="Harness status is {}, not UNRESOLVED".format(harness_status),
            harness_status=harness_status, test_log_exists=test_log_exists,
            test_log_size=test_log_size, witness_valid=witness_valid,
            failed_test_count=failed_count, has_error_message=has_err,
            stack_frame_count=frame_count,
        )

    # UNRESOLVED but missing test log
    if not test_log_exists:
        return EligibilityReport(
            ok=False, status="ineligible_missing_test_log",
            reason="test log missing or empty",
            harness_status=harness_status, test_log_exists=False,
            test_log_size=0, witness_valid=witness_valid,
            failed_test_count=failed_count, has_error_message=has_err,
            stack_frame_count=frame_count,
        )

    # UNRESOLVED but no valid witness
    if not witness_valid:
        return EligibilityReport(
            ok=False, status="ineligible_empty_witness",
            reason="FailureWitness has no failed_tests, error_message, or stack_frames",
            harness_status=harness_status, test_log_exists=test_log_exists,
            test_log_size=test_log_size, witness_valid=False,
            failed_test_count=failed_count, has_error_message=has_err,
            stack_frame_count=frame_count,
        )

    return EligibilityReport(
        ok=True, status="eligible", reason="R1 UNRESOLVED with valid witness",
        harness_status=harness_status, test_log_exists=test_log_exists,
        test_log_size=test_log_size, witness_valid=True,
        failed_test_count=failed_count, has_error_message=has_err,
        stack_frame_count=frame_count,
    )
