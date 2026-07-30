"""Workspace Snapshot — capture and verify persistent workspace state."""
from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def sha256_full(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_short(text: str) -> str:
    return sha256_full(text)[:16]


@dataclass
class UntrackedFile:
    """A file present in the workspace but not tracked by git."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}


@dataclass
class CaptureResult:
    """Result of a workspace snapshot capture operation.

    ok=False + reason means capture failed — episode should be blocked.
    ok=True + snapshot contains the captured workspace state.
    """

    ok: bool = False
    snapshot: Any | None = None  # WorkspaceSnapshot
    reason: str = ""


@dataclass
class WorkspaceSnapshot:
    tracked_diff: str = ""
    untracked_manifest: list[UntrackedFile] = field(default_factory=list)
    untracked_archive_path: str = ""
    base_commit_sha: str = ""

    @property
    def tracked_diff_sha(self) -> str:
        return sha256_short(self.tracked_diff) if self.tracked_diff else ""

    @property
    def untracked_manifest_sha(self) -> str:
        if not self.untracked_manifest:
            return ""
        raw = json.dumps([u.to_dict() for u in self.untracked_manifest], sort_keys=True)
        return sha256_short(raw)

    @property
    def workspace_state_sha(self) -> str:
        raw = (self.base_commit_sha or "") + self.tracked_diff_sha + self.untracked_manifest_sha
        return sha256_short(raw) if raw.strip() else ""

    def to_dict(self) -> dict:
        return {
            "tracked_diff_sha": self.tracked_diff_sha,
            "untracked_manifest_sha": self.untracked_manifest_sha,
            "workspace_state_sha": self.workspace_state_sha,
            "untracked_count": len(self.untracked_manifest),
            "untracked_archive_exists": bool(self.untracked_archive_path),
            "base_commit_sha": self.base_commit_sha,
        }


def _exec(agent, cmd: str) -> tuple[int, str]:
    """Execute a command in the agent container and return (returncode, output)."""
    r = agent.env.execute({"command": cmd})
    rc = r.get("returncode", -1)
    out = r.get("output", "") or ""
    return rc, out


def capture_workspace_fingerprint(agent: Any, base_commit: str) -> CaptureResult:
    """Capture full workspace fingerprint: tracked diff + untracked manifest.

    This is the ONLY function used for:
      - R1 workspace capture
      - Restore validation
      - Branch preflight fairness

    All three phases call this same function — single source of truth.

    Any critical command failure returns CaptureResult(ok=False, reason=...),
    NOT a silent empty default. Episodes must be blocked on failure.
    """
    # Base commit SHA
    rc, head_sha = _exec(agent, "cd /testbed && git rev-parse HEAD 2>/dev/null")
    if rc != 0 or not head_sha.strip():
        return CaptureResult(ok=False, reason="git rev-parse HEAD failed")
    head_sha = head_sha.strip()

    # Tracked diff
    rc, tracked = _exec(agent, f"cd /testbed && git diff --binary --no-ext-diff {base_commit} 2>/dev/null")
    if rc != 0:
        return CaptureResult(ok=False, reason=f"git diff failed (rc={rc})")

    # Untracked file fingerprint via inline Python (shell-safe for all filenames)
    untracked_paths: list[str] = []
    py_script = """
import hashlib, json, os, subprocess
raw = subprocess.check_output(["git", "ls-files", "-z", "--others", "--exclude-standard"])
items = []
for raw_path in raw.split(b"\\0"):
    if not raw_path:
        continue
    path = os.fsdecode(raw_path)
    try:
        with open(path, "rb") as f:
            data = f.read()
        items.append({"path": path, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    except (OSError, PermissionError) as e:
        items.append({"path": path, "size": -1, "sha256": f"error:{e}"})
print(json.dumps(items, ensure_ascii=False))
"""
    rc, ut_json = _exec(agent, f"cd /testbed && python3 -c {shlex.quote(py_script)} 2>/dev/null")
    if rc != 0:
        return CaptureResult(ok=False, reason=f"untracked fingerprint failed (rc={rc})")

    try:
        untracked_items = json.loads(ut_json)
    except json.JSONDecodeError as e:
        return CaptureResult(ok=False, reason=f"untracked JSON parse failed: {e}")

    manifest: list[UntrackedFile] = []
    for item in untracked_items:
        if item.get("size", 0) < 0:
            return CaptureResult(ok=False, reason=f"untracked read error: {item['path']}: {item.get('sha256', '')}")
        manifest.append(UntrackedFile(
            path=item["path"],
            size=item["size"],
            sha256=item["sha256"],
        ))

    snapshot = WorkspaceSnapshot(
        tracked_diff=tracked,
        untracked_manifest=manifest,
        base_commit_sha=head_sha,
    )
    return CaptureResult(ok=True, snapshot=snapshot)


def archive_untracked_files(agent: Any, snapshot_dir: Path) -> str:
    """Create a tar archive of untracked files and copy it out of the container.

    Uses null-delimited file list for safety with special characters.
    Returns the local archive path, or empty string on failure.
    """
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    archive_path = str(snapshot_dir / "untracked.tar")

    # Create archive with null-delimited file list + pipefail
    # Use bash -o pipefail to detect failures in either git ls-files or tar
    archive_script = (
        "git ls-files -z --others --exclude-standard 2>/dev/null | "
        "tar --null --verbatim-files-from -T - -cf /tmp/untracked.tar"
    )
    rc, _ = _exec(agent, f"cd /testbed && bash -o pipefail -c {shlex.quote(archive_script)} 2>&1")
    if rc != 0:
        return ""

    # Copy out
    import subprocess
    cp_r = subprocess.run(
        ["docker", "cp", f"{agent.env.container_id}:/tmp/untracked.tar", archive_path],
        capture_output=True, timeout=10,
    )
    if cp_r.returncode != 0:
        return ""

    return archive_path


def check_tracked_code_fairness(
    r1_snapshot: "WorkspaceSnapshot | None",
    sf_snapshot: "WorkspaceSnapshot | None",
    cd_snapshot: "WorkspaceSnapshot | None" = None,
) -> dict[str, Any]:
    """Verify that all branches start from the same TRACKED code state.

    This is the BLOCKING preflight gate before any R2 agent step runs.
    We compare tracked_diff_sha only — untracked files (scratch, reproduce
    scripts) are allowed to vary between containers because they do not
    participate in Harness evaluation.

    Returns a dict of per-pair booleans plus an `all_ok` summary.

    Two snapshots are considered equal at the tracked-code level iff
    `tracked_diff_sha` matches. Missing snapshots are treated as a failure
    (we require an explicit baseline).
    """
    result: dict[str, Any] = {}
    if r1_snapshot is None or sf_snapshot is None:
        result["r1_vs_sf_tracked_ok"] = False
    else:
        result["r1_vs_sf_tracked_ok"] = (
            r1_snapshot.tracked_diff_sha == sf_snapshot.tracked_diff_sha
        )
    if cd_snapshot is not None:
        if r1_snapshot is None or cd_snapshot is None:
            result["r1_vs_cd_tracked_ok"] = False
        else:
            result["r1_vs_cd_tracked_ok"] = (
                r1_snapshot.tracked_diff_sha == cd_snapshot.tracked_diff_sha
            )
    pairs_with_value = [v for v in result.values() if isinstance(v, bool)]
    result["all_ok"] = bool(pairs_with_value) and all(pairs_with_value)
    return result


def check_full_workspace_equivalence(
    r1_snapshot: "WorkspaceSnapshot | None",
    sf_snapshot: "WorkspaceSnapshot | None",
    cd_snapshot: "WorkspaceSnapshot | None" = None,
) -> dict[str, Any]:
    """Audit-only: verify that all branches match at the full-workspace level.

    Compares both tracked diff SHA AND untracked manifest SHA. This is the
    stricter equivalence check. Use it ONLY for forensic / audit purposes —
    it WILL fail whenever containers differ in reproduce scripts or other
    scratch artifacts, even when the underlying code state is identical.

    Recorded values (per-pair) plus a summary `all_ok`. Missing snapshots
    fail the comparison.
    """
    result: dict[str, Any] = {}
    if r1_snapshot is None or sf_snapshot is None:
        result["r1_vs_sf_tracked_ok"] = False
        result["r1_vs_sf_state_ok"] = False
    else:
        result["r1_vs_sf_tracked_ok"] = (
            r1_snapshot.tracked_diff_sha == sf_snapshot.tracked_diff_sha
        )
        result["r1_vs_sf_state_ok"] = (
            r1_snapshot.workspace_state_sha == sf_snapshot.workspace_state_sha
        )
    if cd_snapshot is not None:
        if r1_snapshot is None or cd_snapshot is None:
            result["r1_vs_cd_tracked_ok"] = False
            result["r1_vs_cd_state_ok"] = False
        else:
            result["r1_vs_cd_tracked_ok"] = (
                r1_snapshot.tracked_diff_sha == cd_snapshot.tracked_diff_sha
            )
            result["r1_vs_cd_state_ok"] = (
                r1_snapshot.workspace_state_sha == cd_snapshot.workspace_state_sha
            )
    pairs_with_value = [v for v in result.values() if isinstance(v, bool)]
    result["all_ok"] = bool(pairs_with_value) and all(pairs_with_value)
    return result


def check_workspace_fairness(
    r1_snapshot: WorkspaceSnapshot,
    sf_snapshot: WorkspaceSnapshot,
    cd_snapshot: WorkspaceSnapshot | None = None,
) -> dict[str, bool]:
    """DEPRECATED alias for backward compatibility.

    Splits into:
      - check_tracked_code_fairness(...) — BLOCKING preflight gate
      - check_full_workspace_equivalence(...) — audit-only forensic check

    Existing callers should migrate to the two explicit functions. This
    alias keeps the historical "all pairs AND all states" union so legacy
    tests that assert `fairness["all_ok"]` on mixed mismatch (untracked
    differs but tracked matches) still behave the same.
    """
    tracked = check_tracked_code_fairness(r1_snapshot, sf_snapshot, cd_snapshot)
    full = check_full_workspace_equivalence(r1_snapshot, sf_snapshot, cd_snapshot)
    merged: dict[str, bool] = {}
    for k, v in tracked.items():
        if k == "all_ok":
            continue
        merged[k] = bool(v)
    for k, v in full.items():
        if k == "all_ok":
            continue
        if k.endswith("_state_ok"):
            merged[k] = bool(v)
    merged["all_ok"] = all(merged.values()) if merged else False
    return merged
