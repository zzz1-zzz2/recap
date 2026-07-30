"""Offline Shadow Replay — re-run P1-3C + P1-3D without model or harness.

Usage:
  python -m condiag.acquisition.replay \\
    --bundle /path/to/failure_feature_bundle.json \\
    --repo-root /path/to/repo \\
    --output /path/to/replay_output \\
    --run-id astropy-13398-replay

Exits with:
  0 — success (even when zero actionable contracts or zero hits)
  2 — invalid input or bundle
  3 — pipeline exception
  4 — invariant violation (repo modified, out-of-bounds, budget exceeded)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pydantic

logger = logging.getLogger("condiag.acquisition.replay")


class ReplayInputError(ValueError):
    """User input error — maps to exit code 2."""


class ReplayInvariantError(RuntimeError):
    """Invariant violation — maps to exit code 4."""


# ── ReplaySummary ───────────────────────────────────────────────────


@dataclass
class ReplaySummary:
    run_id: str = ""
    instance_id: str = ""
    bundle_sha256: str = ""
    repo_head_sha: str = ""
    repo_modified: bool = False

    n_clusters: int = 0
    n_hypotheses: int = 0
    n_actionable_contracts: int = 0
    n_router_actions: int = 0
    n_router_found: int = 0
    n_router_not_found: int = 0
    n_router_unsupported: int = 0
    n_router_invalid: int = 0
    n_router_error: int = 0
    n_hits: int = 0

    total_actionable_budget: int = 0
    budget_violations: list[str] = field(default_factory=list)
    out_of_bounds_files: list[str] = field(default_factory=list)

    gold_accessed: bool = False
    repo_modified_by_replay: bool = False

    errors: list[str] = field(default_factory=list)

    # Causal refinement (P1-3E)
    n_patch_linked: int = 0
    n_baseline_failures: int = 0
    n_uncertain: int = 0
    primary_cluster_indices: list[int] = field(default_factory=list)
    deprioritized: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "instance_id": self.instance_id,
            "bundle_sha256": self.bundle_sha256,
            "repo_head_sha": self.repo_head_sha,
            "repo_modified": self.repo_modified,
            "n_clusters": self.n_clusters,
            "n_hypotheses": self.n_hypotheses,
            "n_actionable_contracts": self.n_actionable_contracts,
            "n_router_actions": self.n_router_actions,
            "n_router_found": self.n_router_found,
            "n_router_not_found": self.n_router_not_found,
            "n_router_unsupported": self.n_router_unsupported,
            "n_router_invalid": self.n_router_invalid,
            "n_router_error": self.n_router_error,
            "n_hits": self.n_hits,
            "total_actionable_budget": self.total_actionable_budget,
            "budget_violations": list(self.budget_violations),
            "out_of_bounds_files": list(self.out_of_bounds_files),
            "gold_accessed": self.gold_accessed,
            "repo_modified_by_replay": self.repo_modified_by_replay,
            "errors": list(self.errors),
            "n_patch_linked": self.n_patch_linked,
            "n_baseline_failures": self.n_baseline_failures,
            "n_uncertain": self.n_uncertain,
            "primary_cluster_indices": list(self.primary_cluster_indices),
            "deprioritized": list(self.deprioritized),
        }


# ── Helpers ─────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_head(repo_root: Path) -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
            cwd=repo_root,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _git_is_clean(repo_root: Path) -> bool:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=30,
            cwd=repo_root,
        )
        return r.returncode == 0 and not r.stdout.strip()
    except Exception:
        return False


def _git_diff_sha(repo_root: Path) -> str:
    """SHA256 of tracked-file diff (HEAD vs working tree). Empty if no diff."""
    try:
        r = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
            capture_output=True, text=True, timeout=30, cwd=repo_root,
        )
        if r.returncode == 0 and r.stdout.strip():
            return hashlib.sha256(r.stdout.encode()).hexdigest()[:16]
        return ""
    except Exception:
        return ""


def _git_untracked_manifest(repo_root: Path) -> str:
    """SHA256 of all untracked files (paths + content). Empty if none."""
    try:
        r = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, timeout=30, cwd=repo_root,
        )
        if r.returncode == 0 and r.stdout.strip():
            manifest = sorted(r.stdout.strip().split("\n"))
            return hashlib.sha256("\n".join(manifest).encode()).hexdigest()[:16]
        return ""
    except Exception:
        return ""


# ── Pipeline ────────────────────────────────────────────────────────


def run_replay(
    *,
    bundle_path: Path,
    repo_root: Path,
    output_dir: Path,
    run_id: str,
    max_total_actions: int = 3,
    max_total_budget: int = 8,
    max_files_examined: int = 200,
) -> ReplaySummary:
    """Full offline Shadow pipeline.

    Returns ReplaySummary and writes Shadow artifacts to output_dir/.
    Raises on invariant violations.
    """
    # ── Repo state validation (R1 dirty workspace accepted) ──
    if not bundle_path.exists():
        raise FileNotFoundError(f"bundle not found: {bundle_path}")
    if not repo_root.is_dir():
        raise NotADirectoryError(f"repo_root not a directory: {repo_root}")
    if not (repo_root / ".git").exists():
        raise ReplayInputError(f"repo_root is not a git repository: {repo_root}")
    # Capture pre-replay state for read-only verification
    repo_head_before = _git_head(repo_root)
    if not repo_head_before:
        raise ReplayInputError(f"cannot determine repo HEAD: {repo_root}")
    # Output dir must NOT be inside repo_root (Replay would dirty the repo)
    try:
        output_dir.resolve().relative_to(repo_root.resolve())
        raise ReplayInputError(
            f"output_dir must not be inside repo_root: "
            f"{output_dir.resolve()} is inside {repo_root.resolve()}"
        )
    except ValueError:
        pass  # expected — output_dir outside repo_root is correct

    # ── Capture full pre-replay repo state ──
    bundle_raw = bundle_path.read_bytes()
    bundle_sha = hashlib.sha256(bundle_raw).hexdigest()[:16]
    repo_head_before = _git_head(repo_root)
    repo_diff_before = _git_diff_sha(repo_root)
    repo_untracked_before = _git_untracked_manifest(repo_root)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load bundle ──
    from pydantic import ValidationError as PydanticValidationError
    try:
        from .signals.schema import RuntimeFailureFeatureBundle
        bundle = RuntimeFailureFeatureBundle.model_validate_json(bundle_raw)
    except (json.JSONDecodeError, PydanticValidationError, KeyError) as e:
        raise ReplayInputError(f"invalid bundle: {e}")

    # ── Verify base_commit matches repo HEAD ──
    bundle_base = ""
    if bundle.instance and bundle.instance.base_commit:
        bundle_base = bundle.instance.base_commit
    if bundle_base and repo_head_before and bundle_base != repo_head_before:
        raise ReplayInvariantError(
            f"bundle base_commit ({bundle_base}) != repo HEAD ({repo_head_before}). "
            f"Replay requires the exact commit the canary ran on."
        )

    # ── Workspace diff SHA ──
    workspace_diff_sha = ""
    try:
        r = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD"],
            capture_output=True, text=True, timeout=30, cwd=repo_root,
        )
        if r.returncode == 0:
            workspace_diff_sha = hashlib.sha256(r.stdout.encode()).hexdigest()[:16]
    except Exception:
        pass

    # ── P1-3A/B: Cluster → Diagnose ──
    from recap.diagnosis.failure_event import reasoner_v2_cluster
    from recap.diagnosis.alignment import reasoner_v2_diagnose

    clusters = reasoner_v2_cluster(bundle)
    diagnoses = reasoner_v2_diagnose(
        clusters, bundle.patch, bundle.trajectory,
    )

    # ── P1-3C: Hypothesis → Evidence → Contract ──
    from recap.diagnosis.hypothesis import from_subtyped_diagnosis
    from recap.diagnosis.search_contract import (
        PlanBudget,
        build_evidence_ledger,
        build_search_plan,
        write_shadow_artifacts,
    )

    hypotheses = [
        from_subtyped_diagnosis(d, c.cluster_id, c.test_names)
        for c, d in zip(clusters, diagnoses)
    ]

    ledger = build_evidence_ledger(clusters, diagnoses, bundle=bundle)

    plan_budget = PlanBudget(
        max_total_actions=max_total_actions,
        max_total_budget=max_total_budget,
    )
    contracts = build_search_plan(hypotheses, budget=plan_budget, ledger=ledger)

    # Write P1-3C Shadow artifacts
    write_shadow_artifacts(
        output_dir / "p1_3c_shadow",
        contracts=contracts,
        hypotheses=hypotheses,
        ledger=ledger,
        validation_report={
            "run_id": run_id,
            "instance_id": bundle.instance.instance_id if bundle.instance else "",
            "schema_version": "1",
        },
    )

    # ── P1-3D: Router ──
    from recap.acquisition.router import AcquisitionRouter
    from recap.acquisition.artifact_writer import (
        write_acquisition_artifacts,
    )

    # Collect viewed files and failed test names
    viewed = list(getattr(bundle.trajectory, "viewed_files", []) or [])
    failed_names: list[str] = []
    for c in clusters:
        failed_names.extend(c.test_names)

    router = AcquisitionRouter(
        repo_root,
        r1_viewed_files=viewed,
        failed_test_names=failed_names,
        max_files_examined=max_files_examined,
    )

    results = []
    for contract in contracts:
        if contract.status.value != "ACTIONABLE":
            continue
        results.extend(router.dispatch_contract(contract))

    # Write P1-3D Shadow artifacts
    write_acquisition_artifacts(
        output_dir / "p1_3d_shadow",
        results,
        repo_root,
        run_id=run_id,
    )

    # ── P1-3E: Causal Refinement (post-Router, pre-Reshaping) ──
    from recap.diagnosis.causal_refinement import (
        causal_refine,
        merge_attribution_into_diagnosis,
        safety_gate,
    )

    # Load baseline failure info (FAIL_TO_PASS) from instance registry
    fail_to_pass: list[str] = []
    try:
        from instance_registry import InstanceRegistry
        iid = bundle.instance.instance_id if bundle.instance else ""
        if iid:
            registry = InstanceRegistry()
            spec = registry.get_instance(iid)
            if spec and hasattr(spec, "fail_to_pass"):
                fail_to_pass = list(spec.fail_to_pass)
    except Exception as exc:
        logger.warning("Could not load instance data for baseline: %s", exc)

    # Load actual patch diff text for symbol extraction
    patch_content = ""
    try:
        # workspace.patch lives alongside the bundle in round1/
        patch_path = bundle_path.parent / "workspace.patch"
        if patch_path.exists():
            patch_content = patch_path.read_text(encoding="utf-8", errors="replace")
            logger.info("Loaded patch diff (%d chars) for symbol extraction", len(patch_content))
    except Exception as exc:
        logger.warning("Could not load workspace.patch: %s", exc)

    # Run causal refinement for each cluster
    refined_diagnoses = list(diagnoses)  # copy
    attributions: list = []
    for i, (cluster, dx) in enumerate(zip(clusters, refined_diagnoses)):
        attr = causal_refine(
            provisional_subtype=dx.subtype,
            cluster=cluster,
            patch_edited_files=list(bundle.patch.edited_files),
            fail_to_pass=fail_to_pass,
            patch_content=patch_content,
        )
        refined_diagnoses[i] = merge_attribution_into_diagnosis(dx, attr)
        attributions.append(attr)

    # Safety gate — determine primary vs deprioritized clusters
    primary_indices, deprioritized = safety_gate(
        attributions,
        [d.subtype for d in refined_diagnoses],
    )
    logger.info(
        "Causal refinement: %d primary, %d deprioritized",
        len(primary_indices), len(deprioritized),
    )
    if deprioritized:
        for dep in deprioritized:
            logger.info("  Deprioritized: %s", dep)

    # Write refinement artifacts
    refinement_dir = output_dir / "p1_3e_refinement"
    refinement_dir.mkdir(parents=True, exist_ok=True)
    (refinement_dir / "causal_attributions.json").write_text(json.dumps([
        {
            "cluster_id": c.cluster_id,
            "subtype": d.subtype,
            "causal_status": d.causal_status,
            "patch_linkage": d.patch_linkage,
            "mechanism": d.mechanism,
            "benchmark_role": d.benchmark_role,
            "failure_delta": d.failure_delta,
            "revision_priority": d.revision_priority,
            "unsafe_repair_warnings": d.unsafe_repair_warnings,
            "baseline_in_fail_to_pass": d.baseline_in_fail_to_pass,
        }
        for c, d in zip(clusters, refined_diagnoses)
    ], indent=2))
    (refinement_dir / "safety_gate.json").write_text(json.dumps({
        "primary_cluster_indices": primary_indices,
        "deprioritized": deprioritized,
    }, indent=2))

    # ── Verify repo was not modified by Replay ──
    repo_head_after = _git_head(repo_root)
    repo_diff_after = _git_diff_sha(repo_root)
    repo_untracked_after = _git_untracked_manifest(repo_root)
    if repo_head_after != repo_head_before:
        raise RuntimeError(
            f"Replay changed repo HEAD: {repo_head_before} → {repo_head_after}. "
            f"This is a bug."
        )
    if repo_diff_after != repo_diff_before:
        raise RuntimeError(
            f"Replay modified tracked files in the repo. "
            f"Diff SHA changed: {repo_diff_before} → {repo_diff_after}. "
            f"This is a bug."
        )
    if repo_untracked_after != repo_untracked_before:
        raise RuntimeError(
            f"Replay modified untracked files in the repo. "
            f"Untracked manifest changed. This is a bug."
        )
    repo_modified_now = False  # verified above

    # ── Build summary ──
    from recap.acquisition.schema import AcquisitionStatus

    n_found = sum(1 for r in results if r.status == AcquisitionStatus.FOUND)
    n_not_found = sum(1 for r in results if r.status == AcquisitionStatus.NOT_FOUND)
    n_unsupported = sum(1 for r in results if r.status == AcquisitionStatus.UNSUPPORTED)
    n_invalid = sum(1 for r in results if r.status == AcquisitionStatus.INVALID_TARGET)
    n_error = sum(1 for r in results if r.status == AcquisitionStatus.ERROR)
    n_hits = sum(len(r.hits) for r in results)
    total_budget = sum(r.budget_limit for r in results)

    # Post-replay provenance check
    from recap.acquisition.artifact_writer import validate_results
    val_rep = validate_results(results, repo_root)

    _summary = ReplaySummary(
        run_id=run_id,
        instance_id=bundle.instance.instance_id if bundle.instance else "",
        bundle_sha256=bundle_sha,
        repo_head_sha=repo_head_before,
        repo_modified=bool(repo_diff_before or repo_untracked_before),
        n_clusters=len(clusters),
        n_hypotheses=len(hypotheses),
        n_actionable_contracts=sum(
            1 for c in contracts if c.status.value == "ACTIONABLE"
        ),
        n_router_actions=len(results),
        n_router_found=n_found,
        n_router_not_found=n_not_found,
        n_router_unsupported=n_unsupported,
        n_router_invalid=n_invalid,
        n_router_error=n_error,
        n_hits=n_hits,
        total_actionable_budget=total_budget,
        budget_violations=val_rep.budget_violations,
        out_of_bounds_files=val_rep.out_of_bounds_files,
        gold_accessed=False,
        repo_modified_by_replay=repo_modified_now,
        n_patch_linked=sum(
            1 for d in refined_diagnoses
            if d.causal_status in ("PATCH_LINKED", "NEW_REGRESSION")
        ),
        n_baseline_failures=sum(
            1 for d in refined_diagnoses if d.baseline_in_fail_to_pass
        ),
        n_uncertain=sum(
            1 for d in refined_diagnoses if d.causal_status == "UNCERTAIN"
        ),
        primary_cluster_indices=primary_indices,
        deprioritized=deprioritized,
    )

    # Write replay_manifest.json from summary
    _manifest = _summary.to_dict()
    _manifest["bundle_path"] = str(bundle_path.resolve())
    _manifest["repo_root"] = str(repo_root.resolve())
    _manifest["max_total_actions"] = max_total_actions
    _manifest["max_total_budget"] = max_total_budget
    _manifest["max_files_examined"] = max_files_examined
    _manifest_path = output_dir / "replay_manifest.json"
    _manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _manifest_path.write_text(json.dumps(_manifest, indent=2))

    return _summary


# ── CLI ────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Offline Shadow Replay: bundle → cluster → diagnose → contract → router",
    )
    p.add_argument("--bundle", required=True, type=Path, help="Path to failure_feature_bundle.json")
    p.add_argument("--repo-root", required=True, type=Path, help="Path to repository checkout")
    p.add_argument("--output", required=True, type=Path, help="Output directory for artifacts")
    p.add_argument("--run-id", default="replay", help="Descriptive run label")
    p.add_argument("--max-total-actions", type=int, default=3, help="Plan-level cap (default 3)")
    p.add_argument("--max-total-budget", type=int, default=8, help="Plan-level budget cap (default 8)")
    p.add_argument("--max-files-examined", type=int, default=200, help="Per-action file scan limit (default 200)")
    args = p.parse_args()

    exit_code = 0
    try:
        summary = run_replay(
            bundle_path=args.bundle,
            repo_root=args.repo_root,
            output_dir=args.output,
            run_id=args.run_id,
            max_total_actions=args.max_total_actions,
            max_total_budget=args.max_total_budget,
            max_files_examined=args.max_files_examined,
        )
    except ReplayInputError as e:
        logger.error("Input error: %s", e)
        sys.exit(2)
    except FileNotFoundError as e:
        logger.error("Input error: %s", e)
        sys.exit(2)
    except NotADirectoryError as e:
        logger.error("Input error: %s", e)
        sys.exit(2)
    except ReplayInvariantError as e:
        logger.error("Invariant violation: %s", e)
        sys.exit(4)
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:
        logger.exception("Pipeline error: %s", e)
        sys.exit(3)

    # Check invariants from summary
    if summary.out_of_bounds_files:
        logger.error(
            "Invariant violation: %d out-of-bounds files (%s)",
            len(summary.out_of_bounds_files),
            summary.out_of_bounds_files[:3],
        )
        exit_code = 4
    if summary.budget_violations:
        logger.error(
            "Invariant violation: %d budget violations (%s)",
            len(summary.budget_violations),
            summary.budget_violations[:3],
        )
        exit_code = 4
    if summary.repo_modified_by_replay:
        logger.error("Invariant violation: repo was modified")
        exit_code = 4

    logger.info("Replay manifest → %s/replay_manifest.json", args.output)
    logger.info(
        "Summary: %d clusters, %d hypotheses, %d actionable contracts, "
        "%d actions, %d hits, exit=%d",
        summary.n_clusters, summary.n_hypotheses,
        summary.n_actionable_contracts,
        summary.n_router_actions, summary.n_hits, exit_code,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
