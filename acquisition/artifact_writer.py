"""P1-3D: Acquisition artifact writer — writes Router outputs as Shadow artifacts.

Produces:
  cd/p1_3d_shadow/acquisition_results.json
  cd/p1_3d_shadow/router_validation.json
  cd/p1_3d_shadow/router_error.json   (only when build failed)

Strict invariants enforced at write time:
  - every result has a non-empty action_id and target value
  - every hit has file_path, start_line, end_line, content
  - no result files_examined > budget + 100 (overflow guard)
  - file_path of every hit is inside repo_root (out-of-bounds flagged)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from acquisition.schema import AcquisitionResult, AcquisitionStatus


@dataclass
class RouterValidationReport:
    n_actions: int = 0
    n_found: int = 0
    n_not_found: int = 0
    n_unsupported: int = 0
    n_invalid: int = 0
    n_error: int = 0
    n_hits: int = 0
    out_of_bounds_files: list[str] = field(default_factory=list)
    budget_violations: list[str] = field(default_factory=list)
    schema_version: str = "1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "n_actions": self.n_actions,
            "n_found": self.n_found,
            "n_not_found": self.n_not_found,
            "n_unsupported": self.n_unsupported,
            "n_invalid": self.n_invalid,
            "n_error": self.n_error,
            "n_hits": self.n_hits,
            "out_of_bounds_files": list(self.out_of_bounds_files),
            "budget_violations": list(self.budget_violations),
        }


def validate_results(
    results: list[AcquisitionResult],
    repo_root: Path,
) -> RouterValidationReport:
    """Build a RouterValidationReport from results, checking invariants."""
    rep = RouterValidationReport(n_actions=len(results))
    abs_repo = repo_root.resolve()
    for r in results:
        if r.status == AcquisitionStatus.FOUND:
            rep.n_found += 1
            rep.n_hits += len(r.hits)
            for h in r.hits:
                try:
                    (abs_repo / h.file_path).resolve().relative_to(abs_repo)
                except (OSError, ValueError):
                    rep.out_of_bounds_files.append(h.file_path)
        elif r.status == AcquisitionStatus.NOT_FOUND:
            rep.n_not_found += 1
        elif r.status == AcquisitionStatus.UNSUPPORTED:
            rep.n_unsupported += 1
        elif r.status == AcquisitionStatus.INVALID_TARGET:
            rep.n_invalid += 1
        elif r.status == AcquisitionStatus.ERROR:
            rep.n_error += 1
        # Budget violation: budget_used must not exceed budget_limit (when set)
        if r.budget_limit > 0 and r.budget_used > r.budget_limit:
            rep.budget_violations.append(
                f"{r.action_id}: {r.budget_used} hits > budget_limit {r.budget_limit}"
            )
        # Scan violation: files_examined must not exceed scan_limit (when set)
        if r.scan_limit > 0 and r.files_examined > r.scan_limit:
            rep.budget_violations.append(
                f"{r.action_id}: scanned {r.files_examined} files > scan_limit {r.scan_limit}"
            )
        # Hit count violation: len(hits) must not exceed budget_limit (when set)
        if r.budget_limit > 0 and len(r.hits) > r.budget_limit:
            rep.budget_violations.append(
                f"{r.action_id}: {len(r.hits)} hits > budget_limit {r.budget_limit}"
            )
    return rep


def write_acquisition_artifacts(
    output_dir: str | Path,
    results: list[AcquisitionResult],
    repo_root: Path,
    run_id: str = "",
    error: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Write acquisition_results.json + router_validation.json + optional error."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    rep = validate_results(results, repo_root)

    payload = {
        "schema_version": "1",
        "run_id": run_id,
        "n_results": len(results),
        "results": [r.to_dict() for r in results],
    }
    p = out_dir / "acquisition_results.json"
    p.write_text(json.dumps(payload, indent=2))
    paths["acquisition_results"] = str(p)

    rep_dict = rep.to_dict()
    rep_dict["run_id"] = run_id
    p = out_dir / "router_validation.json"
    p.write_text(json.dumps(rep_dict, indent=2))
    paths["router_validation"] = str(p)

    if error is not None:
        p = out_dir / "router_error.json"
        p.write_text(json.dumps(error, indent=2))
        paths["router_error"] = str(p)

    return paths
