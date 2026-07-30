"""ConDiag v4 Instance Registry — load instance data from ContextBench + SWE-bench.

Provides a single-source-of-truth for instance parameters used across the
V2c pipeline: Round 1 agent, official evaluation, ContextBench trajectory
analysis, and MER/MRP computation.

Architecture:
  ContextBench full.parquet  ─┬─ instance_id, repo, base_commit
                              ├─ problem_statement
                              ├─ gold_context, gold_patch
                              └─ f2p, p2p (as JSON strings)
  SWE-bench Verified dataset ─┬─ version, FAIL_TO_PASS, PASS_TO_PASS
                              └─ environment_setup_commit

  The registry merges both sources on instance_id == original_inst_id.
  The canonical SWE-bench row is attached as _swebench_row for harness use.

Pilot scope: 16 Verified/python first-failed instances.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

logger = logging.getLogger("condiag.registry")

CB_PARQUET = Path(
    os.environ.get(
        "CONDIAG_CB_PARQUET",
        "/home/zz/桌面/condiag/ContextBench/data/full.parquet",
    )
)
MANIFEST = Path(
    os.environ.get(
        "CONDIAG_MANIFEST",
        "/home/zz/桌面/condiag/artifacts/v2c_canary_manifest.jsonl",
    )
)
SWEBENCH_VERIFIED = "princeton-nlp/SWE-bench_Verified"


@dataclass
class InstanceSpec:
    """All parameters needed to run one instance through the V2c pipeline."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    gold_patch: str
    gold_context: str
    test_patch: str
    fail_to_pass: list[str]
    pass_to_pass: list[str]
    version: str
    environment_setup_commit: str
    source: str
    language: str
    cb_instance_id: str
    pool: str
    _swebench_row: dict = field(default_factory=dict)


class InstanceRegistry:
    """Loads and caches instance data from ContextBench + SWE-bench datasets."""

    def __init__(self, cb_path: str | Path = CB_PARQUET, manifest_path: str | Path = MANIFEST):
        self.cb_path = Path(cb_path)
        self.manifest_path = manifest_path
        self._cb_df = None
        self._manifest = None
        self._swebench_map: dict[str, dict] = {}

    def list_pilot(self) -> list[InstanceSpec]:
        """Return the 16 Verified/python first-failed instances for V2c pilot."""
        return self.list_instances(pool="first_failed", source="Verified", language="python")

    def list_instances(self, pool: str | None = None, source: str | None = None,
                       language: str | None = None) -> list[InstanceSpec]:
        results = []
        for spec in self._iter_all():
            if pool and spec.pool != pool: continue
            if source and spec.source != source: continue
            if language and spec.language != language: continue
            results.append(spec)
        return results

    def get_instance(self, instance_id: str) -> InstanceSpec | None:
        for spec in self._iter_all():
            if spec.instance_id == instance_id:
                return spec
        return None

    def get_cb_gold_context(self, instance_id: str) -> list[dict] | None:
        spec = self.get_instance(instance_id)
        if spec is None or not spec.gold_context: return None
        try: return json.loads(spec.gold_context)
        except json.JSONDecodeError: return None

    @property
    def cb_df(self):
        if self._cb_df is None:
            logger.info("Loading ContextBench parquet from %s", self.cb_path)
            self._cb_df = pq.read_table(str(self.cb_path)).to_pandas()
        return self._cb_df

    @property
    def manifest(self) -> list[dict]:
        if self._manifest is None:
            logger.info("Loading manifest from %s", self.manifest_path)
            self._manifest = [json.loads(line) for line in open(self.manifest_path)]
        return self._manifest

    def _load_swebench_map(self) -> dict[str, dict]:
        if self._swebench_map: return self._swebench_map
        try:
            from datasets import load_dataset
            logger.info("Loading SWE-bench Verified dataset from HuggingFace...")
            ds = load_dataset(SWEBENCH_VERIFIED, split="test")
            self._swebench_map = {inst["instance_id"]: dict(inst) for inst in ds}
            logger.info("Loaded %s instances from SWE-bench Verified", len(self._swebench_map))
        except Exception as e:
            logger.warning("Failed to load SWE-bench dataset: %s (continuing without enrichment)", e)
        return self._swebench_map

    def _parse_list_field(self, field: Any) -> list[str]:
        if field is None: return []
        if isinstance(field, list):
            if field and isinstance(field[0], str) and len(field[0]) == 1:
                try: return eval("".join(field))
                except Exception: return []
            return field
        if isinstance(field, str):
            try: return json.loads(field)
            except (json.JSONDecodeError, TypeError): return [field]
        return []

    def _iter_all(self):
        manifest_map = {d["instance_id"]: d for d in self.manifest}
        sb_map = self._load_swebench_map() if not self._swebench_map else self._swebench_map
        ids_in_manifest = set(manifest_map.keys())
        cb_subset = self.cb_df[self.cb_df["original_inst_id"].isin(ids_in_manifest)]

        for _, row in cb_subset.iterrows():
            inst_id = row["original_inst_id"]
            man_entry = manifest_map.get(inst_id, {})
            sb_entry = sb_map.get(inst_id, {})

            yield InstanceSpec(
                _swebench_row=sb_entry or {},
                instance_id=inst_id,
                repo=row.get("repo", ""),
                base_commit=row.get("base_commit", ""),
                problem_statement=row.get("problem_statement", ""),
                gold_patch=row.get("patch", ""),
                gold_context=row.get("gold_context", ""),
                test_patch=row.get("test_patch", ""),
                fail_to_pass=(
                    json.loads(sb_entry.get("FAIL_TO_PASS", "[]"))
                    if sb_entry else self._parse_list_field(row.get("f2p", []))
                ),
                pass_to_pass=(
                    json.loads(sb_entry.get("PASS_TO_PASS", "[]"))
                    if sb_entry else self._parse_list_field(row.get("p2p", []))
                ),
                version=sb_entry.get("version", "") if sb_entry else "",
                environment_setup_commit=sb_entry.get("environment_setup_commit", "") if sb_entry else "",
                source=row.get("source", ""),
                language=row.get("language", ""),
                cb_instance_id=row.get("instance_id", ""),
                pool=man_entry.get("pool", "unknown"),
            )

    def __len__(self): return len(self.manifest)
    def __repr__(self):
        pools = {}
        for spec in self._iter_all(): pools[spec.pool] = pools.get(spec.pool, 0) + 1
        return f"<InstanceRegistry: {len(self)} instances, pools={pools}>"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    reg = InstanceRegistry()
    print(reg)
    pilot = reg.list_pilot()
    print(f"\nPilot ({len(pilot)} instances):")
    for p in pilot:
        print(f"  {p.instance_id:45s} v={p.version:6s} f2p={len(p.fail_to_pass):2d} sb={bool(p._swebench_row)}")
