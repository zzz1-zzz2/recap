"""ConDiag v4 CheckpointManager — capture and restore agent state for forking.

Captures a snapshot of the ConDiagIntegratedAgent + environment at any point
in the episode, and restores it to create an identical branch.

Checkpoint contents:
  - messages (deep-copied conversation history)
  - agent config + counters (cost, n_calls, phase, etc.)
  - model identity (name + kwargs, for reconstruction)
  - workspace patch (canonical git diff from base_commit, with untracked files)
  - Round 1 patch (if already submitted)
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("condiag.checkpoint")


@dataclass
class AgentSnapshot:
    messages: list[dict] = field(default_factory=list)
    phase: str = "round1"
    cost: float = 0.0
    n_calls: int = 0
    n_consecutive_format_errors: int = 0
    start_time_epoch: float = 0.0
    config: dict = field(default_factory=dict)
    extra_template_vars: dict = field(default_factory=dict)
    model_name: str = ""
    model_kwargs: dict = field(default_factory=dict)
    workspace_patch: str = ""
    base_commit: str = ""
    cwd: str = ""
    env_vars: dict = field(default_factory=dict)
    round1_patch: str = ""
    round1_eval_result: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> AgentSnapshot:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot: AgentSnapshot | None = None

    def capture(
        self, agent: Any, *, base_commit: str = "",
        round1_patch: str = "", round1_eval_result: dict | None = None,
    ) -> AgentSnapshot:
        """Capture current agent state into a snapshot (deep-copied messages)."""
        import copy
        snapshot = AgentSnapshot(
            messages=copy.deepcopy(agent.messages),
            phase=getattr(agent, "phase", "round1"),
            cost=agent.cost,
            n_calls=agent.n_calls,
            n_consecutive_format_errors=getattr(agent, "n_consecutive_format_errors", 0),
            start_time_epoch=getattr(agent, "_start_time", time.time()),
            config=agent.config.model_dump(mode="json") if hasattr(agent.config, "model_dump") else {},
            extra_template_vars=dict(getattr(agent, "extra_template_vars", {})),
            model_name=agent.model.model_name if hasattr(agent.model, "model_name") else "",
            model_kwargs=dict(getattr(agent.model, "model_kwargs", {})),
            cwd=self._get_cwd(agent),
            base_commit=base_commit,
            round1_patch=round1_patch,
            round1_eval_result=round1_eval_result or {},
        )
        # Snapshot workspace via canonical diff (with untracked files)
        snapshot.workspace_patch = self._canonical_diff(snapshot.cwd, base_commit)
        self._snapshot = snapshot
        self._write(snapshot)
        logger.info("Checkpoint: phase=%s cost=%.4f calls=%d msgs=%d ws=%dch",
                     snapshot.phase, snapshot.cost, snapshot.n_calls,
                     len(snapshot.messages), len(snapshot.workspace_patch))
        return snapshot

    def _get_cwd(self, agent) -> str:
        env = getattr(agent, "env", None)
        if env and hasattr(env, "config") and hasattr(env.config, "cwd"):
            return str(env.config.cwd)
        return os.getcwd()

    def _canonical_diff(self, cwd: str, base_commit: str = "") -> str:
        """Capture full workspace diff including untracked files.

        Uses: git add -N (dry-run add new files) → git diff --binary <base> → git reset -N
        """
        if not cwd or not os.path.isdir(cwd):
            return ""
        git_dir = self._find_git_dir(cwd)
        if not git_dir:
            return ""
        try:
            base = base_commit if base_commit else "HEAD"
            result = subprocess.run(
                ["git", "add", "-N", "."],
                capture_output=True, timeout=10, cwd=git_dir,
            )
            result = subprocess.run(
                ["git", "diff", "--binary", base],
                capture_output=True, text=True, timeout=10, cwd=git_dir,
            )
            subprocess.run(
                ["git", "reset", "-N", "."],
                capture_output=True, timeout=10, cwd=git_dir,
            )
            if result.returncode == 0:
                return result.stdout
        except Exception as e:
            logger.warning("Failed to snapshot workspace: %s", e)
        return ""

    def _find_git_dir(self, cwd: str) -> str | None:
        current = Path(cwd).resolve()
        while current.exists():
            if (current / ".git").exists():
                return str(current)
            parent = current.parent
            if parent == current: break
            current = parent
        return None

    def _write(self, snapshot: AgentSnapshot):
        path = self.checkpoint_dir / "checkpoint.json"
        data = snapshot.to_dict()
        data["messages"] = data["messages"][-100:]  # keep last 100
        path.write_text(json.dumps(data, indent=2))

    def fork_agent(self, agent_factory: callable, *,
                   snapshot: AgentSnapshot | None = None,
                   modify_messages: callable | None = None) -> Any:
        """Create a new agent from checkpoint with optional message mutation."""
        import copy
        snap = snapshot or self.load()
        if snap is None:
            raise ValueError("No checkpoint available")
        agent = agent_factory()
        agent.messages = copy.deepcopy(snap.messages)
        if modify_messages:
            agent.messages = modify_messages(copy.deepcopy(snap.messages))
        agent.phase = snap.phase
        agent.cost = snap.cost
        agent.n_calls = snap.n_calls
        agent.n_consecutive_format_errors = snap.n_consecutive_format_errors
        agent._start_time = snap.start_time_epoch
        agent.extra_template_vars = dict(snap.extra_template_vars)
        logger.info("Forked agent: phase=%s cost=%.4f calls=%d msgs=%d",
                     agent.phase, agent.cost, agent.n_calls, len(agent.messages))
        return agent

    def load(self) -> AgentSnapshot | None:
        path = self.checkpoint_dir / "checkpoint.json"
        if not path.exists(): return None
        return AgentSnapshot.from_dict(json.loads(path.read_text()))

    def load_messages(self) -> list[dict] | None:
        path = self.checkpoint_dir / "checkpoint.json"
        if not path.exists(): return None
        return json.loads(path.read_text()).get("messages", [])

    def has_checkpoint(self) -> bool:
        return (self.checkpoint_dir / "checkpoint.json").exists()

    @property
    def path(self) -> Path:
        return self.checkpoint_dir

    @property
    def snapshot_path(self) -> Path:
        return self.checkpoint_dir / "checkpoint.json"
