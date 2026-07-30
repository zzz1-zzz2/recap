"""ConDiag Agent Configuration — single source of truth for all Agent configurations.

Design:
  - One factory function, one set of locked prompt templates
  - Protocol-versioned: "baseline_reproduction" vs "persistent_revision"
  - Config SHA is checked at runtime to detect drift
  - All secrets come from env, not hardcoded strings
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

LOCKED_CONFIG_PATH = Path(__file__).parent.parent.parent / "configs" / "locked" / "minisweagent_swebench_v2.4.1.yaml"

# Full SHA256 hex digest of the locked YAML at the time of writing.
# If upgrading the locked YAML, recalculate and update this value — no auto-follow.
LOCKED_YAML_SHA256 = "229f178a07faa109e3a020c75dfe603b6a24200e1f3aebe257b97098fd3a6fee"


class ConfigDriftError(RuntimeError):
    """Raised when the locked YAML config has drifted from expected SHA."""


def sha256_full(text: str) -> str:
    """Full 64-character SHA256 hex digest. Use for permanent records."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_short(text: str) -> str:
    """First 16 characters of SHA256. Use for display/logging."""
    return sha256_full(text)[:16]


@dataclass(frozen=True)
class RevisionProtocolConfig:
    """Constraints that differ between R1 and R2 but form a single experiment protocol.

    These are NOT in AgentConfig because R1 and R2 apply them differently,
    but they MUST be recorded as part of the episode config SHA.
    """

    r1_wall_time_limit_seconds: int = 3600
    r2_wall_time_limit_seconds: int = 3600
    r1_max_consecutive_format_errors: int = 15
    r2_max_consecutive_format_errors: int = 3

    @property
    def sha(self) -> str:
        raw = f"R1wall={self.r1_wall_time_limit_seconds}|R1fmt={self.r1_max_consecutive_format_errors}"
        raw += f"|R2wall={self.r2_wall_time_limit_seconds}|R2fmt={self.r2_max_consecutive_format_errors}"
        return sha256_short(raw)


@dataclass(frozen=True)
class AgentConfig:
    """Immutable agent configuration with protocol tracking.

    config_sha covers ALL fields below — used as the episode config identity.
    source_yaml_sha is auto-filled from the locked YAML on construction
    and CANNOT be overridden by callers.
    """

    protocol_name: str = "persistent_revision"
    protocol_version: str = "1.0"
    model_name: str = "openai/deepseek-v4-pro"
    temperature: float = 0.0
    max_tokens: int = 4096
    step_limit: int = 0
    cost_limit: float = 5.0
    revision_protocol: RevisionProtocolConfig = RevisionProtocolConfig()

    # Computed fields — not settable by caller to prevent forgery
    config_sha: str = field(default="", init=False)
    source_yaml_sha: str = field(default="", init=False)

    def __post_init__(self):
        # Auto-fill source_yaml_sha from locked YAML — always reads from disk
        if LOCKED_CONFIG_PATH.exists():
            raw = LOCKED_CONFIG_PATH.read_text("utf-8")
            object.__setattr__(self, "source_yaml_sha", sha256_full(raw))

        # Compute config SHA covering ALL protocol-relevant fields
        if not self.config_sha:
            raw = (
                f"protocol={self.protocol_name}:{self.protocol_version}"
                f"|model={self.model_name}"
                f"|temp={self.temperature}"
                f"|maxtok={self.max_tokens}"
                f"|step={self.step_limit}"
                f"|cost={self.cost_limit}"
                f"|yaml={self.source_yaml_sha}"
                f"|rev={self.revision_protocol.sha}"
            )
            full_sha = sha256_full(raw)
            object.__setattr__(self, "config_sha", full_sha[:16])


def load_locked_yaml() -> dict:
    """Load the locked YAML, verify its SHA, return parsed config dict."""
    if not LOCKED_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Locked config not found: {LOCKED_CONFIG_PATH}")

    raw = LOCKED_CONFIG_PATH.read_text(encoding="utf-8")
    actual_sha = sha256_full(raw)

    if actual_sha != LOCKED_YAML_SHA256:
        raise ConfigDriftError(
            f"Locked YAML SHA mismatch:\n"
            f"  expected: {LOCKED_YAML_SHA256}\n"
            f"  actual:   {actual_sha}\n"
            f"File: {LOCKED_CONFIG_PATH}\n"
            f"If you intentionally upgraded the locked YAML, update "
            f"LOCKED_YAML_SHA256 in condiag/agent/config.py"
        )

    return yaml.safe_load(raw)


_REDACTED_KEYS = frozenset({"api_key", "authorization", "token", "password"})


def redact_trajectory(trajectory: dict) -> dict:
    """Remove sensitive fields (API keys, tokens) from a serialized trajectory.

    Call this before saving trajectory.json to prevent secret leakage.
    Modifies the dict in-place and returns it.
    """
    import copy
    result = copy.deepcopy(trajectory)
    model_info = result.get("info", {}).get("config", {}).get("model", {})
    kwargs = model_info.get("model_kwargs", {})
    if isinstance(kwargs, dict):
        for key in _REDACTED_KEYS:
            if key in kwargs:
                kwargs[key] = "***REDACTED***"
    return result


def build_agent_factory(
    config: AgentConfig,
    instance_id: str,
) -> Callable[[], Any]:
    """Build and return a callable that creates a new agent instance.

    Validates source_yaml_sha against the locked YAML on disk.
    Each call to the returned factory creates a fresh agent with its own
    Docker container — suitable for forked branches in Round 2.

    Usage:
        factory = build_agent_factory(config, instance_id)
        agent = factory()  # new container, new agent
    """
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.models.litellm_model import LitellmModel
    from minisweagent.run.benchmarks.swebench import get_swebench_docker_image_name

    # Validate source_yaml_sha against disk
    raw_yaml = LOCKED_CONFIG_PATH.read_text("utf-8")
    actual_yaml_sha = sha256_full(raw_yaml)
    if config.source_yaml_sha and config.source_yaml_sha != actual_yaml_sha:
        raise ConfigDriftError(
            f"AgentConfig.source_yaml_sha ({config.source_yaml_sha[:16]}...) "
            f"does not match disk ({actual_yaml_sha[:16]}...)"
        )

    pred = {"instance_id": instance_id}
    image_name = get_swebench_docker_image_name(pred)

    yaml_config = load_locked_yaml()
    agent_cfg = yaml_config.get("agent", {})
    env_cfg = yaml_config.get("environment", {})
    model_cfg = yaml_config.get("model", {})

    system_template = agent_cfg.get("system_template", "")
    instance_template = agent_cfg.get("instance_template", "")
    env_timeout = env_cfg.get("timeout", 120)
    env_interpreter = env_cfg.get("interpreter", ["bash", "-c"])
    env_vars = env_cfg.get("env", {})
    env_cwd = env_cfg.get("cwd", "/testbed")

    model_kwargs = dict(model_cfg.get("model_kwargs", {}))
    model_kwargs.update({
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "api_base": os.environ.get(
            "DEEPSEEK_API_BASE", "https://api.deepseek.com/v1"
        ),
    })
    observation_template = model_cfg.get("observation_template", "")
    format_error_template = model_cfg.get("format_error_template", "")

    def factory():
        env = DockerEnvironment(
            image=image_name,
            cwd=env_cwd,
            timeout=env_timeout,
            interpreter=env_interpreter,
        )
        for k, v in env_vars.items():
            env.config.env[k] = v

        model = LitellmModel(
            model_name=config.model_name,
            model_kwargs=model_kwargs,
            cost_tracking="ignore_errors",
            observation_template=observation_template,
            format_error_template=format_error_template,
        )

        agent = DefaultAgent(
            model=model,
            env=env,
            system_template=system_template,
            instance_template=instance_template,
            step_limit=config.step_limit,
            cost_limit=config.cost_limit,
            output_path=None,
        )
        return agent

    return factory


def require_api_key():
    """Exit if DEEPSEEK_API_KEY is not set. Call at entry points."""
    if not os.environ.get("DEEPSEEK_API_KEY"):
        import sys
        print("FATAL: DEEPSEEK_API_KEY environment variable is not set.", file=sys.stderr)
        print("Create a .env file with DEEPSEEK_API_KEY=sk-... or export it.", file=sys.stderr)
        sys.exit(1)
