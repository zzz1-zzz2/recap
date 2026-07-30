"""Evaluation harness wrapper."""
import logging
from typing import Any

logger = logging.getLogger("recap.evaluation")


def evaluate_patch(
    instance_id: str,
    patch_text: str,
    *,
    run_id: str | None = None,
    timeout: int = 600,
    force_rebuild: bool = False,
) -> dict:
    """Evaluate a patch using the SWE-bench official harness.

    Returns dict with 'resolved' bool and 'report' dict.
    """
    from evaluators.official_harness import OfficialHarnessGateway
    from instance_registry import InstanceRegistry

    reg = InstanceRegistry()
    spec = reg.get_instance(instance_id)
    if spec is None:
        return {"resolved": False, "error": f"Instance {instance_id} not found"}

    import time
    run_id = run_id or f"eval_{instance_id}_{int(time.time())}"
    gateway = OfficialHarnessGateway(
        run_id=run_id, rm_image=False,
        force_rebuild=force_rebuild, timeout=timeout,
    )
    try:
        result = gateway.evaluate(spec, patch_text, run_id=run_id)
        resolved = getattr(result, "resolved", None)
        if resolved is None:
            resolved = result.report.get("resolved", False) if hasattr(result, "report") else False
        return {
            "resolved": resolved,
            "report": getattr(result, "report", {}) if hasattr(result, "report") else {},
        }
    except Exception as e:
        logger.error("Evaluation failed: %s", e)
        return {"resolved": False, "error": str(e)}
