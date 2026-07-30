"""P1-3D: Acquisition schema — Router output data structures."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from recap.diagnosis.search_contract import SearchActionType, SearchTarget


class AcquisitionStatus(str, Enum):
    """Router outcome for one SearchAction."""

    FOUND = "FOUND"                  # at least one hit returned
    NOT_FOUND = "NOT_FOUND"           # searched, zero hits
    INVALID_TARGET = "INVALID_TARGET"  # target value malformed
    UNSUPPORTED = "UNSUPPORTED"      # action_type not implemented in v1
    ERROR = "ERROR"                  # exception during search


@dataclass
class AcquisitionHit:
    """One retrieval hit returned by a Router executor."""

    file_path: str = ""
    start_line: int = 0
    end_line: int = 0
    symbol: str = ""
    content: str = ""

    retrieval_method: str = ""
    relevance_reason: str = ""

    action_id: str = ""
    evidence_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "content": self.content,
            "retrieval_method": self.retrieval_method,
            "relevance_reason": self.relevance_reason,
            "action_id": self.action_id,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass
class AcquisitionResult:
    """One Router execution result for one SearchAction."""

    action_id: str = ""
    action_type: SearchActionType = SearchActionType.FIND_DEFINITION
    target: SearchTarget = field(default_factory=SearchTarget)
    status: AcquisitionStatus = AcquisitionStatus.NOT_FOUND

    hits: list[AcquisitionHit] = field(default_factory=list)

    files_examined: int = 0
    budget_used: int = 0
    budget_limit: int = 0       # maximum hits Router may return
    scan_limit: int = 0         # maximum files Router may scan
    stop_reason: str = ""
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "target": self.target.to_dict(),
            "status": self.status.value,
            "hits": [h.to_dict() for h in self.hits],
            "files_examined": self.files_examined,
            "budget_used": self.budget_used,
            "budget_limit": self.budget_limit,
            "scan_limit": self.scan_limit,
            "stop_reason": self.stop_reason,
            "errors": list(self.errors),
        }
