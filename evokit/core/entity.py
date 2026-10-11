"""
Data contracts and core entities for EvoKit.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class MetricRecord(BaseModel):
    """Structured evaluation metrics for an individual."""
    model_config = ConfigDict(extra="allow")

    fitness: Optional[float] = None
    is_valid: bool = False
    metrics: Dict[str, Any] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    error_trace: Optional[str] = None
    runtime_seconds: float = 0.0


class Individual(BaseModel):
    """
    The fundamental unit of evolution in EvoKit.
    Encapsulates synthesized algorithm code, natural-language reasoning tokens,
    scalar fitness, multi-objective metrics, parentage lineage, and metadata.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    id: str = Field(default_factory=lambda: f"ind_{uuid.uuid4().hex[:12]}")
    code: str
    thought: Optional[str] = None
    fitness: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    is_valid: bool = False
    parent_ids: List[str] = Field(default_factory=list)
    generation: int = 0
    operator: Optional[str] = None
    error_trace: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    def apply_evaluation(self, record: MetricRecord) -> None:
        """Incorporate an evaluation result into this individual."""
        self.fitness = record.fitness
        self.is_valid = record.is_valid
        self.metrics.update(record.metrics)
        self.error_trace = record.error_trace or record.stderr
        self.metadata["runtime_seconds"] = record.runtime_seconds
        self.metadata["stdout"] = record.stdout
        self.metadata["stderr"] = record.stderr

    def clone(self, new_id: bool = True) -> Individual:
        """Create a deep copy of this individual."""
        data = self.model_dump()
        if new_id:
            data["id"] = f"ind_{uuid.uuid4().hex[:12]}"
            data["created_at"] = time.time()
        return Individual(**data)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Individual:
        return cls(**data)
