"""
Base abstraction for Evaluator Sandbox Services in EvoKit.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.benchmarks.base import BaseBenchmarkTask
from evokit.core.base import BaseComponent, ComponentConfig
from evokit.core.entity import Individual, MetricRecord


class EvaluatorConfig(ComponentConfig):
    timeout_seconds: float = Field(default=30.0, description="Per-evaluation wall-clock timeout limit")
    memory_limit_mb: int = Field(default=2048, description="Maximum virtual memory in MB")
    max_workers: int = Field(default=1, description="Parallel evaluation worker pool size")


class BaseEvaluator(BaseComponent):
    """
    Contract for evaluating unverified LLM-generated code under isolated sandbox boundaries.
    """

    def __init__(self, config: Optional[Dict[str, Any] | EvaluatorConfig] = None) -> None:
        super().__init__(config)
        self.cfg = config if isinstance(config, EvaluatorConfig) else EvaluatorConfig(**(config or {}))

    @abstractmethod
    def evaluate(
        self,
        individual: Individual,
        task: BaseBenchmarkTask,
        split: str = "train",
    ) -> MetricRecord:
        """
        Execute candidate individual code against the benchmark task in the sandbox.
        Returns a complete MetricRecord with fitness, metrics, and stdout/stderr traces.
        """
        pass

    def evaluate_batch(
        self,
        individuals: List[Individual],
        task: BaseBenchmarkTask,
        split: str = "train",
    ) -> List[MetricRecord]:
        """
        Evaluate a batch of candidate individuals against the benchmark task.
        Default implementation evaluates sequentially; subclasses can execute concurrently.
        """
        return [self.evaluate(ind, task, split=split) for ind in individuals]
