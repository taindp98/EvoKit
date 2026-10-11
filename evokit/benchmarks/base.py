"""
Base abstraction for Evolutionary Benchmark Tasks in EvoKit.
Strictly segregates Train / Test dataset splits and abstracts domain prompts as data.
Supports parallel instance-level evaluation.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from evokit.core.base import BaseComponent, ComponentConfig
from evokit.core.entity import MetricRecord


class PromptSpec(BaseModel):
    """Declarative domain specification provided to LLM mutation plugins."""
    task_name: str
    description: str
    function_name: str
    signature: str
    docstring: str
    input_description: str
    output_description: str
    initial_code: str
    evaluation_criteria: str = ""


class BenchmarkConfig(ComponentConfig):
    num_train_instances: int = Field(default=5, description="Number of training instances for evolutionary loop")
    num_test_instances: int = Field(default=10, description="Number of held-out test instances for generalization")
    parallel_workers: int = Field(default=1, description="Number of worker threads for parallel instance evaluation")
    seed: int = Field(default=42, description="Random seed for deterministic instance generation")


class BaseBenchmarkTask(BaseComponent):
    """
    Abstract contract for benchmark problems.
    Enforces train/test split isolation and provides prompt metadata.
    """

    def __init__(self, config: Optional[Dict[str, Any] | BenchmarkConfig] = None) -> None:
        super().__init__(config)
        self.cfg = config if isinstance(config, BenchmarkConfig) else BenchmarkConfig(**(config or {}))
        self.train_instances: List[Any] = []
        self.test_instances: List[Any] = []
        self._prompt_spec: Optional[PromptSpec] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name (e.g., 'obp_online', 'tsp_construct')."""
        pass

    @property
    @abstractmethod
    def direction(self) -> str:
        """Optimization goal: 'MAXIMIZE' or 'MINIMIZE'."""
        pass

    @property
    @abstractmethod
    def prompt_spec(self) -> PromptSpec:
        """Domain prompt templates and function signature."""
        pass

    def _setup_internal(self) -> None:
        self._generate_instances()
        self.logger.info(
            f"Initialized {self.name} benchmark with {len(self.train_instances)} train and {len(self.test_instances)} test instances."
        )

    @abstractmethod
    def _generate_instances(self) -> None:
        """Populate self.train_instances and self.test_instances."""
        pass

    @abstractmethod
    def evaluate_code(self, code_str: str, split: str = "train", timeout: float = 30.0) -> MetricRecord:
        """
        Evaluate a candidate Python code string on train or test instances.
        Returns a structured MetricRecord.
        """
        pass

    def reference_bound(self, split: str = "train") -> Optional[float]:
        """Optional domain reference value for computing an optimality gap.

        Benchmarks that have a meaningful reference point — a theoretical lower
        bound, a known optimum, or a strong baseline score — may override this
        to return that value (in the same units/direction as fitness). The
        engine uses it only to annotate the trajectory with a ``gap``; a gap is
        computed solely when this returns a non-None value, so benchmarks
        without such a reference (the default) simply omit the gap. This keeps
        the invariant engine free of any per-benchmark metric assumptions.
        """
        return None

    def get_context_spec(self) -> Dict[str, Any]:
        """Serializable specification dictionary for EvolutionContext."""
        p = self.prompt_spec
        return {
            "benchmark_name": self.name,
            "direction": self.direction,
            "function_name": p.function_name,
            "signature": p.signature,
            "description": p.description,
            "input_description": p.input_description,
            "output_description": p.output_description,
            "initial_code": p.initial_code,
            "docstring": p.docstring,
            "evaluation_criteria": p.evaluation_criteria,
            "num_train_instances": len(self.train_instances),
            "num_test_instances": len(self.test_instances),
        }
