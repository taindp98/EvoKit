"""
EvolutionContext: Immutable state container passed into mutation and search operators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from evokit.core.gateway import LLMGateway


@dataclass(frozen=True)
class EvolutionContext:
    """
    Immutable state container injected into mutation operators during each generational step.
    Provides isolated read access to the current generation, population snapshot,
    benchmark task specifications, runtime configuration, and the active LLM gateway.
    """
    generation: int
    population_view: Any
    task_spec: Dict[str, Any] = field(default_factory=dict)
    runtime_config: Dict[str, Any] = field(default_factory=dict)
    llm_gateway: Optional[LLMGateway] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **kwargs) -> EvolutionContext:
        """Return a new EvolutionContext with selected updated fields."""
        new_data = {
            "generation": self.generation,
            "population_view": self.population_view,
            "task_spec": self.task_spec,
            "runtime_config": self.runtime_config,
            "llm_gateway": self.llm_gateway,
            "metadata": self.metadata,
        }
        new_data.update(kwargs)
        return EvolutionContext(**new_data)
