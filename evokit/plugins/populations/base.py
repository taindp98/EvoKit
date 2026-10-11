"""
Base abstraction for Population and Archive Management in EvoKit (Pillar 1).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.core.base import BaseComponent, ComponentConfig
from evokit.core.entity import Individual


class PopulationConfig(ComponentConfig):
    capacity: int = Field(default=20, description="Maximum population/archive capacity")
    direction: str = Field(default="MAXIMIZE", description="Optimization direction (MAXIMIZE or MINIMIZE)")


class BasePopulationManager(BaseComponent):
    """
    Contract for managing candidate solutions, archives, parent selection,
    and lineage graphs across evolutionary generations.
    """

    def __init__(self, config: Optional[Dict[str, Any] | PopulationConfig] = None) -> None:
        super().__init__(config)
        self.cfg = config if isinstance(config, PopulationConfig) else PopulationConfig(**(config or {}))

    @property
    def is_maximize(self) -> bool:
        return self.cfg.direction.upper() == "MAXIMIZE"

    @abstractmethod
    def add(self, individual: Individual) -> bool:
        """
        Ingest an evaluated individual and determine archive admission/replacement.
        Returns True if admitted, False otherwise.
        """
        pass

    @abstractmethod
    def sample_parents(self, k: int = 1) -> List[Individual]:
        """Sample k parent candidates for mutation or crossover."""
        pass

    @abstractmethod
    def get_elites(self, top_k: int = 1) -> List[Individual]:
        """Return top_k best performing individuals in the population."""
        pass

    @abstractmethod
    def get_all(self) -> List[Individual]:
        """Return all active individuals."""
        pass

    @abstractmethod
    def size(self) -> int:
        """Return count of active individuals."""
        pass

    def get_view(self) -> List[Dict[str, Any]]:
        """Return a lightweight serializable view of current population state."""
        return [
            {
                "id": ind.id,
                "fitness": ind.fitness,
                "operator": ind.operator,
                "thought": ind.thought,
                "generation": ind.generation,
            }
            for ind in self.get_all()
        ]
