"""
MAP-Elites Grid (OpenEvolve Pattern) for EvoKit.
N-dimensional behavioral tessellation grid storing high-performing algorithms
across orthogonal complexity and runtime characteristics.
"""

from __future__ import annotations

import ast
import random
from typing import Any, Dict, List, Optional, Tuple
from pydantic import Field

from evokit.core.entity import Individual
from evokit.plugins.populations.base import BasePopulationManager, PopulationConfig


class MAPElitesConfig(PopulationConfig):
    grid_dim_1: int = Field(default=5, description="Number of bins along feature 1 (code complexity)")
    grid_dim_2: int = Field(default=5, description="Number of bins along feature 2 (runtime / metric)")
    max_complexity: int = Field(default=1000, description="Max character length for feature 1 discretization")
    max_metric_val: float = Field(default=1.0, description="Max metric/runtime for feature 2 discretization")


class MAPElitesGrid(BasePopulationManager):
    """
    Quality-Diversity archive maintaining an N-dimensional grid of elite programs.
    """

    def __init__(self, config: Optional[Dict[str, Any] | MAPElitesConfig] = None) -> None:
        if isinstance(config, MAPElitesConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = MAPElitesConfig(**config)
        else:
            cfg = MAPElitesConfig()
        super().__init__(cfg)
        self.map_cfg: MAPElitesConfig = cfg

        # Grid cells: (bin_x, bin_y) -> Individual
        self.grid: Dict[Tuple[int, int], Individual] = {}

    def _compute_features(self, individual: Individual) -> Tuple[int, int]:
        """Compute behavioral descriptor coordinates (bin_x, bin_y)."""
        # Feature 1: Code AST length / complexity
        code_len = len(individual.code.strip())
        norm_feat1 = min(1.0, code_len / max(1, self.map_cfg.max_complexity))
        bin_x = min(self.map_cfg.grid_dim_1 - 1, int(norm_feat1 * self.map_cfg.grid_dim_1))

        # Feature 2: Runtime or primary metric
        runtime = individual.metadata.get("runtime_seconds", 0.0)
        norm_feat2 = min(1.0, runtime / max(0.001, self.map_cfg.max_metric_val))
        bin_y = min(self.map_cfg.grid_dim_2 - 1, int(norm_feat2 * self.map_cfg.grid_dim_2))

        return (bin_x, bin_y)

    def add(self, individual: Individual) -> bool:
        if not individual.is_valid or individual.fitness is None:
            return False

        cell = self._compute_features(individual)
        if cell not in self.grid:
            self.grid[cell] = individual
            return True

        current = self.grid[cell]
        is_better = (
            individual.fitness > current.fitness
            if self.is_maximize
            else individual.fitness < current.fitness
        )

        if is_better:
            self.grid[cell] = individual
            return True

        return False

    def sample_parents(self, k: int = 1) -> List[Individual]:
        occupied = list(self.grid.values())
        if not occupied:
            return []
        if len(occupied) <= k:
            return list(occupied)
        return random.sample(occupied, k)

    def get_elites(self, top_k: int = 1) -> List[Individual]:
        occupied = list(self.grid.values())
        occupied.sort(
            key=lambda x: (x.fitness if x.fitness is not None else float("-inf" if self.is_maximize else "inf")),
            reverse=self.is_maximize,
        )
        return occupied[:top_k]

    def get_all(self) -> List[Individual]:
        return list(self.grid.values())

    def size(self) -> int:
        return len(self.grid)

    def coverage(self) -> float:
        """Fraction of grid cells currently occupied."""
        total_cells = self.map_cfg.grid_dim_1 * self.map_cfg.grid_dim_2
        return len(self.grid) / max(1, total_cells)
