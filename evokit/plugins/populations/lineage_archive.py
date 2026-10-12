"""
Lineage Graph Archive (EoH Pattern) for EvoKit.
Preserves explicit Directed Acyclic Graph (DAG) structures mapping how textual ideas
and code variants transition across generations, with elitist replacement and parent selection.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Set, Tuple
from pydantic import Field

from evokit.core.entity import Individual
from evokit.plugins.populations.base import BasePopulationManager, PopulationConfig


class LineageArchiveConfig(PopulationConfig):
    tournament_size: int = Field(default=2, description="Tournament size for parent selection")
    dedup_by_score: bool = Field(
        default=True,
        description=(
            "Reject candidates whose code or exact fitness duplicates an existing "
            "population member (LLM4AD EoH `has_duplicate_function`). Forces "
            "behaviorally distinct heuristics and prevents the Best-Fit baseline "
            "from cloning across the whole pool."
        ),
    )


class LineageGraphArchive(BasePopulationManager):
    """
    Maintains active population and full historical Directed Acyclic Graph (DAG)
    of all synthesized individuals and idea lineage.
    """

    def __init__(self, config: Optional[Dict[str, Any] | LineageArchiveConfig] = None) -> None:
        if isinstance(config, LineageArchiveConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = LineageArchiveConfig(**config)
        else:
            cfg = LineageArchiveConfig()
        super().__init__(cfg)
        self.archive_cfg: LineageArchiveConfig = cfg

        # Full historical graph
        self.nodes: Dict[str, Individual] = {}
        self.edges: List[Tuple[str, str]] = []  # (parent_id, child_id)

        # Active elite pool
        self.population: List[Individual] = []

    def add(self, individual: Individual) -> bool:
        # Record into lineage DAG regardless of validity
        self.nodes[individual.id] = individual
        for p_id in individual.parent_ids:
            if p_id in self.nodes:
                self.edges.append((p_id, individual.id))

        if not individual.is_valid or individual.fitness is None:
            return False

        # Score/code de-duplication (LLM4AD EoH `has_duplicate_function`): reject a
        # candidate whose code or exact fitness already exists in the pool, so the
        # population keeps behaviorally distinct heuristics instead of cloning the
        # baseline. (Disable via dedup_by_score=False for the old behavior.)
        if self.archive_cfg.dedup_by_score:
            for member in self.population:
                if member.code == individual.code or member.fitness == individual.fitness:
                    return False

        # (parents + offspring) survival: admit, then keep the best `capacity`.
        # While the pool is below capacity this admits any valid, non-duplicate
        # candidate — including equal/intermediate ones — so the search is not
        # frozen by a strict "must beat the worst" test on a tiny pool.
        self.population.append(individual)
        self._sort_population()
        if len(self.population) > self.cfg.capacity:
            self.population = self.population[: self.cfg.capacity]
        # Admitted iff it survived the trim.
        return individual.id in {m.id for m in self.population}

    def _sort_population(self) -> None:
        """Sort active population best-first."""
        self.population.sort(
            key=lambda x: (x.fitness if x.fitness is not None else float("-inf" if self.is_maximize else "inf")),
            reverse=self.is_maximize,
        )

    def sample_parents(self, k: int = 1) -> List[Individual]:
        if not self.population:
            return []
        if len(self.population) <= k:
            return list(self.population)

        selected = []
        for _ in range(k):
            # Tournament selection
            candidates = random.sample(self.population, min(self.archive_cfg.tournament_size, len(self.population)))
            winner = max(candidates, key=lambda x: x.fitness if x.fitness is not None else float("-inf")) if self.is_maximize else min(candidates, key=lambda x: x.fitness if x.fitness is not None else float("inf"))
            selected.append(winner)
        return selected

    def get_elites(self, top_k: int = 1) -> List[Individual]:
        return self.population[:top_k]

    def get_all(self) -> List[Individual]:
        return list(self.population)

    def size(self) -> int:
        return len(self.population)

    def get_lineage(self, individual_id: str) -> List[Individual]:
        """Trace full ancestry chain of an individual back to generation 0."""
        lineage = []
        curr_id = individual_id
        visited: Set[str] = set()

        while curr_id and curr_id in self.nodes and curr_id not in visited:
            visited.add(curr_id)
            node = self.nodes[curr_id]
            lineage.append(node)
            curr_id = node.parent_ids[0] if node.parent_ids else None

        return lineage

    def to_dot_graph(self) -> str:
        """Generate Graphviz DOT representation of the evolutionary lineage."""
        lines = ["digraph EvoLineage {", '  rankdir="TB";', '  node [shape="box", style="rounded"];']
        for ind_id, ind in self.nodes.items():
            color = "green" if ind.is_valid else "red"
            label = f"{ind.id}\\nGen:{ind.generation} Op:{ind.operator or ''}\\nFit:{ind.fitness}"
            lines.append(f'  "{ind_id}" [label="{label}", color="{color}"];')
        for p_id, c_id in self.edges:
            lines.append(f'  "{p_id}" -> "{c_id}";')
        lines.append("}")
        return "\n".join(lines)
