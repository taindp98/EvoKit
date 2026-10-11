"""
Reflection Buffer (ReEvo Pattern) for EvoKit.
Maintains execution failure logs, stdout/stderr diagnostic traces, and textual reflection memory
alongside elite candidate solutions.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.core.entity import Individual
from evokit.plugins.populations.base import BasePopulationManager, PopulationConfig


class ReflectionBufferConfig(PopulationConfig):
    max_failure_history: int = Field(default=20, description="Maximum number of failure logs to retain")
    max_reflection_memory: int = Field(default=10, description="Maximum reflection notes in short-term buffer")
    selection_strategy: str = Field(
        default="non_elitist",
        description=(
            "Survival after each generation. 'non_elitist' (default, faithful to original "
            "ReEvo): each generation's offspring unconditionally replace the population. "
            "'elitist' (myirace --elitist-selection variant): keep the best `capacity` of "
            "parents + offspring. The global best-so-far incumbent is preserved in both modes."
        ),
    )


class ReflectionBuffer(BasePopulationManager):
    """
    Dual-buffer archive storing both successful algorithmic variants and execution diagnostics
    for reflective LLM prompt construction.
    """

    def __init__(self, config: Optional[Dict[str, Any] | ReflectionBufferConfig] = None) -> None:
        if isinstance(config, ReflectionBufferConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = ReflectionBufferConfig(**config)
        else:
            cfg = ReflectionBufferConfig()
        super().__init__(cfg)
        self.ref_cfg: ReflectionBufferConfig = cfg

        self.population: List[Individual] = []
        self.failure_buffer: List[Individual] = []
        self.short_term_reflections: List[str] = []
        self.long_term_reflections: List[str] = []
        # Global best-so-far incumbent, kept OUTSIDE the active population so it is
        # never lost under non-elitist generational replacement (as ReEvo tracks
        # its `elitist`). Exposed via get_elites so search progress is monotone.
        self._incumbent: Optional[Individual] = None
        self._cur_gen: Optional[int] = None

    @property
    def _non_elitist(self) -> bool:
        return str(self.ref_cfg.selection_strategy).lower() != "elitist"

    def _is_better_than(self, a: Individual, b: Optional[Individual]) -> bool:
        if b is None or b.fitness is None:
            return True
        if a.fitness is None:
            return False
        return a.fitness > b.fitness if self.is_maximize else a.fitness < b.fitness

    def add(self, individual: Individual) -> bool:
        if not individual.is_valid or individual.error_trace:
            self.failure_buffer.append(individual)
            if len(self.failure_buffer) > self.ref_cfg.max_failure_history:
                self.failure_buffer.pop(0)

        if not individual.is_valid or individual.fitness is None:
            return False

        # Track the global incumbent regardless of selection strategy.
        if self._is_better_than(individual, self._incumbent):
            self._incumbent = individual

        if self._non_elitist:
            # Non-elitist (faithful original ReEvo): offspring replace the
            # population generationally. The engine adds candidates one at a time,
            # so we detect a generation change and flush the active pool, then
            # accumulate this generation's members (capped at capacity).
            gen = individual.generation
            if gen != self._cur_gen:
                self._cur_gen = gen
                self.population = []
            self.population.append(individual)
            self._sort_population()
            if len(self.population) > self.cfg.capacity:
                self.population = self.population[: self.cfg.capacity]
            return True

        # Elitist (myirace --elitist-selection): keep the best `capacity` of
        # parents + offspring (replace-worst-if-better).
        if len(self.population) < self.cfg.capacity:
            self.population.append(individual)
            self._sort_population()
            return True
        worst = self.population[-1]
        if self._is_better_than(individual, worst):
            self.population[-1] = individual
            self._sort_population()
            return True
        return False

    def _sort_population(self) -> None:
        self.population.sort(
            key=lambda x: (x.fitness if x.fitness is not None else float("-inf" if self.is_maximize else "inf")),
            reverse=self.is_maximize,
        )

    def add_reflection(self, reflection_text: str, is_long_term: bool = False) -> None:
        """Store natural-language diagnostic reflections."""
        cleaned = reflection_text.strip()
        if not cleaned:
            return
        if is_long_term:
            self.long_term_reflections.append(cleaned)
        else:
            self.short_term_reflections.append(cleaned)
            if len(self.short_term_reflections) > self.ref_cfg.max_reflection_memory:
                self.short_term_reflections.pop(0)

    def get_recent_reflections(self, k: int = 3) -> List[str]:
        return self.short_term_reflections[-k:]

    def get_failed_cases(self, k: int = 2) -> List[Individual]:
        return self.failure_buffer[-k:]

    def sample_parents(self, k: int = 1) -> List[Individual]:
        if not self.population:
            return []
        if len(self.population) <= k:
            return list(self.population)
        return random.sample(self.population, k)

    def get_elites(self, top_k: int = 1) -> List[Individual]:
        # Front-load the preserved global incumbent so best-so-far is never lost
        # under non-elitist generational replacement (it may have fallen out of
        # the current population). De-duplicate by id while keeping order.
        ranked: List[Individual] = []
        seen = set()
        for ind in ([self._incumbent] if self._incumbent is not None else []) + self.population:
            if ind is not None and ind.id not in seen:
                ranked.append(ind)
                seen.add(ind.id)
        return ranked[:top_k]

    def get_all(self) -> List[Individual]:
        return list(self.population)

    def size(self) -> int:
        return len(self.population)

    def get_view(self) -> List[Dict[str, Any]]:
        view = super().get_view()
        for item in view:
            item["recent_reflections_count"] = len(self.short_term_reflections)
            item["failure_history_count"] = len(self.failure_buffer)
        return view
