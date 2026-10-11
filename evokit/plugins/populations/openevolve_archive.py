"""
OpenEvolve Archive (Pillar 1) for EvoKit.

A faithful port of OpenEvolve's program database
(references/openevolve/openevolve/database.py, and the self-contained
reimplementation in references/myirace/src/reprod/openevolve_tsp_gls.py):

  - **Island model**: num_islands sub-populations with periodic ring migration.
  - **Per-island MAP-Elites feature grid**: programs map to (complexity,
    diversity) cells; a cell is replaced only by a fitter occupant, preserving
    behavioural diversity.
  - **Archive**: a separate top-K-by-fitness set used for exploitation sampling.
  - **Mixed-strategy parent selection**: exploration (fitness-weighted from the
    island) / exploitation (fitness-weighted from the archive) / random.
  - **Novelty rejection**: near-duplicate code (line-level SequenceMatcher
    similarity >= 1 - novelty_threshold) is rejected.
  - **Worst-first eviction**: MAP-Elites cell owners are protected; non-owners
    are evicted first when the store exceeds max_db_size.

Direction handling: OpenEvolve maximises fitness. EvoKit benchmarks may MINIMIZE
(e.g. OBP mean bins), so an internal ``_elite_fitness`` normalises every
candidate to "higher = better" (negating MINIMIZE fitness), and all of the
reference's comparisons run on that normalised value.
"""

from __future__ import annotations

import math
import random
from difflib import SequenceMatcher
from threading import Lock
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from pydantic import Field

from evokit.core.entity import Individual
from evokit.plugins.populations.base import BasePopulationManager, PopulationConfig


class OpenEvolveArchiveConfig(PopulationConfig):
    num_islands: int = Field(default=5, description="Number of island sub-populations")
    archive_size: int = Field(default=100, description="Top-K archive size for exploitation sampling")
    feature_bins: int = Field(default=10, description="MAP-Elites bins per feature dimension")
    exploration_ratio: float = Field(default=0.2, description="P(sample fitness-weighted from the island)")
    exploitation_ratio: float = Field(default=0.7, description="P(sample fitness-weighted from the archive)")
    elite_selection_ratio: float = Field(default=0.1, description="Fraction used when picking top-K inspirations")
    migration_interval: int = Field(default=10, description="Generations between ring migrations")
    migration_rate: float = Field(default=0.1, description="Fraction of each island migrated")
    novelty_threshold: float = Field(default=0.02, description="Min edit-distance ratio to accept (0 disables)")
    max_db_size: int = Field(default=1000, description="Max programs retained before worst-first eviction")


class OpenEvolveArchive(BasePopulationManager):
    """OpenEvolve program database: islands + per-island MAP-Elites + archive."""

    def __init__(self, config: Optional[Dict[str, Any] | OpenEvolveArchiveConfig] = None) -> None:
        if isinstance(config, OpenEvolveArchiveConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = OpenEvolveArchiveConfig(**config)
        else:
            cfg = OpenEvolveArchiveConfig()
        super().__init__(cfg)
        self.oe_cfg: OpenEvolveArchiveConfig = cfg

        self._programs: Dict[str, Individual] = {}
        self.islands: List[Set[str]] = [set() for _ in range(cfg.num_islands)]
        self.island_feature_maps: List[Dict[str, str]] = [{} for _ in range(cfg.num_islands)]
        self._archive: Set[str] = set()
        self._best_id: Optional[str] = None
        self._lock = Lock()

        self._feature_stats: Dict[str, Dict[str, Any]] = {}
        self._diversity_refs: List[str] = []
        self._diversity_ref_size = 20
        self._rr_island = 0          # round-robin island assignment counter
        self._generation = 0

    # ---- direction-normalised fitness ("higher is better") ----------------
    def _elite_fitness(self, ind: Individual) -> float:
        if ind.fitness is None or not math.isfinite(ind.fitness):
            return float("-inf")
        return ind.fitness if self.is_maximize else -ind.fitness

    # ---- accessors --------------------------------------------------------
    def get_all(self) -> List[Individual]:
        return list(self._programs.values())

    def size(self) -> int:
        return len(self._programs)

    @property
    def cell_owner_ids(self) -> Set[str]:
        ids: Set[str] = set()
        for fm in self.island_feature_maps:
            ids.update(fm.values())
        return ids

    def get_best(self) -> Optional[Individual]:
        if self._best_id and self._best_id in self._programs:
            return self._programs[self._best_id]
        if not self._programs:
            return None
        return max(self._programs.values(), key=self._elite_fitness)

    def get_elites(self, top_k: int = 1) -> List[Individual]:
        return self.get_top(top_k)

    def get_top(self, k: int) -> List[Individual]:
        valid = [c for c in self._programs.values() if math.isfinite(self._elite_fitness(c))]
        valid.sort(key=self._elite_fitness, reverse=True)
        return valid[:k]

    def coverage(self) -> float:
        max_cells = (self.oe_cfg.feature_bins ** 2) * self.oe_cfg.num_islands
        occupied = sum(len(fm) for fm in self.island_feature_maps)
        return occupied / max_cells if max_cells else 0.0

    # ---- MAP-Elites feature computation -----------------------------------
    def _update_feature_stats(self, dim: str, value: float) -> None:
        stats = self._feature_stats.setdefault(dim, {"min": value, "max": value, "n": 0})
        stats["min"] = min(stats["min"], value)
        stats["max"] = max(stats["max"], value)
        stats["n"] += 1

    def _scale_feature_value(self, dim: str, value: float) -> float:
        stats = self._feature_stats.get(dim)
        if not stats:
            return 0.5
        rng = stats["max"] - stats["min"]
        if rng <= 0:
            return 0.5
        return max(0.0, min(1.0, (value - stats["min"]) / rng))

    def _compute_diversity(self, ind: Individual) -> float:
        if not self._diversity_refs:
            return 0.5
        cand_lines = ind.code.strip().splitlines()
        dists = [1.0 - SequenceMatcher(None, cand_lines, ref.strip().splitlines()).ratio()
                 for ref in self._diversity_refs]
        return float(np.mean(dists)) if dists else 0.5

    def _update_diversity_refs(self, ind: Individual) -> None:
        if len(self._diversity_refs) < self._diversity_ref_size:
            self._diversity_refs.append(ind.code)
        elif random.random() < 0.1:
            self._diversity_refs[random.randint(0, self._diversity_ref_size - 1)] = ind.code

    def _feature_coords(self, ind: Individual) -> List[int]:
        coords = []
        for dim in ("complexity", "diversity"):
            val = float(len(ind.code)) if dim == "complexity" else self._compute_diversity(ind)
            self._update_feature_stats(dim, val)
            bin_idx = int(self._scale_feature_value(dim, val) * self.oe_cfg.feature_bins)
            coords.append(max(0, min(self.oe_cfg.feature_bins - 1, bin_idx)))
        return coords

    # ---- novelty ----------------------------------------------------------
    def _is_novel(self, ind: Individual, island_idx: int) -> bool:
        if self.oe_cfg.novelty_threshold <= 0:
            return True
        min_sim = 1.0 - self.oe_cfg.novelty_threshold
        cand_lines = ind.code.strip().splitlines()
        for pid in self.islands[island_idx]:
            other = self._programs.get(pid)
            if other is None:
                continue
            if SequenceMatcher(None, cand_lines, other.code.strip().splitlines()).ratio() >= min_sim:
                return False
        return True

    # ---- add --------------------------------------------------------------
    def add(self, individual: Individual) -> bool:
        with self._lock:
            # Island assignment: inherit parent's island, else round-robin.
            island_idx = None
            for pid in individual.parent_ids:
                if pid in self._programs:
                    island_idx = self._programs[pid].island_idx  # type: ignore[attr-defined]
                    break
            if island_idx is None:
                island_idx = self._rr_island % self.oe_cfg.num_islands
                self._rr_island += 1
            individual.island_idx = island_idx  # stash on the Individual (extra='allow')

            if individual.fitness is None or not individual.is_valid:
                return False
            if not self._is_novel(individual, island_idx):
                return False

            self._programs[individual.id] = individual
            self.islands[island_idx].add(individual.id)

            coords = self._feature_coords(individual)
            key = str(coords)
            fm = self.island_feature_maps[island_idx]
            should_replace = key not in fm
            if not should_replace:
                existing_id = fm[key]
                should_replace = (existing_id not in self._programs or
                                  self._elite_fitness(individual) > self._elite_fitness(self._programs[existing_id]))
            if should_replace:
                old_id = fm.get(key)
                fm[key] = individual.id
                if old_id and old_id != individual.id and old_id in self._programs:
                    self.islands[island_idx].discard(old_id)
                    if old_id in self._archive:
                        self._archive.discard(old_id)
                        self._archive.add(individual.id)
                    self._remove_if_orphaned(old_id)

            if self._best_id is None or self._elite_fitness(individual) > self._elite_fitness(
                self._programs.get(self._best_id, individual)
            ):
                self._best_id = individual.id

            self._update_archive(individual)
            self._update_diversity_refs(individual)
            self._enforce_population_limit(exclude_id=individual.id)
            return should_replace

    def _remove_if_orphaned(self, program_id: str) -> None:
        if program_id == self._best_id:
            return
        for fm in self.island_feature_maps:
            if program_id in fm.values():
                return
        self._programs.pop(program_id, None)
        for isl in self.islands:
            isl.discard(program_id)
        self._archive.discard(program_id)

    def _update_archive(self, ind: Individual) -> None:
        if len(self._archive) < self.oe_cfg.archive_size:
            self._archive.add(ind.id)
            return
        self._archive = {pid for pid in self._archive if pid in self._programs}
        if len(self._archive) < self.oe_cfg.archive_size:
            self._archive.add(ind.id)
            return
        progs = [self._programs[pid] for pid in self._archive if pid in self._programs]
        if not progs:
            self._archive.add(ind.id)
            return
        worst = min(progs, key=self._elite_fitness)
        if self._elite_fitness(ind) > self._elite_fitness(worst):
            self._archive.discard(worst.id)
            self._archive.add(ind.id)

    def _enforce_population_limit(self, exclude_id: Optional[str] = None) -> None:
        if len(self._programs) <= self.oe_cfg.max_db_size:
            return
        num_to_remove = len(self._programs) - self.oe_cfg.max_db_size
        cell_owners = self.cell_owner_ids
        protected = {self._best_id, exclude_id} - {None}
        progs = list(self._programs.values())
        non_elite = sorted([p for p in progs if p.id not in cell_owners and p.id not in protected],
                           key=self._elite_fitness)
        elite = sorted([p for p in progs if p.id in cell_owners and p.id not in protected],
                       key=self._elite_fitness)
        to_remove = non_elite[:num_to_remove]
        if len(to_remove) < num_to_remove:
            to_remove.extend(elite[: num_to_remove - len(to_remove)])
        for prog in to_remove:
            self._programs.pop(prog.id, None)
            for isl in self.islands:
                isl.discard(prog.id)
            for fm in self.island_feature_maps:
                for k in [k for k, v in fm.items() if v == prog.id]:
                    del fm[k]
            self._archive.discard(prog.id)

    # ---- mixed-strategy parent selection ----------------------------------
    def sample_parents(self, k: int = 1) -> List[Individual]:
        if not self._programs:
            return []
        return [self._sample_one() for _ in range(k)]

    def _sample_one(self) -> Individual:
        # Scope to a round-robin island (mirrors offspring island assignment).
        island_idx = self._rr_island % self.oe_cfg.num_islands
        pool = [self._programs[pid] for pid in self.islands[island_idx] if pid in self._programs]
        if not pool:
            pool = list(self._programs.values())
        r = random.random()
        if r < self.oe_cfg.exploration_ratio:
            return self._weighted_sample(pool)
        if r < self.oe_cfg.exploration_ratio + self.oe_cfg.exploitation_ratio:
            return self._sample_exploitation(pool)
        return random.choice(pool)

    def _weighted_sample(self, pool: List[Individual]) -> Individual:
        if len(pool) == 1:
            return pool[0]
        weights = [max(self._elite_fitness(p) if math.isfinite(self._elite_fitness(p)) else 0.0, 0.001)
                   for p in pool]
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1.0 / len(pool)] * len(pool)
        return random.choices(pool, weights=weights, k=1)[0]

    def _sample_exploitation(self, fallback_pool: List[Individual]) -> Individual:
        progs = [self._programs[pid] for pid in self._archive if pid in self._programs]
        return self._weighted_sample(progs) if progs else self._weighted_sample(fallback_pool)

    def sample_inspirations(self, parent: Individual, n: int = 2) -> List[Individual]:
        inspirations: List[Individual] = []
        best = self.get_best()
        if best and best.id != parent.id:
            inspirations.append(best)
        top_n = max(1, int(n * self.oe_cfg.elite_selection_ratio))
        for prog in self.get_top(top_n):
            if prog.id != parent.id and prog.id not in {i.id for i in inspirations}:
                inspirations.append(prog)
        if len(inspirations) < n:
            exclude = {parent.id} | {i.id for i in inspirations}
            cands = [p for p in self._programs.values()
                     if p.id not in exclude and math.isfinite(self._elite_fitness(p))]
            if cands:
                inspirations.extend(random.sample(cands, min(n - len(inspirations), len(cands))))
        return inspirations[:n]

    # ---- ring migration ---------------------------------------------------
    def on_generation_end(self, generation: int) -> None:
        """Engine hook: trigger periodic ring migration between islands."""
        self.maybe_migrate(generation)

    def maybe_migrate(self, generation: int) -> None:
        self._generation = generation
        if self.oe_cfg.num_islands < 2 or self.oe_cfg.migration_interval <= 0:
            return
        if generation == 0 or generation % self.oe_cfg.migration_interval != 0:
            return
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            for i in range(self.oe_cfg.num_islands):
                pids = [pid for pid in self.islands[i] if pid in self._programs]
                if not pids:
                    continue
                progs = sorted((self._programs[pid] for pid in pids),
                               key=self._elite_fitness, reverse=True)
                num = max(1, int(len(progs) * self.oe_cfg.migration_rate))
                targets = [(i + 1) % self.oe_cfg.num_islands, (i - 1) % self.oe_cfg.num_islands]
                for migrant in progs[:num]:
                    if migrant.metadata.get("_migrant"):
                        continue
                    for t in targets:
                        self.islands[t].add(migrant.id)
                    migrant.metadata["_migrant"] = True
