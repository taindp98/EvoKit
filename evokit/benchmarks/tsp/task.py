"""
Traveling Salesperson Problem (TSP) Constructive Benchmark Task Implementation.
Supports parallel instance-level simulation.
"""

from __future__ import annotations

import concurrent.futures
import copy
import time
import numpy as np
from typing import Any, Dict, List, Optional, Set
from pydantic import Field

from evokit.benchmarks.base import BaseBenchmarkTask, BenchmarkConfig, PromptSpec
from evokit.benchmarks.tsp.prompts import TSP_PROMPT_SPEC, INITIAL_TSP_CODE
from evokit.core.entity import MetricRecord


class TSPTaskConfig(BenchmarkConfig):
    train_cities: int = Field(default=30, description="Number of cities per training instance")
    test_cities: int = Field(default=50, description="Number of cities per test instance")


class TSPBenchmarkTask(BaseBenchmarkTask):
    """Constructive TSP Benchmark with 2D Euclidean coordinate instances."""

    def __init__(self, config: Optional[Dict[str, Any] | TSPTaskConfig] = None) -> None:
        if isinstance(config, TSPTaskConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = TSPTaskConfig(**config)
        else:
            cfg = TSPTaskConfig()
        super().__init__(cfg)
        self.tsp_cfg: TSPTaskConfig = cfg

    @property
    def name(self) -> str:
        return "tsp_construct"

    @property
    def direction(self) -> str:
        return "MINIMIZE"  # Minimizing tour_length

    @property
    def prompt_spec(self) -> PromptSpec:
        return TSP_PROMPT_SPEC

    def _generate_instances(self) -> None:
        rng = np.random.default_rng(self.tsp_cfg.seed)

        self.train_instances = []
        for i in range(self.tsp_cfg.num_train_instances):
            coords = rng.uniform(0.0, 1.0, size=(self.tsp_cfg.train_cities, 2))
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
            self.train_instances.append({
                "id": f"train_tsp_{i}",
                "coords": coords,
                "dist_matrix": dist_mat,
                "num_cities": self.tsp_cfg.train_cities,
            })

        self.test_instances = []
        for i in range(self.tsp_cfg.num_test_instances):
            coords = rng.uniform(0.0, 1.0, size=(self.tsp_cfg.test_cities, 2))
            diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            dist_mat = np.sqrt(np.sum(diff**2, axis=-1))
            self.test_instances.append({
                "id": f"test_tsp_{i}",
                "coords": coords,
                "dist_matrix": dist_mat,
                "num_cities": self.tsp_cfg.test_cities,
            })

    def simulate(self, select_fn, instance: Dict[str, Any]) -> float:
        """Construct a tour from node 0 using select_fn and calculate total Euclidean length."""
        n = instance["num_cities"]
        dist_mat = instance["dist_matrix"]

        start_node = 0
        tour = [start_node]
        unvisited: Set[int] = set(range(n))
        unvisited.remove(start_node)

        for _ in range(n - 1):
            next_node = select_fn(
                current_node=tour[-1],
                destination_node=start_node,
                unvisited_nodes=copy.copy(unvisited),
                distance_matrix=dist_mat.copy(),
            )
            if next_node not in unvisited:
                raise ValueError(f"Selected city {next_node} is already visited or invalid.")
            tour.append(next_node)
            unvisited.remove(next_node)

        # Tour length including return to start
        total_len = 0.0
        for i in range(n):
            c1 = tour[i]
            c2 = tour[(i + 1) % n]
            total_len += float(dist_mat[c1, c2])

        return total_len

    def evaluate_code(self, code_str: str, split: str = "train", timeout: float = 30.0) -> MetricRecord:
        if not self._is_setup:
            self.setup()

        instances = self.train_instances if split == "train" else self.test_instances
        start_t = time.time()

        local_env: Dict[str, Any] = {"np": np, "numpy": np}
        try:
            exec(code_str, local_env, local_env)
        except Exception as e:
            return MetricRecord(
                fitness=None,
                is_valid=False,
                error_trace=f"Compilation/Syntax Error: {e}",
                runtime_seconds=time.time() - start_t,
            )

        fn = local_env.get("select_next_node")
        if not callable(fn):
            return MetricRecord(
                fitness=None,
                is_valid=False,
                error_trace="Function 'select_next_node' not defined in candidate code.",
                runtime_seconds=time.time() - start_t,
            )

        tour_lengths = []
        try:
            p_workers = max(1, int(self.tsp_cfg.parallel_workers))
            if p_workers > 1 and len(instances) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(p_workers, len(instances))) as pool:
                    tour_lengths = list(pool.map(lambda inst: self.simulate(fn, inst), instances))
            else:
                for inst in instances:
                    l = self.simulate(fn, inst)
                    tour_lengths.append(l)
        except Exception as e:
            return MetricRecord(
                fitness=None,
                is_valid=False,
                error_trace=f"Runtime simulation exception: {e}",
                runtime_seconds=time.time() - start_t,
            )

        avg_length = float(np.mean(tour_lengths))
        fitness = avg_length  # Direct positive tour length (MINIMIZE)

        return MetricRecord(
            fitness=fitness,
            is_valid=True,
            metrics={
                "avg_tour_length": avg_length,
                "min_tour_length": float(np.min(tour_lengths)),
                "max_tour_length": float(np.max(tour_lengths)),
                "split": split,
            },
            runtime_seconds=time.time() - start_t,
        )
