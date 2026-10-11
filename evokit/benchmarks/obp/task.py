"""
Online Bin Packing (OBP) Benchmark Task Implementation.
Supports parallel instance-level simulation and exact Weibull dataset generation.
"""

from __future__ import annotations

import concurrent.futures
import time
import numpy as np
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.benchmarks.base import BaseBenchmarkTask, BenchmarkConfig, PromptSpec
from evokit.benchmarks.obp.prompts import OBP_PROMPT_SPEC, INITIAL_OBP_CODE
from evokit.core.entity import MetricRecord


class OBPTaskConfig(BenchmarkConfig):
    capacity: float = Field(default=100.0, description="Bin capacity")
    train_items_per_instance: int = Field(default=200, description="Number of items per training instance")
    test_items_per_instance: int = Field(default=500, description="Number of items per test instance")
    dataset_type: str = Field(default="mixed", description="Distribution type: 'mixed' or 'weibull_llm4ad'")


class OBPBenchmarkTask(BaseBenchmarkTask):
    """Online Bin Packing Benchmark with Train/Test instance segregation."""

    def __init__(self, config: Optional[Dict[str, Any] | OBPTaskConfig] = None) -> None:
        if isinstance(config, OBPTaskConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = OBPTaskConfig(**config)
        else:
            cfg = OBPTaskConfig()
        super().__init__(cfg)
        self.bpp_cfg: OBPTaskConfig = cfg

    @property
    def name(self) -> str:
        return "obp_online"

    @property
    def direction(self) -> str:
        return "MINIMIZE"  # Minimizing num_bins

    @property
    def prompt_spec(self) -> PromptSpec:
        return OBP_PROMPT_SPEC

    def reference_bound(self, split: str = "train") -> Optional[float]:
        """Mean L1 lower bound (ceil(sum(items)/capacity)) over the split's
        instances — the theoretical minimum bin count. Used by the engine to
        report the optimality gap (score - lb) / lb."""
        instances = self.train_instances if split == "train" else self.test_instances
        if not instances:
            return None
        return float(np.mean([inst["l1_bound"] for inst in instances]))

    def _generate_instances(self) -> None:
        rng = np.random.default_rng(self.bpp_cfg.seed)
        cap = self.bpp_cfg.capacity
        llm4ad_seed = 2024 if (self.bpp_cfg.seed is None or self.bpp_cfg.seed == 0) else self.bpp_cfg.seed
        weibull_train_rng = np.random.RandomState(llm4ad_seed)
        weibull_test_rng = np.random.RandomState(llm4ad_seed + 1000)

        self.train_instances = []
        for i in range(self.bpp_cfg.num_train_instances):
            if self.bpp_cfg.dataset_type == "weibull_llm4ad":
                # Matches LLM4AD generate_weibull_dataset bit-for-bit: Weibull(3) * 45 * (cap/100)
                samples = weibull_train_rng.weibull(3, self.bpp_cfg.train_items_per_instance) * 45.0 * (cap / 100.0)
                samples = np.clip(samples, 1.0, cap)
                items = np.round(samples).astype(int)
            elif i % 2 == 0:
                items = rng.uniform(10.0, 50.0, size=self.bpp_cfg.train_items_per_instance)
                items = np.clip(items, 1.0, cap - 1.0)
            else:
                items = rng.weibull(1.5, size=self.bpp_cfg.train_items_per_instance) * 30.0 + 5.0
                items = np.clip(items, 1.0, cap - 1.0)

            self.train_instances.append({
                "id": f"train_{i}",
                "capacity": cap,
                "items": items,
                "l1_bound": float(np.ceil(np.sum(items) / cap)),
            })

        self.test_instances = []
        for i in range(self.bpp_cfg.num_test_instances):
            if self.bpp_cfg.dataset_type == "weibull_llm4ad":
                samples = weibull_test_rng.weibull(3, self.bpp_cfg.test_items_per_instance) * 45.0 * (cap / 100.0)
                samples = np.clip(samples, 1.0, cap)
                items = np.round(samples).astype(int)
            elif i % 2 == 0:
                items = rng.uniform(5.0, 70.0, size=self.bpp_cfg.test_items_per_instance)
                items = np.clip(items, 1.0, cap - 1.0)
            else:
                items = rng.exponential(scale=25.0, size=self.bpp_cfg.test_items_per_instance) + 2.0
                items = np.clip(items, 1.0, cap - 1.0)

            self.test_instances.append({
                "id": f"test_{i}",
                "capacity": cap,
                "items": items,
                "l1_bound": float(np.ceil(np.sum(items) / cap)),
            })

    def simulate(self, priority_fn, instance: Dict[str, Any]) -> int:
        """Simulate packing for a single instance using the priority heuristic."""
        cap = instance["capacity"]
        items = instance["items"]
        bins: List[float] = []

        for item in items:
            valid_indices = [idx for idx, rem in enumerate(bins) if rem >= item]
            if valid_indices:
                candidate_rem = np.array([bins[idx] for idx in valid_indices], dtype=float)
                priorities = priority_fn(float(item), candidate_rem)
                priorities = np.asarray(priorities, dtype=float).flatten()
                if len(priorities) != len(valid_indices):
                    raise ValueError(
                        f"Expected {len(valid_indices)} priority values, but got {len(priorities)}"
                    )
                best_choice = valid_indices[int(np.argmax(priorities))]
                bins[best_choice] -= item
            else:
                bins.append(cap - item)

        return len(bins)

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

        fn = local_env.get("priority")
        if not callable(fn):
            return MetricRecord(
                fitness=None,
                is_valid=False,
                error_trace="Function 'priority' not defined in candidate code.",
                runtime_seconds=time.time() - start_t,
            )

        used_bins_list = []
        l1_ratios = []
        try:
            p_workers = max(1, int(self.bpp_cfg.parallel_workers))
            if p_workers > 1 and len(instances) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(p_workers, len(instances))) as pool:
                    used_bins_list = list(pool.map(lambda inst: self.simulate(fn, inst), instances))
                    l1_ratios = [inst["l1_bound"] / max(1, n_bins) for inst, n_bins in zip(instances, used_bins_list)]
            else:
                for inst in instances:
                    n_bins = self.simulate(fn, inst)
                    used_bins_list.append(n_bins)
                    l1_ratios.append(inst["l1_bound"] / max(1, n_bins))
        except Exception as e:
            return MetricRecord(
                fitness=None,
                is_valid=False,
                error_trace=f"Runtime evaluation exception: {e}",
                runtime_seconds=time.time() - start_t,
            )

        avg_bins = float(np.mean(used_bins_list))
        avg_l1_ratio = float(np.mean(l1_ratios))
        fitness = avg_bins  # Direct positive bin count (MINIMIZE)

        return MetricRecord(
            fitness=fitness,
            is_valid=True,
            metrics={
                "avg_bins": avg_bins,
                "min_bins": int(np.min(used_bins_list)),
                "max_bins": int(np.max(used_bins_list)),
                "avg_l1_ratio": avg_l1_ratio,
                "split": split,
            },
            runtime_seconds=time.time() - start_t,
        )
