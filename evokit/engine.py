"""
Invariant Evolution Engine for EvoKit.
Coordinates Population Management, Mutation Strategies, Evaluator Services,
LLM Gateway, and Dual-Tier Experiment Storage across the evolutionary lifecycle.
Supports parallel multi-processing evaluation of candidate batches.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from evokit.core.base import BaseComponent
from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual, MetricRecord
from evokit.core.gateway import LLMGateway
from evokit.core.storage import ExperimentStorage
from evokit.benchmarks.base import BaseBenchmarkTask
from evokit.plugins.populations.base import BasePopulationManager
from evokit.plugins.mutations.base import BaseMutationStrategy
from evokit.plugins.evaluators.base import BaseEvaluator


class EngineConfig(BaseModel):
    max_generations: int = Field(default=5, description="Number of evolutionary iterations")
    candidates_per_generation: int = Field(default=2, description="Number of program variants synthesized per step")
    test_top_k: int = Field(default=1, description="Number of elite algorithms to test on hold-out test set")
    evaluate_test_split: bool = Field(default=True, description="Perform hold-out generalization evaluation at end")
    seed_evaluation: bool = Field(default=True, description="Evaluate baseline initial code as generation 0")


class EvolutionEngine(BaseComponent):
    """
    Central invariant runtime coordinator for EvoKit.
    Decoupled from specific algorithmic variants via 3-pillar interfaces.
    """

    def __init__(
        self,
        gateway: LLMGateway,
        storage: ExperimentStorage,
        benchmark: BaseBenchmarkTask,
        population: BasePopulationManager,
        mutation: BaseMutationStrategy,
        evaluator: BaseEvaluator,
        config: Optional[Dict[str, Any] | EngineConfig] = None,
    ) -> None:
        super().__init__()
        self.engine_cfg = config if isinstance(config, EngineConfig) else EngineConfig(**(config or {}))
        self.gateway = gateway
        self.storage = storage
        self.benchmark = benchmark
        self.population = population
        self.mutation = mutation
        self.evaluator = evaluator

    def _setup_internal(self) -> None:
        self.logger.info("Initializing EvoKit Engine components...")
        self.storage.setup()
        self.gateway.setup()
        self.benchmark.setup()
        self.population.setup()
        self.mutation.setup()
        self.evaluator.setup()

    def _teardown_internal(self) -> None:
        self.logger.info("Tearing down EvoKit Engine components...")
        self.evaluator.teardown()
        self.mutation.teardown()
        self.population.teardown()
        self.benchmark.teardown()
        self.gateway.teardown()
        self.storage.teardown()

    def run(self) -> Dict[str, Any]:
        """Execute full evolutionary lifecycle."""
        if not self._is_setup:
            self.setup()

        start_time = time.time()
        self.logger.info(f"Starting evolutionary search for task: {self.benchmark.name}")

        # Generation 0: Seed Baseline Evaluation
        if self.engine_cfg.seed_evaluation:
            self._evaluate_seed()

        # Generational Evolutionary Loop
        for gen in range(1, self.engine_cfg.max_generations + 1):
            gen_start = time.time()
            self.logger.info(f"── Generation {gen}/{self.engine_cfg.max_generations} ──")

            # 1. Construct Immutable EvolutionContext
            ctx = EvolutionContext(
                generation=gen,
                population_view=self.population.get_all(),
                task_spec=self.benchmark.get_context_spec(),
                runtime_config=self.engine_cfg.model_dump(),
                llm_gateway=self.gateway,
                metadata={"time_elapsed": time.time() - start_time},
            )

            # 2. Search & Mutation: Generate un-evaluated candidates
            candidates = self.mutation.mutate(ctx, num_candidates=self.engine_cfg.candidates_per_generation)

            # 3. Sandboxed Evaluation (Parallel Batch) & Storage
            records = self.evaluator.evaluate_batch(candidates, self.benchmark, split="train")

            valid_count = 0
            for cand, record in zip(candidates, records):
                cand.apply_evaluation(record)

                # Persist to Relational DB & Artifact Filesystem
                artifact_path = self.storage.record_individual(cand, generation=gen)
                cand.metadata["artifact_path"] = str(artifact_path)

                # Update Population / Archive
                admitted = self.population.add(cand)
                if cand.is_valid:
                    valid_count += 1
                    status_str = "Admitted to archive" if admitted else "Evaluated (not admitted)"
                    self.logger.info(
                        f"  [Gen {gen}] Ind {cand.id} | Fit: {cand.fitness:.4f} | Op: {cand.operator} | {status_str}"
                    )
                else:
                    self.logger.warning(
                        f"  [Gen {gen}] Ind {cand.id} | FAILED | Op: {cand.operator} | Error: {cand.error_trace}"
                    )

            # Generational Summary Telemetry
            elites = self.population.get_elites(top_k=1)
            best_fit = elites[0].fitness if elites and elites[0].fitness is not None else float("nan")
            gen_duration = time.time() - gen_start
            self.logger.info(
                f"Gen {gen} complete ({gen_duration:.2f}s) | Valid: {valid_count}/{len(candidates)} | Best Train Fitness: {best_fit}"
            )

        # 4. Out-of-Distribution Generalization Testing on Test Split
        test_results = []
        if self.engine_cfg.evaluate_test_split:
            self.logger.info("Evaluating top elite algorithms on held-out Test split...")
            elites = self.population.get_elites(top_k=self.engine_cfg.test_top_k)
            if elites:
                test_records = self.evaluator.evaluate_batch(elites, self.benchmark, split="test")
                for elite, test_rec in zip(elites, test_records):
                    test_results.append({
                        "individual_id": elite.id,
                        "train_fitness": elite.fitness,
                        "test_fitness": test_rec.fitness,
                        "test_metrics": test_rec.metrics,
                        "test_valid": test_rec.is_valid,
                    })
                    self.logger.info(
                        f"Elite {elite.id} -> Train Fitness: {elite.fitness} | Held-out Test Fitness: {test_rec.fitness}"
                    )

        total_runtime = time.time() - start_time
        summary = {
            "benchmark": self.benchmark.name,
            "direction": self.benchmark.direction,
            "total_generations": self.engine_cfg.max_generations,
            "total_runtime_seconds": total_runtime,
            "population_size": self.population.size(),
            "best_elites": [e.to_dict() for e in self.population.get_elites(top_k=3)],
            "test_generalization": test_results,
            "token_usage": self.gateway.get_token_usage(),
        }
        self.logger.info(f"Evolution complete in {total_runtime:.2f}s.")
        return summary

    def _evaluate_seed(self) -> None:
        """Evaluate baseline initial algorithm as generation 0."""
        seed_code = self.benchmark.prompt_spec.initial_code
        seed_ind = Individual(
            code=seed_code,
            thought="Initial baseline algorithm",
            operator="initial_baseline",
            generation=0,
        )
        record = self.evaluator.evaluate(seed_ind, self.benchmark, split="train")
        seed_ind.apply_evaluation(record)
        self.storage.record_individual(seed_ind, generation=0)
        self.population.add(seed_ind)
        self.logger.info(f"Seed evaluation complete: Fitness = {seed_ind.fitness}")
