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
from evokit.core.tb_logger import TensorBoardLogger
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
    initial_population_size: Optional[int] = Field(default=None, description="If set, LLM-sample this many candidates as generation 0 (EoH i1 cold-start) instead of the single baseline seed")
    tb_log_dir: Optional[str] = Field(default=None, description="Directory for TensorBoard telemetry logs (disabled if None)")
    experiment_id: Optional[str] = Field(default=None, description="Unique run identifier (timestamp_study_seed_hash)")
    seed: Optional[int] = Field(default=None, description="Random seed for this run (for provenance)")
    parent_selection_num: Optional[int] = Field(default=None, description="If set, draw this many parents via population.sample_parents() as the EoH selection pool (paper 'p'/selection_num); else expose full archive")
    run_metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary run provenance surfaced to TensorBoard")


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

        # Progress counters for TensorBoard x-axes.
        self._cum_function_evaluations = 0
        self._best_objective: Optional[float] = None
        self._tb: Optional[TensorBoardLogger] = None
        # Per-generation best-so-far trajectory (reference-reproduction compatible).
        self.trajectory: List[Dict[str, Any]] = []

    def _is_better(self, candidate: float, incumbent: Optional[float]) -> bool:
        if incumbent is None:
            return True
        maximize = str(self.benchmark.direction).upper() == "MAXIMIZE"
        return candidate > incumbent if maximize else candidate < incumbent

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

        self._tb = TensorBoardLogger(
            self.engine_cfg.tb_log_dir, experiment_id=self.engine_cfg.experiment_id
        )
        self._log_run_metadata()
        n_train = max(1, len(self.benchmark.train_instances))

        # Generation 0: initial population.
        if self.engine_cfg.initial_population_size:
            n0 = self._sample_initial_population(self.engine_cfg.initial_population_size)
            self._cum_function_evaluations += n0 * n_train
            self._emit_telemetry(generation=0, start_time=start_time)
        elif self.engine_cfg.seed_evaluation:
            self._evaluate_seed()
            self._cum_function_evaluations += n_train
            self._emit_telemetry(generation=0, start_time=start_time)

        # Generational Evolutionary Loop
        for gen in range(1, self.engine_cfg.max_generations + 1):
            gen_start = time.time()
            self.logger.info(f"── Generation {gen}/{self.engine_cfg.max_generations} ──")

            # 1. Construct Immutable EvolutionContext.
            # If parent_selection_num is set, draw a tournament-selected pool of
            # that size (EoH's 'p'/selection_num); operators then sample their
            # 1-2 parents from this pool. Otherwise expose the full archive.
            if self.engine_cfg.parent_selection_num:
                parent_view = self.population.sample_parents(k=self.engine_cfg.parent_selection_num)
            else:
                parent_view = self.population.get_all()
            ctx = EvolutionContext(
                generation=gen,
                population_view=parent_view,
                task_spec=self.benchmark.get_context_spec(),
                runtime_config=self.engine_cfg.model_dump(),
                llm_gateway=self.gateway,
                metadata={"time_elapsed": time.time() - start_time},
            )

            # 2. Search & Mutation: Generate un-evaluated candidates
            candidates = self.mutation.mutate(ctx, num_candidates=self.engine_cfg.candidates_per_generation)

            # 3. Sandboxed Evaluation (Parallel Batch) & Storage
            records = self.evaluator.evaluate_batch(candidates, self.benchmark, split="train")
            self._cum_function_evaluations += len(candidates) * len(self.benchmark.train_instances)

            valid_count = 0
            for cand, record in zip(candidates, records):
                cand.apply_evaluation(record)
                if cand.is_valid and cand.fitness is not None and self._is_better(
                    cand.fitness, self._best_objective
                ):
                    self._best_objective = cand.fitness

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

            self._emit_telemetry(generation=gen, start_time=start_time)

            # Optional end-of-generation population hook (e.g. island migration).
            self.population.on_generation_end(gen)

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
            "trajectory": list(self.trajectory),
            "experiment_id": self.engine_cfg.experiment_id,
            "seed": self.engine_cfg.seed,
            "total_function_evaluations": self._cum_function_evaluations,
        }
        if self._tb is not None:
            self._tb.close()
        self.logger.info(f"Evolution complete in {total_runtime:.2f}s.")
        return summary

    def _log_run_metadata(self) -> None:
        """Surface run provenance (id, seed, timestamp, settings, log paths) to
        TensorBoard's TEXT tab so the run is self-describing for debugging."""
        if self._tb is None or not self._tb.enabled:
            return
        import datetime as _dt

        gw = getattr(self.gateway, "cfg", None)
        metadata: Dict[str, Any] = {
            "experiment_id": self.engine_cfg.experiment_id or "(unset)",
            "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
            "seed": self.engine_cfg.seed if self.engine_cfg.seed is not None else "(unset)",
            "benchmark": self.benchmark.name,
            "direction": self.benchmark.direction,
            "max_generations": self.engine_cfg.max_generations,
            "candidates_per_generation": self.engine_cfg.candidates_per_generation,
            "train_instances": len(self.benchmark.train_instances),
            "test_instances": len(self.benchmark.test_instances),
            "llm_provider": getattr(gw, "provider", "?"),
            "llm_model": getattr(gw, "model", "?"),
            "llm_base_url": getattr(gw, "base_url", "?"),
            "temperature": getattr(gw, "temperature", "?"),
            "mutation_strategy": type(self.mutation).__name__,
            "population": type(self.population).__name__,
            "tb_log_dir": self._tb.exp_dir,
        }
        # Any extra provenance threaded in from the CLI (config path, terminal.txt, etc.).
        metadata.update(self.engine_cfg.run_metadata or {})
        self._tb.log_metadata(metadata)

    def _emit_telemetry(self, *, generation: int, start_time: float) -> None:
        """Record one best-so-far telemetry point (trajectory + TensorBoard)."""
        if self._best_objective is None:
            return
        usage = self.gateway.get_token_usage()
        llm_calls = int(usage.get("request_count", 0))

        # Generic trajectory point. The optimality `gap` is only added when the
        # benchmark exposes a reference bound via its optional contract method;
        # benchmarks without one simply omit it (no engine-side assumptions).
        point: Dict[str, Any] = {
            "gen_id": generation,
            "score": float(self._best_objective),
            "n_instances": len(self.benchmark.train_instances),
            "used_budget": self._cum_function_evaluations,
            "llm_calls": llm_calls,
            "wall_time_seconds": time.time() - start_time,
        }
        try:
            ref = self.benchmark.reference_bound(split="train")
        except Exception:
            ref = None
        if ref:
            point["gap"] = (self._best_objective - ref) / ref
            point["reference_bound"] = ref
        self.trajectory.append(point)

        if self._tb is None or not self._tb.enabled:
            return
        self._tb.log_generation(
            generation=generation,
            best_objective=float(self._best_objective),
            llm_calls=llm_calls,
            function_evaluations=self._cum_function_evaluations,
            wall_time_seconds=time.time() - start_time,
            extra_scalars={"cumulative_tokens": float(usage.get("total_tokens", 0))},
        )

    def _sample_initial_population(self, size: int) -> int:
        """Generation 0: LLM-sample `size` candidates via the mutation strategy's
        cold-start (empty population -> EoH i1), evaluate, store, and seed the
        archive. Mirrors the reference reproduction's initial population.
        Returns the number of candidates sampled."""
        ctx = EvolutionContext(
            generation=0,
            population_view=[],
            task_spec=self.benchmark.get_context_spec(),
            runtime_config=self.engine_cfg.model_dump(),
            llm_gateway=self.gateway,
            metadata={},
        )
        candidates = self.mutation.mutate(ctx, num_candidates=size)
        records = self.evaluator.evaluate_batch(candidates, self.benchmark, split="train")
        valid = 0
        for cand, record in zip(candidates, records):
            cand.apply_evaluation(record)
            if cand.is_valid and cand.fitness is not None and self._is_better(
                cand.fitness, self._best_objective
            ):
                self._best_objective = cand.fitness
            artifact_path = self.storage.record_individual(cand, generation=0)
            cand.metadata["artifact_path"] = str(artifact_path)
            self.population.add(cand)
            if cand.is_valid:
                valid += 1
        self.logger.info(
            f"Initial population (gen 0): sampled {len(candidates)} | Valid: {valid}/{len(candidates)} "
            f"| Best: {self._best_objective}"
        )
        return len(candidates)

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
        if seed_ind.is_valid and seed_ind.fitness is not None and self._is_better(
            seed_ind.fitness, self._best_objective
        ):
            self._best_objective = seed_ind.fitness
        self.logger.info(f"Seed evaluation complete: Fitness = {seed_ind.fitness}")
