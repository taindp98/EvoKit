"""
Unit and integration tests for parallel multi-processing sandbox evaluation in EvoKit.
"""

import time
import unittest
from evokit.core.entity import Individual
from evokit.core.gateway import LLMGateway, GatewayConfig
from evokit.core.storage import ExperimentStorage, StorageConfig
from evokit.engine import EvolutionEngine, EngineConfig
from evokit.benchmarks.tsp import TSPBenchmarkTask
from evokit.plugins.evaluators.subprocess_sandbox import SubprocessSandboxEvaluator
from evokit.plugins.evaluators.mcp_evaluator import MCPEvaluator
from evokit.plugins.populations.lineage_archive import LineageGraphArchive
from evokit.plugins.mutations.diff import DiffMutationStrategy

VALID_TSP_CODE = """
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    return min(unvisited_nodes, key=lambda city: distance_matrix[current_node, city])
"""

SYNTAX_ERROR_CODE = """
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    this is invalid python syntax !!!
"""

INFINITE_LOOP_CODE = """
import time
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    while True:
        time.sleep(0.1)
"""


class TestParallelEvaluator(unittest.TestCase):
    def setUp(self):
        self.task = TSPBenchmarkTask({"num_train_instances": 4, "train_cities": 15})
        self.task.setup()

    def test_subprocess_sandbox_batch_parallel(self):
        evaluator = SubprocessSandboxEvaluator({"timeout_seconds": 10.0, "max_workers": 4})
        evaluator.setup()

        individuals = [
            Individual(code=VALID_TSP_CODE, thought=f"Candidate {i}")
            for i in range(4)
        ]

        t0 = time.time()
        records = evaluator.evaluate_batch(individuals, self.task, split="train")
        elapsed = time.time() - t0

        self.assertEqual(len(records), 4)
        for i, rec in enumerate(records):
            self.assertTrue(rec.is_valid, f"Record {i} should be valid")
            self.assertIsNotNone(rec.fitness)
            self.assertIn("avg_tour_length", rec.metrics)

    def test_subprocess_sandbox_batch_fault_tolerance(self):
        evaluator = SubprocessSandboxEvaluator({"timeout_seconds": 2.0, "max_workers": 3})
        evaluator.setup()

        individuals = [
            Individual(code=VALID_TSP_CODE, thought="Valid candidate"),
            Individual(code=SYNTAX_ERROR_CODE, thought="Syntax error candidate"),
            Individual(code=INFINITE_LOOP_CODE, thought="Infinite loop candidate"),
        ]

        records = evaluator.evaluate_batch(individuals, self.task, split="train")
        self.assertEqual(len(records), 3)

        # 1. Valid individual
        self.assertTrue(records[0].is_valid)
        self.assertIsNotNone(records[0].fitness)

        # 2. Syntax error individual
        self.assertFalse(records[1].is_valid)
        self.assertIn("Syntax", records[1].error_trace or records[1].stderr)

        # 3. Timeout individual
        self.assertFalse(records[2].is_valid)
        self.assertTrue("timed out" in records[2].error_trace.lower() or "timeoutexpired" in records[2].stderr.lower())

    def test_tsp_parallel_instance_workers(self):
        parallel_task = TSPBenchmarkTask({
            "num_train_instances": 6,
            "train_cities": 20,
            "parallel_workers": 3,
        })
        parallel_task.setup()

        rec = parallel_task.evaluate_code(VALID_TSP_CODE, split="train")
        self.assertTrue(rec.is_valid)
        self.assertIsNotNone(rec.fitness)
        self.assertIn("avg_tour_length", rec.metrics)

    def test_engine_parallel_generation_run(self):
        storage = ExperimentStorage(StorageConfig(db_path=":memory:", artifact_dir="./tmp_test_parallel_artifacts"))
        gateway = LLMGateway(GatewayConfig(provider="mock"))
        pop = LineageGraphArchive({"capacity": 10})
        mut = DiffMutationStrategy()
        evaluator = SubprocessSandboxEvaluator({"timeout_seconds": 5.0, "max_workers": 2})

        engine = EvolutionEngine(
            gateway=gateway,
            storage=storage,
            benchmark=self.task,
            population=pop,
            mutation=mut,
            evaluator=evaluator,
            config=EngineConfig(max_generations=2, candidates_per_generation=2, evaluate_test_split=True),
        )

        res = engine.run()
        self.assertEqual(res["total_generations"], 2)
        self.assertGreater(res["population_size"], 0)


if __name__ == "__main__":
    unittest.main()
