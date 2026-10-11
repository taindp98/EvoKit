"""
Unit tests for EvoKit Benchmarks:
Online Bin Packing (OBP) and Constructive Traveling Salesperson (TSP).
"""

import unittest
from evokit.benchmarks.obp import OBPBenchmarkTask, OBPTaskConfig, INITIAL_OBP_CODE
from evokit.benchmarks.tsp import TSPBenchmarkTask, TSPTaskConfig, INITIAL_TSP_CODE


class TestBenchmarks(unittest.TestCase):
    def test_bpp_benchmark(self):
        cfg = OBPTaskConfig(
            num_train_instances=3,
            num_test_instances=2,
            train_items_per_instance=50,
            test_items_per_instance=50,
            seed=42,
        )
        task = OBPBenchmarkTask(cfg)
        task.setup()

        self.assertEqual(len(task.train_instances), 3)
        self.assertEqual(len(task.test_instances), 2)
        self.assertEqual(task.direction, "MINIMIZE")

        # Test initial baseline code
        record = task.evaluate_code(INITIAL_OBP_CODE, split="train")
        self.assertTrue(record.is_valid)
        self.assertIsNotNone(record.fitness)
        self.assertGreater(record.fitness, 0.0)  # Positive bin count (MINIMIZE)
        self.assertIn("avg_bins", record.metrics)
        self.assertIn("avg_l1_ratio", record.metrics)

        # Test evaluation on test split
        test_record = task.evaluate_code(INITIAL_OBP_CODE, split="test")
        self.assertTrue(test_record.is_valid)
        self.assertEqual(test_record.metrics["split"], "test")

        # Test invalid syntax
        bad_syntax = "def priority(item, bins): return unknown_var + 1"
        err_record = task.evaluate_code(bad_syntax, split="train")
        self.assertFalse(err_record.is_valid)
        self.assertIn("unknown_var", err_record.error_trace)

        # Context spec
        spec = task.get_context_spec()
        self.assertEqual(spec["benchmark_name"], "obp_online")
        self.assertEqual(spec["function_name"], "priority")
        self.assertEqual(spec["direction"], "MINIMIZE")

    def test_tsp_benchmark(self):
        cfg = TSPTaskConfig(
            num_train_instances=2,
            num_test_instances=2,
            train_cities=15,
            test_cities=20,
            seed=42,
        )
        task = TSPBenchmarkTask(cfg)
        task.setup()

        self.assertEqual(len(task.train_instances), 2)
        self.assertEqual(len(task.test_instances), 2)
        self.assertEqual(task.direction, "MINIMIZE")

        # Test initial baseline code (Nearest Neighbor)
        record = task.evaluate_code(INITIAL_TSP_CODE, split="train")
        self.assertTrue(record.is_valid)
        self.assertIsNotNone(record.fitness)
        self.assertGreater(record.fitness, 0.0)  # Positive tour length (MINIMIZE)
        self.assertIn("avg_tour_length", record.metrics)

        # Test invalid city selection (runtime error)
        bad_logic = """
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    return -999  # Invalid city
"""
        err_record = task.evaluate_code(bad_logic, split="train")
        self.assertFalse(err_record.is_valid)
        self.assertIsNotNone(err_record.error_trace)


if __name__ == "__main__":
    unittest.main()
