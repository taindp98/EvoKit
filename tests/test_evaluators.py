"""
Unit tests for EvoKit Evaluator Plugins:
SubprocessSandboxEvaluator and MCPEvaluator.
"""

import unittest
from evokit.benchmarks.bpp import BPPBenchmarkTask, BPPTaskConfig, INITIAL_BPP_CODE
from evokit.core.entity import Individual
from evokit.plugins.evaluators.subprocess_sandbox import SubprocessSandboxEvaluator
from evokit.plugins.evaluators.base import EvaluatorConfig
from evokit.plugins.evaluators.mcp_evaluator import MCPEvaluator


class TestEvaluators(unittest.TestCase):
    def setUp(self):
        self.task = BPPBenchmarkTask(BPPTaskConfig(
            num_train_instances=2,
            train_items_per_instance=20,
            seed=1,
        ))
        self.task.setup()

    def test_subprocess_sandbox_success(self):
        evaluator = SubprocessSandboxEvaluator(EvaluatorConfig(timeout_seconds=15.0, memory_limit_mb=1024))
        ind = Individual(code=INITIAL_BPP_CODE)

        record = evaluator.evaluate(ind, self.task, split="train")
        self.assertTrue(record.is_valid)
        self.assertIsNotNone(record.fitness)
        self.assertIn("avg_bins", record.metrics)
        self.assertGreater(record.runtime_seconds, 0.0)

    def test_subprocess_sandbox_syntax_error(self):
        evaluator = SubprocessSandboxEvaluator(EvaluatorConfig(timeout_seconds=15.0))
        ind = Individual(code="def broken_func(): syntax error here !!!")

        record = evaluator.evaluate(ind, self.task, split="train")
        self.assertFalse(record.is_valid)
        self.assertIsNone(record.fitness)
        self.assertIsNotNone(record.error_trace)

    def test_subprocess_sandbox_timeout(self):
        evaluator = SubprocessSandboxEvaluator(EvaluatorConfig(timeout_seconds=2.0))
        infinite_loop_code = """
import time
def priority(item, bins):
    time.sleep(10.0)
    return bins
"""
        ind = Individual(code=infinite_loop_code)
        record = evaluator.evaluate(ind, self.task, split="train")
        self.assertFalse(record.is_valid)
        self.assertIn("timed out", (record.error_trace or "").lower())

    def test_mcp_evaluator(self):
        mcp = MCPEvaluator(EvaluatorConfig(timeout_seconds=10.0))
        ind = Individual(code=INITIAL_BPP_CODE)

        # Test request formatting
        req = mcp.format_mcp_request(123, ind, self.task, split="train")
        self.assertEqual(req["jsonrpc"], "2.0")
        self.assertEqual(req["method"], "tools/call")
        self.assertEqual(req["params"]["name"], "evaluate_algorithm")
        self.assertEqual(req["params"]["arguments"]["individual_id"], ind.id)

        # Test local evaluate flow
        record = mcp.evaluate(ind, self.task, split="train")
        self.assertTrue(record.is_valid)
        self.assertIsNotNone(record.fitness)


if __name__ == "__main__":
    unittest.main()
