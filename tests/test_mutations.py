"""
Unit tests for EvoKit Mutation Strategies:
EoHMutationStrategy, ReEvoMutationStrategy, and DiffMutationStrategy.
"""

import unittest
from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual
from evokit.core.gateway import LLMGateway, GatewayConfig
from evokit.plugins.mutations.eoh import EoHMutationStrategy
from evokit.plugins.mutations.reevo import ReEvoMutationStrategy
from evokit.plugins.mutations.diff import DiffMutationStrategy


class TestMutations(unittest.TestCase):
    def setUp(self):
        self.gateway = LLMGateway(GatewayConfig(provider="mock"))
        self.task_spec = {
            "description": "Online Bin Packing",
            "signature": "def priority(item, bins):",
            "docstring": "Compute priority scores",
            "initial_code": "def priority(item, bins):\n    return -(bins - item)",
        }

    def test_eoh_mutation(self):
        strategy = EoHMutationStrategy()
        # Mock LLM output with thought in braces and code block
        self.gateway.set_mock_responses([
            "{Idea: Prefer fuller bins first}\n```python\ndef priority(item, bins):\n    return -(bins - item) * 2\n```"
        ])

        parent = Individual(id="p1", code="def priority(item, bins):\n    return -(bins - item)", fitness=100.0)
        ctx = EvolutionContext(
            generation=1,
            population_view=[parent],
            task_spec=self.task_spec,
            llm_gateway=self.gateway,
        )

        candidates = strategy.mutate(ctx, num_candidates=1)
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertIn("priority", cand.code)
        self.assertIn("fuller bins", cand.thought)
        self.assertTrue(cand.operator.startswith("eoh_"))

    def test_reevo_mutation(self):
        strategy = ReEvoMutationStrategy()
        # 1st mock response: reflection, 2nd mock response: offspring code
        self.gateway.set_mock_responses([
            "Algorithm A handles small items with less fragmentation.",
            "{Idea: Dynamic thresholding}\n```python\ndef priority(item, bins):\n    return -(bins - item) + 0.5\n```",
        ])

        p1 = Individual(id="p1", code="def priority(item, bins):\n    return -(bins - item)", fitness=150.0)
        p2 = Individual(id="p2", code="def priority(item, bins):\n    return bins", fitness=80.0)

        ctx = EvolutionContext(
            generation=2,
            population_view=[p1, p2],
            task_spec=self.task_spec,
            llm_gateway=self.gateway,
        )

        candidates = strategy.mutate(ctx, num_candidates=1)
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertIn("priority", cand.code)
        self.assertEqual(cand.operator, "reevo_reflect")

    def test_diff_mutation(self):
        strategy = DiffMutationStrategy()
        self.gateway.set_mock_responses([
            "{Idea: Square the remaining difference}\n```python\ndef priority(item, bins):\n    return -((bins - item) ** 2)\n```"
        ])

        parent = Individual(id="p1", code="def priority(item, bins):\n    return -(bins - item)", fitness=100.0)
        ctx = EvolutionContext(
            generation=1,
            population_view=[parent],
            task_spec=self.task_spec,
            llm_gateway=self.gateway,
        )

        candidates = strategy.mutate(ctx, num_candidates=1)
        self.assertEqual(len(candidates), 1)
        cand = candidates[0]
        self.assertIn("** 2", cand.code)
        self.assertEqual(cand.operator, "diff_patch")


if __name__ == "__main__":
    unittest.main()
