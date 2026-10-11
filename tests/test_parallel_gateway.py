"""
Unit tests for parallel LLM Gateway sampling and concurrent mutation strategies.
"""

import unittest
from evokit.core.gateway import LLMGateway, GatewayConfig
from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual
from evokit.plugins.mutations.eoh import EoHMutationStrategy
from evokit.plugins.mutations.reevo import ReEvoMutationStrategy
from evokit.plugins.mutations.diff import DiffMutationStrategy


class TestParallelGateway(unittest.TestCase):
    def test_gateway_generate_batch_mock(self):
        gw = LLMGateway(GatewayConfig(provider="mock", max_parallel_requests=4))
        gw.set_mock_responses([
            "Mock Response 0",
            "Mock Response 1",
            "Mock Response 2",
            "Mock Response 3",
        ])

        prompts = [f"Prompt {i}" for i in range(4)]
        responses = gw.generate_batch(prompts, max_workers=4)

        self.assertEqual(len(responses), 4)
        for i, resp in enumerate(responses):
            self.assertIn("Mock Response", resp.content)
            self.assertGreater(resp.total_tokens, 0)

        usage = gw.get_token_usage()
        self.assertGreater(usage["total_tokens"], 0)

    def test_gateway_chat_batch_mock(self):
        gw = LLMGateway(GatewayConfig(provider="mock", max_parallel_requests=4))
        gw.set_mock_responses(["Answer A", "Answer B", "Answer C"])

        msg_list = [
            [{"role": "user", "content": f"Query {i}"}]
            for i in range(3)
        ]
        responses = gw.chat_batch(msg_list)
        self.assertEqual(len(responses), 3)

    def test_eoh_parallel_candidate_sampling(self):
        gw = LLMGateway(GatewayConfig(provider="mock"))
        gw.set_mock_responses([
            "{Idea: Variant 1}\n```python\ndef priority(item, bins): return -bins\n```",
            "{Idea: Variant 2}\n```python\ndef priority(item, bins): return bins * 2\n```",
            "{Idea: Variant 3}\n```python\ndef priority(item, bins): return bins + item\n```",
        ])

        ctx = EvolutionContext(
            generation=1,
            population_view=[
                Individual(code="def priority(item, bins): return -bins", fitness=10.0, is_valid=True)
            ],
            task_spec={"description": "BPP", "initial_code": "def priority(item, bins): pass"},
            runtime_config={},
            llm_gateway=gw,
        )

        strat = EoHMutationStrategy()
        candidates = strat.mutate(ctx, num_candidates=3)

        self.assertEqual(len(candidates), 3)
        for cand in candidates:
            self.assertIn("priority", cand.code)
            self.assertIsNotNone(cand.thought)

    def test_reevo_parallel_candidate_sampling(self):
        gw = LLMGateway(GatewayConfig(provider="mock"))
        ctx = EvolutionContext(
            generation=1,
            population_view=[
                Individual(code="def f(): return 1", fitness=5.0, is_valid=True),
                Individual(code="def f(): return 2", fitness=10.0, is_valid=True),
            ],
            task_spec={"description": "Test", "initial_code": "def f(): pass"},
            runtime_config={},
            llm_gateway=gw,
        )

        strat = ReEvoMutationStrategy()
        candidates = strat.mutate(ctx, num_candidates=2)
        self.assertEqual(len(candidates), 2)


if __name__ == "__main__":
    unittest.main()
