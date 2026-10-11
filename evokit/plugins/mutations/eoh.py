"""
EoH (Evolution of Heuristics) Mutation Strategy Plugin for EvoKit.
Implements dual Thought-Code search across E1/E2 exploration and M1/M2 mutation operators.
Supports parallel multi-request LLM sampling.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual
from evokit.plugins.mutations.base import BaseMutationStrategy, MutationConfig


class EoHConfig(MutationConfig):
    operator_weights: Dict[str, float] = Field(
        default_factory=lambda: {"e1": 0.25, "e2": 0.25, "m1": 0.25, "m2": 0.25}
    )
    selection_num: int = Field(default=5, description="Number of parents passed to E1/E2 crossover prompts (reference EoH 'selection_num')")
    use_e2_operator: bool = Field(default=True, description="Enable the E2 (backbone synthesis) operator")
    use_m1_operator: bool = Field(default=True, description="Enable the M1 (form mutation) operator")
    use_m2_operator: bool = Field(default=True, description="Enable the M2 (parameter mutation) operator")
    cycle_operators: bool = Field(default=True, description="Reference EoH behavior: cycle through all enabled operators across a generation (round-robin) instead of weighted-random single-operator selection")


class EoHMutationStrategy(BaseMutationStrategy):
    """
    Search strategy encapsulating the Evolution of Heuristics (EoH) paradigm:
    E1 (Cross-parent Exploration), E2 (Backbone Synthesis),
    M1 (Form Mutation), and M2 (Parameter Mutation).
    """

    def __init__(self, config: Optional[Dict[str, Any] | EoHConfig] = None) -> None:
        if isinstance(config, EoHConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = EoHConfig(**config)
        else:
            cfg = EoHConfig()
        super().__init__(cfg)
        self.eoh_cfg: EoHConfig = cfg

    def _enabled_operators(self) -> List[str]:
        """Operators active this run. Reference EoH always uses E1 and toggles
        E2/M1/M2 (no M3)."""
        ops = ["e1"]
        if self.eoh_cfg.use_e2_operator:
            ops.append("e2")
        if self.eoh_cfg.use_m1_operator:
            ops.append("m1")
        if self.eoh_cfg.use_m2_operator:
            ops.append("m2")
        return ops

    def _select_operator(self, has_parents: bool) -> str:
        if not has_parents:
            # Cold start: initialization exploration
            return "i1"

        weights = self.eoh_cfg.operator_weights
        ops = list(weights.keys())
        w_values = [weights[op] for op in ops]
        return random.choices(ops, weights=w_values, k=1)[0]

    def _build_prompt(self, op: str, task_spec: Dict[str, Any], parents: List[Individual]) -> str:
        desc = task_spec.get("description", "")
        func_spec = f"Function Name: {task_spec.get('function_name', 'solve')}\nSignature: {task_spec.get('signature', '')}\nDocstring: {task_spec.get('docstring', '')}"

        if op == "i1":
            return f"""You are an expert algorithm designer.
Design an initial novel heuristic to solve the following problem:
{desc}

Interface:
{func_spec}

Instructions:
1. Enclose your core algorithmic thought and design rationale in curly braces: {{thought}}
2. Provide the complete executable Python function in a single ```python code block.
"""

        parent_blocks = []
        for i, p in enumerate(parents):
            parent_blocks.append(f"Algorithm {i+1} (Fitness: {p.fitness}):\nIdea: {p.thought or 'N/A'}\n```python\n{p.code}\n```")
        parent_str = "\n\n".join(parent_blocks)

        if op == "e1":
            # Exploration across disparate parents
            return f"""You are an expert algorithm designer. Analyze these existing algorithms:
{parent_str}

Problem context:
{desc}
{func_spec}

Operator: Cross-parent Exploration (E1)
Task: Identify the underlying ideas and contrasting principles of the above algorithms. Synthesize a brand-new heuristic algorithm that combines complementary insights.
1. Put your core idea in curly braces: {{thought}}
2. Write complete executable Python function inside ```python ... ``` block.
"""

        elif op == "e2":
            # Backbone synthesis
            return f"""You are an expert algorithm designer. Analyze these top-performing algorithms:
{parent_str}

Problem context:
{desc}
{func_spec}

Operator: Common Backbone Synthesis (E2)
Task: Identify the common successful algorithmic structure shared by these algorithms and synthesize a clean, refined version.
1. Put your core idea in curly braces: {{thought}}
2. Write complete executable Python code inside ```python ... ``` block.
"""

        elif op == "m1":
            # Form Mutation
            return f"""You are an expert algorithm designer. Analyze this algorithm:
{parent_str}

Problem context:
{desc}
{func_spec}

Operator: Component Substitution (M1)
Task: Keep the high-level strategy of the algorithm, but replace one key sub-heuristic or decision rule with a novel design.
1. Put your core idea in curly braces: {{thought}}
2. Write complete executable Python code inside ```python ... ``` block.
"""

        elif op == "m2":
            # Parameter Mutation
            return f"""You are an expert algorithm designer. Analyze this algorithm:
{parent_str}

Problem context:
{desc}
{func_spec}

Operator: Parameter & Threshold Mutation (M2)
Task: Refine scoring coefficients, thresholds, normalization constants, and edge condition logic of this algorithm.
1. Put your core idea in curly braces: {{thought}}
2. Write complete executable Python code inside ```python ... ``` block.
"""

        return f"{desc}\n{func_spec}"

    def mutate(self, context: EvolutionContext, num_candidates: int = 1) -> List[Individual]:
        candidates: List[Individual] = []
        task_spec = context.task_spec
        pop_pool = context.population_view if isinstance(context.population_view, list) else []
        gateway = context.llm_gateway

        if num_candidates <= 0:
            return []

        # 1. Plan operators and construct prompts for all candidates.
        #    Reference EoH applies ALL enabled operators each step; we reproduce
        #    that by cycling e1/e2/m1/m2 round-robin across the generation
        #    (cycle_operators=True), so offspring are evenly distributed across
        #    operators rather than drawn from a weighted-random single choice.
        has_parents = len(pop_pool) > 0
        enabled = self._enabled_operators()
        sel_num = max(1, self.eoh_cfg.selection_num)
        plan = []
        for idx in range(num_candidates):
            if not has_parents:
                op = "i1"
            elif self.eoh_cfg.cycle_operators:
                op = enabled[idx % len(enabled)]
            else:
                op = self._select_operator(has_parents)

            parents: List[Individual] = []
            if op in ("e1", "e2"):
                # E1/E2 crossover prompts take selection_num parents (reference).
                parents = random.sample(pop_pool, min(sel_num, len(pop_pool)))
            elif op in ("m1", "m2"):
                parents = random.sample(pop_pool, 1)

            prompt = self._build_prompt(op, task_spec, parents)
            plan.append((op, parents, prompt))

        # 2. Parallel LLM generation across all planned prompts
        if gateway is not None:
            prompts = [p for _, _, p in plan]
            responses = gateway.generate_batch(
                prompts,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            for (op, parents, prompt), resp in zip(plan, responses):
                raw_text = resp.content
                thought = self.extract_thought(raw_text)
                code = self.extract_code(raw_text, default_fallback=task_spec.get("initial_code", ""))
                parent_ids = [p.id for p in parents if hasattr(p, "id")]
                cand = Individual(
                    code=code,
                    thought=thought,
                    parent_ids=parent_ids,
                    generation=context.generation,
                    operator=f"eoh_{op}",
                    metadata={"prompt_preview": prompt[:150], "llm_latency": resp.latency_seconds},
                )
                candidates.append(cand)
        else:
            for op, parents, prompt in plan:
                raw_text = task_spec.get("initial_code", "def solve(): pass")
                thought = self.extract_thought(raw_text)
                code = self.extract_code(raw_text, default_fallback=task_spec.get("initial_code", ""))
                parent_ids = [p.id for p in parents if hasattr(p, "id")]
                cand = Individual(
                    code=code,
                    thought=thought,
                    parent_ids=parent_ids,
                    generation=context.generation,
                    operator=f"eoh_{op}",
                    metadata={"prompt_preview": prompt[:150]},
                )
                candidates.append(cand)

        return candidates
