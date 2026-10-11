"""
ReEvo (Reflective Evolution) Mutation Strategy Plugin for EvoKit.
Implements multi-step reflective loops analyzing failure diagnostics and comparative algorithmic traces.
Supports concurrent multi-request LLM sampling.
"""

from __future__ import annotations

import concurrent.futures
import random
from typing import Any, Dict, List, Optional
from pydantic import Field

from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual
from evokit.plugins.mutations.base import BaseMutationStrategy, MutationConfig


class ReEvoConfig(MutationConfig):
    enable_reflection_loop: bool = Field(default=True, description="Perform separate LLM reflection step before generation")


class ReEvoMutationStrategy(BaseMutationStrategy):
    """
    Search strategy encapsulating the Reflective Evolution (ReEvo) paradigm:
    Synthesizes short-term reflections across contrasting algorithm performances
    and error feedback before emitting code candidates.
    Supports concurrent generation across candidate variants.
    """

    def __init__(self, config: Optional[Dict[str, Any] | ReEvoConfig] = None) -> None:
        if isinstance(config, ReEvoConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = ReEvoConfig(**config)
        else:
            cfg = ReEvoConfig()
        super().__init__(cfg)
        self.reevo_cfg: ReEvoConfig = cfg

    def _reflect(self, better: Individual, worse: Individual, context: EvolutionContext) -> str:
        gateway = context.llm_gateway
        if gateway is None:
            return "Comparative baseline reflection."

        prompt = f"""Problem Context:
{context.task_spec.get('description', '')}

Algorithm A (Higher Performing, Fitness: {better.fitness}):
Idea: {better.thought or 'N/A'}
```python
{better.code}
```

Algorithm B (Lower Performing, Fitness: {worse.fitness}):
Idea: {worse.thought or 'N/A'}
```python
{worse.code}
```

Task:
Compare Algorithm A and Algorithm B. Formulate a short, concrete natural language reflection explaining:
1. Why Algorithm A outperforms Algorithm B.
2. What structural mechanism, heuristic feature, or edge-case handling contributed to this difference.
3. How this insight can be extended into a more optimal heuristic.
Keep your reflection under 150 words.
"""
        resp = gateway.generate(prompt=prompt, temperature=self.cfg.temperature, max_tokens=512)
        return resp.content.strip()

    def _generate_offspring(
        self,
        better: Individual,
        worse: Optional[Individual],
        reflection: str,
        context: EvolutionContext,
    ) -> Individual:
        gateway = context.llm_gateway
        task_spec = context.task_spec

        prompt = f"""Problem Context:
{task_spec.get('description', '')}

Function Interface:
```python
{task_spec.get('signature', '')}
    \"\"\"{task_spec.get('docstring', '')}\"\"\"
```

High-Performing Baseline:
```python
{better.code}
```

Synthesized Reflection & Guidance:
{reflection}

Task:
Synthesize an improved Python heuristic algorithm that acts on the guidance in the reflection.
1. Enclose your core algorithmic thought in curly braces: {{thought}}
2. Output the full executable Python code in a single ```python ... ``` code block.
"""
        if gateway is not None:
            resp = gateway.generate(prompt=prompt, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens)
            raw_text = resp.content
        else:
            raw_text = task_spec.get("initial_code", "def solve(): pass")

        thought = self.extract_thought(raw_text)
        code = self.extract_code(raw_text, default_fallback=task_spec.get("initial_code", ""))

        parent_ids = [better.id]
        if worse is not None and hasattr(worse, "id"):
            parent_ids.append(worse.id)

        return Individual(
            code=code,
            thought=thought or "ReEvo reflective candidate",
            parent_ids=parent_ids,
            generation=context.generation,
            operator="reevo_reflect",
            metadata={"reflection": reflection},
        )

    def _mutate_single(self, context: EvolutionContext, pop_pool: List[Individual]) -> Individual:
        if len(pop_pool) >= 2:
            p1, p2 = random.sample(pop_pool, 2)
            f1 = p1.fitness if p1.fitness is not None else float("inf")
            f2 = p2.fitness if p2.fitness is not None else float("inf")
            better, worse = (p1, p2) if f1 <= f2 else (p2, p1)

            reflection = (
                self._reflect(better, worse, context)
                if self.reevo_cfg.enable_reflection_loop
                else "Analyze differential performance and advance search."
            )
            return self._generate_offspring(better, worse, reflection, context)
        elif len(pop_pool) == 1:
            better = pop_pool[0]
            reflection = "Refine the existing baseline by exploring alternative decision boundaries."
            return self._generate_offspring(better, None, reflection, context)
        else:
            init_code = context.task_spec.get("initial_code", "def solve(): pass")
            return Individual(
                code=init_code,
                thought="Cold start seed individual",
                generation=context.generation,
                operator="reevo_seed",
            )

    def mutate(self, context: EvolutionContext, num_candidates: int = 1) -> List[Individual]:
        if num_candidates <= 0:
            return []
        pop_pool = context.population_view if isinstance(context.population_view, list) else []

        if num_candidates == 1 or context.llm_gateway is None:
            return [self._mutate_single(context, pop_pool) for _ in range(num_candidates)]

        # Parallelize across candidates
        pool_size = min(num_candidates, 8)
        results: List[Optional[Individual]] = [None] * num_candidates
        with concurrent.futures.ThreadPoolExecutor(max_workers=pool_size) as executor:
            future_to_idx = {
                executor.submit(self._mutate_single, context, pop_pool): i
                for i in range(num_candidates)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                results[idx] = future.result()

        return [r for r in results if r is not None]
