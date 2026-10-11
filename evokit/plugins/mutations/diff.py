"""
Diff / Patch Mutation Strategy Plugin for EvoKit.
Generates targeted modifications and localized algorithmic adjustments to parent code.
Supports concurrent multi-request LLM sampling.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual
from evokit.plugins.mutations.base import BaseMutationStrategy, MutationConfig


class DiffMutationStrategy(BaseMutationStrategy):
    """
    Produces focused modifications and surgical edits to existing candidate codebases.
    """

    def mutate(self, context: EvolutionContext, num_candidates: int = 1) -> List[Individual]:
        candidates: List[Individual] = []
        pop_pool = context.population_view if isinstance(context.population_view, list) else []
        task_spec = context.task_spec
        gateway = context.llm_gateway

        if num_candidates <= 0:
            return []

        plan = []
        for _ in range(num_candidates):
            if pop_pool:
                parent = random.choice(pop_pool)
                prompt = f"""Problem Context:
{task_spec.get('description', '')}

Function Interface:
```python
{task_spec.get('signature', '')}
    \"\"\"{task_spec.get('docstring', '')}\"\"\"
```

Existing Algorithm:
```python
{parent.code}
```

Task:
Perform a targeted surgical optimization on the existing algorithm.
Modify a specific heuristic term, tie-breaking mechanism, or priority formula to improve performance without rewriting the structure unnecessarily.
First, summarize your targeted modification in one sentence inside curly braces {{Idea: ...}}.
Next, output the full updated Python function inside a ```python ... ``` code block.
"""
                parent_id = parent.id
                default_code = parent.code
            else:
                prompt = f"""Problem Context:
{task_spec.get('description', '')}

Create an initial algorithm implementation for:
```python
{task_spec.get('signature', '')}
```
"""
                parent_id = None
                default_code = task_spec.get("initial_code", "")

            plan.append((parent_id, default_code, prompt))

        if gateway is not None:
            prompts = [p for _, _, p in plan]
            responses = gateway.generate_batch(
                prompts,
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            for (parent_id, default_code, _), resp in zip(plan, responses):
                raw_text = resp.content
                thought = self.extract_thought(raw_text)
                code = self.extract_code(raw_text, default_fallback=default_code)
                cand = Individual(
                    code=code,
                    thought=thought or "Targeted heuristic patch",
                    parent_ids=[parent_id] if parent_id else [],
                    generation=context.generation,
                    operator="diff_patch",
                )
                candidates.append(cand)
        else:
            for parent_id, default_code, _ in plan:
                cand = Individual(
                    code=default_code,
                    thought="Offline fallback patch",
                    parent_ids=[parent_id] if parent_id else [],
                    generation=context.generation,
                    operator="diff_patch",
                )
                candidates.append(cand)

        return candidates
