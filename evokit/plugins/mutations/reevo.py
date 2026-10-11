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
    mutation_rate: float = Field(default=0.5, description="Fraction of offspring produced by elitist mutation (rest via reflective crossover), matching ReEvo")


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
        # ReEvo long-term reflection memory: accumulated verbal "insights" carried
        # across generations (the strategy instance persists for the whole run).
        # This is the mechanism that lets ReEvo build on prior reasoning rather
        # than doing isolated pairwise rewrites.
        self.long_term_reflection: str = ""

    def _update_long_term_reflection(self, short_reflections: List[str], context: EvolutionContext) -> None:
        """Synthesize accumulated insights from prior + new short-term reflections
        (ReEvo's long-term reflection step). Stored on the instance for reuse."""
        gateway = context.llm_gateway
        new = "\n".join(r for r in short_reflections if r)
        if not new:
            return
        if gateway is None:
            # No LLM: just append (bounded) so the signal still carries forward.
            self.long_term_reflection = (self.long_term_reflection + "\n" + new).strip()[-2000:]
            return
        prompt = f"""You are maintaining a long-term list of transferable insights for designing better heuristics.

Problem:
{context.task_spec.get('description', '')}

Prior accumulated insights:
{self.long_term_reflection or '(none yet)'}

New observations from the latest generation:
{new}

Task: Update the list of durable, generalizable design principles (not specifics of one program). Return a concise bulleted list, under 200 words."""
        try:
            resp = gateway.generate(prompt=prompt, temperature=self.cfg.temperature, max_tokens=512)
            self.long_term_reflection = resp.content.strip()
        except Exception:
            self.long_term_reflection = (self.long_term_reflection + "\n" + new).strip()[-2000:]

    @staticmethod
    def _filter_code(code: str) -> str:
        """Strip signature/import lines from a program body (ReEvo `filter_code`),
        so parent snippets shown in prompts are just the logic to reason about."""
        out = []
        for line in (code or "").split("\n"):
            s = line.lstrip()
            if s.startswith(("def ", "import ", "from ")):
                continue
            out.append(line)
            if s.startswith("return"):
                break
        return "\n".join(out).strip("\n")

    def _reflect(self, better: Individual, worse: Individual, context: EvolutionContext) -> str:
        gateway = context.llm_gateway
        if gateway is None:
            return "Comparative baseline reflection."

        prompt = f"""Problem Context:
{context.task_spec.get('description', '')}

Algorithm A (Higher Performing, Fitness: {better.fitness}):
Idea: {better.thought or 'N/A'}
```python
{self._filter_code(better.code)}
```

Algorithm B (Lower Performing, Fitness: {worse.fitness}):
Idea: {worse.thought or 'N/A'}
```python
{self._filter_code(worse.code)}
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

Accumulated Long-Term Insights (from prior generations):
{self.long_term_reflection or '(none yet)'}

Synthesized Reflection & Guidance (this pair):
{reflection}

Task:
Synthesize an improved Python heuristic algorithm that acts on the guidance in the reflection.

Strict requirements:
1. Keep the EXACT function name and signature shown in the Function Interface — do not add, remove, or rename parameters.
2. Write your reasoning on ONE line BEFORE the code block, wrapped in curly braces, e.g. {{your idea here}}. Do NOT put braces or prose inside the function body.
3. Output the full, self-contained, executable Python function in a single ```python ... ``` code block (include the `import numpy as np` if used).
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

    @staticmethod
    def _select_distinct_pair(pop_pool: List[Individual]) -> Optional[tuple]:
        """ReEvo random_select: pick two valid parents with DISTINCT objectives
        (reflecting between equal-scoring programs is meaningless). Returns
        (better, worse) or None if no distinct-objective pair exists."""
        valid = [p for p in pop_pool if p.fitness is not None]
        if len(valid) < 2:
            return None
        for _ in range(1000):
            a, b = random.sample(valid, 2)
            if a.fitness != b.fitness:
                better, worse = (a, b) if a.fitness <= b.fitness else (b, a)
                return better, worse
        return None

    def _mutate_single(self, context: EvolutionContext, pop_pool: List[Individual]) -> Individual:
        """Reflective CROSSOVER: short-term reflection on a distinct-objective pair,
        then reflection-guided offspring."""
        pair = self._select_distinct_pair(pop_pool)
        if pair is not None:
            better, worse = pair
            reflection = (
                self._reflect(better, worse, context)
                if self.reevo_cfg.enable_reflection_loop
                else "Analyze differential performance and advance search."
            )
            return self._generate_offspring(better, worse, reflection, context)
        elif pop_pool:
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

    def _elitist(self, pop_pool: List[Individual], task_spec: Dict[str, Any]) -> Optional[Individual]:
        valid = [p for p in pop_pool if p.fitness is not None]
        if not valid:
            return None
        maximize = str(task_spec.get("direction", "MINIMIZE")).upper() == "MAXIMIZE"
        return (max if maximize else min)(valid, key=lambda p: p.fitness)

    def _mutate_elitist(self, context: EvolutionContext, pop_pool: List[Individual]) -> Individual:
        """ReEvo elitist mutation: mutate the single best individual, guided by the
        accumulated long-term reflection (not a pairwise reflection)."""
        elitist = self._elitist(pop_pool, context.task_spec)
        if elitist is None:
            return self._mutate_single(context, pop_pool)
        gateway = context.llm_gateway
        task_spec = context.task_spec
        prompt = f"""Problem Context:
{task_spec.get('description', '')}

Function Interface:
```python
{task_spec.get('signature', '')}
    \"\"\"{task_spec.get('docstring', '')}\"\"\"
```

Current Elitist (best so far, Fitness: {elitist.fitness}):
```python
{self._filter_code(elitist.code)}
```

Accumulated Long-Term Insights (from prior generations):
{self.long_term_reflection or '(none yet)'}

Task:
Write a mutated version of the elitist that applies the long-term insights to explore a
genuinely different strategy.

Strict requirements:
1. Keep the EXACT function name and signature shown in the Function Interface — do not add, remove, or rename parameters.
2. Write your reasoning on ONE line BEFORE the code block, wrapped in curly braces, e.g. {{your idea here}}. Do NOT put braces or prose inside the function body.
3. Output the full, self-contained, executable Python function in a single ```python ... ``` code block.
"""
        if gateway is not None:
            raw_text = gateway.generate(prompt=prompt, temperature=self.cfg.temperature, max_tokens=self.cfg.max_tokens).content
        else:
            raw_text = task_spec.get("initial_code", "def solve(): pass")
        thought = self.extract_thought(raw_text)
        code = self.extract_code(raw_text, default_fallback=elitist.code or task_spec.get("initial_code", ""))
        return Individual(
            code=code,
            thought=thought or "ReEvo elitist mutation",
            parent_ids=[elitist.id],
            generation=context.generation,
            operator="reevo_mutate",
            metadata={"reflection": self.long_term_reflection},
        )

    def _run_parallel(self, fn, n: int) -> List[Individual]:
        if n <= 0:
            return []
        if n == 1:
            return [fn()]
        results: List[Optional[Individual]] = [None] * n
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n, 8)) as executor:
            fut = {executor.submit(fn): i for i in range(n)}
            for f in concurrent.futures.as_completed(fut):
                results[fut[f]] = f.result()
        return [r for r in results if r is not None]

    def mutate(self, context: EvolutionContext, num_candidates: int = 1) -> List[Individual]:
        """One ReEvo iteration, faithful to references/reevo/reevo.py::evolve:
        reflective CROSSOVER on distinct-objective pairs -> fold short-term
        reflections into the long-term reflection -> elitist MUTATION guided by
        that long-term reflection. `mutation_rate` splits the offspring budget."""
        if num_candidates <= 0:
            return []
        pop_pool = context.population_view if isinstance(context.population_view, list) else []
        serial = (context.llm_gateway is None)

        # Split the budget: mutation_rate fraction to elitist mutation, rest crossover.
        n_mutate = int(round(num_candidates * self.reevo_cfg.mutation_rate)) if pop_pool else 0
        n_cross = num_candidates - n_mutate

        # 1. Reflective crossover offspring (distinct-objective pairs).
        cross = ([self._mutate_single(context, pop_pool) for _ in range(n_cross)]
                 if serial else self._run_parallel(lambda: self._mutate_single(context, pop_pool), n_cross))

        # 2. Long-term reflection: fold this generation's short-term reflections in,
        #    so the elitist mutation below (and next gen) build on accumulated insight.
        if self.reevo_cfg.enable_reflection_loop and cross:
            short = [c.metadata.get("reflection", "") for c in cross
                     if isinstance(c.metadata, dict) and c.metadata.get("reflection")]
            self._update_long_term_reflection(short, context)

        # 3. Elitist mutation offspring, guided by the (updated) long-term reflection.
        mut = ([self._mutate_elitist(context, pop_pool) for _ in range(n_mutate)]
               if serial else self._run_parallel(lambda: self._mutate_elitist(context, pop_pool), n_mutate))

        return cross + mut
