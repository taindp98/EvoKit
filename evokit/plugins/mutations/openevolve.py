"""
OpenEvolve Mutation Strategy (Pillar 2) for EvoKit.

Ports OpenEvolve's LLM program-evolution step (references/openevolve/openevolve/
prompt/ + controller, and the reimplementation in
references/myirace/src/reprod/openevolve_tsp_gls.py):

  - **Inspiration-rich prompt**: a system message (task description) plus a user
    message carrying the parent program, its metrics, the current top programs,
    a set of diverse "inspiration" programs, and concrete improvement areas.
  - **Edit modes**: ``full-rewrite`` (LLM returns the whole new function) or
    ``diff-based`` (LLM returns SEARCH/REPLACE blocks applied to the parent).
  - **Fitness-weighted parent selection + inspirations** drawn from the current
    population view (the OpenEvolveArchive supplies the island / MAP-Elites /
    archive membership that shapes that view).

The island model, per-island MAP-Elites grid, novelty rejection, worst-first
eviction, and ring migration live in the Pillar-1 ``OpenEvolveArchive`` — this
strategy only performs the prompt construction + LLM generation step, keeping
the two concerns cleanly separated per EvoKit's 3-pillar contract.
"""

from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, List, Optional

from pydantic import Field

from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual
from evokit.plugins.mutations.base import BaseMutationStrategy, MutationConfig


class OpenEvolveConfig(MutationConfig):
    edit_mode: str = Field(default="diff-based", description="'diff-based' or 'full-rewrite'")
    num_top_programs: int = Field(default=3, description="Top programs shown in the prompt")
    num_inspirations: int = Field(default=2, description="Diverse inspiration programs shown in the prompt")


def apply_diff(original: str, response: str) -> Optional[str]:
    """Apply OpenEvolve SEARCH/REPLACE diff blocks to ``original``."""
    pattern = re.compile(r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE", re.DOTALL)
    blocks = pattern.findall(response)
    if not blocks:
        return None
    result = original
    for search, replace in blocks:
        if search.strip() in result:
            result = result.replace(search.strip(), replace.strip(), 1)
            continue
        # Fuzzy fallback: best-aligned window with >=60% line match.
        search_lines = search.strip().splitlines()
        result_lines = result.splitlines()
        best_i, best_len = -1, 0
        for i in range(len(result_lines) - len(search_lines) + 1):
            match_len = sum(1 for a, b in zip(search_lines, result_lines[i:]) if a.strip() == b.strip())
            if match_len > best_len:
                best_len, best_i = match_len, i
        if best_i >= 0 and best_len >= len(search_lines) * 0.6:
            before = result_lines[:best_i]
            after = result_lines[best_i + len(search_lines):]
            result = "\n".join(before + replace.splitlines() + after)
    return result


class OpenEvolveMutationStrategy(BaseMutationStrategy):
    """OpenEvolve prompt construction + full-rewrite/diff LLM generation."""

    def __init__(self, config: Optional[Dict[str, Any] | OpenEvolveConfig] = None) -> None:
        if isinstance(config, OpenEvolveConfig):
            cfg = config
        elif isinstance(config, dict):
            cfg = OpenEvolveConfig(**config)
        else:
            cfg = OpenEvolveConfig()
        super().__init__(cfg)
        self.oe_cfg: OpenEvolveConfig = cfg

    # ---- direction-aware helpers -----------------------------------------
    @staticmethod
    def _is_min(task_spec: Dict[str, Any]) -> bool:
        return str(task_spec.get("direction", "MINIMIZE")).upper() == "MINIMIZE"

    def _better(self, a: Individual, b: Individual, minimize: bool) -> Individual:
        fa = a.fitness if a.fitness is not None else float("inf")
        fb = b.fitness if b.fitness is not None else float("inf")
        if minimize:
            return a if fa <= fb else b
        return a if fa >= fb else b

    # ---- selection from the population view -------------------------------
    def _fitness_weighted(self, pool: List[Individual], minimize: bool) -> Individual:
        if len(pool) == 1:
            return pool[0]
        # Normalise to "higher is better" weights.
        fits = []
        for p in pool:
            f = p.fitness if (p.fitness is not None and math.isfinite(p.fitness)) else None
            fits.append(f)
        norm = [(-f if minimize else f) if f is not None else None for f in fits]
        finite = [n for n in norm if n is not None]
        base = min(finite) if finite else 0.0
        weights = [max((n - base) if n is not None else 0.0, 0.001) for n in norm]
        total = sum(weights)
        weights = [w / total for w in weights] if total > 0 else [1.0 / len(pool)] * len(pool)
        return random.choices(pool, weights=weights, k=1)[0]

    def _top_programs(self, pool: List[Individual], k: int, minimize: bool) -> List[Individual]:
        valid = [p for p in pool if p.fitness is not None and math.isfinite(p.fitness)]
        valid.sort(key=lambda p: p.fitness, reverse=not minimize)
        return valid[:k]

    def _inspirations(self, parent: Individual, pool: List[Individual], n: int, minimize: bool) -> List[Individual]:
        insp: List[Individual] = []
        top = self._top_programs(pool, max(1, n), minimize)
        for p in top:
            if p.id != parent.id and p.id not in {i.id for i in insp}:
                insp.append(p)
        if len(insp) < n:
            exclude = {parent.id} | {i.id for i in insp}
            rest = [p for p in pool if p.id not in exclude and p.fitness is not None]
            if rest:
                insp.extend(random.sample(rest, min(n - len(insp), len(rest))))
        return insp[:n]

    # ---- prompt construction (OpenEvolve-faithful) ------------------------
    def _build_messages(self, parent: Individual, inspirations: List[Individual],
                        top_programs: List[Individual], task_spec: Dict[str, Any]) -> List[Dict[str, str]]:
        desc = task_spec.get("description", "")
        func = task_spec.get("function_name", "solve")
        sig = task_spec.get("signature", "")
        minimize = self._is_min(task_spec)

        system = (
            "You are an expert algorithm designer improving a heuristic through "
            "iterative evolution. You will be shown a current program, its metrics, "
            "top-performing programs, and inspirations. Propose an improved version."
        )

        metrics_str = f"- fitness: {parent.fitness}\n"
        for k, v in parent.metrics.items():
            if not k.startswith("_"):
                metrics_str += f"- {k}: {v}\n"

        def _fmt(progs: List[Individual]) -> str:
            parts = [f"**Program {i+1}** (fitness={p.fitness}):\n```python\n{p.code}\n```"
                     for i, p in enumerate(progs)]
            return "\n\n".join(parts) if parts else "(none yet)"

        goal = "minimize" if minimize else "maximize"
        improvement = (
            f"- The objective is to {goal} fitness. Improve on the current program.\n"
            "- Consider alternative decision rules, scoring functions, and edge-case handling.\n"
            "- Learn from the structure of the top programs and inspirations above."
        )

        if self.oe_cfg.edit_mode == "diff-based":
            instr = (
                "Output ONE OR MORE SEARCH/REPLACE diff blocks that transform the current "
                "program into an improved one, using exactly this format:\n"
                "<<<<<<< SEARCH\n<exact lines to find>\n=======\n<replacement lines>\n>>>>>>> REPLACE\n"
                "Also put a one-line rationale in curly braces: {thought}."
            )
        else:
            instr = (
                "Output the COMPLETE improved Python function in a single ```python ... ``` "
                "block. Put a one-line rationale in curly braces: {thought}."
            )

        user = f"""Problem:
{desc}

Target function: {func}
Signature: {sig}

Current program:
```python
{parent.code}
```

Current program metrics:
{metrics_str}
Top programs so far:
{_fmt(top_programs)}

Inspirations (diverse prior programs):
{_fmt(inspirations)}

Improvement areas:
{improvement}

Instructions:
{instr}
"""
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # ---- main entrypoint --------------------------------------------------
    def mutate(self, context: EvolutionContext, num_candidates: int = 1) -> List[Individual]:
        if num_candidates <= 0:
            return []
        task_spec = context.task_spec
        gateway = context.llm_gateway
        pool = context.population_view if isinstance(context.population_view, list) else []
        minimize = self._is_min(task_spec)
        fallback = task_spec.get("initial_code", "")

        # Cold start: no population yet -> sample initial programs from scratch.
        if not pool:
            return self._cold_start(context, num_candidates)

        # Plan one (parent, inspirations) per offspring, then batch the LLM calls.
        plan = []
        for _ in range(num_candidates):
            parent = self._fitness_weighted(pool, minimize)
            inspirations = self._inspirations(parent, pool, self.oe_cfg.num_inspirations, minimize)
            top = self._top_programs(pool, self.oe_cfg.num_top_programs, minimize)
            messages = self._build_messages(parent, inspirations, top, task_spec)
            plan.append((parent, messages))

        candidates: List[Individual] = []
        if gateway is not None:
            responses = gateway.chat_batch(
                [m for _, m in plan],
                temperature=self.cfg.temperature,
                max_tokens=self.cfg.max_tokens,
            )
            for (parent, _), resp in zip(plan, responses):
                candidates.append(self._make_child(resp.content, parent, context, fallback))
        else:
            for parent, _ in plan:
                candidates.append(self._make_child(fallback, parent, context, fallback))
        return candidates

    def _make_child(self, raw_text: str, parent: Individual, context: EvolutionContext,
                    fallback: str) -> Individual:
        thought = self.extract_thought(raw_text)
        if self.oe_cfg.edit_mode == "diff-based":
            code = apply_diff(parent.code, raw_text)
            if not code:
                code = self.extract_code(raw_text, default_fallback=parent.code or fallback)
        else:
            code = self.extract_code(raw_text, default_fallback=parent.code or fallback)
        return Individual(
            code=code,
            thought=thought or "OpenEvolve offspring",
            parent_ids=[parent.id],
            generation=context.generation,
            operator=f"openevolve_{self.oe_cfg.edit_mode}",
            metadata={"edit_mode": self.oe_cfg.edit_mode},
        )

    def _cold_start(self, context: EvolutionContext, num_candidates: int) -> List[Individual]:
        """Generation 0: sample pop_size diverse initial programs from the LLM."""
        task_spec = context.task_spec
        gateway = context.llm_gateway
        desc = task_spec.get("description", "")
        func = task_spec.get("function_name", "solve")
        sig = task_spec.get("signature", "")
        docstring = task_spec.get("docstring", "")
        fallback = task_spec.get("initial_code", "")

        system = "You are an expert algorithm designer creating a novel heuristic."
        user = f"""Design an initial heuristic for the following problem:
{desc}

Function: {func}
Signature: {sig}
Docstring: {docstring}

Instructions:
1. Put a one-line rationale in curly braces: {{thought}}.
2. Output the complete executable Python function in a single ```python ... ``` block.
"""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        candidates: List[Individual] = []
        if gateway is not None:
            responses = gateway.chat_batch([messages] * num_candidates,
                                           temperature=self.cfg.temperature + 0.3,
                                           max_tokens=self.cfg.max_tokens)
            raws = [r.content for r in responses]
        else:
            raws = [fallback] * num_candidates
        for raw in raws:
            thought = self.extract_thought(raw)
            code = self.extract_code(raw, default_fallback=fallback)
            candidates.append(Individual(
                code=code,
                thought=thought or "OpenEvolve initial program",
                generation=context.generation,
                operator="openevolve_init",
            ))
        return candidates
