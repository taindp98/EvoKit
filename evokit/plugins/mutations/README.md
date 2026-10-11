# `evokit.plugins.mutations`: Search Strategies & Genetic Operators (Pillar 2)

This module implements genetic operators, prompt generation pipelines, and LLM-guided code search strategies for evolving algorithmic heuristics.

---

## Component Implementations

| Strategy | Primary Class | Search Paradigm | Prompting Structure |
|---|---|---|---|
| **Evolution of Heuristics (EoH)** (`eoh.py`) | `EoHMutationStrategy` | Dual Thought-Code search decoupling algorithmic reasoning from Python syntax. | Four distinct genetic operators:<br>• **E1 (Cross-parent Exploration)**: Synthesizes novel heuristics inspired by multiple distinct parents.<br>• **E2 (Backbone Synthesis)**: Merges the shared backbone logic of top parents.<br>• **M1 (Form Mutation)**: Replaces heuristic components while preserving core principles.<br>• **M2 (Parameter Mutation)**: Tunes constants, weights, and scoring thresholds. |
| **Reflective Evolution (ReEvo)** (`reevo.py`) | `ReEvoMutationStrategy` | Reflective comparative critique and diagnostic debugging. | Two-stage pipeline:<br>1. **Comparative Reflection**: Synthesizes short-term reflections across contrasting candidate fitness scores, execution traces, and stderr logs.<br>2. **Targeted Generation**: Proposes algorithmic improvements grounded in the synthesized reflection. |
| **Diff / Patch Mutation** (`diff.py`) | `DiffMutationStrategy` | Localized code refinement and surgical edits. | Generates focused patch proposals targeting specific blocks or lines of code rather than rewriting the entire function from scratch. |

---

## Contract Interface (`BaseMutationStrategy`)

Every mutation strategy inherits from `BaseMutationStrategy` and implements:
- `mutate(context: EvolutionContext, num_candidates: int = 1) -> List[Individual]`: Generates un-evaluated candidate program variants given the active `EvolutionContext`.
- `extract_thought(text: str) -> Optional[str]`: Extracts reasoning, algorithmic ideas, or descriptions from LLM completions.
- `extract_code(text: str, default_fallback: str = "") -> str`: Robustly extracts executable Python code blocks (` ```python ... ``` ` or raw function definitions) from completions.

---

## Quick Example: Using a Mutation Strategy

```python
from evokit.core.gateway import LLMGateway
from evokit.core.context import EvolutionContext
from evokit.plugins.mutations.eoh import EoHMutationStrategy

# Configure strategy
strategy = EoHMutationStrategy({
    "temperature": 0.8,
    "max_tokens": 2048,
    "operator_weights": {"e1": 0.3, "e2": 0.2, "m1": 0.3, "m2": 0.2}
})

# Mutate within an active EvolutionContext
new_candidates = strategy.mutate(context, num_candidates=2)
for cand in new_candidates:
    print(f"Candidate: {cand.id} | Parent: {cand.parent_id}")
    print(f"Thought: {cand.thought}")
    print(cand.code)
```
