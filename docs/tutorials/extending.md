# Extending EvoKit with Custom Plugins

[← Back to README](../../README.md)

A new search paradigm is a **plugin + config swap**, not a new script. The invariant
`EvolutionEngine`, benchmark, and evaluator stay fixed; you implement one or both pillars.

1. **Custom population (Pillar 1):** subclass `BasePopulationManager`
   ([`evokit/plugins/populations/base.py`](../../evokit/plugins/populations/base.py)); implement
   `add()`, `sample_parents()`, `get_elites()`, `get_all()`, `size()`. Optionally override
   `on_generation_end(generation)` for cross-generational structure (e.g. island migration).
2. **Custom mutation (Pillar 2):** subclass `BaseMutationStrategy`
   ([`evokit/plugins/mutations/base.py`](../../evokit/plugins/mutations/base.py)); implement
   `mutate(context, num_candidates)`.
3. **Custom benchmark:** subclass `BaseBenchmarkTask`
   ([`evokit/benchmarks/base.py`](../../evokit/benchmarks/base.py)); provide a `PromptSpec`, implement
   `_generate_instances()` with train/test splits, and optionally `reference_bound()` for an
   optimality-gap annotation.

Reference your class in YAML by registry alias or Python dot path — no core edits needed:

```yaml
population:
  type: "my_package.custom_archive:ParetoArchive"
  capacity: 25
```

## Worked examples already in the tree

The three shipped methods are themselves the template — each is the same engine with two
swapped plugins:

| Method | Pillar 1 | Pillar 2 |
|---|---|---|
| EoH | `lineage_archive` | `eoh` |
| ReEvo | `reflection_buffer` | `reevo` |
| OpenEvolve | `openevolve_archive` | `openevolve` |

## Method expansion blueprints

| Method | Core paradigm | Pillar 1 | Pillar 2 |
|---|---|---|---|
| **MEoH** | Multi-objective Pareto frontier | `ParetoArchive` (non-dominated sorting) | trade-off prompt synthesis |
| **EoH-S** (AAAI '26) | Automated heuristic-set design | `HeuristicSetArchive` (instance-coverage portfolio) | hard-instance targeted prompts |
| **HSEvo** | Harmony-search diversity | diversity-aware archive | harmony-guided mutation |
| **LLaMEA** | (1+λ) LLM evolution | elitist archive | mutation-with-feedback |
| **MCTS-AHD** | Monte-Carlo tree search | tree-structured archive | node-expansion prompts |

For full blueprints, code templates, and step-by-step guides, see
[`plugin_guide.md`](plugin_guide.md).
