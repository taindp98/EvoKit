# `evokit.plugins`: The 3 Pluggable Pillars

The `plugins` package houses the implementations of the **3 Pluggable Pillars** in EvoKit:

1. **`populations/`** (Pillar 1): Archive and population management, diversity preservation, lineage tracking, and parent selection.
2. **`mutations/`** (Pillar 2): Search paradigms, prompt engineering, structured LLM generation, and code extraction.
3. **`evaluators/`** (Pillar 3): Sandboxed program execution environments, timeout limits, and Model Context Protocol (MCP) clients.

---

## Pillar Responsibilities & Interactions

```text
              ┌─────────────────────────┐
              │   Population Manager    │  <── Samples parent candidates
              │  (Pillar 1: Archive)    │  ──> Admits evaluated individuals
              └────────────┬────────────┘
                           │
       Parent references   │   Context snapshot
                           ▼
              ┌─────────────────────────┐
              │    Mutation Strategy    │  <── Queries LLM Gateway
              │ (Pillar 2: Synthesis)   │  ──> Emits un-evaluated Individual
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │    Evaluator Service    │  <── Executes in isolated sandbox
              │  (Pillar 3: Sandbox)    │  ──> Returns MetricRecord (fitness & logs)
              └─────────────────────────┘
```

All plugins conform to the base lifecycle interface (`BaseComponent`) and communicate via standardized data models (`Individual`, `EvolutionContext`, `MetricRecord`).

---

## Method Expansion Taxonomy

Every state-of-the-art LLM-guided evolutionary algorithm in the literature maps directly into EvoKit's 3-pillar taxonomy:

| Research Method | Core Paradigm | Pillar 1: Population Manager | Pillar 2: Mutation Strategy | Pillar 3: Sandbox Evaluator |
|---|---|---|---|---|
| **EoH** *(Evolution of Heuristics)* | Dual Thought-Code search & genetic operators | `LineageGraphArchive` (DAG lineage & tournament) | `EoHMutationStrategy` (E1, E2, M1, M2 operators) | `SubprocessSandboxEvaluator` |
| **ReEvo** *(Reflective Evolution)* | Comparative critique & diagnostic feedback | `ReflectionBuffer` (execution logs & reflections) | `ReEvoMutationStrategy` (comparative reflection) | `SubprocessSandboxEvaluator` |
| **OpenEvolve** | Quality-Diversity / MAP-Elites exploration | `MAPElitesGrid` (N-dim complexity & runtime grid) | `DiffMutationStrategy` (surgical code patches) | `SubprocessSandboxEvaluator` |
| **MEoH** *(Multi-Objective EoH)* | Pareto-optimal frontier algorithm search | `ParetoFrontArchive` (non-dominated sorting) | `MEoHMutationStrategy` (trade-off prompts) | `SubprocessSandboxEvaluator` |
| **EoH-S** *(Evolution of Heuristic Set)* | Complementary portfolio coverage for hard instances | `ComplementarySetArchive` (instance portfolio cover) | `EoHSMutationStrategy` (hard-instance targeted prompts) | `SubprocessSandboxEvaluator` |

> For complete code blueprints and recipes for implementing **OpenEvolve**, **MEoH**, and **EoH-S**, see the [Method Expansion Guide in docs/plugin_guide.md](../../docs/tutorials/plugin_guide.md#7-method-expansion-recipes-openevolve-meoh-eoh-s).
