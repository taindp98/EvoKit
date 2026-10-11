# `evokit.plugins.populations`: Population & Archive Management (Pillar 1)

This module implements candidate solution storage, solution diversity maintenance, lineage tracking, and parent selection dynamics.

---

## Component Implementations

| Implementation | Primary Class | Key Characteristics | Parent Selection |
|---|---|---|---|
| **Lineage Graph Archive** (`lineage_archive.py`) | `LineageGraphArchive` | Preserves an explicit Directed Acyclic Graph (DAG) mapping how textual ideas transition into concrete programs across generations. Traces ancestry chains and exports Graphviz DOT graphs. | Fitness-ranked tournament selection |
| **Reflection Buffer** (`reflection_buffer.py`) | `ReflectionBuffer` | Dual-buffer archive maintaining successful individuals alongside short-term and long-term execution failure logs, stderr traces, and diagnostic reflections. | Uniform random or differential pair sampling |
| **MAP-Elites Grid** (`map_elites.py`) | `MAPElitesGrid` | Quality-Diversity archive maintaining an $N$-dimensional behavioral tessellation grid across orthogonal complexity (e.g. AST length) and runtime dimensions. | Uniform random across occupied behavioral cells |

---

## Contract Interface (`BasePopulationManager`)

Every population manager implements:
- `add(individual: Individual) -> bool`: Ingests evaluated candidate and applies archive admission/replacement policy.
- `sample_parents(k: int) -> List[Individual]`: Samples $k$ parent candidates for mutation or crossover.
- `get_elites(top_k: int) -> List[Individual]`: Returns the highest-performing individuals.
- `get_all() -> List[Individual]`: Returns all active individuals in the population.
- `size() -> int`: Returns current active population size.
- `get_view() -> List[Dict[str, Any]]`: Emits a lightweight serializable snapshot for `EvolutionContext`.
