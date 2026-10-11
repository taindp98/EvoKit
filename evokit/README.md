# `evokit`: Core Framework Package

`evokit` is the root package of the EvoKit framework. It houses the invariant evolutionary engine, component lifecycle abstractions, benchmark problems, the 3 pluggable pillars, declarative loaders, and the command-line orchestrator.

---

## Package Architecture

```text
evokit/
├── core/             # Invariant runtime contracts, data entities, storage, and LLM gateway
├── benchmarks/       # Problem tasks, train/test split datasets, and domain prompt specifications
├── plugins/          # Pluggable implementations across the 3 pillars
│   ├── populations/  # Lineage DAG archives, reflection buffers, MAP-Elites grids
│   ├── mutations/    # EoH, ReEvo, and Diff mutation strategies
│   └── evaluators/   # Subprocess sandbox and MCP JSON-RPC protocol evaluators
├── configs/          # Declarative YAML execution specifications
├── engine.py         # EvolutionEngine: Invariant generational loop coordinator
├── loader.py         # Dynamic YAML loader and component registry
└── main.py           # CLI entrypoint (run, evaluate, dashboard)
```

---

## Primary Modules

- **`engine.py`**: Contains `EvolutionEngine`. Drives the step-by-step evolutionary cycle:
  $$\text{Context Construction} \to \text{Mutation} \to \text{Sandboxed Evaluation} \to \text{Dual-Tier Storage} \to \text{Archive Update} \to \text{Hold-out Test Generalization}$$
- **`loader.py`**: Dynamically instantiates components from declarative YAML files, resolves shorthand aliases or Python import paths (`module:Class`), and expands environment variables.
- **`main.py`**: Provides the CLI (`evokit run`, `evokit evaluate`, `evokit dashboard`) with Rich terminal formatting.
