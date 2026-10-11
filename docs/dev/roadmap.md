# EvoKit Development Roadmap

High-level status of what's built and what's planned, for contributors. For the detailed,
timestamped implementation history see [changelog.md](changelog.md); for the
architectural baseline see [spec.md](../tutorials/spec.md).

_Last updated: 2026-10-11._

---

## ✅ Done

### Core engine & contracts
- [x] `BaseComponent` lifecycle (setup/teardown, context-manager) and `ComponentConfig`.
- [x] Data contracts: `Individual`, `MetricRecord`, immutable `EvolutionContext`.
- [x] Invariant `EvolutionEngine` (context → mutate → evaluate → store → select; held-out test split).
- [x] LLM-sampled initial population (`initial_population_size`) and tournament parent pool (`parent_selection_num`).
- [x] Generic end-of-generation population hook (`on_generation_end`) — e.g. island migration.
- [x] Declarative YAML loader + plugin registry (alias or `module:Class`) and dynamic CLI.

### Pillar 1 — Population / Archive
- [x] `LineageGraphArchive` (EoH DAG + tournament selection).
- [x] `ReflectionBuffer` (ReEvo short/long-term reflection memory).
- [x] `MAPElitesGrid` (behavioural tessellation).
- [x] `OpenEvolveArchive` (islands + per-island MAP-Elites + archive + mixed-strategy selection + novelty rejection + worst-first eviction + ring migration).

### Pillar 2 — Mutation / Search
- [x] `EoHMutationStrategy` (E1/E2 exploration, M1/M2 mutation).
- [x] `ReEvoMutationStrategy` (reflection-guided offspring).
- [x] `OpenEvolveMutationStrategy` (inspiration-rich prompt; diff-based / full-rewrite edit modes).
- [x] `DiffMutationStrategy` (surgical patches).

### Pillar 3 — Evaluation sandbox
- [x] `SubprocessSandboxEvaluator` (isolated subprocess, wall-clock timeout, batch/parallel workers).
- [x] `MCPEvaluator` (JSON-RPC 2.0 `tools/call` client).

### Benchmarks
- [x] `OBPBenchmarkTask` (`obp_online`, Weibull, L1 bound, `reference_bound()` contract).
- [x] `TSPBenchmarkTask` (`tsp_construct`, 2D Euclidean, NN baseline).
- [x] Native MINIMIZE direction standardised across benchmarks + presets.

### Infrastructure
- [x] `LLMGateway` (vLLM / cloud / mock; concurrent batch sampling; token + request-count tracking).
- [x] Dual-tier `ExperimentStorage`: Optuna-compatible SQLite (runtime-derived schema/alembic head) + `.py` artifacts.
- [x] TensorBoard telemetry: per-experiment runs, switchable x-axis (generations / LLM calls / FEs / wall-clock), run-metadata TEXT panel, per-run `terminal.txt`.
- [x] Two-tier parallelism (inter-candidate + intra-task) and concurrent gateway sampling.

### Examples, scripts & docs
- [x] `examples/{eoh,reevo,openevolve}_obp/` — faithful reproductions on the modular engine.
- [x] `scripts/run_{eoh,reevo,openevolve}_obp_3seeds.sh` — parallel multi-seed, per-seed DB, shared TensorBoard root.
- [x] Restructured README + `docs/tutorials/` (community) and `docs/dev/` (milestones) sub-guides; per-package READMEs.
- [x] 27/27 unit + integration tests passing.

### Fidelity audits (plugins vs. original implementations)
- [x] **ReEvo** vs `references/reevo/reevo.py`: code-sanitizer + pinned signature, long-term reflection memory, distinct-objective pairing, elitist-mutation operator, `filter_code`, and configurable non-elitist/elitist survival (default non-elitist). See [bugfixes.md](bugfixes.md).
- [x] **EoH** vs `references/LLM4AD/llm4ad/method/eoh`: all-operators-per-step cycling + `selection_num` parents for E1/E2.
- [x] **OpenEvolve** vs `references/openevolve`: island-scoped (current_island) uniform mixed-strategy selection, island-only inspirations, faithful copy/flag/skip-duplicate migration; strategy delegates selection to the archive via `EvolutionContext.metadata["population_manager"]`.

---

## 🔭 To do

### New methods (Pillars 1 & 2)
- [ ] **LLaMEA** — (1+λ) LLM evolutionary algorithm (elitist archive + mutation-with-feedback).
- [ ] **HSEvo** — harmony-search diversity (diversity-aware archive + harmony-guided mutation).
- [ ] **EoH-S** (AAAI '26) — automated heuristic-set design (`HeuristicSetArchive` instance-coverage portfolio + hard-instance prompts).
- [ ] **MCTS-AHD** — Monte-Carlo tree search (tree-structured archive + node-expansion prompts).
- [ ] **MEoH** — multi-objective Pareto frontier (`ParetoArchive` non-dominated sorting).

### Benchmarks
- [ ] TSP-GLS (guided local search; perturbation heuristics) to match the reference reproductions.
- [ ] Additional LLM4AD-style tasks (e.g. admissible heuristics, scheduling) with train/test splits.
- [ ] A benchmark registry doc describing the `reference_bound()` / `PromptSpec` contract per task.

### Engine & fidelity
- [ ] Function-evaluation budget mode (`max_fe`) alongside generation-count, matching ReEvo/OpenEvolve budgeting.
- [x] ~~Island-scoped per-offspring parent routing for OpenEvolve~~ — done (Phase 25); strategy delegates to the archive's island-scoped sampling.
- [ ] Per-island generation counters for OpenEvolve migration trigger (currently uses the engine's global generation; see [bugfixes.md](bugfixes.md) known approximations).
- [ ] Resume / checkpoint a run from its SQLite DB + artifacts.
- [ ] Async / streaming evaluation to overlap LLM generation with sandbox execution.

### Infrastructure & robustness
- [ ] LLM response cache (reproducible, cost-saving) analogous to the reference `CachedLLM`.
- [ ] Richer cloud-provider coverage in `LLMGateway` (Anthropic, OpenAI, Mistral) with retries/backoff tests.
- [ ] Docker/WASM sandbox evaluator option beyond subprocess + MCP.
- [ ] Harden `extract_thought`/code extraction against malformed LLM output (known fragility).

### Analysis & reporting
- [ ] Across-seed aggregate plots (mean ± std trajectory) and a small results-summary CLI.
- [ ] Cross-method comparison report (EoH vs. ReEvo vs. OpenEvolve on the same benchmark/seeds).

### Project hygiene
- [ ] CI (lint + `pytest`) on push; coverage badge wired to reality.
- [ ] Packaging polish for PyPI; pin a tested dependency set.
- [ ] Contribution guide (`CONTRIBUTING.md`) referencing [extending.md](../tutorials/extending.md).

---

## How to pick up a task

1. Read the pillar contracts in [extending.md](../tutorials/extending.md) and the
   [architecture spec](../tutorials/spec.md).
2. For a new method, implement a Pillar-1 and/or Pillar-2 plugin, register it in
   `evokit/loader.py`, add a `evokit/configs/<method>_<task>.yaml`, and mirror the
   `examples/<method>_obp/` runner + 3-seed script pattern.
3. Add unit coverage under `tests/` and record the work as a new phase in
   [changelog.md](changelog.md).
