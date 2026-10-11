# EvoKit Task Tracking Log

This document tracks implementation progress across all development phases of **EvoKit**, following production and industrial standards.

## Status Legend
- ⏳ **Planning**: Architectural design in progress.
- 🚧 **Implementing**: Active code development.
- 🧪 **Testing**: Unit, integration, or smoke testing.
- ✅ **Done**: Verified and documented.

---

## Phase Overview

| Phase | Description | Status | Verification Summary |
|---|---|---|---|
| **Phase 1** | Foundation Scaffolding & Core Architecture | ✅ Done | 5/5 unit tests passed (`test_core.py`) |
| **Phase 2** | Benchmark & Isolated Sandbox Layer | ✅ Done | 6/6 unit tests passed (`test_benchmarks.py`, `test_evaluators.py`) |
| **Phase 3** | The Population & Mutation Plugin Stack | ✅ Done | 6/6 unit tests passed (`test_populations.py`, `test_mutations.py`) |
| **Phase 4** | Orchestration & Declarative Configuration Assembly | ✅ Done | 1/1 end-to-end integration test passed (`test_engine.py`) |
| **Phase 5** | Validation & Ecosystem Integration | ✅ Done | Live run verified on vLLM server (`Devstral-Small-2-24B-Instruct-2512`) |
| **Phase 12** | Optuna Dashboard Schema Compatibility Fixes | ✅ Done | Dashboard serves the live DB (previously crashed); Optuna loads the study |
| **Phase 13** | TensorBoard Telemetry & Run Provenance | ✅ Done | Objective curves with switchable x-axis + metadata TEXT panel verified |
| **Phase 14** | BPP → OBP Rename & Faithful EoH-OBP Reproduction | ✅ Done | 27/27 tests pass; modular reproduction mirrors `src/reprod/eoh_obp.py` |

---

## Detailed Task Breakdown

### Phase 1: Foundation Scaffolding & Core Architecture
- [x] **Task 1.1**: Initialize workspace structure and task logging in `docs/tasks/`
  - Created modular directories (`evokit/core`, `evokit/benchmarks`, `evokit/plugins`, `evokit/configs`, `tests`, `docs/tasks`).
  - Added `pyproject.toml` and root `__init__.py`.
- [x] **Task 1.2**: Implement `BaseComponent` lifecycle contracts (`evokit/core/base.py`)
  - Defined idempotent `setup()` and `teardown()` hooks, context manager protocols (`__enter__`, `__exit__`), and `ComponentConfig` Pydantic models.
- [x] **Task 1.3**: Implement Core Data Models (`Individual`, `MetricRecord`, `EvolutionContext`) (`evokit/core/entity.py`, `evokit/core/context.py`)
  - Enforced immutable state containment in `EvolutionContext`.
  - Implemented `Individual` entity with unique IDs, lineage tracking, multidimensional metrics, and cloning.
- [x] **Task 1.4**: Implement Unified `LLMGateway` (`evokit/core/gateway.py`)
  - Integrated local OpenAI-compatible vLLM endpoints and cloud providers.
  - Implemented token budgeting, temperature, top-p, retry resilience with backoff, and mock testing backend.
- [x] **Task 1.5**: Implement Dual-Tier `ExperimentStorage` (`evokit/core/storage.py`)
  - Tier 1: Relational SQLite database strictly aligned with Optuna schema (`studies`, `trials`, `trial_values`, `trial_params`, `trial_user_attributes`).
  - Tier 2: Physical `.py` code artifact persistence with trial IDs, generational hierarchy, and metadata headers.
  - Custom `evokit_individuals` table for evolutionary lineage tracking.
- [x] **Task 1.6**: Implement unit tests for Phase 1 (`tests/test_core.py`)
  - Verified component lifecycle, individual serialization, context immutability, gateway budgeting, and Optuna SQLite tables.

### Phase 2: Benchmark & Sandbox Layer
- [x] **Task 2.1**: Implement `BaseBenchmarkTask` interface (`evokit/benchmarks/base.py`)
  - Enforced strict train/test instance segregation and abstracted domain prompts as data via `PromptSpec`.
- [x] **Task 2.2**: Implement Online Bin Packing benchmark (`evokit/benchmarks/obp/`)
  - Built `OBPBenchmarkTask` with uniform and Weibull arrival streams, capacity constraints, L1 bound comparison, and baseline Best-Fit heuristic.
- [x] **Task 2.3**: Implement Traveling Salesperson constructive benchmark (`evokit/benchmarks/tsp/`)
  - Built `TSPBenchmarkTask` with 2D Euclidean coordinate instances, distance matrix generation, tour simulation, and Nearest-Neighbor baseline.
- [x] **Task 2.4**: Implement `SubprocessSandboxEvaluator` (`evokit/plugins/evaluators/subprocess_sandbox.py`)
  - Isolated subprocess execution with hard wall-clock timeouts, complete stdout/stderr capture, and clean process termination.
- [x] **Task 2.5**: Implement `MCPEvaluator` (`evokit/plugins/evaluators/mcp_evaluator.py`)
  - Model Context Protocol (MCP) JSON-RPC 2.0 tool-calling client (`tools/call` for `evaluate_algorithm`).
- [x] **Task 2.6**: Implement unit tests for Phase 2 (`tests/test_benchmarks.py`, `tests/test_evaluators.py`)
  - Verified benchmark simulation accuracy, train/test split isolation, timeout handling, and MCP payload formatting.

### Phase 3: The Population & Mutation Plugin Stack
- [x] **Task 3.1**: Implement Base Plugin Interfaces (`BasePopulationManager`, `BaseMutationStrategy`)
  - Defined parent sampling, archive admission, and code/thought regex extraction utilities.
- [x] **Task 3.2**: Implement `LineageGraphArchive` (`evokit/plugins/populations/lineage_archive.py`)
  - Preserved historical DAG of idea-to-code transitions, tournament selection, and Graphviz DOT graph export.
- [x] **Task 3.3**: Implement `ReflectionBuffer` (`evokit/plugins/populations/reflection_buffer.py`)
  - Dual-buffer architecture retaining execution failure logs, stderr traces, and short-term/long-term reflection memory.
- [x] **Task 3.4**: Implement `MAPElitesGrid` (`evokit/plugins/populations/map_elites.py`)
  - N-dimensional behavioral tessellation grid across code complexity and runtime metrics with cell coverage metrics.
- [x] **Task 3.5**: Implement `EoHMutationStrategy` (`evokit/plugins/mutations/eoh.py`)
  - Implemented E1/E2 exploration operators and M1/M2 parameter mutation operators with structured thought/code extraction.
- [x] **Task 3.6**: Implement `ReEvoMutationStrategy` (`evokit/plugins/mutations/reevo.py`)
  - Multi-step reflective prompt loop analyzing performance differentials and failure feedback before synthesizing offspring.
- [x] **Task 3.7**: Implement `DiffMutationStrategy` (`evokit/plugins/mutations/diff.py`)
  - Targeted patch-style heuristic modifications.
- [x] **Task 3.8**: Implement unit tests for Phase 3 (`tests/test_populations.py`, `tests/test_mutations.py`)
  - Validated archive admission, tournament selection, reflection storage, grid discretization, and operator prompt synthesis.

### Phase 4: Orchestration & Configuration Assembly
- [x] **Task 4.1**: Implement Declarative YAML Loader and Plugin Registry (`evokit/loader.py`)
  - Dynamic class resolution by alias or Python import path (`module:Class`) with environment variable interpolation.
- [x] **Task 4.2**: Implement Invariant `EvolutionEngine` (`evokit/engine.py`)
  - Coordinated generational lifecycle: Context construction -> Mutation -> Sandboxed Evaluation -> Relational & Artifact Storage -> State update -> Hold-out test generalization.
- [x] **Task 4.3**: Implement Dynamic CLI Entrypoint (`evokit/main.py`)
  - Implemented `run` (with `--dry-run`, `--vllm-url`, `--model`), `evaluate` (direct artifact evaluation), and `dashboard` commands with Rich formatting.
- [x] **Task 4.4**: Create Declarative YAML Configuration Presets (`evokit/configs/`)
  - `configs/eoh_obp.yaml` (EoH on Online Bin Packing).
  - `configs/reevo_tsp.yaml` (ReEvo on Constructive TSP).
  - `configs/map_elites_obp.yaml` (MAP-Elites on Online Bin Packing).
- [x] **Task 4.5**: Implement End-to-End Integration Tests (`tests/test_engine.py`)
  - Validated complete evolutionary cycle, seed evaluation, generational updates, and storage persistence.

### Phase 5: Validation & Ecosystem Integration
- [x] **Task 5.1**: Live Smoke Test on vLLM Server
  - Endpoint: `http://172.22.4.1:8000/v1`
  - Model: `mistralai/Devstral-Small-2-24B-Instruct-2512`
  - Verified live prompt inference, thought/code parsing, sandboxed evaluation, and test split evaluation (total runtime 40.47s, 1778 tokens).
- [x] **Task 5.2**: Validate Optuna Dashboard SQLite Compatibility
  - Verified `studies`, `trials`, `trial_values`, `trial_params`, and `trial_user_attributes` tables exist with valid trials.
  - ⚠️ **Correction (Phase 12):** this task only checked that tables/rows existed; it never launched `optuna-dashboard` against the DB. The dashboard in fact crashed on three real schema mismatches, fixed in **Phase 12** below.
- [x] **Task 5.3**: Author Comprehensive Documentation
  - Written `docs/plugin_guide.md` (plugin development guide for community extensions).
  - Written `README.md` (architecture, quick start, CLI commands, and visual schemas).
- [x] **Task 5.4**: Finalize Task Tracking Logs
  - Updated `docs/tasks/task_log.md` with complete implementation logs and test verification records.

### Phase 6: Documentation Polish & Package Accessibility
- [x] **Task 6.1**: Baseline Specification Preservation (`docs/spec.md`)
  - Preserved the full approved System Architecture Specification (v0.0.1) under `EvoKit/docs/spec.md`.
- [x] **Task 6.2**: TorchRL-Style Top-Level Documentation (`README.md`)
  - Restructured top-level documentation with TorchRL design philosophy, ASCII pipeline diagrams, comparison matrix, method-benchmark tutorials (`eoh-tsp`, `reevo-tsp`, `eoh-obp`, `reevo-obp`, `map_elites-obp`), and Optuna dashboard instructions.
- [x] **Task 6.3**: Per-Directory Descriptive README Coverage (`evokit/` hierarchy)
  - Created dedicated `README.md` documentation for every individual package and submodule across `evokit/`:
    - `evokit/README.md`: Package architecture, pillar breakdown, and import conventions.
    - `evokit/core/README.md`: Core abstractions, lifecycle contracts, and immutable contexts.
    - `evokit/benchmarks/README.md`: Contract interface and partition requirements.
    - `evokit/benchmarks/obp/README.md`: Online Bin Packing problem formulation, Weibull distribution, and L1 bound.
    - `evokit/benchmarks/tsp/README.md`: Traveling Salesperson 2D Euclidean problem formulation and greedy baselines.
    - `evokit/configs/README.md`: Declarative experiment configuration anatomy, schema breakdown, and execution reference.
    - `evokit/plugins/README.md`: Three-pillar plugin registry overview and discovery rules.
    - `evokit/plugins/populations/README.md`: Archive and selection algorithms (`LineageGraphArchive`, `ReflectionBuffer`, `MAPElitesGrid`).
    - `evokit/plugins/mutations/README.md`: Search strategies and genetic operators (`EoHMutationStrategy`, `ReEvoMutationStrategy`, `DiffMutationStrategy`).
    - `evokit/plugins/evaluators/README.md`: Isolated sandbox execution and MCP protocol client (`SubprocessSandboxEvaluator`, `MCPEvaluator`).

### Phase 7: Multi-Processing & Parallel Evaluation Infrastructure
- [x] **Task 7.1**: Implement Concurrent Batch Sandbox Evaluation (`BaseEvaluator.evaluate_batch`)
  - Added `evaluate_batch(individuals, task, split)` contract in `evokit/plugins/evaluators/base.py`.
- [x] **Task 7.2**: Parallel Subprocess Worker Pool in Sandbox Evaluator (`SubprocessSandboxEvaluator`)
  - Multi-worker parallel execution via `ThreadPoolExecutor(max_workers)` spawning concurrent isolated OS subprocesses on separate CPU cores without GIL bottlenecks.
  - Implemented fault isolation: individual syntax errors and timeouts (`TimeoutExpired`) are caught per-worker without crashing or stalling parallel batches.
- [x] **Task 7.3**: Engine Batch Parallelization (`EvolutionEngine`)
  - Integrated `evaluate_batch` across generational candidate loops and hold-out test split evaluation.
- [x] **Task 7.4**: Intra-Task Instance-Level Parallelization (`BenchmarkConfig.parallel_workers`)
  - Supported multi-instance concurrent simulation across CPU cores in `TSPBenchmarkTask` and `OBPBenchmarkTask` for computationally demanding heuristics (e.g. perturbation TSP with GLS).
- [x] **Task 7.5**: CLI and Configuration Support
  - Added `--max-workers` CLI argument to `evokit run`.
- [x] **Task 7.6**: Comprehensive Parallel Test Suite (`tests/test_parallel_evaluator.py`)
  - 4/4 new unit and integration tests passing; total test suite 22/22 passing.

### Phase 8: Optimization Direction Standardization (Native MINIMIZE)
- [x] **Task 8.1**: Standardize Benchmark Fitness & Direction to Native MINIMIZE
  - Switched `TSPBenchmarkTask` from negated fitness (`-tour_length`, `MAXIMIZE`) to native positive tour length (`fitness = avg_tour_length`, `direction = "MINIMIZE"`).
  - Switched `OBPBenchmarkTask` from negated fitness (`-avg_bins`, `MAXIMIZE`) to native positive bin count (`fitness = avg_bins`, `direction = "MINIMIZE"`).
- [x] **Task 8.2**: Update Declarative Preset Configurations
  - Updated all YAML presets (`eoh_obp.yaml`, `eoh_tsp.yaml`, `reevo_tsp.yaml`, `reevo_obp.yaml`, `map_elites_obp.yaml`) to use `direction: "MINIMIZE"`.
- [x] **Task 8.3**: Full Direction Validation across Population & Storage
  - Validated that `LineageGraphArchive`, `ReflectionBuffer`, and `MAPElitesGrid` accurately order elites ascendingly (`min`), replace candidates on lower fitness, and populate Optuna with `MINIMIZE` studies.
  - Added `test_lineage_graph_archive_minimize` to `tests/test_populations.py`.
  - Full suite: 23/23 tests passing.

### Phase 9: High-Throughput Concurrent LLM Gateway Sampling
- [x] **Task 9.1**: Implement Thread-Safe Batch LLM Sampling (`LLMGateway`)
  - Added `generate_batch(...)` and `chat_batch(...)` supporting concurrent multi-request dispatch via `ThreadPoolExecutor(max_parallel_requests)`.
  - Configured `httpx.Limits(max_connections=50, max_keepalive_connections=20)` with thread-safe token budget counting (`threading.Lock`).
  - Validated live continuous batching on vLLM server (`3 parallel requests completed in 0.44s total`, `n=3 single-request completed in 0.98s`).
- [x] **Task 9.2**: Parallelize Search & Mutation Operators
  - Updated `EoHMutationStrategy` to batch operator prompts and dispatch them in parallel to vLLM.
  - Updated `DiffMutationStrategy` to execute patch generations concurrently.
  - Updated `ReEvoMutationStrategy` to concurrently synthesize reflections and offspring candidates.
- [x] **Task 9.3**: Add Gateway Parallel Test Suite (`tests/test_parallel_gateway.py`)
  - 4/4 new tests passed; 27/27 total test suite passing.

### Phase 10: Installation & Inference Deployment Documentation
- [x] **Task 10.1**: Add Standalone `requirements.txt`
  - Created `requirements.txt` containing core runtime dependencies, optional Optuna dashboard extras, pytest development packages, and vLLM notes.
- [x] **Task 10.2**: Add Comprehensive Installation Section to `README.md`
  - Documented prerequisites, editable install (`pip install -e .`), `requirements.txt` install, optional extras (`pip install -e ".[dashboard,dev]"`), and CLI verification (`evokit --help`).
- [x] **Task 10.3**: Add Comprehensive vLLM Setup & Deployment Guide to `README.md`
  - Detailed pip and Docker deployment steps.
  - Documented production server startup commands (`python -m vllm.entrypoints.openai.api_server`) with parameter breakdown (`tensor-parallel-size`, `max-model-len`, `gpu-memory-utilization`).
  - Added curl verification endpoints and SSH port forwarding instructions for remote compute nodes.
  - Updated test pass badge to 27 tests passed.

### Phase 11: Method Expansion Architecture & Blueprints
- [x] **Task 11.1**: Author Method Expansion Taxonomy in `evokit/plugins/README.md`
  - Mapped core research paradigms (**OpenEvolve**, **MEoH**, **EoH-S**) directly onto the 3-pillar taxonomy (Pillar 1: Archive, Pillar 2: Mutation, Pillar 3: Sandbox).
- [x] **Task 11.2**: Write Comprehensive Implementation Recipes in `docs/plugin_guide.md`
  - **OpenEvolve Blueprint**: Quality-Diversity with `MAPElitesGrid` and `DiffMutationStrategy`.
  - **MEoH Blueprint**: Multi-objective Pareto frontier tracking with `ParetoArchive` (non-dominated sorting and dominance checking) and trade-off mutation prompting.
  - **EoH-S Blueprint**: Automated Heuristic Set Design with `HeuristicSetArchive` (portfolio minimum cost coverage $\min_{h \in S} \text{cost}(h, i)$) and hard-instance targeted prompt mutations.
- [x] **Task 11.3**: Update Top-Level `README.md`
  - Added method expansion blueprints table and links to the detailed recipe guide.

### Phase 12: Optuna Dashboard Schema Compatibility Fixes
- [x] **Task 12.1**: Diagnose dashboard crash on the live SQLite DB (`evokit/core/storage.py`)
  - `optuna-dashboard` failed immediately against EvoKit-written DBs. Three real schema mismatches vs. the installed Optuna's RDB schema were found by launching the dashboard (which Phase 5.2 never actually did).
- [x] **Task 12.2**: Fix `version_info` table
  - Added the missing `library_version` column and the single-row `CHECK (version_info_id=1)` constraint so Optuna's ORM query (`version_info.library_version`) succeeds.
- [x] **Task 12.3**: Add the `alembic_version` migration-tracking table
  - EvoKit hand-builds the schema and omitted `alembic_version`; `RDBStorage` reads the Alembic head revision and rejected the DB when it was absent. Now created and populated with the installed Optuna's head revision.
- [x] **Task 12.4**: Rename `trial_params` primary key
  - `trial_param_id` → `param_id` to match Optuna's model, fixing trial loading.
- [x] **Task 12.5**: Make the schema identifiers Optuna-version-aware
  - Added `_optuna_schema_identifiers()` deriving `schema_version`, `library_version`, and the Alembic head from the installed Optuna at runtime (safe fallbacks if absent), instead of hardcoding a single release.
- [x] **Task 12.6**: Fix in-memory DB handling
  - `StorageConfig(db_path=":memory:")` was `Path.resolve()`-d into a stray on-disk file named `:memory:`; now kept verbatim so SQLite opens a true in-memory DB.
- **Verification:** `optuna-dashboard` serves the live DB (HTTP 200, study + trials rendered) and `optuna.load_study(...)` loads it; 27/27 tests pass.

### Phase 13: TensorBoard Telemetry & Run Provenance
- [x] **Task 13.1**: Add an LLM request counter to the gateway (`evokit/core/gateway.py`)
  - Thread-safe `request_count` (incremented in mock and real success paths), exposed via `get_token_usage()["request_count"]`.
- [x] **Task 13.2**: Implement the multi-run TensorBoard logger (`evokit/core/tb_logger.py`)
  - Logs best-so-far objective into separate runs — `by_generation`, `by_llm_calls`, `by_fevals` (each using that counter as the step) plus a `scalars` run — giving a **switchable x-axis** in TensorBoard (objective vs. generations / LLM calls / function evaluations / wall-clock via STEP/RELATIVE/WALL). No-op if no SummaryWriter backend is installed.
- [x] **Task 13.3**: Instrument the engine (`evokit/engine.py`)
  - Tracks cumulative function evaluations, best-so-far incumbent (direction-aware), and per-generation telemetry; records a `trajectory` list; emits points after seed/initial-population and every generation.
- [x] **Task 13.4**: Run provenance (resolves "which run / when / settings / raw logs")
  - Per-run **experiment ID** (`<timestamp>_<study>_seed<seed>_<hash>`) namespaces each TensorBoard run directory so runs never collide.
  - A markdown **TEXT panel** (`run_metadata`) records experiment id, ISO timestamp, seed, model/provider/base_url, generations, candidates, operators, strategy, and the path to the per-run `terminal.txt`.
  - A per-run `terminal.txt` file-log handler is attached for debugging.
- [x] **Task 13.5**: CLI + config surface (`evokit/main.py`, `EngineConfig`)
  - `EngineConfig` gains `tb_log_dir`, `experiment_id`, `seed`, `run_metadata`; `evokit run` gains `--tb-log-dir` / `--no-tensorboard`; TensorBoard is on by default at `<db_dir>/tensorboard`.
- **Dependencies:** installed `tensorboard` + `tensorboardX`. Note: the env's `torch` is broken (missing `libtorch_cpu.dylib`), so `tensorboardX` is used as the SummaryWriter backend.
- **Verification:** event files confirmed with the three distinct step scales; dashboard serves all runs; TEXT panel + `terminal.txt` populated.

### Phase 14: BPP → OBP Rename & Faithful EoH-OBP Reproduction
- [x] **Task 14.1**: Full BPP → OBP rename across the codebase
  - Directory `evokit/benchmarks/bpp/` → `evokit/benchmarks/obp/`; `BPPBenchmarkTask` → `OBPBenchmarkTask`; `BPPTaskConfig` → `OBPTaskConfig`; `BPP_PROMPT_SPEC`/`INITIAL_BPP_CODE` → `OBP_*`; registry key `bpp_online` → `obp_online`; config files `*_bpp.yaml` → `*_obp.yaml`; all references in source, configs, docs, and tests. Upstream `references/` left untouched.
- [x] **Task 14.2**: Engine support for faithful EoH gen-0 and parent selection
  - `EngineConfig.initial_population_size`: generation 0 is an LLM-sampled initial population via EoH i1 cold-start (mirrors reference `fix_init_pop=False`), not a single baseline seed.
  - `EngineConfig.parent_selection_num`: draws a tournament-selected pool of size `p` via `population.sample_parents()` (EoH's `selection_num`), instead of exposing the full archive.
- [x] **Task 14.3**: Revise `examples/eoh_obp/` to a modular reproduction
  - `eoh_obp_reprod.py` is now a thin runner over the modular `EvolutionEngine` (via `assemble_engine`), faithfully mirroring `references/myirace/src/reprod/eoh_obp.py`: 25 Weibull instances, 5000 items, capacity 100, pop 20, selection_num 5, timeout 30 s, temperature 0.9, operators E1/E2/M1/M2, train-only trajectory.
  - Emits a reference-compatible `trajectory.json` (`gen_id`, `score`=mean bins, `gap`, `n_instances`, `used_budget`) plus `summary.json`, `args.yaml`, `terminal.txt`, the Optuna DB, and TensorBoard runs.
- [x] **Task 14.4**: 3-seed local reproduction script (`scripts/run_eoh_obp_3seeds.sh`)
  - Runs seeds 0/1/2, 10 generations, single evaluation core and single LLM call at a time (local overrides), against `http://localhost:8000/v1`.
- [x] **Task 14.5**: Correct stale configuration
  - All default vLLM base URLs point to `http://localhost:8000/v1`.
- **Verification:** 27/27 tests pass; mock smoke test confirms gen-0 LLM sampling, tournament selection, OBP naming, and reference-format trajectory output.

### Phase 15: ReEvo-OBP Example — Plug-in Mechanism Demonstration
- [x] **Task 15.1**: Add `examples/reevo_obp/` as a plug-in swap of `examples/eoh_obp/`
  - `reevo_obp_reprod.py` reuses the EoH example's `run_experiment` over the **same** `EvolutionEngine`, benchmark (`obp_online`), and sandbox evaluator; only the config's Pillar-1/Pillar-2 plugins change: `mutation: eoh → reevo`, `population: lineage_archive → reflection_buffer`. This exercises EvoKit's contract-driven friendly plug-in mechanism — a new search paradigm is a config swap, not a new script.
- [x] **Task 15.2**: Faithful ReEvo settings (`examples/reevo_obp/config.yaml`)
  - Mirrors `references/myirace/src/reprod/reevo_tsp_gls.py` / `references/reevo/reevo.py`: pop_size 10, init_pop_size 10, mutation_rate 0.5, temperature 1.0, budget by function evaluations; ReEvo loop = sample initial population → short-term reflection → reflection-guided crossover → long-term reflection → elitist mutation. OBP task/metric (25 instances, 5000 items, mean bins MINIMIZE, L1-bound gap) identical to the EoH example.
- [x] **Task 15.3**: 3-seed parallel script (`scripts/run_reevo_obp_3seeds.sh`)
  - Seeds 0/1/2 in parallel, single eval core + single LLM call per seed, per-seed study/DB, shared TensorBoard root (matching the EoH script).
- [x] **Task 15.4**: Fix remaining stale cluster URLs
  - Replaced `http://172.22.4.1:8000/v1` with `http://localhost:8000/v1` across the remaining presets (`eoh_tsp.yaml`, `reevo_tsp.yaml`, `reevo_obp.yaml`, `map_elites_obp.yaml`) and `evokit/configs/README.md`.
- **Verification:** mock smoke test confirms the plug-in swap runs end-to-end — ReEvo mutation (`Op: reevo_reflect`) + ReflectionBuffer population + OBP benchmark, with gap via the benchmark contract and reference-format trajectory output.

### Phase 16: OpenEvolve Plug-in & Example
- [x] **Task 16.1**: Port OpenEvolve's program database as a Pillar-1 plugin (`evokit/plugins/populations/openevolve_archive.py`)
  - `OpenEvolveArchive`: island model (num_islands) + per-island MAP-Elites feature grid ((complexity, diversity), feature_bins) + top-K archive + mixed-strategy parent selection (exploration/exploitation/random) + novelty rejection (line-level SequenceMatcher) + worst-first eviction (cell owners protected) + ring migration. Direction-aware (`_elite_fitness` normalises MINIMIZE/MAXIMIZE to "higher is better"). Faithful to `references/openevolve/openevolve/database.py` and `references/myirace/src/reprod/openevolve_tsp_gls.py`.
- [x] **Task 16.2**: Port OpenEvolve's generation step as a Pillar-2 plugin (`evokit/plugins/mutations/openevolve.py`)
  - `OpenEvolveMutationStrategy`: inspiration-rich prompt (parent + metrics + top programs + diverse inspirations + improvement areas); `diff-based` (SEARCH/REPLACE blocks) or `full-rewrite` edit modes; fitness-weighted parent + inspiration selection; LLM-sampled cold-start initial population.
- [x] **Task 16.3**: Generic end-of-generation population hook
  - Added `BasePopulationManager.on_generation_end(generation)` (no-op default); the engine invokes it each generation; `OpenEvolveArchive` overrides it for periodic ring migration — keeping the invariant engine free of OpenEvolve specifics.
- [x] **Task 16.4**: Register plugins (`evokit/loader.py`)
  - `population: openevolve_archive`, `mutation: openevolve`.
- [x] **Task 16.5**: Add `examples/openevolve_obp/`
  - `config.yaml` (OpenEvolve DatabaseConfig defaults on the OBP task) + `openevolve_obp_reprod.py` (thin runner reusing the EoH example's `run_experiment` — same engine, swapped pillars) + `scripts/run_openevolve_obp_3seeds.sh` (3-seed parallel, per-seed DB, shared TensorBoard root).
- **Verification:** 27/27 tests pass; unit checks confirm per-island novelty rejection, direction-aware elite fitness, mixed-strategy sampling, and migration; mock smoke test runs the example end-to-end (`openevolve_init` → `openevolve_diff-based`, islands + MAP-Elites + migration, reference-format trajectory with gap via the benchmark contract).

### Phase 17: README Restructure, NEWS & Positioning
- [x] **Task 17.1**: Slim the top-level `README.md` (540 → ~150 lines)
  - Kept overview, a "Why EvoKit?" comparison, NEWS, install, quickstart, the 3-pillar table, and a documentation index. Moved deep detail into focused sub-files under `evokit/docs/`: `setup.md` (install + vLLM), `tutorials.md` (presets + reproductions), `cli.md` (CLI + parallelism), `telemetry.md` (Optuna + TensorBoard + artifacts), `extending.md` (custom plugins + blueprints).
- [x] **Task 17.2**: Add a timestamped NEWS section
  - Notes the repo is newly launched (2026-10-11); methods done as checkboxes (OpenEvolve, EoH, ReEvo) and roadmap items (LLaMEA, HSEvo, EoH-S, MCTS-AHD).
- [x] **Task 17.3**: Add an OpenEvolve vs. LLM4AD vs. EvoKit comparison table
  - Near the top; emphasises that OpenEvolve hard-wires one QD algorithm and LLM4AD couples methods to tasks, versus EvoKit's invariant engine + three independently swappable pillars (add a method = one Pillar-2 plugin named in YAML; any method × any benchmark; swappable sandbox; built-in Optuna/TensorBoard telemetry; enforced held-out test split).
- **Verification:** quickstart Python API confirmed working; all README documentation links resolve to existing files.

### Phase 18: Development Roadmap Doc
- [x] **Task 18.1**: Author `docs/ROADMAP.md`
  - High-level, forward-looking "what's done / to-do" for contributors: done items across engine, 3 pillars, benchmarks, infra, examples/tests; to-do covering new methods (LLaMEA, HSEvo, EoH-S, MCTS-AHD, MEoH), benchmarks (TSP-GLS, more LLM4AD tasks), engine fidelity (max_fe budget, island-scoped parent routing, resume/checkpoint, async eval), infra (LLM cache, more providers, Docker/WASM sandbox, extraction hardening), analysis/reporting, and project hygiene (CI, packaging, CONTRIBUTING). Includes a "how to pick up a task" guide. Linked from the README documentation index.

### Phase 19: LLM Backend Setup Docs (vLLM self-host + cloud API keys)
- [x] **Task 19.1**: Document both gateway backends in `evokit/docs/setup.md`
  - Added an "LLM Backends" overview (self-hosted vLLM / cloud API / mock) and relabelled the existing vLLM steps as "Option A — Self-hosted vLLM".
- [x] **Task 19.2**: Add "Option B — Cloud API providers (OpenAI, OpenRouter, …)"
  - Export the API key as an env var; wire it into the `gateway` block with `${VAR}` interpolation (loader-supported); OpenAI and OpenRouter YAML examples (OpenRouter via the OpenAI protocol); verification + cost/rate-limit guidance. Verified `${OPENAI_API_KEY}` interpolation and `provider: "openai"` config build.
- [x] **Task 19.3**: Update README pointers to mention self-host + cloud API keys.

### Phase 20: Environment Configuration (.env)
- [x] **Task 20.1**: Add `.env.example` (tracked template) and `.env` (gitignored local)
  - Document the real variables used by the codebase: `VLLM_BASE_URL`, `LLM_MODEL` (read by the run scripts), and cloud keys `OPENAI_API_KEY` / `OPENROUTER_API_KEY` (referenced from configs via `${VAR}` interpolation). `.gitignore` already ignores `.env`/`*.env` and allows `.env.example`.
- [x] **Task 20.2**: Auto-load `.env` in the 3-seed run scripts
  - Each `scripts/run_*_3seeds.sh` sources `$EVOKIT_DIR/.env` (`set -a; . .env; set +a`) if present, so VLLM_BASE_URL/LLM_MODEL/keys flow through without manual export. Verified sourcing picks up values.
- [x] **Task 20.3**: Document the `.env` workflow in `evokit/docs/setup.md` (copy template, gitignored, auto-loaded by scripts, `set -a; source .env` for other entrypoints).

### Phase 21: docs/ Restructure (community vs. development)
- [x] **Task 21.1**: Split `docs/` by audience
  - **`docs/tutorials/`** (community, step-by-step): `setup.md`, `methods.md` (was `tutorials.md`), `cli.md`, `telemetry.md`, `extending.md`, `plugin_guide.md`, `spec.md` — the former `evokit/docs/*` guides plus the moved `spec.md`/`plugin_guide.md`. `evokit/docs/` removed.
  - **`docs/dev/`** (maintainers): `roadmap.md` (was `docs/ROADMAP.md`), `changelog.md` (this file, was `docs/tasks/task_log.md`), and a new `bugfixes.md` (consolidated defects + known issues).
- [x] **Task 21.2**: Index READMEs for both folders (`docs/tutorials/README.md`, `docs/dev/README.md`) with reading order / workflow.
- [x] **Task 21.3**: Repoint all cross-links
  - Fixed relative links in the moved tutorial files, the dev docs, the main README (split into a "Tutorials" table and a "Development notes" table), and `evokit/plugins/README.md`. Historical path mentions in earlier changelog phases are left as-is for accuracy.

### Phase 22: MIT License & Results GIF
- [x] **Task 22.1**: Add `LICENSE` (MIT, "2026 EvoKit Contributors"); declare `license = {text="MIT"}` + the OSI classifier in `pyproject.toml` so metadata matches the README badge.
- [x] **Task 22.2**: Add a "See it in action" section near the top of the README showing `assets/tensorboard_demo.gif` (TensorBoard results demo, 820x426, ~0.5MB) with a link to the telemetry guide. Added `scripts/make_demo_gif.sh` (ffmpeg palette conversion); GIF generated from the recording of the live EoH/ReEvo OBP runs.
