# Bug Fixes & Known Issues

Developer record of defects found and fixed (and issues still open). Detailed phase context
is in [changelog.md](changelog.md).

_Last updated: 2026-10-11._

---

## Fixed

### Optuna Dashboard schema incompatibility (Phase 12)
**Symptom:** `optuna-dashboard` crashed immediately on an EvoKit-written SQLite DB; the earlier
"dashboard validated" claim had only checked that tables/rows existed, never launched the
dashboard.
**Root causes & fixes** (`evokit/core/storage.py`):
- `version_info` table was missing the `library_version` column the dashboard queries → added it (+ single-row `CHECK (version_info_id=1)`).
- No `alembic_version` migration-tracking table → `RDBStorage` rejected the DB → created and populated it with the installed Optuna's head revision.
- `trial_params` primary key was `trial_param_id` instead of Optuna's `param_id` → renamed.
- Hardcoded schema/version constants → replaced with `_optuna_schema_identifiers()` which derives schema version, library version, and Alembic head from the installed Optuna at runtime.

### Stray `:memory:` file (Phase 12)
**Symptom:** a file literally named `:memory:` appeared in the repo root.
**Cause:** `StorageConfig(db_path=":memory:")` was `Path(...).resolve()`-d into an on-disk path.
**Fix:** keep `:memory:` verbatim and skip the parent-dir `mkdir`, so SQLite opens a true
in-memory DB.

### Benchmark-specific `gap` leaked into the invariant engine (post-Phase 13)
**Symptom:** the engine computed the optimality `gap` by reading `l1_bound` off
`benchmark.train_instances` — a field only OBP has. Any other benchmark (TSP, future tasks)
would silently get a wrong/None gap.
**Fix:** added an optional `BaseBenchmarkTask.reference_bound(split)` contract (default `None`);
OBP overrides it to return the mean L1 bound; the engine annotates `gap` only when it is
non-None. The invariant engine no longer knows any per-benchmark metric.

### Stale vLLM cluster URLs (Phases 12, 14, 15)
**Symptom:** configs/examples pointed at a private cluster IP (`172.22.*`), breaking local runs.
**Fix:** standardised every default to `http://localhost:8000/v1`.

### Example/runner cleanups (Phase 14)
- `examples/eoh_obp` was a stale standalone loop; rewritten as a thin runner over the modular
  `EvolutionEngine`, with `config.yaml` actually loaded, dry-run provider correctly reported,
  and tournament parent selection wired through (`selection_num` was previously dead).

---

### ReEvo poor convergence on OBP (Phase 23)
**Symptom:** on OBP, ReEvo barely improved over 10 generations (2 of 3 seeds flat at the
Best-Fit baseline), while EoH improved on all seeds under identical settings.
**Root causes & fixes:**
- **Malformed-code failures (~50–70% invalid).** Two sub-causes:
  (a) the LLM dropped the literal `{thought}` placeholder / brace-wrapped prose *inside* the
  function body → `name 'thought' is not defined` / syntax errors; fixed by
  `BaseMutationStrategy._sanitize_code()` (strips leaked placeholder/prose lines while
  preserving real dict/set literals) — benefits all strategies.
  (b) the LLM changed the `priority(...)` signature (added params) → "missing argument" /
  "Expected 2 priority values"; fixed by pinning the exact signature and forbidding
  body-braces in the ReEvo offspring prompt.
- **Missing long-term reflection memory.** The strategy generated a fresh pairwise reflection
  per candidate but never accumulated ReEvo's cross-generational "long-term insights", so it
  behaved like isolated rewrites. Fixed: the (persistent) strategy now maintains a
  `long_term_reflection`, folds each generation's short-term reflections into it, and feeds the
  accumulated insights into subsequent offspring prompts.

**Second-pass fidelity audit vs `references/reevo/reevo.py`** found three more deviations, all fixed:
- **Non-distinct parent pairing.** EvoKit paired any two parents; the reference's `random_select`
  requires two parents with *distinct* objectives (reflecting between equal-scoring programs is
  meaningless, and common when a strong baseline dominates). Fixed with `_select_distinct_pair`.
- **No separate elitist-mutation operator.** The reference's loop is reflective **crossover** on
  pairs *then* **elitist mutation** (mutate the single best, guided by the long-term reflection),
  producing `pop_size * mutation_rate` offspring. EvoKit only did crossover. Added
  `_mutate_elitist` + `reevo_mutate` operator; `mutate()` now splits the budget by `mutation_rate`
  and runs crossover → long-term-reflect → elitist-mutate in one call, matching `evolve()`.
- **No `filter_code`.** The reference strips signature/import lines from parent snippets shown in
  prompts. Added `_filter_code`, applied to parent code in the reflection and mutation prompts.

---

### EoH fidelity audit vs `llm4ad/method/eoh` (Phase 24)
**Fixed:**
- **Operator application.** Reference EoH applies *all* enabled operators (E1→E2→M1→M2) each
  step; EvoKit picked one operator at random by weight. Added `cycle_operators` (default on):
  offspring cycle round-robin through the enabled operators across a generation.
- **E1/E2 parent count.** Reference uses `selection_num` parents for E1/E2; EvoKit hardcoded 2.
  Added `EoHConfig.selection_num` (default 5); M1/M2 keep 1 parent.

**Known approximations (not changed):**
- **Parent selection distribution.** Reference `population.selection()` is rank-based
  (`p ∝ 1/(rank+N)`); EvoKit samples uniformly from the `population_view` the engine supplies,
  which is already a tournament-selected pool (`parent_selection_num`). Close but not identical.
- **Score-based dedup.** Reference marks any function sharing an existing score as `-inf`
  (aggressive novelty-by-score). EvoKit's `LineageGraphArchive` does not.
- **Prompt wording.** Behaviorally equivalent (boxed-`{}` thought + function template) but not
  byte-identical to the reference templates.

---

### OpenEvolve island model bypassed (Phase 25)
**Symptom (fidelity):** the island model existed in the Pillar-1 archive but did not affect
search — `OpenEvolveMutationStrategy` selected the parent and inspirations globally
(fitness-weighted) from the flat `population_view`, ignoring islands entirely.
**Root causes & fixes (vs `references/openevolve/openevolve/database.py`):**
- **Global vs island-scoped selection.** Now the strategy delegates to the archive, which does
  `current_island` rotation (`advance_island` = reference `next_island`) and island-scoped
  mixed-strategy parent sampling.
- **Fitness-weighted vs uniform.** Reference uses `random.choice` (uniform) for exploration
  (current island), exploitation (archive, preferring current island), and random (all). EvoKit
  was fitness-weighting; switched to uniform to match.
- **Global vs island-only inspirations.** Reference `_sample_inspirations` draws ONLY from the
  parent's island plus that island's best. EvoKit used the global top/best; fixed.
- **Migration semantics.** Reference migrates a COPY (new id, `migrant` flag, skip-duplicate-code
  in target, register into target MAP-Elites). EvoKit previously just added the original's id to
  the neighbor island set; fixed to copy+flag+skip-duplicate.
- Engine now exposes the live population manager via `EvolutionContext.metadata["population_manager"]`
  so a strategy can opt into manager-specific selection.

**Known approximations (not changed):**
- **Migration trigger.** Reference uses per-island generation counters
  (`max(island_generations) - last_migration >= interval`); EvoKit triggers on the engine's global
  generation index. Equivalent cadence for the synchronous generational loop.
- **No MVCC/snapshot transaction layer or embedding-based novelty.** The reference's full
  `ProgramDatabase` has an MVCC commit/abort layer and optional LLM/embedding novelty; EvoKit uses
  a simpler locked store with line-level SequenceMatcher novelty.

---

## Known issues / open

- **LLM output extraction is fragile.** `extract_thought` grabs the first `{...}`, which can
  collide with literal braces in code; code extraction can mis-handle malformed responses.
  Tracked in [roadmap.md](roadmap.md) → "harden extraction".
- **EoH `M3` operator not implemented.** The spec lists E1/E2 + M1/M2/M3; only M1/M2 exist
  (matches the reference reproduction, which also enables only E1/E2/M1/M2).
- **OpenEvolve parent routing is approximated.** Island-scoped per-offspring parent selection
  from the reference is approximated by fitness-weighted selection over the full population
  view; the island/MAP-Elites/migration machinery itself is faithful. Tracked in the roadmap.
- **`GatewayConfig.provider` docstring lists `anthropic`** but all non-mock providers route
  through the OpenAI-compatible path; native Anthropic/Gemini APIs are not yet supported.

---

## Reporting a bug

Open an issue (or add an entry here during development) with: the command/config, the expected
vs. actual behaviour, and the relevant `terminal.txt` from the run's log directory.
