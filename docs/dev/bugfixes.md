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
