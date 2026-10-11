# Method–Benchmark Tutorials

[← Back to README](../../README.md)

EvoKit ships declarative YAML presets in [`evokit/configs/`](../../evokit/configs/) and faithful,
reference-matching reproductions under [`examples/`](../../examples/).

Run any preset with the CLI:

```bash
evokit run -c evokit/configs/<preset>.yaml           # live run
evokit run -c evokit/configs/<preset>.yaml --dry-run  # zero-cost mock run
```

## Presets

| Preset | Method | Benchmark | Pillars (population / mutation) |
|---|---|---|---|
| `eoh_obp.yaml` | EoH | Online Bin Packing | `lineage_archive` / `eoh` |
| `eoh_tsp.yaml` | EoH | TSP (constructive) | `lineage_archive` / `eoh` |
| `reevo_obp.yaml` | ReEvo | Online Bin Packing | `reflection_buffer` / `reevo` |
| `reevo_tsp.yaml` | ReEvo | TSP (constructive) | `reflection_buffer` / `reevo` |
| `map_elites_obp.yaml` | MAP-Elites | Online Bin Packing | `map_elites` / `diff` |

- **EoH** — Evolution of Heuristics: dual Thought–Code operators (E1 cross-parent, E2 backbone, M1 form, M2 parameter).
- **ReEvo** — Reflective Evolution: short/long-term reflections over contrasting algorithms guide offspring synthesis.
- **OpenEvolve** — island model + per-island MAP-Elites + archive + mixed-strategy selection + novelty + ring migration (`openevolve_archive` / `openevolve`).
- **MAP-Elites** — behavioural tessellation across complexity vs. runtime, paired with diff mutations.

## Faithful reproductions (`examples/`)

Each example is a thin runner over the **same** modular `EvolutionEngine` — only the two
pillar plugins differ between them, which is the point of the plug-in design. They mirror
the reference reproductions and emit a reference-compatible `trajectory.json` plus the full
Optuna + TensorBoard telemetry.

| Example | Reference | 3-seed script |
|---|---|---|
| `examples/eoh_obp/` | `references/myirace/src/reprod/eoh_obp.py` | `scripts/run_eoh_obp_3seeds.sh` |
| `examples/reevo_obp/` | `references/myirace/src/reprod/reevo_tsp_gls.py` | `scripts/run_reevo_obp_3seeds.sh` |
| `examples/openevolve_obp/` | `references/openevolve/` + `.../openevolve_tsp_gls.py` | `scripts/run_openevolve_obp_3seeds.sh` |

The OBP reproductions use 25 Weibull instances, 5000 items, capacity 100, generation 0 as an
LLM-sampled initial population, and report `score` = mean bins (MINIMIZE) with
`gap = (score − L1_bound) / L1_bound`.

```bash
# 3 seeds (0,1,2) in parallel, single eval core + single LLM call per seed:
bash scripts/run_eoh_obp_3seeds.sh
```

Per-run logs land in `examples/<method>_obp/logs/<experiment_id>/`
(`trajectory.json`, `summary.json`, `args.yaml`, `terminal.txt`), with per-seed Optuna DBs
and a shared TensorBoard root under `artifacts/<method>_obp/`.
