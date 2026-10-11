# Telemetry: Optuna Dashboard, TensorBoard & Artifacts

[← Back to README](../../README.md)

EvoKit uses dual-tier persistence: an Optuna-compatible SQLite DB (metrics/metadata) plus
standalone `.py` code artifacts on disk, and optional TensorBoard scalar logs.

## Optuna Dashboard

`ExperimentStorage` writes a native SQLite schema compatible with
[Optuna Dashboard](https://github.com/optuna/optuna-dashboard). The schema version, library
version, and Alembic migration head are derived from the **installed Optuna at runtime**
(see `_optuna_schema_identifiers()` in [`evokit/core/storage.py`](../../evokit/core/storage.py)), so the
DB matches whatever dashboard you have.

```bash
optuna-dashboard sqlite:///artifacts/eoh_obp/experiments.db --port 8080
```

Open `http://127.0.0.1:8080/dashboard` to inspect generational objective trajectories, trial
states (`COMPLETE` vs. `FAIL`), and per-trial user attributes (`individual_id`, `operator`,
`metrics`, `artifact_path`, `thought_preview`).

> **Tip:** use `127.0.0.1`, not `localhost`, to avoid IPv6 resolution landing on an unrelated
> listener on the same port.

## TensorBoard (configurable x-axis)

Each `evokit run` writes TensorBoard logs (default `<db_dir>/tensorboard`; disable with
`--no-tensorboard`, relocate with `--tb-log-dir`). Runs are namespaced by a unique
**experiment ID** (`<timestamp>_<study>_seed<seed>_<hash>`) so they never collide.

```bash
tensorboard --logdir artifacts/eoh_obp/tensorboard --port 6006
```

The best-so-far objective is logged into sub-runs that each use a different progress counter
as the step, giving a **switchable x-axis**:

| Sub-run | X-axis |
|---|---|
| `by_generation` | generation index |
| `by_llm_calls` | cumulative LLM calls |
| `by_fevals` | cumulative function evaluations |
| *(any run)* | wall-clock — TensorBoard's `STEP` / `RELATIVE` / `WALL` toggle |

Toggle the active sub-run in the **Runs** panel to switch the x-axis. The **TEXT** tab shows a
`run_metadata` panel (experiment id, timestamp, seed, model, settings, and the path to the
per-run `terminal.txt`).

> **Backend:** uses `torch.utils.tensorboard` or, as a torch-free fallback, `tensorboardX`
> (`pip install tensorboardX`). Logging is a no-op if neither is installed.

## Physical code artifacts

Every generated algorithm is written as a standalone, runnable script under
`artifacts/{study_name}/code/{study_name}/gen_{NNN}/{individual_id}.py`, with a metadata
header:

```python
# EvoKit Generated Algorithm
# Individual ID: ind_636ac2ea774a
# Generation: 1
# Operator: eoh_m1
# Fitness: 68.6
# Valid: True
# Parents: ['ind_c34cc121328a']
# Metrics: {"avg_bins": 68.6, "avg_l1_ratio": 0.9126, "split": "train"}
# Thought: "Idea: Modified Best-Fit heuristic favoring more efficient packing."

import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    remaining = bins - item
    return -(remaining / 100.0)
```
