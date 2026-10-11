# Command-Line Interface & Parallelism

[← Back to README](../../README.md)

The dynamic CLI (`evokit` or `python evokit/main.py`) runs searches, evaluates standalone
artifacts, and inspects telemetry.

## `evokit run`

```bash
# Basic
evokit run -c evokit/configs/eoh_obp.yaml

# Override endpoint / model / generations / workers
evokit run -c evokit/configs/eoh_obp.yaml \
  --vllm-url http://localhost:8000/v1 \
  --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
  -g 10 --max-workers 4 --instance-workers 8

# Zero-cost dry run (deterministic mock LLM)
evokit run -c evokit/configs/eoh_obp.yaml --dry-run

# TensorBoard logging controls
evokit run -c evokit/configs/eoh_obp.yaml --tb-log-dir ./tb        # custom dir
evokit run -c evokit/configs/eoh_obp.yaml --no-tensorboard          # disable
```

## `evokit evaluate`

Evaluate a single code artifact against a benchmark without the search loop:

```bash
evokit evaluate \
  --code artifacts/eoh_obp/code/eoh_obp_online/gen_001/ind_636ac2ea774a.py \
  --benchmark obp_online \
  --split test --timeout 30.0 --instance-workers 4
```

## `evokit dashboard`

Print the Optuna SQLite connection string and launch instructions:

```bash
evokit dashboard --db ./artifacts/eoh_obp/experiments.db
```

---

## Parallel Execution (two tiers)

```
Tier 1 — Inter-candidate concurrency (--max-workers)
         Multiple synthesized candidates evaluated in parallel isolated subprocesses.

Tier 2 — Intra-task instance concurrency (--instance-workers)
         Problem instances distributed across CPU cores within one candidate's evaluation.
```

Configure in YAML:

```yaml
evaluator:
  max_workers: 4        # Tier 1
benchmark:
  parallel_workers: 8   # Tier 2
```

or via CLI:

```bash
evokit run -c evokit/configs/reevo_tsp.yaml --max-workers 2 --instance-workers 8
```

LLM request concurrency is separate: `gateway.max_parallel_requests` dispatches that many
prompts to vLLM at once (vLLM batches them server-side).
