# `evokit.configs`: Declarative Execution Configurations

This directory contains declarative YAML execution specifications for configuring complete evolutionary runs in EvoKit.

---

## Configuration Schema

Every configuration file defines seven distinct component blocks:

```yaml
experiment:
  study_name: "my_experiment"             # Study name for relational DB and artifacts
  max_generations: 5                      # Number of evolutionary generations
  candidates_per_generation: 2            # Offspring synthesized per generation
  evaluate_test_split: true               # Run held-out test split evaluation at end
  test_top_k: 1                           # Number of top elites to test
  seed_evaluation: true                   # Evaluate generation 0 seed baseline

storage:
  db_path: "./artifacts/exp/experiments.db" # Optuna-compatible SQLite database
  artifacts_dir: "./artifacts/exp/code"     # Filesystem directory for .py code files
  direction: "MINIMIZE"                     # MAXIMIZE or MINIMIZE

gateway:
  provider: "vllm"                        # vllm, openai, anthropic, or mock
  base_url: "http://172.22.4.1:8000/v1"   # Inference endpoint
  model: "mistralai/Devstral-Small-2-24B-Instruct-2512"
  temperature: 0.8
  max_tokens: 2048
  timeout: 120.0

benchmark:
  type: "bpp_online"                      # bpp_online or tsp_construct
  # Problem-specific parameters...

population:
  type: "lineage_archive"                 # lineage_archive, reflection_buffer, map_elites
  capacity: 10
  direction: "MINIMIZE"

mutation:
  type: "eoh"                             # eoh, reevo, or diff
  temperature: 0.8

evaluator:
  type: "subprocess_sandbox"              # subprocess_sandbox or mcp
  timeout_seconds: 30.0
```

---

## Bundled Presets

| Config File | Strategy | Problem Benchmark | Population Manager |
|---|---|---|---|
| `eoh_bpp.yaml` | EoH (E1/E2 & M1/M2) | Online Bin Packing (`bpp_online`) | Lineage Graph Archive |
| `eoh_tsp.yaml` | EoH (E1/E2 & M1/M2) | Constructive TSP (`tsp_construct`) | Lineage Graph Archive |
| `reevo_tsp.yaml` | ReEvo (Reflective Loop) | Constructive TSP (`tsp_construct`) | Reflection Buffer |
| `reevo_bpp.yaml` | ReEvo (Reflective Loop) | Online Bin Packing (`bpp_online`) | Reflection Buffer |
| `map_elites_bpp.yaml` | Diff Mutations | Online Bin Packing (`bpp_online`) | MAP-Elites Grid |

---

## Environment Variable Expansion & CLI Overrides

EvoKit's loader automatically interpolates environment variables:
```yaml
gateway:
  base_url: "${VLLM_URL:-http://172.22.4.1:8000/v1}"
  api_key: "${OPENAI_API_KEY:-EMPTY}"
```

Parameters can also be dynamically overridden from the CLI:
```bash
python evokit/main.py run -c evokit/configs/eoh_bpp.yaml \
  --vllm-url http://172.22.4.1:8000/v1 \
  --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
  -g 10
```
