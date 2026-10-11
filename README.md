# EvoKit: Modular Framework for LLM-Guided Algorithm Design

[![Python version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Unit-tests](https://img.shields.io/badge/tests-27%20passed-brightgreen.svg)](tests/)
[![Dashboard](https://img.shields.io/badge/dashboard-Optuna-orange.svg)](https://optuna.org/)
[![vLLM](https://img.shields.io/badge/inference-vLLM%20%7C%20Cloud-purple.svg)](https://github.com/vllm-project/vllm)

**EvoKit** is an open, production-grade toolkit for Automated Heuristic Design (AHD) and LLM-guided program synthesis—analogous to what **PyTorch** provides for deep learning and **TorchRL** provides for reinforcement learning.

Current academic repositories (e.g. EoH, ReEvo, OpenEvolve, LLM4AD) bundle search strategies, execution sandboxes, and problem benchmarks into tight, monolithic scripts. EvoKit decouples these concerns into pluggable, contract-driven components, providing an invariant evolution engine that coordinates population management, mutation operators, isolated sandboxed evaluators, and dual-tier persistence.

---

## Navigation

[Installation](#installation) |
[vLLM Setup Guide](#vllm-setup--deployment-guide) |
[Core Mental Model](#a-quick-mental-model) |
[Quickstart: Python API](#quickstart-python-api) |
[Method-Benchmark Tutorials](#method-benchmark-tutorials) |
[CLI Workflow](#command-line-interface) |
[Parallel & Multi-Processing](#parallel-execution--multi-processing) |
[Optuna Dashboard](#real-time-optuna-dashboard-visualization) |
[Extending EvoKit](#extending-evokit-with-custom-plugins) |
[Architecture Spec](docs/spec.md) |
[Plugin Guide](docs/plugin_guide.md)

---

## Installation

### Prerequisites
- **Python**: Version 3.10 or higher.
- **Operating System**: Linux (recommended for robust subprocess isolation and timeout enforcement) or macOS.

### 1. Install from Source (Editable Mode)

Clone the repository and install EvoKit in editable mode:

```bash
git clone https://github.com/taindp98/myirace.git
cd myirace/EvoKit

# Install in editable mode
pip install -e .
```

### 2. Install Dependencies via `requirements.txt`

Alternatively, you can install dependencies directly using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 3. Optional Extras (Dashboard & Developer Tools)

To install Optuna Dashboard for real-time visualization and development testing packages:

```bash
# Optuna Dashboard and testing tools:
pip install -e ".[dashboard,dev]"
```

### 4. Verify Installation

Verify that the `evokit` CLI entrypoint is correctly registered:

```bash
evokit --help
```

---

## vLLM Setup & Deployment Guide

EvoKit is engineered for high-throughput local LLM inference via **[vLLM](https://github.com/vllm-project/vllm)**. vLLM uses PagedAttention and continuous batching, allowing multiple concurrent algorithm mutation requests to be processed simultaneously on your GPU.

### Step 1: Install vLLM on the GPU Host

On your GPU-enabled machine or compute node (requires CUDA 12.1+):

```bash
pip install vllm
```

*Or run directly using the official Docker image:*
```bash
docker pull vllm/vllm-openai:latest
```

### Step 2: Launch the OpenAI-Compatible API Server

Start the vLLM server with your chosen model (e.g. `mistralai/Devstral-Small-2-24B-Instruct-2512` or similar coding LLM):

```bash
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code
```

> **Key Server Flags:**
> - `--host 0.0.0.0`: Binds to all network interfaces so worker nodes can connect.
> - `--port 8000`: Default listening port.
> - `--tensor-parallel-size <N>`: Set to the number of GPUs to shard across (e.g. `2` or `4` for multi-GPU setups).
> - `--max-model-len`: Maximum context window (ensure sufficient length for long code prompts).
> - `--gpu-memory-utilization`: Fraction of GPU VRAM dedicated to model weights and KV cache.

*Docker Alternative:*
```bash
docker run --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
    --max-model-len 8192
```

### Step 3: Verify the vLLM Endpoint

Test that the endpoint is responsive and serving completions:

```bash
# Check available models
curl http://localhost:8000/v1/models

# Test a completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Devstral-Small-2-24B-Instruct-2512",
    "messages": [{"role": "user", "content": "Write a python function returning 42."}],
    "max_tokens": 50
  }'
```

### Step 4: Configure EvoKit to Connect to vLLM

In your experiment YAML configuration (e.g. `evokit/configs/eoh_bpp.yaml`):

```yaml
gateway:
  provider: "vllm"
  base_url: "http://172.22.4.1:8000/v1"   # Point to your vLLM server IP/hostname
  model: "mistralai/Devstral-Small-2-24B-Instruct-2512"
  max_parallel_requests: 8               # Dispatches up to 8 candidate prompts concurrently
  temperature: 0.8
  max_tokens: 4096
  timeout: 120.0
```

Or override dynamically via the CLI:
```bash
evokit run -c evokit/configs/eoh_bpp.yaml \
  --vllm-url http://localhost:8000/v1 \
  --model mistralai/Devstral-Small-2-24B-Instruct-2512
```

> **Remote Cluster / SSH Tip**: If your vLLM server is running on a private remote compute node, forward the port to your local machine:
> ```bash
> ssh -L 8000:localhost:8000 user@gpu-cluster-node
> ```

---

## Core Philosophy

EvoKit is built around four central tenets:

1. **Invariant Core Engine**: The evolutionary lifecycle, telemetry synchronization, and artifact management remain invariant across diverse search paradigms (EoH, ReEvo, MAP-Elites, Diff mutations).
2. **3-Pillar Separation**: Complete operational independence between **Population Management** (Pillar 1), **Mutation Strategies** (Pillar 2), and **Isolated Evaluation Services** (Pillar 3).
3. **Data Contracts as First-Class Entities**: Candidate programs are encapsulated as structured `Individual` entities carrying unique IDs, code, reasoning tokens (`thought`), scalar fitness, multidimensional metrics, and parentage DAG edges.
4. **Zero-Leakage Generalization**: Every benchmark strictly segregates **Training Instances** (used during search) from held-out **Testing Instances** (evaluated post-evolution to measure true out-of-distribution generalization).

---

## A Quick Mental Model

Similar to how TorchRL passes `TensorDict` instances across environments, collectors, replay buffers, and loss modules, EvoKit passes structured `Individual` and `EvolutionContext` instances across pluggable components:

```text
EvolutionContext (immutable generation state, population snapshot, task prompt)
  │
  ▼
Mutation Strategy (EoH, ReEvo, Diff) writes thoughts and code
  │
  ▼
Physical Storage persists .py source artifact with metadata header
  │
  ▼
Evaluator Sandbox (Subprocess / MCP) executes unverified code with timeout/memory limits
  │
  ▼
MetricRecord captures scalar fitness, validity, runtime, and diagnostic traces
  │
  ▼
Population Manager (Lineage DAG, Reflection Buffer, MAP-Elites) admits / selects elites
  │
  ▼
Relational Storage updates SQLite database (Optuna-compatible schema)
```

---

## Component Matrix

| Component Type | Included Implementations | Key Characteristics |
|---|---|---|
| **Population Management** | `LineageGraphArchive`<br>`ReflectionBuffer`<br>`MAPElitesGrid` | • Lineage DAG idea-to-code tracking & tournament selection<br>• Execution failure log & short/long-term reflection memory<br>• N-dimensional behavioral tessellation grid across complexity & runtime |
| **Mutation Strategies** | `EoHMutationStrategy`<br>`ReEvoMutationStrategy`<br>`DiffMutationStrategy` | • Dual Thought-Code search (E1/E2 exploration, M1/M2 parameter mutation)<br>• Comparative diagnostic reflection loops across contrasting algorithms<br>• Targeted surgical edits and localized patch adjustments |
| **Evaluation Sandboxes** | `SubprocessSandboxEvaluator`<br>`MCPEvaluator` | • Isolated subprocess execution with hard wall-clock timeouts & process cleanup<br>• Model Context Protocol (MCP) JSON-RPC 2.0 tool-calling client (`tools/call`) |
| **Benchmark Problems** | `BPPBenchmarkTask`<br>`TSPBenchmarkTask` | • Online 1D Bin Packing (`bpp_online`) with Best-Fit baseline & L1 bound metrics<br>• Constructive Traveling Salesperson (`tsp_construct`) with 2D Euclidean instances |
| **LLM Inference** | `LLMGateway` | • vLLM OpenAI-compatible local GPU endpoints, cloud APIs, mock backend, and parallel continuous batching |
| **Storage & Telemetry** | `ExperimentStorage` | • Optuna-compatible SQLite relational database (schema v13) + standalone `.py` filesystem artifacts |

---

## Quickstart: Python API

You can construct and execute an entire evolutionary pipeline in just a few lines of Python:

```python
from evokit.core.gateway import LLMGateway, GatewayConfig
from evokit.core.storage import ExperimentStorage, StorageConfig
from evokit.benchmarks.bpp import BPPBenchmarkTask, BPPTaskConfig
from evokit.plugins.populations.lineage_archive import LineageGraphArchive
from evokit.plugins.mutations.eoh import EoHMutationStrategy
from evokit.plugins.evaluators.subprocess_sandbox import SubprocessSandboxEvaluator
from evokit.engine import EvolutionEngine, EngineConfig

# 1. Initialize Infrastructure
gateway = LLMGateway(GatewayConfig(
    provider="vllm",
    base_url="http://172.22.4.1:8000/v1",
    model="mistralai/Devstral-Small-2-24B-Instruct-2512",
    max_parallel_requests=4,
))
storage = ExperimentStorage(StorageConfig(
    db_path="./artifacts/eoh_bpp/experiments.db",
    artifacts_dir="./artifacts/eoh_bpp/code",
    direction="MINIMIZE",
))

# 2. Assemble the 3 Pillars & Benchmark
benchmark = BPPBenchmarkTask(BPPTaskConfig(num_train_instances=5, num_test_instances=10))
population = LineageGraphArchive(config={"direction": "MINIMIZE"})
mutation = EoHMutationStrategy()
evaluator = SubprocessSandboxEvaluator(config={"timeout_seconds": 30.0, "max_workers": 2})

# 3. Coordinate via Invariant Engine
engine = EvolutionEngine(
    gateway=gateway,
    storage=storage,
    benchmark=benchmark,
    population=population,
    mutation=mutation,
    evaluator=evaluator,
    config=EngineConfig(max_generations=5, candidates_per_generation=2),
)

# 4. Run Optimization & Generalization Testing
summary = engine.run()
print(f"Best Elite Train Fitness: {summary['best_elites'][0]['fitness']}")
print(f"Held-out Test Fitness: {summary['test_generalization'][0]['test_fitness']}")
```

---

## Method-Benchmark Tutorials

EvoKit ships with declarative YAML specifications for standard method-benchmark combinations located in `evokit/configs/`.

### 1. EoH on Online Bin Packing (`eoh-obp`)
Explores diverse packing heuristic designs using the **Evolution of Heuristics** (EoH) dual Thought-Code operators (E1 cross-parent exploration, E2 backbone idea extraction, M1 form mutation, M2 parameter tuning).

* **Config**: `evokit/configs/eoh_bpp.yaml`
* **Run Command**:
  ```bash
  python evokit/main.py run -c evokit/configs/eoh_bpp.yaml
  ```
* **Dry-Run (Zero Inference Cost)**:
  ```bash
  python evokit/main.py run -c evokit/configs/eoh_bpp.yaml --dry-run
  ```

---

### 2. EoH on Traveling Salesperson (`eoh-tsp`)
Applies the EoH exploration-mutation operators to synthesize constructive node-selection heuristics for the symmetric Traveling Salesperson Problem (TSP).

* **Config**: `evokit/configs/eoh_tsp.yaml`
* **Run Command**:
  ```bash
  python evokit/main.py run -c evokit/configs/eoh_tsp.yaml
  ```

---

### 3. ReEvo on Traveling Salesperson (`reevo-tsp`)
Leverages **Reflective Evolution** (ReEvo) to analyze comparative performance differentials. Contrasts a higher-performing tour constructor with a lower-performing variant, generates a diagnostic reflection, and uses that critique to synthesize an improved offspring algorithm.

* **Config**: `evokit/configs/reevo_tsp.yaml`
* **Run Command**:
  ```bash
  python evokit/main.py run -c evokit/configs/reevo_tsp.yaml
  ```

---

### 4. ReEvo on Online Bin Packing (`reevo-obp`)
Applies reflective comparative loops to online bin packing. Contrasts algorithms that leave high residual fragmentation against tighter heuristics, extracting mathematical insights on bin capacity utilization.

* **Config**: `evokit/configs/reevo_bpp.yaml`
* **Run Command**:
  ```bash
  python evokit/main.py run -c evokit/configs/reevo_bpp.yaml
  ```

---

### 5. MAP-Elites on Online Bin Packing (`map_elites-obp`)
Maintains an $N$-dimensional behavioral tessellation grid across orthogonal algorithm characteristics (e.g. code complexity vs. execution runtime), pairing with `DiffMutationStrategy` to evolve diverse high-performing heuristics.

* **Config**: `evokit/configs/map_elites_bpp.yaml`
* **Run Command**:
  ```bash
  python evokit/main.py run -c evokit/configs/map_elites_bpp.yaml
  ```

---

## Parallel Execution & Multi-Processing

EvoKit provides a two-tier parallel execution architecture designed for computationally intensive heuristic discovery (such as perturbation TSP with Guided Local Search):

```
Tier 1: Inter-Candidate Concurrency (--max-workers)
        Evaluates multiple synthesized candidates in parallel isolated OS subprocesses.

Tier 2: Intra-Task Instance Concurrency (--instance-workers)
        Distributes problem instances (e.g. 20 TSPLIB instances) across CPU cores simultaneously.
```

### Configuring Concurrency

- **In YAML**:
  ```yaml
  evaluator:
    max_workers: 4       # Tier 1: 4 candidates evaluated simultaneously
  benchmark:
    parallel_workers: 8  # Tier 2: 8 workers solving instances per candidate
  ```
- **Via CLI Overrides**:
  ```bash
  evokit run -c evokit/configs/reevo_tsp.yaml --max-workers 2 --instance-workers 8
  ```

---

## Command-Line Interface

The dynamic CLI (`evokit/main.py` or `evokit` command) provides tools for executing runs, evaluating standalone artifacts, and inspecting telemetry.

### `evokit run`

Execute an evolutionary search from a declarative YAML config:
```bash
# Basic run
evokit run -c evokit/configs/eoh_bpp.yaml

# Override vLLM endpoint, model, or number of generations
evokit run -c evokit/configs/eoh_bpp.yaml \
  --vllm-url http://172.22.4.1:8000/v1 \
  --model mistralai/Devstral-Small-2-24B-Instruct-2512 \
  -g 10 \
  --max-workers 4 \
  --instance-workers 8

# Zero-cost dry run using deterministic mock LLM
evokit run -c evokit/configs/eoh_bpp.yaml --dry-run
```

### `evokit evaluate`

Directly evaluate any individual Python code artifact against a benchmark task without running the full search loop:
```bash
evokit evaluate \
  --code artifacts/eoh_bpp/code/eoh_bpp_online/gen_001/ind_636ac2ea774a.py \
  --benchmark bpp_online \
  --split test \
  --timeout 30.0 \
  --instance-workers 4
```

### `evokit dashboard`

Print the Optuna SQLite connection string and launch instructions:
```bash
evokit dashboard --db ./artifacts/eoh_bpp/experiments.db
```

---

## Real-Time Optuna Dashboard Visualization

EvoKit's `ExperimentStorage` implements a native SQLite schema strictly compatible with [Optuna Dashboard](https://github.com/optuna/optuna-dashboard) (Optuna schema v3–v13).

To visualize optimization trajectories, generational fitness curves, and parameter distributions in real time:

```bash
optuna-dashboard sqlite:///artifacts/eoh_bpp/experiments.db
```

Open `http://127.0.0.1:8080/` in your browser to inspect:
- Generational objective trajectories (minimization / maximization curves)
- Trial completion states (`COMPLETE` vs. `FAIL`)
- Detailed user attributes per trial: `individual_id`, `operator`, `metrics`, `artifact_path`, and `thought_preview`.

---

## Physical Code Artifacts

Every generated algorithm is written to disk as a standalone, runnable Python script under:
```text
artifacts/{study_name}/code/{study_name}/gen_{generation:03d}/{individual_id}.py
```

Each artifact includes structured metadata comments:
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

---

## Extending EvoKit with Custom Plugins

EvoKit makes it straightforward to expand to emerging research paradigms (e.g. **OpenEvolve**, **MEoH**, **EoH-S**):

1. **Custom Population**: Inherit from `BasePopulationManager` (`evokit/plugins/populations/base.py`) and implement `add()`, `sample_parents()`, and `get_elites()`.
2. **Custom Mutation**: Inherit from `BaseMutationStrategy` (`evokit/plugins/mutations/base.py`) and implement `mutate(context)`.
3. **Custom Benchmark**: Inherit from `BaseBenchmarkTask` (`evokit/benchmarks/base.py`), provide a `PromptSpec`, and define `_generate_instances()` with train/test splits.

### Method Expansion Blueprints

| Method | Core Paradigm | Pillar 1: Population Archive | Pillar 2: Mutation Strategy |
|---|---|---|---|
| **OpenEvolve** | Quality-Diversity MAP-Elites | `MAPElitesGrid` (N-dim complexity & runtime grid) | `DiffMutationStrategy` (surgical code patches) |
| **MEoH** | Multi-Objective Pareto Frontier | `ParetoArchive` (non-dominated sorting & trade-offs) | `MEoHMutationStrategy` (trade-off prompt synthesis) |
| **EoH-S** (AAAI '26) | Automated Heuristic Set Design | `HeuristicSetArchive` (instance coverage portfolio) | `EoHSMutationStrategy` (hard-instance targeted prompts) |

Custom classes can be specified directly in declarative YAML via standard Python dot notation without modifying core EvoKit code:
```yaml
population:
  type: "my_package.custom_archive:ParetoArchive"
  capacity: 25
```

For complete implementation blueprints, code templates, and step-by-step expansion guides, see **[docs/plugin_guide.md](docs/plugin_guide.md#7-method-expansion-recipes-openevolve-meoh-eoh-s)**.

---

## Running the Test Suite

Execute all 27 unit and integration tests across the framework:
```bash
PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py"
```

Test coverage includes:
- Component lifecycle contracts and context immutability (`test_core.py`)
- Benchmark deterministic instance generation and simulation (`test_benchmarks.py`)
- Sandbox timeout enforcement and MCP protocol handling (`test_evaluators.py`)
- Multi-processing sandbox batch evaluation and fault isolation (`test_parallel_evaluator.py`)
- High-throughput LLM gateway batch sampling (`test_parallel_gateway.py`)
- Population Lineage DAG, Reflection Buffer, and MAP-Elites grids (`test_populations.py`)
- EoH, ReEvo, and Diff mutation prompt formatting and extraction (`test_mutations.py`)
- End-to-end integration loop across all 3 pillars and Optuna storage (`test_engine.py`)

---

## Documentation

- [Full Architecture Specification](docs/spec.md): Complete architectural baseline (v0.0.1) and system requirements.
- [Plugin Creation Guide](docs/plugin_guide.md): Developer guide for implementing custom algorithms and archives.
- [Task Tracking Log](docs/tasks/task_log.md): Phased engineering log and verification records.
