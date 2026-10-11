# `evokit.benchmarks`: Problem Formulations & Benchmark Environments

This package defines evaluation tasks, ground-truth fitness functions, instance generation pipelines, and prompt specifications for LLM-driven algorithm discovery.

---

## Architecture & Responsibilities

1. **Strict Dataset Splitting**: Every benchmark provides isolated `train_instances` and `test_instances` generated with a fixed seed. Generalization to unseen instances is evaluated explicitly on the held-out test split.
2. **Prompts as Structured Data**: Benchmark prompt guidance, function signatures, docstrings, and baseline heuristics are defined declaratively via [`PromptSpec`](base.py) objects.
3. **Multi-Processing / Parallel Instance Simulation**: Computationally intensive optimization benchmarks (e.g. constructive TSP, Guided Local Search / GLS, large-scale bin packing) support concurrent evaluation across problem instances via the `parallel_workers` configuration parameter.

---

## Directory Map

- [`obp/`](obp): Online 1D Bin Packing Problem (`obp_online`).
  - `prompts.py`: Function contract (`priority(item, bins)`) and Best-Fit baseline heuristic.
  - `task.py`: Online item stream simulation, Weibull/Uniform distributions, $L_1$ lower bound, and instance parallelization.
- [`tsp/`](tsp): Constructive Traveling Salesperson Problem (`tsp_construct`).
  - `prompts.py`: Function contract (`select_next_node(...)`) and Nearest-Neighbor baseline heuristic.
  - `task.py`: 2D Euclidean coordinate tour generation, distance matrix calculation, and instance parallelization.

---

## Contract Interface (`BaseBenchmarkTask`)

Every benchmark inherits from [`BaseBenchmarkTask`](base.py) and implements:
- `name -> str`: Benchmark identifier (e.g., `'obp_online'`, `'tsp_construct'`).
- `direction -> str`: Optimization direction (`'MAXIMIZE'` or `'MINIMIZE'`).
- `prompt_spec -> PromptSpec`: Declarative prompt templates, function interface, and input/output schema.
- `_generate_instances() -> None`: Generates train and held-out test instance collections.
- `evaluate_code(code_str: str, split: str = "train", timeout: float = 30.0) -> MetricRecord`: Executes code against the instance split and returns metrics.
