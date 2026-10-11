# `evokit.plugins.evaluators`: Sandbox Execution & Evaluation Services

This module provides isolated sandbox environments and standardized remote protocol clients for safely executing untrusted LLM-generated Python code against benchmark tasks.

---

## Component Implementations

| Evaluator | Primary Class | Execution Architecture | Safety & Concurrency Guarantees |
|---|---|---|---|
| **Subprocess Sandbox** (`subprocess_sandbox.py`) | `SubprocessSandboxEvaluator` | Local isolated subprocess execution via `subprocess.run` with multi-core worker pool. | • **Parallel Batch Execution**: `evaluate_batch(...)` runs multiple candidates concurrently across isolated subprocesses on separate CPU cores via `ThreadPoolExecutor(max_workers)`.<br>• **Hard Wall-Clock Timeout**: Enforces per-evaluation timeouts via `TimeoutExpired` to prevent infinite loops.<br>• **Filesystem Isolation**: Writes code and runner scripts into scoped `tempfile.TemporaryDirectory`.<br>• **Process Tree Cleanup**: Terminated processes are killed cleanly.<br>• **Complete Output Capture**: Captures complete stdout, stderr, and tracebacks without polluting parent runtime. |
| **Model Context Protocol (MCP)** (`mcp_evaluator.py`) | `MCPEvaluator` | Standardized JSON-RPC 2.0 tool execution (`tools/call`). | • **Standardized Interface**: Packages evaluation requests into MCP `evaluate_algorithm` JSON-RPC payloads.<br>• **Concurrent Dispatch**: Supports parallel `evaluate_batch(...)` requests across remote endpoints.<br>• **Remote Dispatch**: Connects to remote MCP HTTP evaluation servers when `server_url` is configured.<br>• **Local Fallback**: Emulates MCP tool invocation locally when no remote endpoint is specified. |

---

## Contract Interface (`BaseEvaluator`)

Every evaluator inherits from `BaseEvaluator` and implements:
- `evaluate(individual: Individual, task: BaseBenchmarkTask, split: str = "train") -> MetricRecord`:
  Executes candidate code against the specified benchmark task and returns a structured `MetricRecord`.
- `evaluate_batch(individuals: List[Individual], task: BaseBenchmarkTask, split: str = "train") -> List[MetricRecord]`:
  Concurrently evaluates a batch of candidate algorithms in parallel using the configured `max_workers` worker pool, strictly preserving individual ordering and fault isolation.

---

## Quick Example: Parallel Batch Evaluation

```python
from evokit.core.entity import Individual
from evokit.benchmarks.tsp import TSPBenchmarkTask
from evokit.plugins.evaluators.subprocess_sandbox import SubprocessSandboxEvaluator

# 1. Initialize task and evaluator with 4 parallel worker cores
task = TSPBenchmarkTask({"num_train_instances": 5, "city_size": 20})
task.setup()
evaluator = SubprocessSandboxEvaluator({"timeout_seconds": 15.0, "max_workers": 4})

# 2. Evaluate a batch of candidate individuals concurrently
candidates = [
    Individual(code="def select_next_node(...): ..."),
    Individual(code="def select_next_node(...): ..."),
    Individual(code="def select_next_node(...): ..."),
    Individual(code="def select_next_node(...): ..."),
]

records = evaluator.evaluate_batch(candidates, task, split="train")
for cand, rec in zip(candidates, records):
    print(f"Ind {cand.id} | Valid: {rec.is_valid} | Fitness: {rec.fitness:.4f}")
```
