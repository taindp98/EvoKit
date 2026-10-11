# `evokit.core`: Foundational Architecture & Invariant Interfaces

The `evokit.core` package establishes the foundational lifecycle contracts, entities, and state boundaries across all EvoKit subsystems.

---

## Core Components

| Module | Primary Class | Role & Invariants |
|---|---|---|
| [`base.py`](base.py) | `BaseComponent`, `ComponentConfig` | Lifecycle contract (`setup()`, `teardown()`, `_setup_internal()`, `_teardown_internal()`) with thread-safe idempotent setup guards and Pydantic configuration validation. |
| [`entity.py`](entity.py) | `Individual`, `MetricRecord` | Representation of evolved code candidates, thoughts, parent lineage, generation tags, and sandboxed evaluation telemetry. |
| [`context.py`](context.py) | `EvolutionContext` | Read-only generational context snapshot passed to mutation strategies. Encapsulates active population view, benchmark specification, and engine metadata. |
| [`gateway.py`](gateway.py) | `LLMGateway`, `GatewayConfig` | High-throughput client for vLLM, OpenAI, and cloud APIs. Supports concurrent multi-request parallel sampling (`generate_batch`, `chat_batch`), connection pooling, and token tracking. |
| [`storage.py`](storage.py) | `ExperimentStorage`, `StorageConfig` | Dual-tier persistence engine: Optuna-compatible SQLite relational schema (schema v13) alongside structured `.py` code artifact directories. |
