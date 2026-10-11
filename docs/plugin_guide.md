# EvoKit Plugin Creation Guidelines

EvoKit is built upon a **modular, contract-driven architecture** where all algorithmic variations, archives, and execution environments are pluggable components conforming to strict lifecycle contracts.

This guide explains how to develop custom plugins for:
1. **Population & Archive Managers** (Pillar 1)
2. **Mutation & Search Strategies** (Pillar 2)
3. **Evaluation Sandboxes** (Pillar 3)
4. **Benchmark Problem Tasks**

---

## 1. The Core Lifecycle Contract

Every pluggable subsystem inherits from `BaseComponent` (`evokit.core.base.BaseComponent`):

```python
from evokit.core.base import BaseComponent, ComponentConfig

class MyCustomPlugin(BaseComponent):
    def __init__(self, config=None):
        super().__init__(config)
        # Ingest configuration

    def _setup_internal(self):
        # Validate dependencies, connect to network/DB, initialize caches
        pass

    def _teardown_internal(self):
        # Gracefully close connections, flush buffers, kill child pools
        pass
```

Components implement context manager semantics (`with MyCustomPlugin(): ...`) and idempotent setup/teardown hooks.

---

## 2. Creating a Custom Population Manager (Pillar 1)

Population managers manage candidate admission, archives, and parent selection.

Inherit from `BasePopulationManager` (`evokit.plugins.populations.base.BasePopulationManager`):

```python
from evokit.plugins.populations.base import BasePopulationManager, PopulationConfig
from evokit.core.entity import Individual
from typing import List

class IslandModelArchive(BasePopulationManager):
    def __init__(self, config=None):
        super().__init__(config)
        self.islands = [[] for _ in range(4)]

    def add(self, individual: Individual) -> bool:
        if not individual.is_valid:
            return False
        island_idx = hash(individual.id) % len(self.islands)
        self.islands[island_idx].append(individual)
        return True

    def sample_parents(self, k: int = 1) -> List[Individual]:
        # Sample parents according to island topology
        pass

    def get_elites(self, top_k: int = 1) -> List[Individual]:
        # Return best individuals across all islands
        pass

    def get_all(self) -> List[Individual]:
        all_inds = []
        for isl in self.islands:
            all_inds.extend(isl)
        return all_inds

    def size(self) -> int:
        return sum(len(isl) for isl in self.islands)
```

---

## 3. Creating a Custom Mutation Strategy (Pillar 2)

Mutation strategies encapsulate prompt engineering, LLM queries, and program synthesis.

Inherit from `BaseMutationStrategy` (`evokit.plugins.mutations.base.BaseMutationStrategy`):

```python
from evokit.plugins.mutations.base import BaseMutationStrategy
from evokit.core.context import EvolutionContext
from evokit.core.entity import Individual
from typing import List

class ShinkaEvolveMutation(BaseMutationStrategy):
    def mutate(self, context: EvolutionContext, num_candidates: int = 1) -> List[Individual]:
        candidates = []
        task_spec = context.task_spec
        gateway = context.llm_gateway

        for _ in range(num_candidates):
            parents = context.population_view
            # Build your custom prompt
            prompt = f"Problem: {task_spec['description']}\nOptimize this heuristic..."

            response = gateway.generate(prompt=prompt, temperature=self.cfg.temperature)
            thought = self.extract_thought(response.content)
            code = self.extract_code(response.content, default_fallback=task_spec["initial_code"])

            candidate = Individual(
                code=code,
                thought=thought,
                generation=context.generation,
                operator="shinka_mutation",
            )
            candidates.append(candidate)

        return candidates
```

---

## 4. Creating a Custom Evaluator Service (Pillar 3)

Evaluators run unverified code inside strict isolation boundaries.

Inherit from `BaseEvaluator` (`evokit.plugins.evaluators.base.BaseEvaluator`):

```python
from evokit.plugins.evaluators.base import BaseEvaluator
from evokit.benchmarks.base import BaseBenchmarkTask
from evokit.core.entity import Individual, MetricRecord

class RemoteClusterEvaluator(BaseEvaluator):
    def evaluate(self, individual: Individual, task: BaseBenchmarkTask, split: str = "train") -> MetricRecord:
        # Submit code to remote worker or container
        # Capture scalar fitness, stdout, stderr, and sub-metrics
        return MetricRecord(
            fitness=105.2,
            is_valid=True,
            metrics={"latency_ms": 14.2},
            stdout="Evaluation passed",
            runtime_seconds=0.015,
        )
```

---

## 5. Creating a Benchmark Task

Inherit from `BaseBenchmarkTask` (`evokit.benchmarks.base.BaseBenchmarkTask`):

```python
from evokit.benchmarks.base import BaseBenchmarkTask, PromptSpec
from evokit.core.entity import MetricRecord

class KnapsackBenchmark(BaseBenchmarkTask):
    @property
    def name(self) -> str:
        return "knapsack_01"

    @property
    def direction(self) -> str:
        return "MAXIMIZE"

    @property
    def prompt_spec(self) -> PromptSpec:
        return PromptSpec(
            task_name="knapsack_01",
            description="0-1 Knapsack Problem heuristic",
            function_name="select_item",
            signature="def select_item(values, weights, capacity):",
            docstring="Returns selected item index.",
            input_description="- values: list, weights: list, capacity: float",
            output_description="- index: int",
            initial_code="def select_item(values, weights, capacity):\n    return 0",
        )

    def _generate_instances(self):
        self.train_instances = [...]  # Fast evaluation set
        self.test_instances = [...]   # Out-of-distribution hold-out set

    def evaluate_code(self, code_str: str, split: str = "train", timeout: float = 30.0) -> MetricRecord:
        # Evaluate program logic deterministically
        pass
```

---

## 6. Registering and Using Custom Plugins in YAML

You can specify your custom plugin directly in declarative YAML configurations using standard Python dot notation:

```yaml
population:
  type: "my_custom_package.archives:IslandModelArchive"
  capacity: 20

mutation:
  type: "my_custom_package.mutations:ShinkaEvolveMutation"
  temperature: 0.8
```

EvoKit's `loader.py` dynamically resolves the class and injects configuration without modifying core framework code.

---

## 7. Method Expansion Recipes (OpenEvolve, MEoH, EoH-S)

EvoKit's 3-pillar decoupling makes expanding to any emerging research paradigm straightforward. Below are complete architectural blueprints and code recipes for three prominent methodologies:

---

### 7.1 OpenEvolve (Quality-Diversity / MAP-Elites with Surgical Code Edits)

**Core Idea**: Instead of searching for a single best algorithm, OpenEvolve explores an $N$-dimensional behavioral feature landscape (e.g., Code Complexity vs. Execution Runtime) using MAP-Elites, applying localized diff/patch mutations to candidate code.

#### Pillar Mapping:
- **Pillar 1 (Population)**: [`MAPElitesGrid`](../evokit/plugins/populations/map_elites.py) discretizes algorithms into an $N \times M$ tessellation grid based on behavioral descriptors.
- **Pillar 2 (Mutation)**: [`DiffMutationStrategy`](../evokit/plugins/mutations/diff.py) samples parents from occupied grid niches and prompts the LLM for targeted, surgical patch improvements.
- **Pillar 3 (Evaluator)**: [`SubprocessSandboxEvaluator`](../evokit/plugins/evaluators/subprocess_sandbox.py) executes candidates in isolation, capturing both fitness and behavioral feature telemetry.

#### Configuration Blueprint (`configs/openevolve_example.yaml`):
```yaml
experiment:
  study_name: "openevolve_bpp"
  max_generations: 10
  candidates_per_generation: 4

population:
  type: "map_elites"
  grid_dim_1: 5               # 5 bins for Feature 1 (AST / Code Complexity)
  grid_dim_2: 5               # 5 bins for Feature 2 (Runtime latency)
  max_complexity: 800         # Max characters/AST nodes for bin normalization
  max_metric_val: 0.05        # Max runtime in seconds
  direction: "MINIMIZE"

mutation:
  type: "diff"
  temperature: 0.7

evaluator:
  type: "subprocess_sandbox"
  max_workers: 4
  timeout_seconds: 30.0
```

---

### 7.2 MEoH (Multi-Objective Evolution of Heuristics)

**Core Idea**: Real-world algorithm design frequently involves conflicting objectives (e.g., **Solution Quality** vs. **Computational Runtime**, or **Tour Length** vs. **Memory Footprint**). MEoH replaces scalar tournament selection with a **Pareto Frontier Archive** and prompts the LLM to navigate trade-offs.

#### Implementation Blueprint:

```python
# plugins/populations/pareto_archive.py
from typing import List, Tuple
from evokit.plugins.populations.base import BasePopulationManager, PopulationConfig
from evokit.core.entity import Individual

class ParetoArchiveConfig(PopulationConfig):
    pass

class ParetoArchive(BasePopulationManager):
    """
    Maintains the non-dominated Pareto frontier across multiple objectives.
    Objective 1: individual.fitness (primary performance metric, e.g. tour length)
    Objective 2: individual.metadata['runtime_seconds'] (efficiency metric)
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.archive: List[Individual] = []

    def _dominates(self, a: Individual, b: Individual) -> bool:
        """Returns True if individual A Pareto-dominates individual B (minimization)."""
        fit_a = a.fitness if a.fitness is not None else float("inf")
        fit_b = b.fitness if b.fitness is not None else float("inf")
        time_a = a.metadata.get("runtime_seconds", 0.0)
        time_b = b.metadata.get("runtime_seconds", 0.0)

        # A is no worse in all objectives and strictly better in at least one
        no_worse = (fit_a <= fit_b) and (time_a <= time_b)
        strictly_better = (fit_a < fit_b) or (time_a < time_b)
        return no_worse and strictly_better

    def add(self, individual: Individual) -> bool:
        if not individual.is_valid or individual.fitness is None:
            return False

        # If any existing individual dominates this candidate, reject
        for elite in self.archive:
            if self._dominates(elite, individual):
                return False

        # Remove existing individuals that this candidate dominates
        self.archive = [e for e in self.archive if not self._dominates(individual, e)]
        self.archive.append(individual)

        # Enforce archive capacity if needed (e.g., crowding distance pruning)
        if len(self.archive) > self.cfg.capacity:
            self.archive.sort(key=lambda x: x.fitness)
            self.archive = self.archive[:self.cfg.capacity]
        return True

    def sample_parents(self, k: int = 2) -> List[Individual]:
        # Sample complementary parents from opposite ends of the Pareto frontier
        if len(self.archive) <= k:
            return list(self.archive)
        sorted_frontier = sorted(self.archive, key=lambda x: x.fitness)
        return [sorted_frontier[0], sorted_frontier[-1]]  # Best quality + fastest runtime

    def get_elites(self, top_k: int = 1) -> List[Individual]:
        return sorted(self.archive, key=lambda x: x.fitness)[:top_k]

    def get_all(self) -> List[Individual]:
        return list(self.archive)

    def size(self) -> int:
        return len(self.archive)
```

#### Mutation Strategy for MEoH:
In `MEoHMutationStrategy`, the LLM is prompted with Pareto-contrasting parents:
```text
Parent A: Ultra-fast execution (0.002s), moderate solution quality (fitness: 12.5).
Parent B: High solution quality (fitness: 9.1), slower execution (0.045s).

Task: Synthesize a Pareto-optimal hybrid that achieves the quality of Parent B
while pruning redundant loop checks to match the speed of Parent A.
```

---

### 7.3 EoH-S (Evolution of Heuristic Set - AAAI 2026)

**Core Idea**: Instead of searching for a *single universal heuristic* that tries (and often fails) to generalize across all problem distributions, **EoH-S** designs a **Complementary Heuristic Set** (portfolio) $S = \{h_1, h_2, \dots, h_k\}$ where each instance $i$ in a benchmark is optimized by whichever heuristic performs best on that specific instance:

$$\text{Portfolio Cost}(S, \mathcal{I}) = \frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \min_{h \in S} \text{cost}(h, i)$$

#### Implementation Blueprint:

```python
# plugins/populations/heuristic_set_archive.py
from typing import List, Dict
import numpy as np
from evokit.plugins.populations.base import BasePopulationManager, PopulationConfig
from evokit.core.entity import Individual

class HeuristicSetConfig(PopulationConfig):
    portfolio_size: int = 4  # Target size of complementary heuristic portfolio

class HeuristicSetArchive(BasePopulationManager):
    """
    Maintains a complementary heuristic portfolio.
    An individual is admitted if it improves coverage on instance subsets
    where the current portfolio heuristics perform poorly.
    """
    def __init__(self, config=None):
        super().__init__(config)
        self.portfolio: List[Individual] = []
        self.instance_costs: Dict[str, List[float]] = {}  # individual_id -> per-instance costs

    def add(self, individual: Individual) -> bool:
        if not individual.is_valid:
            return False

        # Per-instance evaluation records must be emitted by benchmark
        inst_scores = individual.metadata.get("instance_costs")
        if not inst_scores:
            return False

        self.instance_costs[individual.id] = inst_scores

        if len(self.portfolio) < self.cfg.capacity:
            self.portfolio.append(individual)
            return True

        # Check if replacing any portfolio member improves overall set coverage:
        # Score(S) = sum_i min_{h in S} cost(h, i)
        current_set_score = self._compute_portfolio_score(self.portfolio)

        best_candidate_set = None
        best_candidate_score = current_set_score

        for replace_idx in range(len(self.portfolio)):
            trial_set = list(self.portfolio)
            trial_set[replace_idx] = individual
            trial_score = self._compute_portfolio_score(trial_set)
            if trial_score < best_candidate_score:
                best_candidate_score = trial_score
                best_candidate_set = trial_set

        if best_candidate_set is not None:
            self.portfolio = best_candidate_set
            return True
        return False

    def _compute_portfolio_score(self, candidate_set: List[Individual]) -> float:
        cost_matrix = [self.instance_costs[h.id] for h in candidate_set]
        min_costs_per_instance = np.min(cost_matrix, axis=0)
        return float(np.mean(min_costs_per_instance))

    def sample_parents(self, k: int = 1) -> List[Individual]:
        return list(self.portfolio)

    def get_elites(self, top_k: int = 1) -> List[Individual]:
        return self.portfolio[:top_k]

    def get_all(self) -> List[Individual]:
        return list(self.portfolio)

    def size(self) -> int:
        return len(self.portfolio)
```

#### Mutation Strategy for EoH-S (`EoHSMutationStrategy`):
1. **Identify Hard Instances**: Computes the instance index $i^* = \arg\max_i \min_{h \in S} \text{cost}(h, i)$ where the current portfolio has its worst performance.
2. **Targeted Prompting**: Queries the LLM with the structural features of instance $i^*$ and the failure mode of the current heuristics on it, requesting a specialized heuristic targeting those specific properties.

---

### 7.4 Summary: Decoupling Enables Infinite Combinations

Because EvoKit enforces invariant boundaries:
- Any **Population Manager** (`LineageDAG`, `ReflectionBuffer`, `MAPElites`, `ParetoArchive`, `HeuristicSetArchive`)
- Can be combined with any **Mutation Strategy** (`EoH`, `ReEvo`, `Diff`, `MEoH`, `EoH-S`)
- On any **Benchmark Task** (`TSP`, `BPP`, `FSSP`, `CVRP`)
- Executed under any **Evaluator Sandbox** (Local Subprocess, Remote Slurm, MCP JSON-RPC).
