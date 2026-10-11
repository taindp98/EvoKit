# `evokit.benchmarks.tsp`: Constructive Traveling Salesperson

This module implements the **Constructive Traveling Salesperson Problem (TSP)** benchmark.

---

## Problem Formulation

Given $N$ cities with 2D Euclidean coordinates in $[0, 1]^2$, find a permutation of cities that visits every city exactly once and returns to the origin city while minimizing total tour length.

In this constructive setting, an algorithm builds the tour incrementally node by node starting from node 0.

---

## Function Contract

Candidate algorithms must implement the `select_next_node` function:

```python
import numpy as np

def select_next_node(
    current_node: int,
    destination_node: int,
    unvisited_nodes: set,
    distance_matrix: np.ndarray,
) -> int:
    """
    Select the next city to visit from unvisited_nodes.
    """
    return min(unvisited_nodes, key=lambda node: distance_matrix[current_node, node])  # Nearest Neighbor baseline
```

### Arguments
- `current_node` (`int`): Index of the city most recently visited.
- `destination_node` (`int`): Origin city to which the tour must eventually return (typically `0`).
- `unvisited_nodes` (`set[int]`): Set of integer indices of cities not yet visited.
- `distance_matrix` (`np.ndarray`): 2D symmetric float array ($N \times N$) of pairwise Euclidean distances.

### Returns
- `next_node` (`int`): City index chosen from `unvisited_nodes`.

---

## Benchmark Configuration

Controlled via `TSPTaskConfig`:
- `train_cities`: Number of cities per training instance (default: `30`).
- `test_cities`: Number of cities per held-out test instance (default: `50`).
- `num_train_instances`: Training instances generated with random uniform 2D coordinates (default: `5`).
- `num_test_instances`: Out-of-distribution test instances (default: `10`).

### Evaluated Metrics
- `fitness`: Average Euclidean tour length (lower is better, direction: MINIMIZE).
- `avg_tour_length`: Average total tour length including the final leg back to city 0.
- `min_tour_length`, `max_tour_length`: Range of tour lengths across the evaluation split.
