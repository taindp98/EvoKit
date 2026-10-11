# `evokit.benchmarks.obp`: Online 1D Bin Packing

This module implements the **Online 1D Bin Packing Problem (OBP)** benchmark.

---

## Problem Formulation

In online bin packing, items of sizes $s_i \in (0, C]$ arrive sequentially one at a time and must be immediately assigned to an open bin without exceeding the bin capacity $C = 100.0$. Future item arrivals are unknown.

The evolutionary goal is to discover heuristic scoring functions that pack items with minimal bin waste and minimum total bins across diverse arrival distributions.

---

## Function Contract

Candidate algorithms must implement the `priority` function:

```python
import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Calculate priority scores for all open bins that have sufficient remaining capacity.
    The incoming item is placed in the bin with the highest priority: best_bin = argmax(priority).
    """
    remaining = bins - item
    return -remaining  # Best-Fit heuristic baseline
```

### Arguments
- `item` (`float`): Weight/size of the incoming item.
- `bins` (`np.ndarray`): 1D float array of remaining capacities for open bins that can fit the item ($bins_j \ge item$).

### Returns
- `priorities` (`np.ndarray`): 1D float array of priority scores matching the length of `bins`.

---

## Benchmark Configuration

Controlled via `OBPTaskConfig`:
- `capacity`: Bin capacity limit (default: `100.0`).
- `train_items_per_instance`: Items per training stream (default: `200`).
- `test_items_per_instance`: Items per held-out test stream (default: `500`).
- `num_train_instances`: Training instances generated with mixture of Uniform and Weibull distributions (default: `5`).
- `num_test_instances`: Out-of-distribution test instances with Exponential and wide Uniform distributions (default: `10`).

### Evaluated Metrics
- `fitness`: Negative average number of bins used across instances (higher is better).
- `avg_bins`: Average total bins opened.
- `avg_l1_ratio`: Average ratio of theoretical L1 lower bound to used bins ($\frac{\lceil \sum s_i / C \rceil}{\text{used\_bins}}$).
