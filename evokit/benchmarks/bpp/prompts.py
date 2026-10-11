"""
Domain Prompt Specifications for Online Bin Packing (BPP).
"""

from evokit.benchmarks.base import PromptSpec

INITIAL_BPP_CODE = '''import numpy as np

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Best-Fit heuristic: prefer bins that leave the least remaining space after placing the item.
    """
    remaining = bins - item
    return -remaining
'''

BPP_PROMPT_SPEC = PromptSpec(
    task_name="bpp_online",
    description="Online 1D Bin Packing. Items arrive one by one and must be placed immediately into an available bin without exceeding the bin capacity (capacity = 100.0).",
    function_name="priority",
    signature="def priority(item: float, bins: np.ndarray) -> np.ndarray:",
    docstring="Calculate priority scores for all open bins that have sufficient remaining capacity for the given item.",
    input_description="""- item (float): Size/weight of the incoming item.
- bins (np.ndarray): 1D array of remaining capacities for valid candidate bins (each bin has remaining >= item).""",
    output_description="""- priorities (np.ndarray): 1D array of float priority scores, matching the length of bins. The item is placed in the bin with argmax(priorities).""",
    initial_code=INITIAL_BPP_CODE,
    evaluation_criteria="Minimize the total number of bins used across diverse item arrival distributions.",
)
