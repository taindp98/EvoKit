"""
Domain Prompt Specifications for Constructive Traveling Salesperson (TSP).
"""

from evokit.benchmarks.base import PromptSpec

INITIAL_TSP_CODE = '''import numpy as np

def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:
    """
    Nearest Neighbor heuristic: pick the closest unvisited node to current_node.
    """
    return min(unvisited_nodes, key=lambda node: distance_matrix[current_node, node])
'''

TSP_PROMPT_SPEC = PromptSpec(
    task_name="tsp_construct",
    description="Constructive Traveling Salesperson Problem (TSP). Incrementally construct a tour visiting all cities exactly once and returning to the origin city with minimum total tour length.",
    function_name="select_next_node",
    signature="def select_next_node(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:",
    docstring="Select the next node to visit from the set of unvisited nodes.",
    input_description="""- current_node (int): Index of the most recently visited city.
- destination_node (int): Origin city to which the tour must eventually return (typically 0).
- unvisited_nodes (set): Set of integer indices of cities not yet visited.
- distance_matrix (np.ndarray): 2D float array of Euclidean distances between all city pairs.""",
    output_description="""- next_node (int): City index chosen from unvisited_nodes to visit next.""",
    initial_code=INITIAL_TSP_CODE,
    evaluation_criteria="Minimize the total Euclidean tour length over standard 2D TSP problem instances.",
)
