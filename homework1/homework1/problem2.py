"""
Homework 1 part 2 solution
"""


from functools import cache

import networkx as nx

from .graph_search import BreadthFirstSearch, DepthFirstSearch

_MAZE_START_NODE = 0
"""
The starting node of the maze.
"""
_MAZE_END_NODE = 17
"""
The ending node of the maze.
"""


@cache
def make_maze_graph() -> nx.Graph:
    """
    Creates a graph representing the maze.

    Returns:
        The graph that it created.

    """
    graph = nx.Graph()
    graph.add_nodes_from(range(25))
    graph.add_edges_from(
        [
            (1, 0),
            (0, 2),
            (2, 3),
            (2, 4),
            (4, 5),
            (5, 7),
            (5, 6),
            (4, 8),
            (8, 9),
            (8, 10),
            (10, 12),
            (10, 11),
            (11, 13),
            (11, 14),
            (14, 15),
            (14, 16),
            (16, 17),
            (16, 18),
            (18, 19),
            (18, 20),
            (20, 21),
            (20, 22),
            (22, 23),
            (22, 24),
        ]
    )

    return graph


def main() -> None:
    graph = make_maze_graph()

    # Use both search algorithms to find a path through the maze.
    bfs_paths = BreadthFirstSearch(graph).find_paths(
        _MAZE_START_NODE, _MAZE_END_NODE
    )
    dfs_paths = DepthFirstSearch(graph).find_paths(
        _MAZE_START_NODE, _MAZE_END_NODE
    )

    print(f"BFS Paths: {bfs_paths}\nDFS Paths: {dfs_paths}")
