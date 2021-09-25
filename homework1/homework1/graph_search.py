"""
Implements various graph searches.
"""


import abc
import collections
from functools import lru_cache
from typing import Any, Iterable, List

import networkx as nx


class GraphSearch(abc.ABC):
    """
    Common superclass for graph searches.
    """

    def __init__(self, graph: nx.Graph):
        """
        Args:
            graph: The graph to search.

        """

        # The graph to search.
        self.__graph = graph

        # Set of nodes that have been already visited.
        self.__expanded = set()
        # Maps a particular node to its ancestors. This is used for
        # generating paths.
        self.__node_ancestors = {}

    def __update_ancestor(self, parent: Any, child: Any) -> None:
        """
        Updates the recorded ancestors of a node.

        Args:
            parent: The parent node.
            child: The child node.

        """
        if ancestors := self.__node_ancestors.get(child) is None:
            ancestors = set()
            self.__node_ancestors[child] = ancestors

        ancestors.add(parent)

    def __expand_node(self, node: Any) -> None:
        """
        Expands a single node.

        Args:
            node: The node to expand.

        """
        for neighbor in self._graph.neighbors(node):
            if neighbor in self.__expanded:
                # We've already expanded this node.
                continue

            # Mark this node for future expansion.
            self._add_to_frontier(neighbor)
            # Record the backwards path.
            self.__update_ancestor(node, neighbor)

        # Record that we've expanded this node.
        self.__expanded.add(node)

    def __build_paths(self, start: Any, end: Any) -> List[List[Any]]:
        """
        Builds all the paths from a start node to an end node after the
        search has been completed.

        Args:
            start: The starting node.
            end: The ending node.

        Returns:
            The list of paths that it found.

        """

        @lru_cache()
        def _build_paths(_start: Any, _end: Any) -> List[List[Any]]:
            """
            Memoized version of `__build_paths`.

            """
            # Build up paths backwards by following back-references.
            parents = self.__node_ancestors.get(_end)
            if parents is None:
                # This must be a starting node.
                return [[_start]]

            paths = []
            for parent in parents:
                # Find all paths to the parent node.
                partial_paths = self.__build_paths(_start, parent)
                # Add the next node.
                for path in partial_paths:
                    path.append(_end)

                paths.extend(partial_paths)

            return paths

        return _build_paths(start, end)

    @property
    def _graph(self) -> nx.Graph:
        """
        Returns:
            The underlying graph that we are searching.

        """
        return self.__graph

    @abc.abstractmethod
    def _expandable_nodes(self) -> Iterable[Any]:
        """
        Iterates through graph nodes in the order that they should be expanded.

        Yields:
            The next node that should be expanded in the search.

        """

    @abc.abstractmethod
    def _add_to_frontier(self, node: Any) -> None:
        """
        Adds a particular node to the frontier, which should slate it for
        later expansion.

        Args:
            node: The node to add.

        """

    def find_paths(self, start: Any, end: Any) -> List[List[Any]]:
        """
        Finds all paths between a start and end node.

        Notes:
            If the algorithm is optimal, it will produce all optimal paths.
            Otherwise, it will produce an arbitrary subset of all possible
            paths.

        Args:
            start: The starting node.
            end: The ending node.

        Returns:
            The list of all possible paths between the two nodes.

        """
        # At first, only the starting node is in the frontier.
        self._add_to_frontier(start)

        for node in self._expandable_nodes():
            self.__expand_node(node)

            # Check to see if we've found a path and can stop prematurely.
            if end in self.__expanded:
                break

        return self.__build_paths(start, end)


class BreadthFirstSearch(GraphSearch):
    """
    Implements a breadth-first search algorithm.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(*args, **kwargs)

        # The FIFO queue to use for the search.
        self.__queue = collections.deque()

    def _expandable_nodes(self) -> Iterable[Any]:
        while len(self.__queue) > 0:
            yield self.__queue.pop()

    def _add_to_frontier(self, node: Any) -> None:
        self.__queue.appendleft(node)


class DepthFirstSearch(GraphSearch):
    """
    Implements a depth-first search algorithm.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """
        Args:
            *args: Will be forwarded to the superclass.
            **kwargs: Will be forwarded to the superclass.

        """
        super().__init__(*args, **kwargs)

        # The stack to use for the search.
        self.__stack = []

    def _expandable_nodes(self) -> Iterable[Any]:
        while len(self.__stack) > 0:
            yield self.__stack.pop()

    def _add_to_frontier(self, node: Any) -> None:
        self.__stack.append(node)
