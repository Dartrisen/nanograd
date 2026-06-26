from __future__ import annotations

from typing import Protocol, Iterable

class GraphNode(Protocol):
    _prev: Iterable["GraphNode"]

def topo_sort_iterative(root: GraphNode) -> list[GraphNode]:
    """Returns a topologically sorted list of nodes starting from the root."""
    topo: list[GraphNode] = []
    visited: set[GraphNode] = set()
    stack: list[GraphNode] = [root]

    while stack:
        node = stack.pop()
        if node in visited:
            topo.append(node)
            continue

        visited.add(node)
        stack.append(node)

        for child in node._prev:
            if child not in visited:
                stack.append(child)

    return topo
