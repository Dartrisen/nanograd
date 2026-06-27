from __future__ import annotations

from typing import Iterable, Protocol


class GraphNode(Protocol):
    _prev: Iterable["GraphNode"]


def topo_sort_iterative(root: GraphNode) -> list[GraphNode]:
    """Returns a topologically ordered list of nodes starting from the root."""
    topo: list[GraphNode] = []
    visited: set[GraphNode] = set()
    stack: list[tuple[GraphNode, bool]] = [(root, False)]

    while stack:
        node, expanded = stack.pop()
        if expanded:
            topo.append(node)
            continue

        if node in visited:
            continue

        visited.add(node)
        stack.append((node, True))
        for child in reversed(list(node._prev)):
            if child not in visited:
                stack.append((child, False))

    return topo
