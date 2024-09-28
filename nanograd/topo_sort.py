# topological graph sort

def topo_sort_iterative(start):
    topo = []
    visited = set()
    stack = [start]

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
