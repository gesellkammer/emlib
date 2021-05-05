from __future__ import annotations


def depth_first(start, end, neighborsfunc, maxsize=0):
    """
    Find ways between start and end

    Args:
        start: the start node
        end: the end node
        neighborsfunc: a function returning the neighbors of a node
            neighbors = neighborsfunc(node)
        maxsize: the max. size of the connection. 0 to leave it unbound

    Returns:
        a list of connections, where a conection is a list nodes starting
        with start and ending with end
    """

    visited = set()
    visited.add(start)

    nodestack = []
    indexstack = []
    current = start
    i = 0

    while True:
        # get a list of the neighbors of the current node
        neighbors = neighborsfunc(current)

        # find the next unvisited neighbor of this node, if any
        while i < len(neighbors) and neighbors[i] in visited: 
            i += 1

        if i >= len(neighbors) or (maxsize > 0 and len(nodestack) >= maxsize):
            # we've reached the last neighbor of this node, backtrack
            visited.remove(current)
            if len(nodestack) < 1: 
                break  # can't backtrack, stop!
            current = nodestack.pop()
            i = indexstack.pop()
        elif neighbors[i] == end:
            # yay, we found the target node! let the caller process the path
            yield nodestack+[current, end]
            i += 1
        else:
            # push current node and index onto stacks, switch to neighbor
            nodestack.append(current)
            indexstack.append(i+1)
            visited.add(neighbors[i])
            current = neighbors[i]
            i = 0