"""Mechanical connected components from the reference grid and cut codes."""

import numpy as np


def compatible(a, b):
    a, b = int(a), int(b)
    while a or b:
        x, y = a % 3, b % 3
        if x and y and x != y:
            return False
        a //= 3
        b //= 3
    return True


def components(nodes, weights, codes):
    """Label particles connected through shared grid nodes with compatible codes.

    Graph vertices are the particles plus one vertex per (grid node, code) pair.
    Each particle links to the pairs of its stencil nodes; two pairs at the same
    node link when their codes are compatible. Labels are ordered by the lowest
    particle index in each component.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    nodes, weights, codes = np.asarray(nodes), np.asarray(weights), np.asarray(codes, np.int64)
    n = len(codes)
    used = weights > 0
    particle = np.broadcast_to(np.arange(n)[:, None], nodes.shape)[used]
    code_count = int(codes.max()) + 1 if n else 1
    key = nodes[used].astype(np.int64) * code_count + codes[particle]
    unique_keys, key_vertex = np.unique(key, return_inverse=True)
    key_node, key_code = unique_keys // code_count, unique_keys % code_count
    a_side, b_side = [particle], [n + key_vertex]
    distinct = np.unique(key_code)
    for i, a in enumerate(distinct):
        for b in distinct[i + 1 :]:
            if compatible(a, b):
                _, ia, ib = np.intersect1d(
                    key_node[key_code == a], key_node[key_code == b], return_indices=True
                )
                a_side.append(n + np.flatnonzero(key_code == a)[ia])
                b_side.append(n + np.flatnonzero(key_code == b)[ib])
    rows, cols = np.concatenate(a_side), np.concatenate(b_side)
    size = n + len(unique_keys)
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(size, size))
    _, label = connected_components(graph, directed=False)
    # Every (node, code) vertex touches a particle, so each component has one.
    label = label[:n]
    first = np.full(label.max() + 1, n)
    np.minimum.at(first, label, np.arange(n))
    rank = np.empty_like(first)
    rank[np.argsort(first)] = np.arange(len(first))
    labels = rank[label]
    return labels, np.bincount(labels)
