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


class StablePieces:
    """Piece id per particle that survives later cuts.

    Component labels are renumbered whenever a cut adds a piece. Each previous
    piece passes its id to the new component sharing most particles with it;
    the other components get ids never used before.
    """

    def __init__(self):
        self.ids = None
        self.next = 1

    def update(self, labels):
        previous = self.ids
        if labels.max() == 0 or previous is None or len(previous) != len(labels):
            self.next = int(labels.max()) + 1
            self.ids = labels.copy()
            return self.ids
        base = int(previous.max()) + 1
        pairs, counts = np.unique(labels.astype(np.int64) * base + previous, return_counts=True)
        mapping = np.full(int(labels.max()) + 1, -1)
        taken = set()
        for i in np.argsort(-counts, kind="stable"):
            component, old = divmod(int(pairs[i]), base)
            if mapping[component] < 0 and old not in taken:
                mapping[component] = old
                taken.add(old)
        for component in np.flatnonzero(mapping < 0):
            mapping[component] = self.next
            self.next += 1
        self.ids = mapping[labels]
        return self.ids


def recorded_pieces(config, codes_per_frame):
    """Stable piece ids for saved side codes, from the run's reference stencil."""
    from .geometry import points, stencil

    reference, origin, cells = points(
        config["size"], config["grid_spacing"], config["shape"], config.get("sampling_seed", 0)
    )
    nodes, weights, _ = stencil(reference, origin, config["grid_spacing"], cells)
    pieces, last, out = StablePieces(), None, []
    for codes in codes_per_frame:
        codes = np.asarray(codes)
        if len(codes) != len(reference):
            raise ValueError("Saved codes do not match the run's reference particles")
        if last is None or not np.array_equal(codes, last):
            ids = pieces.update(components(nodes, weights, codes)[0])
            last = codes
        out.append(ids)
    return out


def piece_rgb(ids, base_rgb):
    """Colors as in the browser viewer: piece 0 keeps the material color, new pieces get new hues.

    base_rgb is one color or one per particle (e.g. cake layers).
    """
    import colorsys

    ids = np.asarray(ids)
    rgb = np.array(np.broadcast_to(np.asarray(base_rgb, float), (len(ids), 3)))
    for piece in np.unique(ids[ids != 0]):
        rgb[ids == piece] = colorsys.hls_to_rgb((0.55 + piece * 0.618034) % 1, 0.52, 0.62)
    return rgb
