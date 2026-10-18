"""Full material-cell quadrature using partition-local reconstructed kinematics."""

from itertools import product

import numpy as np

from .fig5 import (
    _lattice,
    cut_completion_indices,
    middle_tetrahedra,
    moving_piece_ids,
    signed_tetra_volumes,
    topology_diagnostics,
    protocol,
)


def compatible_codes(a, b):
    a, b = np.array(a, copy=True), np.array(b, copy=True)
    result = np.ones(np.broadcast_shapes(a.shape, b.shape), dtype=bool)
    while np.any(a) or np.any(b):
        da, db = a % 3, b % 3
        result &= (da == 0) | (db == 0) | (da == db)
        a //= 3
        b //= 3
    return result


def local_volume(reference, position, pieces, cell_volume, *, codes=None):
    """Integrate det(dx/dX) over every particle's complete reference cell.

    Weighted least squares on the immediate 26-neighbor reference stencil uses
    only the same material region and compatible crack codes. One-sided fits at
    outer faces include the boundary half cells. Rank loss and inverted local
    cells invalidate the integral instead of dropping volume or taking abs(J).
    """
    ref, x, pieces = np.asarray(reference), np.asarray(position), np.asarray(pieces)
    _, grid = _lattice(ref)
    if x.shape != ref.shape or pieces.shape != (len(ref),) or not np.isfinite(x).all():
        raise ValueError("Invalid volume samples")
    if pieces.dtype.kind not in "iu" or np.any((pieces < 0) | (pieces > 2)):
        raise ValueError("Invalid material partition")
    if not np.isfinite(cell_volume) or cell_volume <= 0:
        raise ValueError("Invalid reference cell volume")
    if codes is not None:
        codes = np.asarray(codes)
        if codes.shape != pieces.shape or codes.dtype.kind not in "iu" or np.any(codes < 0):
            raise ValueError("Invalid cut codes")
    normal = np.zeros((len(ref), 3, 3))
    cross = np.zeros_like(normal)
    excluded = np.zeros(len(ref), dtype=int)
    for offset in product((-1, 0, 1), repeat=3):
        if offset == (0, 0, 0):
            continue
        a = tuple(slice(max(0, -d), min(n, n - d)) for d, n in zip(offset, grid.shape))
        b = tuple(slice(max(0, d), min(n, n + d)) for d, n in zip(offset, grid.shape))
        source, target = grid[a].ravel(), grid[b].ravel()
        use = pieces[source] == pieces[target]
        if codes is not None:
            use &= compatible_codes(codes[source], codes[target])
        excluded[source[~use]] += 1
        source, target = source[use], target[use]
        delta = ref[target] - ref[source]
        current = x[target] - x[source]
        weight = 1 / np.einsum("ij,ij->i", delta, delta)
        normal[source] += np.einsum("ni,nj,n->nij", delta, delta, weight)
        cross[source] += np.einsum("ni,nj,n->nij", current, delta, weight)
    eigenvalues = np.linalg.eigvalsh(normal)
    supported = eigenvalues[:, 0] > 1e-10 * np.maximum(1, eigenvalues[:, -1])
    jacobian = np.full(len(ref), np.nan)
    jacobian[supported] = np.linalg.det(cross[supported] @ np.linalg.inv(normal[supported]))
    result = {}
    for p, name in enumerate(("left", "middle", "right")):
        selected = pieces == p
        count = int(selected.sum())
        try:
            tets = middle_tetrahedra(ref, (), pieces=np.where(selected, 1, 0))
        except ValueError:
            tets = np.empty((0, 4), dtype=int)
        crack_crossing = np.zeros(len(tets), dtype=bool)
        if codes is not None:
            for a in range(4):
                for b in range(a + 1, 4):
                    crack_crossing |= ~compatible_codes(codes[tets[:, a]], codes[tets[:, b]])
        volumes = signed_tetra_volumes(x, tets[~crack_crossing])
        nonpositive = int(np.count_nonzero(jacobian[selected] <= 0))
        inverted_tets = int(np.count_nonzero(volumes <= 0))
        unsupported = int(np.count_nonzero(selected & ~supported))
        valid = bool(count and not (unsupported or nonpositive or inverted_tets or crack_crossing.any()))
        initial = count * cell_volume
        volume = float(jacobian[selected].sum() * cell_volume) if valid else None
        result[name] = {
            "valid": valid,
            "particles": count,
            "reference_m3": initial,
            "current_m3": volume,
            "contraction_percent": 100 * (1 - volume / initial) if valid else None,
            "reference_coverage_fraction": 1.0 if count else 0.0,
            "supported_particles": count - unsupported,
            "unsupported_particles": unsupported,
            "nonpositive_jacobians": nonpositive,
            "nonpositive_interior_tetrahedra": inverted_tets,
            "interior_tetrahedra_crossing_cut_codes": int(crack_crossing.sum()),
            "excluded_neighbor_links": int(excluded[selected].sum()),
        }
    return result


def measure_actual_partition(cfg, data):
    ref, x = np.asarray(data["reference"]), np.asarray(data["position"])
    pieces = moving_piece_ids(cfg, x)
    events = protocol(cfg)[1]
    geometry = []
    radius = cfg.grid_spacing / 4
    first, second = cut_completion_indices(cfg)
    first_left = x[first, :, 0] < cfg.cut_x[0]
    for cut, index in enumerate((first, second)):
        parent = np.ones(len(ref), dtype=bool) if cut == 0 else ~first_left
        position = x[index, parent]
        # Full-width/depth insertion must cover the material cells of the parent.
        spans = bool(
            len(position)
            and np.max(np.abs(position[:, 1])) + radius <= cfg.blade_half_length
            and position[:, 2].max() + radius <= 2 * cfg.blade_half_height - 0.002
            and position[:, 2].min() - radius >= -0.002
            and np.any(position[:, 0] < cfg.cut_x[cut])
            and np.any(position[:, 0] > cfg.cut_x[cut])
        )
        geometry.append({"time_s": index * cfg.interval, "blade_spans_parent": spans})
    try:
        tets = middle_tetrahedra(ref, cfg.cut_x, pieces=pieces)
    except ValueError:
        tets = np.empty((0, 4), dtype=int)
    topology = topology_diagnostics(data, pieces, tets, events)
    populated = bool(np.all(np.bincount(pieces, minlength=3) > 0))
    valid = populated and all(row["blade_spans_parent"] for row in geometry)
    valid &= topology["status"] != "mismatched"
    status = ("matched" if topology["status"] == "matched" else "geometric") if valid else "invalid"
    index = round(events["both_cuts_complete"] / cfg.interval)
    codes = None
    if "cut_codes_by_event" in data:
        names = np.asarray(data["topology_event_names"]).tolist()
        codes = data["cut_codes_by_event"][names.index("both_cuts_complete")]
    volumes = local_volume(ref, x[index], pieces, (cfg.grid_spacing / 2) ** 3, codes=codes)
    if not valid:
        for volume in volumes.values():
            volume.update(valid=False, current_m3=None, contraction_percent=None)
    return (
        pieces,
        {
            "status": status,
            "definition": "material membership frozen at each deformed full-depth cut; second cut subdivides remainder",
            "piece_sizes": np.bincount(pieces, minlength=3).tolist(),
            "cut_geometry": geometry,
            "mechanical_topology": topology,
        },
        {
            "definition": "sum of partition-local reconstructed det(dx/dX) times complete reference particle-cell volume",
            "reference_cell_volume_m3": (cfg.grid_spacing / 2) ** 3,
            "reference_total_m3": len(ref) * (cfg.grid_spacing / 2) ** 3,
            "after_cuts": volumes,
        },
    )
