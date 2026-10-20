"""Dynamic contact coverage on the default 6 mm banana. Particle radius stays h/4.

Five prescribed-tool motions at sampling seeds 0 and 1. The edge starts above the fruit and enters it. Each motion is sampled every 50 ms:

- oblique: one 0.5 rad cut through the fruit
- shallow cross: a vertical cut to mid-height, then a second cut rotated 90 degrees
- thin sheets: three cuts 12 mm apart, stopped above the board
- reversal: the same edge goes down and back up
- stationary: the knife holds still while the material is given a horizontal speed

Records how far particle centres enter the metal, the strict surface crossings
after the shipped repair, and the gap from each surface vertex to the nearest
particle of another piece. Does not change the contact radius or the penalty.

    python -m scripts.contact_coverage
"""

import json
import sys
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.solver import Solver
from blademaster_warp.surface import _mesh_labels, _strict_crossing_counts
from blademaster_warp.surface_runtime import SurfaceRecorder
from blademaster_warp.topology import components
from scripts.force_convergence import banana_config

OUT = Path("runs/contact_coverage_20261019/entered.json")
CHUNK_S = 0.05


def axis_quat(axis, angle):
    axis = np.asarray(axis, float)
    axis /= np.linalg.norm(axis)
    half = angle / 2
    return np.array([*(axis * np.sin(half)), np.cos(half)])


IDENT = np.array([0.0, 0.0, 0.0, 1.0])
OBLIQUE = axis_quat([0, 1, 0], 0.5)
CROSS = axis_quat([0, 0, 1], np.pi / 2)


def segments():
    """Edge heights sit on the analytic banana: centre z=12 mm, top about 24 mm.

    The first pass stopped the shallow edge at 25 mm, above the fruit, so its force was zero.
    """
    down = [0.0, 0.0, -0.04]
    return {
        "oblique": [
            dict(edge=[0, 0, 0.030], quat=OBLIQUE, vel=[0, 0, -0.05], duration=0.6, cut=0, cutting=True)
        ],
        "shallow-cross": [
            dict(edge=[0, 0, 0.030], quat=IDENT, vel=down, duration=0.4, cut=0, cutting=True),
            dict(edge=[0, 0, 0.014], quat=IDENT, vel=[0, 0, 0.08], duration=0.2, cut=0, cutting=False),
            dict(edge=[0, 0, 0.030], quat=CROSS, vel=down, duration=0.4, cut=1, cutting=True),
        ],
        "thin-sheets": [
            dict(edge=[-0.012, 0, 0.031], quat=IDENT, vel=[0, 0, -0.05], duration=0.6, cut=0, cutting=True),
            dict(edge=[-0.012, 0, 0.001], quat=IDENT, vel=[0, 0, 0.10], duration=0.3, cut=0, cutting=False),
            dict(edge=[-0.012, 0, 0.031], quat=IDENT, vel=[0.12, 0, 0], duration=0.1, cut=0, cutting=False),
            dict(edge=[0, 0, 0.031], quat=IDENT, vel=[0, 0, -0.05], duration=0.6, cut=1, cutting=True),
            dict(edge=[0, 0, 0.001], quat=IDENT, vel=[0, 0, 0.10], duration=0.3, cut=1, cutting=False),
            dict(edge=[0, 0, 0.031], quat=IDENT, vel=[0.12, 0, 0], duration=0.1, cut=1, cutting=False),
            dict(edge=[0.012, 0, 0.031], quat=IDENT, vel=[0, 0, -0.05], duration=0.6, cut=2, cutting=True),
        ],
        "reversal": [
            dict(edge=[0, 0, 0.030], quat=IDENT, vel=down, duration=0.55, cut=0, cutting=True),
            dict(edge=[0, 0, 0.008], quat=IDENT, vel=[0, 0, 0.04], duration=0.55, cut=0, cutting=True),
        ],
        "stationary": [
            dict(
                edge=[0.064, 0, 0.012],
                quat=IDENT,
                vel=[0, 0, 0],
                duration=0.25,
                cut=0,
                cutting=True,
                kick=0.30,
            )
        ],
    }


class Pose:
    def __init__(self, position, quat):
        self.position = np.asarray(position, float)
        self.orientation_xyzw = np.asarray(quat, float)


def tool_origin(edge, quat, half):
    offset = Rotation.from_quat(quat).apply([0.0, 0.0, -half[2]])
    return np.asarray(edge, float) - offset


def metal_intrusion(points, origin, quat, half):
    """Centres inside the rectangular blade slab, and the deepest entry in metres."""
    local = (points - origin) @ Rotation.from_quat(quat).as_matrix()
    on_face = (np.abs(local[:, 1]) <= half[1]) & (np.abs(local[:, 2]) <= half[2])
    depth = half[0] - np.abs(local[:, 0])
    inside = on_face & (depth > 0)
    return int(inside.sum()), float(depth[inside].max()) if inside.any() else 0.0


def opposite_sphere_gap(vertices, vertex_codes, particles, particle_codes, radius):
    """Smallest gap from a surface vertex to another piece's particle sphere.

    Positive means every vertex is outside every other piece's contact sphere.
    """
    vertices = np.asarray(vertices, float)
    particles = np.asarray(particles, float)
    if not len(vertices) or not len(particles):
        return dict(
            min_gap_mm=None, inside_sphere=0, farther_than_1mm=0, with_opposite=0, closer_to_opposite=0
        )
    nearest = np.full(len(vertices), np.inf)
    own = np.full(len(vertices), np.inf)
    codes_p = np.asarray(particle_codes, np.int64)
    for start in range(0, len(vertices), 256):
        vc = np.asarray(vertex_codes[start : start + 256], np.int64)
        dist = np.linalg.norm(vertices[start : start + 256, None, :] - particles[None, :, :], axis=2)
        a = np.repeat(vc[:, None], len(codes_p), axis=1)
        b = np.broadcast_to(codes_p, a.shape).copy()
        conflict = np.zeros(a.shape, dtype=bool)
        while np.any(a) or np.any(b):
            digit_v, digit_p = a % 3, b % 3
            conflict |= (digit_v != 0) & (digit_p != 0) & (digit_v != digit_p)
            a //= 3
            b //= 3
        stop = start + len(vc)
        nearest[start:stop] = np.where(conflict, dist, np.inf).min(axis=1)
        own[start:stop] = np.where(conflict, np.inf, dist).min(axis=1)
    valid = np.isfinite(nearest)
    gap = nearest[valid] - radius
    closer = valid & (nearest < own)
    return dict(
        min_gap_mm=None if not valid.any() else float(gap.min() * 1000),
        inside_sphere=int(np.count_nonzero(gap < 0)) if valid.any() else 0,
        farther_than_1mm=int(np.count_nonzero(gap > 0.001)) if valid.any() else 0,
        with_opposite=int(valid.sum()),
        closer_to_opposite=int(closer.sum()),
        max_opposite_lead_mm=None if not closer.any() else float((own[closer] - nearest[closer]).max() * 1000),
    )


def crossings(points, triangles):
    labels = _mesh_labels(len(points), triangles)
    return _strict_crossing_counts(points, triangles, labels)


def reference_inter_hits(reference, triangles, codes):
    """Strict reference crossings between surface components, with plane depths."""
    from scripts.surface_deformed import intersection_pairs, segment_triangle_pairs

    reference = np.asarray(reference, float)
    triangles = np.asarray(triangles, int)
    codes = np.asarray(codes, np.int64)
    labels = _mesh_labels(len(reference), triangles)
    edges, faces = intersection_pairs(reference, triangles)
    if not len(edges):
        return []
    interior = segment_triangle_pairs(
        reference[edges[:, 0]], reference[edges[:, 1]], reference[triangles[faces]], strict=True
    )
    other = labels[edges[:, 0]] != labels[triangles[faces, 0]]
    rows = []
    for edge, face in zip(edges[interior & other], faces[interior & other]):
        ends = reference[edge]
        tri = reference[triangles[face]]
        normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        length = np.linalg.norm(normal)
        normal = normal / length if length else normal
        signed = (ends - tri[0]) @ normal
        rows.append(
            dict(
                edge=[int(edge[0]), int(edge[1])],
                face=int(face),
                edge_codes=[int(codes[edge[0]]), int(codes[edge[1]])],
                face_codes=[int(c) for c in codes[triangles[face]]],
                edge_mm=float(np.linalg.norm(ends[1] - ends[0]) * 1000),
                signed_mm=[float(v * 1000) for v in signed],
                ends_mm=(ends * 1000).round(3).tolist(),
                face_mm=(tri * 1000).round(3).tolist(),
            )
        )
    return rows


def run_case(name, motion, seed):
    cfg = replace(banana_config(), external_tool=True, sampling_seed=seed)
    if any(abs(seg["duration"] / CHUNK_S - round(seg["duration"] / CHUNK_S)) > 1e-9 for seg in motion):
        raise ValueError(f"{name} durations must be multiples of {CHUNK_S}")
    solver = Solver(cfg)
    port = ExternalToolPort(solver)
    surface = SurfaceRecorder(solver, None, embedding_backend="cuda", ccd_backend="cuda")
    half = np.asarray(cfg.blade_half_size, float)
    radius = cfg.grid_spacing / 4.0
    steps = int(round(CHUNK_S / cfg.dt))
    if abs(steps * cfg.dt - CHUNK_S) > 1e-12:
        raise ValueError("50 ms is not an integer number of substeps")
    t = 0.0
    deepest_mm = 0.0
    most_inside = 0
    peak_force = 0.0
    com0 = solver.x.numpy().mean(axis=0)

    def embed(time_s, origin, quat, cutting, cut):
        surface.update(
            time_s,
            solver.x.numpy(),
            solver.F.numpy(),
            solver.codes.numpy(),
            Pose(origin, quat),
            cutting=cutting,
            cut_index=cut,
            record_frame=False,
        )

    for index, seg in enumerate(motion):
        origin = tool_origin(seg["edge"], seg["quat"], half)
        embed(t, origin, seg["quat"], seg["cutting"], seg["cut"])
        if seg.get("kick"):
            velocity = solver.v.numpy()
            velocity[:, 0] = seg["kick"]
            solver.v.assign(velocity)
        remaining = int(round(seg["duration"] / cfg.dt))
        vel = np.asarray(seg["vel"], float)
        while remaining:
            n = min(steps, remaining)
            reaction = port.advance(
                ToolState(origin, seg["quat"], vel, [0.0, 0.0, 0.0]),
                n * cfg.dt,
                cutting=seg["cutting"],
                cut_index=seg["cut"],
            )
            count, depth = metal_intrusion(solver.x.numpy(), origin, seg["quat"], half)
            most_inside = max(most_inside, count)
            deepest_mm = max(deepest_mm, depth * 1000)
            peak_force = max(peak_force, float(np.linalg.norm(reaction.force_n)))
            origin = origin + vel * (n * cfg.dt)
            remaining -= n
            t += n * cfg.dt
        if index == len(motion) - 1:
            embed(t, origin, seg["quat"], False, seg["cut"])
    current = surface.current
    world = crossings(current["vertices"], current["triangles"])
    reference = crossings(current["reference"], current["triangles"])
    reference_hits = reference_inter_hits(current["reference"], current["triangles"], current["codes"])
    _, piece_sizes = components(solver.nodes.numpy(), solver.weights.numpy(), solver.codes.numpy())
    gap = opposite_sphere_gap(
        current["vertices"], current["codes"], solver.x.numpy(), solver.codes.numpy(), radius
    )
    row = dict(
        motion=name,
        sampling_seed=seed,
        sim_s=t,
        particles=int(solver.n),
        pieces=int(len(piece_sizes)),
        damaged=int(np.count_nonzero(solver.damage.numpy() >= 1)),
        gpu_errors=solver.errors.numpy().astype(int).tolist(),
        max_centres_inside_blade=most_inside,
        max_centre_depth_mm=deepest_mm,
        peak_force_n=peak_force,
        com_displacement_mm=((solver.x.numpy().mean(axis=0) - com0) * 1000).tolist(),
        mean_velocity_mm_s=(solver.v.numpy().mean(axis=0) * 1000).tolist(),
        world_strict_self=world[0],
        world_strict_inter=world[1],
        reference_strict_self=reference[0],
        reference_strict_inter=reference[1],
        reference_inter_hits=reference_hits,
        sphere_gap=gap,
        surface_vertices=int(len(current["vertices"])),
        surface_wall_s=surface.wall_s,
    )
    print(
        f"seed {seed} {name}",
        {k: row[k] for k in ("pieces", "damaged", "peak_force_n", "max_centre_depth_mm", "world_strict_self", "world_strict_inter", "sphere_gap")},
        flush=True,
    )
    return row


def main():
    chosen = sys.argv[1] if len(sys.argv) > 1 else None
    seeds = (int(sys.argv[2]),) if len(sys.argv) > 2 else (0, 1)
    motions = segments()
    if chosen is not None and chosen not in motions:
        raise SystemExit(f"Unknown motion {chosen}")
    output = OUT if chosen is None else OUT.with_name(f"{chosen}-seed{seeds[0]}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        for name, motion in motions.items():
            if chosen is not None and name != chosen:
                continue
            rows.append(run_case(name, motion, seed))
            output.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
