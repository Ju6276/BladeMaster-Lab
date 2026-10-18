"""SI fixture and particle-based measurements for the two-cut Fig. 5 experiment."""

from dataclasses import asdict, dataclass
from itertools import permutations
import numpy as np

from .geometry import points


PAPER_URL = "https://arxiv.org/html/2609.27342v2#S4.F5"
PAPER_RESULTS = {
    "blademaster": {"mean_gap_mm": 0.654, "middle_contraction_percent": 0.506},
    "cressim_mpm": {"mean_gap_mm": 4.059, "middle_contraction_percent": 8.746},
}


@dataclass(frozen=True)
class Fig5Config:
    size: tuple = (0.06, 0.03, 0.03)
    grid_spacing: float = 0.005
    dt: float = 0.00025
    interval: float = 0.002
    young_modulus: float = 6000.0
    poisson_ratio: float = 0.4
    density: float = 1100.0
    cut_x: tuple = (-0.01, 0.01)
    blade_thickness: float = 0.0005
    blade_half_length: float = 0.025
    blade_half_height: float = 0.025
    clearance: float = 0.01
    left_velocity: float = -0.03
    manipulation_s: float = 0.4
    friction: float = 0.0
    cressim_smooth_distance: float = 0.0
    cressim_drag: float = 1.0
    cressim_mass_unit_kg: float = 1.0

    def __post_init__(self):
        values = np.concatenate([np.atleast_1d(v) for v in asdict(self).values()])
        if not np.isfinite(values).all():
            raise ValueError("Fig5 configuration must be finite")
        if len(self.size) != 3 or len(self.cut_x) != 2:
            raise ValueError("Expected three dimensions and two cuts")
        positive = (
            self.dt,
            self.interval,
            self.grid_spacing,
            self.young_modulus,
            self.density,
            self.blade_half_height,
            self.blade_half_length,
            self.clearance,
            self.manipulation_s,
            self.cressim_mass_unit_kg,
        )
        if min(positive) <= 0 or min(self.size) < 2 * self.grid_spacing:
            raise ValueError("Invalid positive parameter or fewer than two cells per axis")
        if not 0 < self.poisson_ratio < 0.5 or self.left_velocity >= 0:
            raise ValueError("Require 0 < Poisson ratio < .5 and a leftward velocity")
        if not np.isclose(self.blade_thickness, 0.0005, atol=1e-12, rtol=0):
            raise ValueError("Fig5 uses a 0.5 mm blade")
        if not -self.size[0] / 2 < self.cut_x[0] < self.cut_x[1] < self.size[0] / 2:
            raise ValueError("Cut planes must be ordered inside the block")
        if self.blade_half_length < self.size[1] / 2 or 2 * self.blade_half_height < self.size[2] + 0.002:
            raise ValueError("Blade must span the block width and full cut depth")
        if min(self.friction, self.cressim_smooth_distance, self.cressim_drag) < 0:
            raise ValueError("Negative contact parameter")
        if self.friction != 0 or self.cressim_drag != 1:
            raise ValueError("The post-withdrawal momentum test requires friction=0 and cressim_drag=1")
        integer_steps(self.interval, self.dt)
        for duration in (0.1, 0.2, 0.4, 0.6, self.manipulation_s):
            integer_steps(duration, self.interval)
        dims = np.asarray(self.size) / self.grid_spacing
        if not np.allclose(dims, np.round(dims), atol=1e-9, rtol=0):
            raise ValueError("Grid spacing must divide all block dimensions")


def integer_steps(duration, dt):
    n = round(duration / dt)
    if n < 1 or not np.isclose(n * dt, duration, atol=1e-12, rtol=1e-10):
        raise ValueError("Duration must be an integer number of steps")
    return n


def reference_particles(cfg):
    return points(cfg.size, cfg.grid_spacing, "block", 0)[0]


def piece_ids(reference, cuts):
    """Material membership is fixed in reference space, including after deformation."""
    return np.searchsorted(cuts, np.asarray(reference)[:, 0], side="right")


def initial_left_velocity(velocity, pieces, speed):
    """Assign the left piece's initial velocity; preserve all other particles bitwise."""
    result = np.array(velocity, dtype=np.result_type(np.asarray(velocity).dtype, np.float32), copy=True)
    if (
        result.shape != (len(pieces), 3)
        or not np.isfinite(result).all()
        or not np.isfinite(speed)
        or speed >= 0
    ):
        raise ValueError("Invalid initial velocity")
    result[np.asarray(pieces) == 0] = (speed, 0.0, 0.0)
    return result


def protocol(cfg):
    """Segments are (end time, blade centre, active cut). Poses are held per interval."""
    top = cfg.size[2]
    hz = cfg.blade_half_height
    rows = [(0.0, [cfg.cut_x[0], 0.0, top + cfg.clearance + hz], -1)]
    events = {}

    def segment(duration, x, edge_z, cut):
        rows.append((round(rows[-1][0] + duration, 10), [x, 0.0, edge_z + hz], cut))

    segment(0.1, cfg.cut_x[0], top + cfg.clearance, -1)
    for index, x in enumerate(cfg.cut_x):
        if index:
            segment(0.2, x, top + cfg.clearance, -1)
        segment(0.2, x, top, index)
        segment(0.4, x, top / 2, index)
        if not index:
            events["first_half_depth"] = rows[-1][0]
        segment(0.4, x, 0.0, index)
        segment(0.1, x, -0.002, index)
        segment(0.1, x, -0.002, index)
        if index:
            events["both_cuts_complete"] = rows[-1][0]
        segment(0.6, x, top + cfg.clearance, -1)
    events["withdrawn"] = rows[-1][0]
    segment(cfg.manipulation_s, cfg.cut_x[1], top + cfg.clearance, -1)
    events["final"] = rows[-1][0]
    return rows, events


def commands(cfg):
    rows, events = protocol(cfg)
    times = np.arange(1, integer_steps(events["final"], cfg.interval) + 1) * cfg.interval
    ends = np.array([r[0] for r in rows])
    result = []
    for t in times:
        j = min(np.searchsorted(ends + 1e-10, t), len(rows) - 1)
        a, b = rows[j - 1], rows[j]
        v = (np.asarray(b[1]) - a[1]) / (b[0] - a[0])
        p = np.asarray(a[1]) + (t - a[0]) * v
        result.append((t, p, v, b[2]))
    return result


def _lattice(reference):
    ref = np.asarray(reference, dtype=float)
    if ref.ndim != 2 or ref.shape[1] != 3 or not np.isfinite(ref).all():
        raise ValueError("Expected finite particle positions")
    axes = [np.unique(ref[:, i]) for i in range(3)]
    if any(len(a) < 2 for a in axes) or np.prod([len(a) for a in axes]) != len(ref):
        raise ValueError("Expected a complete Cartesian particle lattice")
    grid = np.full(tuple(len(a) for a in axes), -1, dtype=int)
    ijk = tuple(np.searchsorted(axes[i], ref[:, i]) for i in range(3))
    grid[ijk] = np.arange(len(ref))
    if (grid < 0).any():
        raise ValueError("Duplicate or missing lattice particles")
    return axes, grid


def middle_tetrahedra(reference, cuts):
    """Six tetrahedra per particle-lattice cell wholly within the middle piece."""
    _, grid = _lattice(reference)
    base = np.indices(np.asarray(grid.shape) - 1).reshape(3, -1).T
    tets = []
    for order in permutations(range(3)):
        offsets = np.zeros((4, 3), dtype=int)
        for j, axis in enumerate(order):
            offsets[j + 1] = offsets[j]
            offsets[j + 1, axis] += 1
        coords = base[:, None, :] + offsets
        tets.append(grid[coords[..., 0], coords[..., 1], coords[..., 2]])
    tets = np.concatenate(tets)
    tets = tets[np.all(piece_ids(reference, cuts)[tets] == 1, axis=1)]
    if len(tets) == 0:
        raise ValueError("Middle piece needs at least two particle columns")
    volumes = signed_tetra_volumes(reference, tets)
    negative = volumes < 0
    tets[negative, 1], tets[negative, 2] = tets[negative, 2].copy(), tets[negative, 1].copy()
    return tets


def signed_tetra_volumes(position, tetrahedra):
    p = np.asarray(position, dtype=float)[tetrahedra]
    return np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) / 6


def half_depth_gap(reference, position, cut_x, edge_z):
    """Signed x-opening of paired columns above the edge, minus their rest spacing."""
    axes, grid = _lattice(reference)
    j = np.searchsorted(axes[0], cut_x)
    if j == 0 or j == len(axes[0]) or np.isclose(axes[0][j], cut_x, atol=1e-12, rtol=0):
        raise ValueError("Cut must lie strictly between particle columns")
    left, right = grid[j - 1].ravel(), grid[j].ravel()
    use = np.asarray(reference)[left, 2] >= edge_z
    left, right = left[use], right[use]
    if not len(left):
        raise ValueError("No paired samples above blade edge")
    distance = position[right, 0] - position[left, 0]
    rest = reference[right, 0] - reference[left, 0]
    opening = distance - rest
    return {
        "mean_opening_increment_mm": float(opening.mean() * 1000),
        "min_opening_increment_mm": float(opening.min() * 1000),
        "max_opening_increment_mm": float(opening.max() * 1000),
        "mean_projected_column_separation_mm": float(distance.mean() * 1000),
        "mean_column_distance_mm": float(
            np.linalg.norm(position[right] - position[left], axis=1).mean() * 1000
        ),
        "mean_reference_column_spacing_mm": float(rest.mean() * 1000),
        "crossed_column_pairs": int((distance <= 0).sum()),
        "pairs": len(left),
        "definition": "paired-column x separation minus reference separation, reference z >= half height",
    }


def evaluate(cfg, data):
    """Evaluate exact event frames; reject partial or nonfinite trajectories."""
    ref = np.asarray(data["reference"], dtype=float)
    time = np.asarray(data["time"], dtype=float)
    x, v = np.asarray(data["position"]), np.asarray(data["velocity"])
    impulse = np.asarray(data["blade_impulse_ns"])
    board_impulse = np.asarray(data["board_impulse_ns"])
    pair_impulse = np.asarray(data["pair_impulse_by_piece_ns"])
    if x.shape != (len(time), len(ref), 3) or v.shape != x.shape or impulse.shape != (len(time), 3):
        raise ValueError("Invalid trajectory shapes")
    if board_impulse.shape != impulse.shape or pair_impulse.shape != (len(time), 3, 3):
        raise ValueError("Invalid contact impulse shapes")
    expected = np.arange(len(commands(cfg)) + 1) * cfg.interval
    if time.shape != expected.shape or not np.allclose(time, expected, rtol=0, atol=1e-10):
        raise ValueError("Missing or misaligned event frames")
    if not all(np.isfinite(a).all() for a in (ref, time, x, v, impulse, board_impulse, pair_impulse)):
        raise ValueError("Nonfinite trajectory")
    if not np.array_equal(ref, reference_particles(cfg)):
        raise ValueError("Particle sampling differs from fixture")
    if not np.allclose(x[0], ref, atol=1e-8, rtol=0) or np.any(v[0] != 0):
        raise ValueError("Initial state differs from the stationary fixture")
    events = protocol(cfg)[1]
    idx = {k: round(t / cfg.interval) for k, t in events.items()}
    pieces = piece_ids(ref, cfg.cut_x)
    tets = middle_tetrahedra(ref, cfg.cut_x)
    rest = signed_tetra_volumes(ref, tets)
    volume = signed_tetra_volumes(x[idx["both_cuts_complete"]], tets)
    displacement = x[-1] - x[idx["withdrawn"]]
    motions = {}
    for p, name in enumerate(("left", "middle", "right")):
        d = displacement[pieces == p]
        motions[name] = {
            "particles": len(d),
            "com_displacement_m": d.mean(0).tolist(),
            "mean_displacement_norm_m": float(np.linalg.norm(d, axis=1).mean()),
        }
    before = np.asarray(data["velocity_before_kick"])
    after = np.asarray(data["velocity_after_kick"])
    expected_kick = initial_left_velocity(before, pieces, cfg.left_velocity)
    if not np.array_equal(after, expected_kick) or not np.array_equal(before, v[idx["withdrawn"]]):
        raise ValueError("Velocity assignment changed middle/right particles or wrong event")
    moving = impulse[idx["withdrawn"] + 1 :]
    board_moving = board_impulse[idx["withdrawn"] + 1 :]
    pair_moving = pair_impulse[idx["withdrawn"] + 1 :]
    clearance = cfg.size[2] + cfg.clearance - float(x[idx["withdrawn"] :, :, 2].max())
    # Check external contacts and global momentum closure before attributing a
    # subset's momentum change to exchange with other material regions.
    contact_clear = bool(
        np.linalg.norm(moving, axis=1).sum() < 1e-9
        and np.linalg.norm(board_moving[:, :2], axis=1).sum() < 1e-7
        and clearance > cfg.grid_spacing / 4
    )
    mass = cfg.density * (cfg.grid_spacing / 2) ** 3
    total_change = mass * (v[-1].astype(float) - after).sum(0)
    gravity_impulse = np.array([0.0, 0.0, -9.81 * mass * len(ref) * cfg.manipulation_s])
    residual = total_change - gravity_impulse - board_moving.sum(0) + moving.sum(0)
    tolerance = max(1e-8, 1e-3 * mass * np.linalg.norm(after.astype(float), axis=1).sum())
    momentum_closed = bool(np.linalg.norm(residual[:2]) <= tolerance)
    clear = contact_clear and momentum_closed
    for p, name in enumerate(("left", "middle", "right")):
        change = mass * (v[-1, pieces == p].astype(float) - after[pieces == p]).sum(0)
        pair = pair_moving[:, p].sum(0)
        motions[name]["momentum_change_ns"] = change.tolist()
        motions[name]["explicit_pair_contact_impulse_ns"] = pair.tolist()
        motions[name]["material_exchange_horizontal_ns"] = change[:2].tolist() if clear else None
        motions[name]["transfer_residual_horizontal_ns"] = (change[:2] - pair[:2]).tolist() if clear else None
    return {
        "events_s": events,
        "first_half_depth_opening": half_depth_gap(
            ref, x[idx["first_half_depth"]], cfg.cut_x[0], cfg.size[2] / 2
        ),
        "middle_volume": {
            "reference_m3": float(rest.sum()),
            "after_cuts_m3": float(np.abs(volume).sum()),
            "contraction_percent": float(100 * (1 - np.abs(volume).sum() / rest.sum())),
            "inverted_or_degenerate_tetrahedra": int((volume <= 0).sum()),
            "reference_region_coverage_fraction": float(
                rest.sum() / ((cfg.cut_x[1] - cfg.cut_x[0]) * cfg.size[1] * cfg.size[2])
            ),
            "tetrahedra": len(tets),
            "definition": "interior reference particle-lattice tetrahedra",
        },
        "manipulation": motions,
        "horizontal_exchange_isolated": clear,
        "momentum_balance": {
            "horizontal_external_contacts_clear": contact_clear,
            "horizontal_balance_closed": momentum_closed,
            "horizontal_tolerance_ns": float(tolerance),
            "residual_ns": residual.tolist(),
        },
        "initial_velocity_assignment": {
            "left_particles": int((pieces == 0).sum()),
            "changed_middle_right_particles": 0,
            "left_velocity_m_s": [cfg.left_velocity, 0.0, 0.0],
        },
        "blade_contact": {
            "minimum_vertical_clearance_after_withdrawal_m": clearance,
            "total_impulse_ns": impulse.sum(0).tolist(),
            "sum_interval_impulse_norms_ns": float(np.linalg.norm(impulse, axis=1).sum()),
            "after_withdrawal_impulse_ns": moving.sum(0).tolist(),
            "after_withdrawal_sum_interval_norms_ns": float(np.linalg.norm(moving, axis=1).sum()),
        },
        "board_contact": {
            "total_impulse_on_material_ns": board_impulse.sum(0).tolist(),
            "after_withdrawal_impulse_on_material_ns": board_moving.sum(0).tolist(),
            "after_withdrawal_sum_interval_norms_ns": float(np.linalg.norm(board_moving, axis=1).sum()),
        },
    }
