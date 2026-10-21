"""Continuous locations of bracketed events within a material step."""

import warp as wp


@wp.func
def crossing_fraction(before: float, after: float, level: float):
    """A rising crossing has before < level <= after; -1 denotes no new event."""
    fraction = float(-1.0)
    if before < level and after >= level:
        fraction = (level - before) / (after - before)
    return fraction


@wp.kernel
def threshold_times(
    before: wp.array(dtype=float),
    trial: wp.array(dtype=float),
    start: float,
    dt: float,
    times: wp.array(dtype=float),
):
    p = wp.tid()
    fraction = crossing_fraction(before[p], trial[p], 1.0)
    times[p] = -1.0
    if fraction >= 0.0:
        times[p] = start + dt * fraction


@wp.kernel
def edge_crossings(
    x_before: wp.array(dtype=wp.vec3),
    x_after: wp.array(dtype=wp.vec3),
    edge_before: wp.array(dtype=wp.vec3),
    edge_after: wp.array(dtype=wp.vec3),
    normal: wp.vec3,
    start: float,
    dt: float,
    times: wp.array(dtype=float),
    positions: wp.array(dtype=wp.vec3),
):
    """The edge-plane normal is fixed over the step; positions use world coordinates in metres."""
    p = wp.tid()
    before = wp.dot(x_before[p] - edge_before[0], normal)
    after = wp.dot(x_after[p] - edge_after[0], normal)
    fraction = crossing_fraction(before, after, 0.0)
    times[p] = -1.0
    positions[p] = wp.vec3(0.0)
    if fraction >= 0.0:
        times[p] = start + dt * fraction
        positions[p] = edge_before[0] + fraction * (edge_after[0] - edge_before[0])
