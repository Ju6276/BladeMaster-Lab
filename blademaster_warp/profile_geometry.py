"""Symmetric two-slope finite blade, with an explicit edge shoulder width."""

import warp as wp

PROFILE_REVISION = "two_slope_blade_v1"


@wp.func
def profile_query(local: wp.vec3, half: wp.vec3, bevel: float, neck: float):
    q = wp.vec2(wp.abs(local[0]), local[2])
    sign = float(1.0)
    if local[0] < 0.0:
        sign = -1.0
    a = wp.vec2(0.0, -half[2])
    b = wp.vec2(neck, -half[2] + bevel)
    c = wp.vec2(half[0], half[2])
    edge = b - a
    t = wp.clamp(wp.dot(q - a, edge) / wp.dot(edge, edge), 0.0, 1.0)
    near = a + t * edge
    best_sq = wp.length_sq(q - near)
    out = wp.normalize(wp.vec2(edge[1], -edge[0]))
    edge2 = c - b
    t2 = wp.clamp(wp.dot(q - b, edge2) / wp.dot(edge2, edge2), 0.0, 1.0)
    candidate = b + t2 * edge2
    distance_sq = wp.length_sq(q - candidate)
    if distance_sq < best_sq:
        best_sq = distance_sq
        near = candidate
        out = wp.normalize(wp.vec2(edge2[1], -edge2[0]))
    candidate = wp.vec2(wp.clamp(q[0], 0.0, half[0]), half[2])
    distance_sq = wp.length_sq(q - candidate)
    if distance_sq < best_sq:
        best_sq = distance_sq
        near = candidate
        out = wp.vec2(0.0, 1.0)
    slope = (half[0] - neck) / (2.0 * half[2] - bevel)
    width = neck + slope * (q[1] + half[2] - bevel)
    inside = (
        q[1] >= -half[2] and q[1] <= half[2] and bevel * q[0] <= neck * (q[1] + half[2]) and q[0] <= width
    )
    best = wp.sqrt(best_sq)
    closest = wp.vec3(sign * near[0], wp.clamp(local[1], -half[1], half[1]), near[1])
    normal = wp.vec3(sign * out[0], 0.0, out[1])
    distance = float(0.0)
    if inside and wp.abs(local[1]) <= half[1]:
        distance = -best
        if half[1] - wp.abs(local[1]) < best:
            side = float(1.0)
            if local[1] < 0.0:
                side = -1.0
            closest = wp.vec3(local[0], side * half[1], local[2])
            normal = wp.vec3(0.0, side, 0.0)
            distance = -(half[1] - wp.abs(local[1]))
    else:
        if inside:
            closest = wp.vec3(local[0], closest[1], local[2])
        delta = local - closest
        distance = wp.length(delta)
        if distance > 1.0e-12:
            normal = delta / distance
    return closest, normal, distance
