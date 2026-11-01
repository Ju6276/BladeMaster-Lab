"""Extrinsic cohesive response with independent normal contact enforcement."""

import warp as wp

from .cohesive import radial_traction, surface_normal, surface_normal_pullback


@wp.kernel
def adhesive_proximal_response(
    free_gap: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    previous: wp.array(dtype=float),
    parameters: wp.array(dtype=wp.vec3),
    area: wp.array(dtype=float),
    gap: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    maximum: wp.array(dtype=float),
    storage: wp.array(dtype=float),
    fracture: wp.array(dtype=float),
):
    # Parameters: strength [Pa], Gc [J/m²], compliance [m/N]; area [m²].
    # The incremental potential is strictly convex when compliance*area*strength < 2*Gc/strength.
    # Normal compression is a free coordinate; storage and fracture are per reference area.
    i = wp.tid()
    strength, gc, compliance = parameters[i][0], parameters[i][1], parameters[i][2]
    measure = area[i]
    failure = 2.0 * gc / strength
    scaled_strength = measure * strength
    n = wp.normalize(normals[i])
    q = free_gap[i]
    compression = wp.min(wp.dot(q, n), 0.0)
    active = q - compression * n
    radius = wp.length(active)
    old = previous[i]
    r = radius
    gain = float(1.0)
    unloading = bool(False)
    if old < failure:
        if old > 0.0 or measure > 0.0:
            denominator = old + compliance * scaled_strength * (1.0 - old / failure)
            gain = old / denominator
            r = radius * gain
            unloading = True
        if r > old or (old == 0.0 and radius > compliance * scaled_strength):
            unloading = False
            r = (radius - compliance * scaled_strength) / (1.0 - compliance * scaled_strength / failure)
            if r >= failure:
                r = radius
    separated = active
    if unloading:
        separated = gain * active
    elif radius > 0.0:
        separated = r / radius * active
    high = wp.max(old, r)
    energy = float(0.0)
    resultant = wp.vec3(0.0)
    if unloading:
        high = old
        denominator = old + compliance * scaled_strength * (1.0 - old / failure)
        energy = (
            0.5 * strength * (1.0 - old / failure) * old * wp.length_sq(active) / (denominator * denominator)
        )
        resultant = scaled_strength * (1.0 - old / failure) / denominator * active
    elif high > 0.0 and high < failure:
        energy = 0.5 * strength * (1.0 - high / failure) * r
        resultant = radial_traction(active, scaled_strength * (1.0 - high / failure))
    gap[i] = separated + compression * n
    force[i] = resultant
    maximum[i] = high
    storage[i] = energy
    fracture[i] = 0.5 * strength * wp.min(high, failure)


@wp.kernel
def adhesive_surface_response(
    gap: wp.array(dtype=wp.vec3),
    deformation: wp.array(dtype=wp.mat33),
    reference_normal: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3),
    surface_moment: wp.array(dtype=wp.mat33),
):
    # Force is N; surface_moment is N*m and is conjugate to the mean-face deformation.
    i = wp.tid()
    n = surface_normal(deformation[i], reference_normal[i])
    compression = wp.min(wp.dot(gap[i], n), 0.0)
    surface_moment[i] = surface_normal_pullback(deformation[i], reference_normal[i], -compression * force[i])
