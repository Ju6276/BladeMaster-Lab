"""Cohesive surface energies and constrained incremental responses."""

from dataclasses import dataclass
import math

import warp as wp


@dataclass(frozen=True)
class CohesiveParameters:
    """Stiffness in Pa/m, strength in Pa and fracture energy in J/m²."""

    stiffness: float
    strength: float
    fracture_energy: float

    def __post_init__(self):
        values = (self.stiffness, self.strength, self.fracture_energy)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Cohesive parameters must be finite and positive')
        if self.strength / self.stiffness >= 2 * self.fracture_energy / self.strength:
            raise ValueError('Damage onset must precede complete separation')

    def as_tuple(self):
        return self.stiffness, self.strength, self.fracture_energy


@dataclass(frozen=True)
class ExtrinsicCohesiveParameters:
    """Strength in Pa and fracture energy per reference area in J/m²."""

    strength: float
    fracture_energy: float

    def __post_init__(self):
        if any(not math.isfinite(v) or v <= 0 for v in (self.strength, self.fracture_energy)):
            raise ValueError('Strength and fracture energy must be finite and positive')

    def proximal_parameters(self, compliance):
        """Compliance is m/Pa; the incremental potential must be strictly convex."""
        if not math.isfinite(compliance) or compliance <= 0:
            raise ValueError('Proximal compliance must be finite and positive')
        if compliance * self.strength >= 2 * self.fracture_energy / self.strength:
            raise ValueError('Proximal stiffness must exceed the cohesive softening slope')
        return self.strength, self.fracture_energy, compliance


@wp.func
def surface_area_vector(deformation: wp.mat33, reference_normal: wp.vec3):
    columns = wp.transpose(deformation)
    return (reference_normal[0] * wp.cross(columns[1], columns[2])
            + reference_normal[1] * wp.cross(columns[2], columns[0])
            + reference_normal[2] * wp.cross(columns[0], columns[1]))


@wp.func
def surface_normal(deformation: wp.mat33, reference_normal: wp.vec3):
    # The mapped reference tangents must span a nonzero area.
    return wp.normalize(surface_area_vector(deformation, reference_normal))


@wp.func
def surface_normal_pullback(deformation: wp.mat33, reference_normal: wp.vec3, adj_normal: wp.vec3):
    columns = wp.transpose(deformation)
    f0, f1, f2 = columns[0], columns[1], columns[2]
    a = surface_area_vector(deformation, reference_normal)
    n = wp.normalize(a)
    adj_a = (adj_normal - wp.dot(adj_normal, n) * n) / wp.length(a)
    p0 = reference_normal[1] * wp.cross(adj_a, f2) + reference_normal[2] * wp.cross(f1, adj_a)
    p1 = reference_normal[0] * wp.cross(f2, adj_a) + reference_normal[2] * wp.cross(adj_a, f0)
    p2 = reference_normal[0] * wp.cross(adj_a, f1) + reference_normal[1] * wp.cross(f0, adj_a)
    return wp.matrix_from_cols(p0, p1, p2)


@wp.kernel
def cohesive_contact_response(
    gap: wp.array(dtype=wp.vec3), deformation: wp.array(dtype=wp.mat33),
    reference_normal: wp.array(dtype=wp.vec3), multiplier: wp.array(dtype=float),
    traction: wp.array(dtype=wp.vec3), surface_stress: wp.array(dtype=wp.mat33),
):
    # multiplier <= 0 is the pressure multiplier (Pa) for g·n >= 0.
    # Both forces are derivatives of multiplier * g·n at fixed multiplier;
    # surface_stress (N/m) is conjugate to the mean-face deformation.
    i = wp.tid()
    traction[i] = multiplier[i] * surface_normal(deformation[i], reference_normal[i])
    surface_stress[i] = surface_normal_pullback(
        deformation[i], reference_normal[i], multiplier[i] * gap[i])


@wp.kernel
def cohesive_proximal_response(
    free_gap: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
    previous: wp.array(dtype=float), parameters: wp.array(dtype=wp.vec3),
    gap: wp.array(dtype=wp.vec3), traction: wp.array(dtype=wp.vec3),
    maximum: wp.array(dtype=float), storage: wp.array(dtype=float),
    fracture: wp.array(dtype=float),
):
    # Minimize |g-q|²/(2 compliance) + storage(g, previous) + fracture(g, previous),
    # subject to g·n >= 0. parameters contains validated (strength, G_c, compliance).
    # Normals are nonzero; previous >= 0. Energies are per reference area in J/m².
    i = wp.tid()
    strength, gc, compliance = parameters[i][0], parameters[i][1], parameters[i][2]
    failure = 2.0 * gc / strength
    n = wp.normalize(normals[i])
    q = free_gap[i]
    compression = wp.min(wp.dot(q, n), 0.0)
    positive = q - compression * n
    radius = wp.length(positive)
    old = previous[i]
    r = radius
    gain = float(1.0)
    unloading = bool(False)
    if old < failure:
        denominator = old + compliance * strength * (1.0 - old / failure)
        gain = old / denominator
        r = radius * gain
        unloading = True
        if r > old or (old == 0.0 and radius > compliance * strength):
            unloading = False
            r = (radius - compliance * strength) / (1.0 - compliance * strength / failure)
            if r >= failure:
                r = radius
    g = wp.vec3(0.0)
    if old >= failure:
        g = positive
    elif unloading:
        g = gain * positive
    elif radius > 0.0:
        g = r / radius * positive
    high = wp.max(old, r)
    energy = float(0.0)
    if unloading:
        high = old
        denominator = old + compliance * strength * (1.0 - old / failure)
        energy = 0.5 * strength * (1.0 - old / failure) * old * wp.length_sq(positive) / (denominator * denominator)
    elif high > 0.0 and high < failure:
        energy = 0.5 * strength * (1.0 - high / failure) / high * r * r
    t = compression / compliance * n
    if unloading:
        denominator = old + compliance * strength * (1.0 - old / failure)
        t += strength * (1.0 - old / failure) / denominator * positive
    elif high > 0.0 and high < failure:
        t += strength * (1.0 - high / failure) / high * g
    gap[i] = g
    traction[i] = t
    maximum[i] = high
    storage[i] = energy
    fracture[i] = 0.5 * strength * wp.min(high, failure)


@wp.kernel
def cohesive_measure_proximal_response(
    free_gap: wp.array(dtype=wp.vec3), normals: wp.array(dtype=wp.vec3),
    previous: wp.array(dtype=float), parameters: wp.array(dtype=wp.vec3),
    area: wp.array(dtype=float), gap: wp.array(dtype=wp.vec3),
    force: wp.array(dtype=wp.vec3), maximum: wp.array(dtype=float),
    storage: wp.array(dtype=float), fracture: wp.array(dtype=float),
):
    # Minimize |g-q|²/(2 compliance) + area*(storage + fracture + contact indicator).
    # Parameters are (strength [Pa], Gc [J/m²], compliance [m/N]); area is m².
    # Require area >= 0 and compliance*area*strength < 2*Gc/strength.
    # Zero measure has no contact constraint. Energy outputs remain J/m².
    i = wp.tid()
    strength, gc, compliance = parameters[i][0], parameters[i][1], parameters[i][2]
    measure = area[i]
    failure = 2.0 * gc / strength
    scaled_strength = measure * strength
    n = wp.normalize(normals[i])
    q = free_gap[i]
    compression = wp.min(wp.dot(q, n), 0.0)
    positive = q - compression * n
    radius = wp.length(positive)
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
    g = positive
    if unloading:
        g = gain * positive
    elif radius > 0.0:
        g = r / radius * positive
    high = wp.max(old, r)
    energy = float(0.0)
    resultant = wp.vec3(0.0)
    if measure > 0.0:
        resultant = compression / compliance * n
    else:
        g += compression * n
    if unloading:
        high = old
        denominator = old + compliance * scaled_strength * (1.0 - old / failure)
        energy = 0.5 * strength * (1.0 - old / failure) * old * wp.length_sq(positive) / (denominator * denominator)
        resultant += scaled_strength * (1.0 - old / failure) / denominator * positive
    elif high > 0.0 and high < failure:
        energy = 0.5 * strength * (1.0 - high / failure) / high * r * r
        resultant += scaled_strength * (1.0 - high / failure) / high * (r / radius * positive)
    gap[i] = g
    force[i] = resultant
    maximum[i] = high
    storage[i] = energy
    fracture[i] = 0.5 * strength * wp.min(high, failure)


@wp.kernel
def cohesive_response(
    gap: wp.array(dtype=wp.vec3), deformation: wp.array(dtype=wp.mat33),
    normal: wp.array(dtype=wp.vec3), previous: wp.array(dtype=float),
    parameters: wp.array(dtype=wp.vec3), traction: wp.array(dtype=wp.vec3),
    surface_stress: wp.array(dtype=wp.mat33), maximum: wp.array(dtype=float),
    damage: wp.array(dtype=float), energy: wp.array(dtype=float), fracture: wp.array(dtype=float),
):
    # deformation maps reference tangents to the mean of the two current faces.
    # Its cofactor applied to normal must be nonzero; previous is nonnegative.
    # traction is Pa; surface_stress is N/m; both energy outputs are J/m².
    # parameters contains a validated (stiffness, strength, fracture energy).
    i = wp.tid()
    reference_normal = normal[i]
    n = surface_normal(deformation[i], reference_normal)
    g = gap[i]
    compression = wp.min(wp.dot(g, n), 0.0)
    active = g - compression * n
    k, strength, gc = parameters[i][0], parameters[i][1], parameters[i][2]
    onset, failure = strength / k, 2.0 * gc / strength
    high = wp.max(previous[i], wp.length(active))
    d = float(0.0)
    if high >= failure:
        d = 1.0
    elif high > onset:
        d = failure * (high - onset) / (high * (failure - onset))
    traction[i] = k * ((1.0 - d) * g + d * compression * n)
    surface_stress[i] = surface_normal_pullback(deformation[i], reference_normal, k * d * compression * g)
    maximum[i] = high
    damage[i] = d
    energy[i] = 0.5 * k * ((1.0 - d) * wp.length_sq(active) + compression * compression)
    fracture[i] = gc * wp.clamp((high - onset) / (failure - onset), 0.0, 1.0)
