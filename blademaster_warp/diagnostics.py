"""Kinetic, elastic and gravitational energy of the material state."""

import warp as wp
import numpy as np


@wp.kernel
def metrics(
    x: wp.array(dtype=wp.vec3),
    v: wp.array(dtype=wp.vec3),
    F: wp.array(dtype=wp.mat33),
    Fp: wp.array(dtype=wp.mat33),
    mass: float,
    volume: float,
    mu: float,
    lam: float,
    out: wp.array(dtype=float),
):
    i = wp.tid()
    fe = F[i] * wp.inverse(Fp[i])
    u, s, vt = wp.svd3(fe)
    r = u * wp.transpose(vt)
    strain = fe - r
    energy = float(0.0)
    for a in range(3):
        for b in range(3):
            energy += mu * strain[a, b] * strain[a, b]
    energy += 0.5 * lam * (wp.determinant(fe) - 1.0) * (wp.determinant(fe) - 1.0)
    wp.atomic_add(out, 0, 0.5 * mass * wp.dot(v[i], v[i]))
    wp.atomic_add(out, 1, mass * 9.81 * x[i][2])
    wp.atomic_add(out, 2, volume * energy)
    wp.atomic_max(out, 3, wp.length(v[i]))
    wp.atomic_max(out, 4, x[i][2])
    wp.atomic_min(out, 5, wp.determinant(F[i]))


class Metrics:
    columns = ["kinetic_J", "gravity_J", "elastic_diagnostic_J", "max_speed_m_s", "max_z_m", "min_det_F"]

    def __init__(self, solver):
        self.s = solver
        self.array = wp.zeros(6, dtype=float, device=solver.device)

    def read(self):
        self.array.assign(np.array([0, 0, 0, 0, -1.0e20, 1.0e20], np.float32))
        s = self.s
        s.launch(metrics, s.n, [s.x, s.v, s.F, s.Fp, s.mass, s.volume, s.mu, s.lam, self.array])
        return self.array.numpy().copy()
