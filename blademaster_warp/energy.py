"""Material energy in joules, evaluated from particle states in SI units."""

import numpy as np


def elastic_density(F, Fp, young, poisson):
    """Fixed-corotated energy per reference volume; F and Fp preserve orientation."""
    F = np.asarray(F, dtype=np.float64)
    Fp = np.asarray(Fp, dtype=np.float64)
    if (F.shape != Fp.shape or F.ndim < 2 or F.shape[-2:] != (3, 3)
            or not np.isfinite(F).all() or not np.isfinite(Fp).all()
            or not np.isfinite([young, poisson]).all()
            or young <= 0 or not -1 < poisson < .5):
        raise ValueError('Invalid deformation tensors or elastic constants')
    if np.any(np.linalg.det(F) <= 0) or np.any(np.linalg.det(Fp) <= 0):
        raise ValueError('Deformation determinants must be positive')
    elastic = F @ np.linalg.inv(Fp)
    stretches = np.linalg.svd(elastic, compute_uv=False)
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    density = (mu * np.sum((stretches - 1) ** 2, axis=-1)
               + .5 * lam * (np.linalg.det(elastic) - 1) ** 2)
    if not np.isfinite(density).all():
        raise FloatingPointError('Nonfinite elastic energy density')
    return density


def material_energy(x, v, F, Fp, mass, volume, young, poisson, gravity=(0., 0., -9.81)):
    """Equal-mass particles; volume is each particle's reference volume in m³."""
    x = np.asarray(x, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    gravity = np.asarray(gravity, dtype=np.float64)
    if (x.ndim != 2 or x.shape[1:] != (3,) or not len(x) or v.shape != x.shape
            or np.shape(F) != (len(x), 3, 3) or np.shape(Fp) != np.shape(F)
            or gravity.shape != (3,) or not np.isfinite(x).all()
            or not np.isfinite(v).all() or not np.isfinite(gravity).all()
            or not np.isfinite([mass, volume]).all() or mass <= 0 or volume <= 0):
        raise ValueError('Invalid particle energy state, mass, volume or gravity')
    result = dict(
        kinetic_j=float(.5 * mass * np.sum(v * v)),
        gravity_j=float(-mass * np.sum(x @ gravity)),
        elastic_j=float(volume * np.sum(elastic_density(F, Fp, young, poisson))),
    )
    if not np.isfinite(list(result.values())).all():
        raise FloatingPointError('Nonfinite material energy')
    return result


def energy_snapshot(model, state, gravity=(0., 0., -9.81)):
    return material_energy(
        *(getattr(state, name).numpy() for name in ('x', 'v', 'F', 'Fp')),
        model.mass, model.volume, float(model.young.numpy()[0]), model.config.poisson_ratio, gravity,
    )
