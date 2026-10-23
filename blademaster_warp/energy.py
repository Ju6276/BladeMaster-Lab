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


def energy_snapshot(model, state, gravity=(0., 0., -9.81), *, affine_transfer=None):
    result = material_energy(
        *(getattr(state, name).numpy() for name in ('x', 'v', 'F', 'Fp')),
        model.mass, model.volume, float(model.young.numpy()[0]), model.config.poisson_ratio, gravity,
    )
    affine = getattr(state, "affine", None)
    if affine is not None:
        if affine_transfer is None or affine_transfer.model is not model:
            raise ValueError("Affine-state energy requires the matching reference transfer")
        if affine.shape != state.v.shape or not np.isfinite(affine.numpy()).all():
            raise ValueError("Affine energy requires finite particle velocity gradients")
        moments = affine.numpy() @ affine_transfer.root.numpy()
        result["affine_kinetic_j"] = float(0.5 * model.mass * np.sum(moments * moments))
        if not np.isfinite(result["affine_kinetic_j"]):
            raise FloatingPointError("Nonfinite affine kinetic energy")
    return result


def contact_storage(gap, history, normal_stiffness, tangent_stiffness):
    """Linear normal penalty and tangential elastic history; SI units per contact."""
    gap = np.asarray(gap, dtype=float)
    history = np.asarray(history, dtype=float)
    kn, kt, gap = np.broadcast_arrays(normal_stiffness, tangent_stiffness, gap)
    if (history.shape != gap.shape + (3,) or not np.isfinite(history).all()
            or not np.isfinite([gap, kn, kt]).all() or np.any(kn <= 0) or np.any(kt <= 0)):
        raise ValueError('Invalid contact energy state or stiffness')
    return dict(normal_j=.5 * kn * np.minimum(gap, 0.) ** 2,
                tangent_j=.5 * kt * np.sum(history * history, axis=-1))


def friction_increment_energy(history, updated, displacement, force, stiffness, damping, duration, theta):
    """History is elastic displacement; total displacement also includes slider slip."""
    h0, h1, dx, force = [np.asarray(a, dtype=float) for a in (history, updated, displacement, force)]
    if (h0.shape != h1.shape or h0.shape != dx.shape or h0.shape != force.shape
            or h0.shape[-1:] != (3,) or not np.isfinite([h0, h1, dx, force]).all()
            or not np.isfinite([stiffness, damping, duration, theta]).all()
            or stiffness <= 0 or damping < 0 or duration <= 0 or not .5 <= theta <= 1):
        raise ValueError('Invalid friction energy increment')
    elastic_increment = h1 - h0
    slip = dx - elastic_increment
    square = np.sum(elastic_increment ** 2, axis=-1)
    return dict(
        input_work_j=-np.sum(force * dx, axis=-1),
        storage_change_j=.5 * stiffness * np.sum(h1 ** 2 - h0 ** 2, axis=-1),
        viscous_j=damping / duration * square,
        sliding_j=-np.sum(force * slip, axis=-1),
        integration_j=(theta - .5) * stiffness * square,
    )
