"""Compensated interface momentum and its virtual-work transpose."""

import numpy as np
import warp as wp

from ._compensated import Expansion
from ._compensated_linalg import divide
from .compensated_state import add_gradient


def _stack(values, axis):
    return Expansion(
        np.stack([a.high for a in values], axis=axis), np.stack([a.low for a in values], axis=axis)
    )


def momentum_residual(b, gradient, area, traction, stress, mass, dt, u, integrated=False):
    B = Expansion(b.numpy())
    G = [Expansion(gradient.numpy()[:, :, c]) for c in range(3)]
    T, S = Expansion(traction.numpy()), Expansion(stress.numpy())
    weight = Expansion(np.ones(len(area)) if integrated else area.numpy())
    m, t = Expansion(mass.numpy()[0]), Expansion(dt.numpy()[0])
    factor = divide(t * t, m)
    force = B.T @ (weight[:, None] * T)
    for c, g in enumerate(G):
        force = force + g.T @ (weight[:, None] * S[:, :, c])
    inputs = [b, gradient, area, traction, stress, mass, dt, u]
    result = wp.empty_like(u, requires_grad=True)
    result.assign((Expansion(u.numpy()) + factor * force).numpy())
    tape = wp._src.context.runtime.tape
    if tape is not None:

        def backward():
            seed = Expansion(result.grad.numpy())
            gap = B @ seed
            frame = [g @ seed for g in G]
            add_gradient(u, seed)
            add_gradient(traction, factor * weight[:, None] * gap)
            add_gradient(stress, factor * weight[:, None, None] * _stack(frame, axis=-1))
            add_gradient(b, factor * weight[:, None] * (T @ seed.T))
            add_gradient(
                gradient,
                factor * weight[:, None, None] * _stack([S[:, :, c] @ seed.T for c in range(3)], axis=-1),
            )
            power = (seed * force).sum()
            add_gradient(mass, -divide(factor * power, m).reshape(1))
            add_gradient(dt, divide(2 * t * power, m).reshape(1))
            if not integrated:
                value = (T * gap).sum(axis=1)
                for c, g in enumerate(frame):
                    value = value + (S[:, :, c] * g).sum(axis=1)
                add_gradient(area, factor * value)

        tape.record_func(backward, [result, *[a for a in inputs if a.requires_grad]])
    return result
