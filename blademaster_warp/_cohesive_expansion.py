"""Compensated evaluation of the generic cohesive residual and Jacobian."""

import numpy as np
from ._compensated import Expansion as E


from ._compensated_linalg import divide


def root(a):
    a = E.from_value(a)
    first = E(np.sqrt(a.high))
    if np.all(first.high == 0):
        return first
    return first + divide(a - first * first, 2 * first)


def scale_power_two(a, exponent):
    a = E.from_value(a)
    return E(np.ldexp(a.high, exponent), np.ldexp(a.low, exponent))


def norm(a):
    a = E.from_value(a)
    largest = np.max(abs(a.high), initial=0.0)
    if largest == 0.0:
        return E(0.0)
    exponent = int(np.frexp(largest)[1])
    scaled = scale_power_two(a, -exponent)
    return scale_power_two(root((scaled * scaled).sum()), exponent)


def stack(values, axis=0):
    values = [E.from_value(v) for v in values]
    return E(np.stack([v.high for v in values], axis=axis), np.stack([v.low for v in values], axis=axis))


def outer(a, b):
    return a[:, None] * b[None, :]


def cross(a, b):
    return stack([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def skew(a):
    return stack([stack([0.0, -a[2], a[1]]), stack([a[2], 0.0, -a[0]]), stack([-a[1], a[0], 0.0])])


def assign(target, index, value):
    value = E.from_value(value)
    target.high[index] = value.high
    target.low[index] = value.low


def positive(a):
    return bool(a.high > 0 or (a.high == 0 and a.low > 0))




class Equations:
    def __init__(
        self,
        base,
        maps,
        reduced,
        length,
        compliance,
        history,
        material,
        factors,
        shared,
        *,
        constrain_normal=True
    ):
        self.base, self.maps, self.reduced = base, maps, reduced
        self.length, self.compliance = length, compliance
        self.history, self.material, self.factors = history, material, factors
        self.shared = shared
        self.size = reduced.shape[0]
        self.count = len(base)
        self.constrain_normal = constrain_normal
        self.fixed_contact_set = None
        self.fixed_loading_set = None

    def contact_set(self, y):
        displacement = self.length * y[: self.size]
        active = np.zeros(self.count, dtype=bool)
        for i, (base, mapping) in enumerate(zip(self.base, self.maps)):
            local = base + mapping @ displacement
            normal = cross(local[3:6], local[6:9])
            normal = divide(normal, norm(normal))
            dual = self.length * y[self.size + 3 * i : self.size + 3 * i + 3]
            component = ((local[:3] + dual) * normal).sum()
            active[i] = positive(-component) and positive(self.compliance[i])
        return active

    def loading_set(self, y):
        u = self.length * y[: self.size]
        result = np.zeros(self.count, dtype=bool)
        for i, (base, mapping) in enumerate(zip(self.base, self.maps)):
            sigma, gc = map(E, self.material[i, :2])
            failure = divide(2 * gc, sigma)
            history = E(self.history[i])
            if not positive(self.compliance[i]) or not positive(failure - history):
                continue
            local = base + mapping @ u
            normal = cross(local[3:6], local[6:9])
            normal = divide(normal, norm(normal))
            z = local[:3] + self.length * y[self.size + 3 * i : self.size + 3 * i + 3]
            component = (z * normal).sum()
            tensile = z - (component * normal if positive(-component) else 0.0)
            limit = history + self.compliance[i] * sigma * (1 - divide(history, failure))
            result[i] = positive(norm(tensile) - limit)
        return result

    def __call__(self, y):
        from ._cohesive_batch import evaluate

        return evaluate(self, y)
