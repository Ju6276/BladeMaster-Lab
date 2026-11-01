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


def local_hessian(g, a, b):
    oriented = cross(a, b)
    length = norm(oriented)
    n = divide(oriented, length)
    c = (g * n).sum()
    z = divide(g - c * n, length)
    dn = divide(E(np.eye(3)) - outer(n, n), length)
    haa = -divide(outer(g, n) + outer(n, g) + c * np.eye(3) - 3 * c * outer(n, n), length * length)
    Ja, Jb = -skew(b), skew(a)
    H = E(np.zeros((9, 9)))
    assign(H, (slice(0, 3), slice(3, 6)), dn @ Ja)
    assign(H, (slice(0, 3), slice(6, 9)), dn @ Jb)
    assign(H, (slice(3, 6), slice(0, 3)), Ja.T @ dn)
    assign(H, (slice(6, 9), slice(0, 3)), Jb.T @ dn)
    assign(H, (slice(3, 6), slice(3, 6)), Ja.T @ haa @ Ja)
    assign(H, (slice(6, 9), slice(6, 9)), Jb.T @ haa @ Jb)
    assign(H, (slice(3, 6), slice(6, 9)), Ja.T @ haa @ Jb - skew(z))
    assign(H, (slice(6, 9), slice(3, 6)), Jb.T @ haa @ Ja + skew(z))
    return H


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
        size, count = self.size, self.count
        u = self.length * y[:size]
        dual = self.length * y[size:]
        ru = self.reduced @ u
        rg = E(np.zeros(3 * count))
        J = E(np.zeros((len(y.high), len(y.high))))
        assign(J, (slice(0, size), slice(0, size)), self.reduced)
        gaps = []
        for i, (offset, A) in enumerate(zip(self.base, self.maps)):
            local = offset + A @ u
            g, a, b = local[:3], local[3:6], local[6:9]
            d = dual[3 * i : 3 * i + 3]
            zero_compliance = not np.any(self.compliance[i].high) and not np.any(self.compliance[i].low)
            free_normal = self.fixed_contact_set is None or not self.fixed_contact_set[i]
            free_loading = self.fixed_loading_set is None or not self.fixed_loading_set[i]
            if ((self.shared[i] or (zero_compliance and free_normal)) and free_loading
                    and not np.any(g.high) and not np.any(g.low)
                    and not np.any(d.high) and not np.any(d.low)):
                span = slice(size + 3 * i, size + 3 * i + 3)
                assign(J, (slice(0, size), span), self.factors[i] * A[:3].T)
                assign(J, (span, span), (1 if self.shared[i] else -1) * np.eye(3))
                gaps.append(E(np.zeros(3)))
                continue
            oriented = cross(a, b)
            measure = norm(oriented)
            n = divide(oriented, measure)
            N = divide(E(np.eye(3)) - outer(n, n), measure) @ (-skew(b) @ A[3:6] + skew(a) @ A[6:9])
            d = dual[3 * i : 3 * i + 3]
            z = g + d
            component = (z * n).sum()
            closed = positive(-component) and positive(self.compliance[i])
            if self.fixed_contact_set is not None:
                closed = bool(self.fixed_contact_set[i])
            compression = component if closed else E(0.0)
            p = z - compression * n
            radius = norm(p)
            sigma, gc = map(E, self.material[i, :2])
            h = E(self.history[i])
            failure = divide(2 * gc, sigma)
            opening = radius
            derivative = E(1.0)
            traction_ratio = E(0.0)
            traction_derivative = E(0.0)
            if positive(self.compliance[i]) and positive(failure - h):
                opening = E(0.0)
                derivative = E(0.0)
                traction_ratio = E(1.0)
                traction_derivative = E(1.0)
                if positive(h):
                    stiffness = divide(sigma * (1 - divide(h, failure)), h)
                    derivative = divide(1, 1 + self.compliance[i] * stiffness)
                    opening = radius * derivative
                    traction_ratio = divide(
                        self.compliance[i] * stiffness, 1 + self.compliance[i] * stiffness
                    )
                    traction_derivative = traction_ratio
                natural_loading = positive(opening - h) or (
                    h.high == 0 and positive(radius - self.compliance[i] * sigma)
                )
                fixed_loading = getattr(self, "fixed_loading_set", None)
                loading = natural_loading if fixed_loading is None else fixed_loading[i]
                if loading:
                    derivative = divide(1, 1 - divide(self.compliance[i] * sigma, failure))
                    opening = (radius - self.compliance[i] * sigma) * derivative
                    traction_ratio = (
                        self.compliance[i] * sigma * (divide(1, radius) - divide(1, failure)) * derivative
                    )
                    traction_derivative = -divide(self.compliance[i] * sigma, failure) * derivative
                    if not positive(failure - opening):
                        opening = radius
                        derivative = E(1.0)
                        traction_ratio = E(0.0)
                        traction_derivative = E(0.0)
            prox = E(np.zeros(3)) if radius.high == 0 else divide(opening, radius) * p
            face = N.T @ prox
            if not self.constrain_normal:
                prox = prox + compression * n
                face = -N.T @ d
            ru = ru + self.factors[i] * (A[:3].T @ d + compression * face)
            use_gap = (
                np.max(abs(g.high)) <= np.max(abs(d.high))
                and positive(failure - h)
                and not positive(opening - h)
            )
            residual = g - prox if use_gap else compression * n + traction_ratio * p - d
            if not self.constrain_normal and not use_gap:
                residual = traction_ratio * p - d
            assign(rg, slice(3 * i, 3 * i + 3), d if self.shared[i] else residual)
            radial_traction = traction_ratio * np.eye(3)
            if radius.high == 0:
                radial = derivative * np.eye(3)
            else:
                direction = divide(p, radius)
                ratio = divide(opening, radius)
                radial = ratio * np.eye(3) + (derivative - ratio) * outer(direction, direction)
                radial_traction = radial_traction + (traction_derivative - traction_ratio) * outer(
                    direction, direction
                )
            Dz = radial @ (E(np.eye(3)) - int(closed) * outer(n, n))
            Dn = -int(closed) * radial @ (component * np.eye(3) + outer(n, z))
            Tz = radial_traction + int(closed) * radial @ outer(n, n)
            if not self.constrain_normal:
                Dz = Dz + int(closed) * outer(n, n)
                Dn = int(closed) * (E(np.eye(3)) - radial) @ (component * np.eye(3) + outer(n, z))
                Tz = radial_traction @ (E(np.eye(3)) - int(closed) * outer(n, n))
            B = A[:3]
            if closed:
                Hc = A.T @ local_hessian(prox, a, b) @ A
                face_gradient = Hc - B.T @ N - N.T @ B + N.T @ (Dz @ B + Dn @ N)
                if not self.constrain_normal:
                    face_gradient = -A.T @ local_hessian(d, a, b) @ A + B.T @ N + N.T @ B
                compression_gradient = int(closed) * (n @ B + z @ N)
                block = J[:size, :size] + self.factors[i] * (
                    compression * face_gradient + outer(face, compression_gradient)
                )
                assign(J, (slice(0, size), slice(0, size)), block)
            span = slice(size + 3 * i, size + 3 * i + 3)
            face_dual = N.T @ Dz if self.constrain_normal else -N.T
            assign(
                J,
                (slice(0, size), span),
                self.factors[i] * (B.T + compression * face_dual + int(closed) * outer(face, n)),
            )
            assign(J, (span, slice(0, size)), Tz @ B - Dn @ N)
            assign(J, (span, span), -Dz)
            if self.shared[i]:
                assign(J, (span, slice(0, size)), np.zeros((3, size)))
                assign(J, (span, span), np.eye(3))
            gaps.append(prox)
        residual = divide(E(np.r_[ru.high, rg.high], np.r_[ru.low, rg.low]), self.length)
        return residual, J, gaps
