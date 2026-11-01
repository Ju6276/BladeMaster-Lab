"""Scalar values and first derivatives with two-component arithmetic."""

from dataclasses import dataclass

import numpy as np

from ._cohesive_expansion import E, divide, positive, root


@dataclass(frozen=True)
class Scalar:
    value: E
    derivative: E

    @classmethod
    def independent(cls, value, index, size):
        return cls(E.from_value(value), E(np.eye(size)[index]))

    def constant(self, value):
        return Scalar(E.from_value(value), E(np.zeros(self.derivative.shape)))

    def other(self, value):
        return value if isinstance(value, Scalar) else self.constant(value)

    def __add__(self, other):
        other = self.other(other)
        return Scalar(self.value + other.value, self.derivative + other.derivative)

    __radd__ = __add__

    def __neg__(self):
        return Scalar(-self.value, -self.derivative)

    def __sub__(self, other):
        return self + -self.other(other)

    def __rsub__(self, other):
        return self.other(other) - self

    def __mul__(self, other):
        other = self.other(other)
        return Scalar(self.value * other.value, self.derivative * other.value + self.value * other.derivative)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.other(other)
        value = divide(self.value, other.value)
        return Scalar(value, divide(self.derivative - value * other.derivative, other.value))

    def __rtruediv__(self, other):
        return self.other(other) / self

    def __lt__(self, other):
        return positive(self.other(other).value - self.value)

    def __le__(self, other):
        return not positive(self.value - self.other(other).value)

    def __gt__(self, other):
        return self.other(other) < self

    def __ge__(self, other):
        return self.other(other) <= self

    def __abs__(self):
        return -self if self < 0.0 else self

    def sqrt(self):
        value = root(self.value)
        if not positive(value):
            raise ValueError("A differentiated square root requires a positive argument")
        return Scalar(value, divide(self.derivative, 2.0 * value))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def rotate(q, vector):
    first = cross(q[:3], vector)
    second = cross(q[:3], first)
    return [v + 2.0 * (q[3] * a + b) for v, a, b in zip(vector, first, second)]


def clamp(value, lower, upper):
    if value < lower:
        return value.other(lower)
    if upper < value:
        return value.other(upper)
    return value
