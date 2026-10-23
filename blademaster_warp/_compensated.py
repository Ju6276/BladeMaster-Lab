"""Two-component float64 expansions for cancellation-sensitive linear algebra."""

from dataclasses import dataclass
import math

import numpy as np

from ._compensated_kernels import matrix_product


def _two_sum(a, b):
    value = a + b
    shifted = value - a
    remainder = (a - (value - shifted)) + (b - shifted)
    return value, remainder


def _two_product(a, b):
    value = a * b
    ma, ea = np.frexp(a)
    mb, eb = np.frexp(b)
    # Split normalized mantissas so the splitter cannot overflow on finite inputs.
    splitter = 134217729.0
    ca, cb = splitter * ma, splitter * mb
    ah, bh = ca - (ca - ma), cb - (cb - mb)
    al, bl = ma - ah, mb - bh
    product = ma * mb
    error = ((ah * bh - product) + ah * bl + al * bh) + al * bl
    exponent = ea + eb
    remainder = np.ldexp((product - np.ldexp(value, -exponent)) + error, exponent)
    return value, remainder


@dataclass(frozen=True)
class Expansion:
    __array_priority__ = 1000

    high: np.ndarray
    low: np.ndarray | float = 0.0

    def __post_init__(self):
        high, low = np.broadcast_arrays(np.asarray(self.high, dtype=float), np.asarray(self.low, dtype=float))
        high, low = _two_sum(high, low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "low", low)

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        value = np.asarray(value)
        high = value.astype(float)
        return cls(high, (value - high.astype(value.dtype)).astype(float))

    @property
    def shape(self):
        return self.high.shape

    @property
    def ndim(self):
        return self.high.ndim

    @property
    def T(self):
        return Expansion(self.high.T, self.low.T)

    def reshape(self, *shape):
        return Expansion(self.high.reshape(*shape), self.low.reshape(*shape))

    def numpy(self, dtype=float):
        return self.high.astype(dtype) + self.low.astype(dtype)

    def __getitem__(self, index):
        return Expansion(self.high[index], self.low[index])

    def __neg__(self):
        return Expansion(-self.high, -self.low)

    def __add__(self, other):
        other = Expansion.from_value(other)
        high, low = _two_sum(self.high, other.high)
        low += self.low + other.low
        return Expansion(high, low)

    __radd__ = __add__

    def __sub__(self, other):
        return self + -Expansion.from_value(other)

    def __rsub__(self, other):
        return Expansion.from_value(other) + -self

    def __mul__(self, other):
        other = Expansion.from_value(other)
        high, low = _two_product(self.high, other.high)
        low += self.high * other.low + self.low * other.high
        low += self.low * other.low
        return Expansion(high, low)

    __rmul__ = __mul__

    def sum(self, axis=None):
        if axis is None:
            return self.reshape(-1).sum(0)
        if not -self.ndim <= axis < self.ndim:
            raise ValueError("Reduction axis exceeds the array dimensions")
        axis %= self.ndim
        high, low = np.moveaxis(self.high, axis, 0), np.moveaxis(self.low, axis, 0)
        result = Expansion(np.zeros(high.shape[1:]))
        for h, l in zip(high, low):
            result = result + Expansion(h, l)
        return result

    def __matmul__(self, other):
        other = Expansion.from_value(other)
        if self.ndim == 0 or other.ndim == 0:
            raise ValueError("Matrix products require at least one dimension")
        left = self.reshape(1, -1) if self.ndim == 1 else self
        right = other.reshape(-1, 1) if other.ndim == 1 else other
        if left.shape[-1] != right.shape[-2]:
            raise ValueError("Matrix product dimensions do not match")
        batch = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
        shape = (*batch, left.shape[-2], right.shape[-1])
        arguments = []
        for operand in (left, right):
            expanded = (*batch, *operand.shape[-2:])
            flat = (math.prod(batch), *operand.shape[-2:])
            for component in (operand.high, operand.low):
                arguments.append(np.ascontiguousarray(np.broadcast_to(component, expanded).reshape(flat)))
        high, low = matrix_product(*arguments)
        result = Expansion(high.reshape(shape), low.reshape(shape))
        if self.ndim == 1:
            result = result.reshape(*batch, right.shape[-1])
        if other.ndim == 1:
            result = result.reshape(batch if self.ndim == 1 else (*batch, left.shape[-2]))
        return result

    def __rmatmul__(self, other):
        return Expansion.from_value(other) @ self
