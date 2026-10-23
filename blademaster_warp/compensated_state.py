"""A float64 Warp state and its CPU rounding remainder represent one variable."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ._compensated import Expansion


def add_gradient(source, value):
    if source.requires_grad:
        result = Expansion(source.grad.numpy()) + value
        source.grad.assign(result.numpy())


@dataclass(frozen=True)
class CompensatedArray:
    high: wp.array
    low: np.ndarray | None = None

    def __post_init__(self):
        value = self.high.numpy()
        if value.dtype != np.float64:
            raise ValueError("Compensated states require float64 Warp storage")
        low = np.zeros_like(value) if self.low is None else np.asarray(self.low, dtype=float).copy()
        if low.shape != value.shape or not np.isfinite(low).all():
            raise ValueError("Rounding remainders must be finite and match the state")
        object.__setattr__(self, "low", low)

    def cpu(self):
        return Expansion(self.high.numpy(), self.low)

    @classmethod
    def from_cpu(cls, value, *, dtype, device, requires_grad=True):
        value = Expansion.from_value(value)
        high = wp.array(value.high, dtype=dtype, device=device, requires_grad=requires_grad)
        return cls(high, value.low)

    def snapshot(self):
        return dict(high=self.high.numpy().copy(), low=self.low.copy())

    @classmethod
    def restore(cls, snapshot, *, dtype, device=None, requires_grad=True):
        return cls(
            wp.array(snapshot["high"], dtype=dtype, device=device, requires_grad=requires_grad),
            snapshot["low"],
        )

    def add_scaled(self, rate, duration, factor=1.0):
        rate, duration = as_compensated(rate), as_compensated(duration)
        if duration.high.dtype != wp.float64 or duration.high.shape != (1,):
            raise ValueError("State increments require a scalar duration")
        if self.high.dtype != rate.high.dtype or self.high.shape != rate.high.shape:
            raise ValueError("State and rate must have matching shapes and types")
        if self.high.device != rate.high.device or self.high.device != duration.high.device:
            raise ValueError("State increments require a shared device")
        if not np.isfinite(factor):
            raise ValueError("State increment factor must be finite")
        base, speed, time = self.cpu(), rate.cpu(), duration.cpu()[0]
        result = self.from_cpu(base + factor * time * speed, dtype=self.high.dtype, device=self.high.device)
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                seed = Expansion(result.high.grad.numpy())
                add_gradient(self.high, seed)
                add_gradient(rate.high, factor * time * seed)
                add_gradient(duration.high, (factor * seed * speed).sum().reshape(1))

            sources = [self.high, rate.high, duration.high]
            tape.record_func(backward, [result.high, *[a for a in sources if a.requires_grad]])
        return result


def as_compensated(value):
    return value if isinstance(value, CompensatedArray) else CompensatedArray(value)
