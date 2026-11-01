"""Different contact geometries in one material/interface momentum equation."""

import numpy as np
import warp as wp

from ._cohesive_expansion import E
from .compensated_state import CompensatedArray, as_compensated, add_gradient


def join_states(states):
    states = tuple(map(as_compensated, states))
    if not states:
        raise ValueError("Joining states requires at least one typed state")
    first = states[0].high
    if any(s.high.dtype != first.dtype or s.high.device != first.device for s in states):
        raise ValueError("Joined states must share a type and device")
    values = [s.cpu() for s in states]
    result = CompensatedArray.from_cpu(
        E(np.concatenate([v.high for v in values]), np.concatenate([v.low for v in values])),
        dtype=first.dtype,
        device=first.device,
    )
    tape = wp._src.context.runtime.tape
    if tape is not None and result.high.size:

        def backward():
            gradient = E(result.high.grad.numpy())
            start = 0
            for state in states:
                stop = start + len(state.high)
                add_gradient(state.high, gradient[start:stop])
                start = stop

        tape.record_func(
            backward, [result.high, *[s.high for s in states if s.high.requires_grad and s.high.size]]
        )
    return result


def split_state(state, counts):
    state = as_compensated(state)
    counts = tuple(counts)
    if any(c < 0 or int(c) != c for c in counts) or sum(counts) != len(state.high):
        raise ValueError("Contact partition sizes must sum to the history length")
    offsets = np.r_[0, np.cumsum(counts)]
    value = state.cpu()
    results = [
        CompensatedArray.from_cpu(value[a:b], dtype=state.high.dtype, device=state.high.device)
        for a, b in zip(offsets[:-1], offsets[1:])
    ]
    tape = wp._src.context.runtime.tape
    if tape is not None and state.high.size:

        def backward():
            gradient = np.concatenate([r.high.grad.numpy() for r in results])
            add_gradient(state.high, E(gradient))

        tape.record_func(
            backward,
            [*[r.high for r in results if r.high.size], *([state.high] if state.high.requires_grad else [])],
        )
    return tuple(results)


class CombinedContacts:
    def __init__(self, batches):
        self.batches = tuple(batches)
        if not self.batches:
            raise ValueError("A combined contact set requires at least one typed batch")
        first = self.batches[0].mapping.high
        if any(
            b.mapping.high.shape[1] != first.shape[1] or b.mapping.high.device != first.device
            for b in self.batches
        ):
            raise ValueError("Contact batches must share material degrees of freedom and device")

    @property
    def mapping(self):
        first = self.batches[0].mapping.high
        maps = [b.mapping.cpu() for b in self.batches]
        return CompensatedArray.from_cpu(
            E(np.concatenate([v.high for v in maps]), np.concatenate([v.low for v in maps])),
            dtype=wp.float64,
            device=first.device,
            requires_grad=False,
        )

    @property
    def count(self):
        return sum(b.count for b in self.batches)

    @property
    def arrays(self):
        return [a for batch in self.batches for a in batch.arrays]

    @property
    def history(self):
        return join_states([batch.history for batch in self.batches])

    def clone(self):
        return CombinedContacts([batch.clone() for batch in self.batches])

    def rows(self):
        rows = []
        for batch in self.batches:
            for original in batch.rows():
                row = dict(original)
                row.setdefault("evaluate", batch.evaluate)
                rows.append(row)
        return rows

    @staticmethod
    def evaluate(row, point, velocity, duration):
        return row["evaluate"](row, point, velocity, duration)

    def length_scale(self, duration):
        return max(batch.length_scale(duration) for batch in self.batches)

    def response(self, u, u_low, duration):
        responses = [batch.response(u, u_low, duration) for batch in self.batches]
        return [join_states(parts) for parts in zip(*responses)]

    def add_momentum(self, original, u, u_low, mass, duration):
        for batch in self.batches:
            original = batch.add_momentum(original, u, u_low, mass, duration)
        return original
