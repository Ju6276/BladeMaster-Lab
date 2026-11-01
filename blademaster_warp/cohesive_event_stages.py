"""Material/contact segments separated by current particle-contact crossings."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .cohesive_pair_events import candidates, guard
from .cohesive_spatial_contact import SpatialPairStep, advance_spatial_pairs, pair_storage
from .cohesive_pairing import PairHistory
from .cohesive_stages import AffineMaterialStep
from .compensated_state import CompensatedArray
from .differentiable_geometry import bracket_time, clock_difference, secant_time


@wp.kernel
def _sum(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64), result: wp.array(dtype=wp.float64)):
    result[0] = a[0] + b[0]


@wp.kernel
def _released(storage: wp.array2d(dtype=wp.float64), result: wp.array(dtype=wp.float64)):
    wp.atomic_add(result, 0, storage[wp.tid(), 1])


def _position(state):
    return CompensatedArray(state.x, state.rounding_remainders.get("x"))


@dataclass(frozen=True)
class ContactSegment:
    step: SpatialPairStep
    start: wp.array
    duration: wp.array


@dataclass(frozen=True)
class ContactEvent:
    time: wp.array
    changes: tuple
    released_energy: wp.array


@dataclass(frozen=True)
class ContactSegments:
    material: AffineMaterialStep
    pair_history: PairHistory
    other_history: CompensatedArray | None
    modes: tuple
    segments: tuple
    events: tuple
    released_energy: wp.array


def apply_contact_event(model, state, pair_history, parameters, diameter, changes):
    """Retire exiting contacts and return their tangential elastic energy in J."""
    event_release = model.zeros(1, float)
    exits = {(p, q) for p, q, active in changes if not active}
    if exits:
        keep = np.asarray(
            [pair for pair in pair_history.pairs if tuple(pair) not in exits], dtype=np.int64
        ).reshape(-1, 2)
        pair_history, retired = pair_history.reindex(keep)
        coefficients = parameters(retired.pairs) if callable(parameters) else parameters
        storage = pair_storage(_position(state), retired, coefficients, diameter)
        if len(storage):
            wp.launch(
                _released,
                len(storage),
                inputs=[storage],
                outputs=[event_release],
                device=model.device,
            )
    return pair_history, event_release


def advance_contact_events(
    *,
    transfer,
    state,
    pair_history,
    codes,
    parameters,
    diameter,
    duration,
    modes=(),
    stop_at_first=False,
    other_contacts=None,
    other_history=None,
    **stage_options,
):
    """Advance current contact geometry, retaining each accepted segment and event.

    modes stores (p, q, active) at a located boundary and belongs in checkpoints.
    Providers and external forces receive elapsed time from this call's start.
    stop_at_first returns the first accepted segment for a coupled event driver.
    Existing cohesive points retain their updated geometry and damage after
    every accepted segment; contact creation/removal is recomputed on each call.
    """
    model = transfer.model
    transfer._check(duration, wp.float64, (1,))
    if not np.isfinite(duration.numpy()).all() or duration.numpy()[0] <= 0:
        raise ValueError("Contact event duration must be positive and finite")
    boundary_modes = {(int(p), int(q)): bool(active) for p, q, active in modes}
    zero = model.array([0.0], float, False)
    elapsed = zero
    released_total = model.zeros(1, float)
    options = dict(stage_options)
    force = options.pop("force", None)
    segments, events = [], []
    horizon = duration

    def clock(kernel, *inputs):
        result = model.zeros(1, float)
        model.launch(kernel, 1, inputs=list(inputs), outputs=[result], device=model.device)
        return result

    def add(a, b):
        result = model.zeros(1, float)
        wp.launch(_sum, 1, inputs=[a, b], outputs=[result], device=model.device)
        return result

    while True:
        remaining = clock(clock_difference, duration, elapsed)
        scale = max(model.config.grid_spacing, diameter.numpy()[0], np.max(abs(state.x.numpy())))
        tolerance = 32 * np.finfo(float).eps * scale
        cache = {}

        def flow(time):
            key = time
            if key in cache:
                return cache[key]
            current = dict(options)
            if force is not None:
                current["force"] = lambda s, z, t: force(s, z, add(elapsed, t))
            provider = None
            if other_contacts is not None:
                provider = lambda x, z, tau, t, h: other_contacts(x, z, tau, add(elapsed, t), h)
            result = advance_spatial_pairs(
                transfer=transfer,
                state=state,
                pair_history=pair_history,
                codes=codes,
                parameters=parameters,
                diameter=diameter,
                duration=time,
                other_contacts=provider,
                other_history=other_history,
                **current,
            )
            cache[key] = result
            return result

        trial = flow(horizon)
        query = candidates(
            _position(state),
            trial.material.path_midpoint,
            _position(trial.material.state),
            model.reference,
            codes,
            float(diameter.numpy()[0]),
            model.config.grid_spacing,
        )
        pending = {}
        signs = {}
        for event in query:
            pair = event.pair
            initial = guard(_position(state), pair, diameter).high.numpy()[0]
            if pair in boundary_modes and abs(initial) > tolerance:
                boundary_modes.pop(pair)
            active = boundary_modes.get(pair, initial < 0 if initial else event.direction > 0)
            if event.direction == (1 if active else -1) and pair not in pending:
                pending[pair] = event
                signs[pair] = -1 if active else 1

        if not pending:
            accepted, accepted_time, changes = trial, horizon, ()
            event_release = None
        else:
            pairs = sorted(pending)

            def boundary(current):
                values = {pair: guard(_position(current), pair, diameter).high for pair in pairs}
                pair = min(pairs, key=lambda key: signs[key] * values[key].numpy()[0])
                value = values[pair]
                return (value if signs[pair] > 0 else clock(clock_difference, zero, value)), pair

            fractions = {event.lower for event in pending.values()} | {
                event.upper for event in pending.values()
            }
            bounds = sorted({0, *fractions, 1})
            fractions |= {(a + b) / 2 for a, b in zip(bounds[:-1], bounds[1:])}
            fractions.add(1)
            lo, glo = zero, boundary(state)[0]
            positive = glo.numpy()[0] > 0
            bracket = None
            for fraction in sorted(fractions):
                if fraction <= 0:
                    continue
                hi = horizon if fraction == 1 else clock(bracket_time, zero, horizon, float(fraction))
                upper = flow(hi)
                ghi, _ = boundary(upper.material.state)
                if ghi.numpy()[0] <= 0 and positive:
                    bracket = (lo, hi, glo, ghi)
                    break
                if ghi.numpy()[0] > 0:
                    lo, glo, positive = hi, ghi, True
            if bracket is None:
                shorter = clock(bracket_time, zero, horizon, 0.5)
                if shorter.numpy()[0] == 0 or shorter.numpy()[0] == horizon.numpy()[0]:
                    raise RuntimeError("Contact candidate has no resolvable material-flow bracket")
                horizon = shorter
                continue
            lo, hi, glo, ghi = bracket
            for _ in range(64):
                time = clock(secant_time, lo, hi, glo, ghi)
                if not lo.numpy()[0] < time.numpy()[0] < hi.numpy()[0]:
                    time = clock(bracket_time, lo, hi, 0.5)
                accepted = flow(time)
                value, pair = boundary(accepted.material.state)
                scalar = value.numpy()[0]
                if 0 < scalar <= tolerance:
                    break
                if scalar > 0:
                    lo, glo = time, value
                else:
                    hi, ghi = time, value
            else:
                raise RuntimeError("Contact event did not converge on its incoming branch")
            accepted_time = time
            changes = tuple(
                (p, q, signs[(p, q)] > 0)
                for p, q in pairs
                if abs(guard(_position(accepted.material.state), (p, q), diameter).high.numpy()[0])
                <= tolerance
            )
            event_release = model.zeros(1, float)

        segments.append(ContactSegment(accepted, elapsed, accepted_time))
        state = accepted.material.state
        pair_history, other_history = accepted.history, accepted.other_history
        options.update(geometry=accepted.material.geometry, history=accepted.material.history)
        next_elapsed = add(elapsed, accepted_time)
        if next_elapsed.numpy()[0] <= elapsed.numpy()[0]:
            raise RuntimeError("Accepted contact segment did not advance time")
        elapsed = next_elapsed
        if changes:
            pair_history, event_release = apply_contact_event(
                model, state, pair_history, parameters, diameter, changes
            )
            boundary_modes.update({(p, q): active for p, q, active in changes})
            events.append(ContactEvent(elapsed, changes, event_release))
            released_total = add(released_total, event_release)
        if stop_at_first:
            break
        if (
            not pending
            and horizon.numpy()[0] == remaining.numpy()[0]
            or elapsed.numpy()[0] >= duration.numpy()[0]
        ):
            break
        horizon = clock(clock_difference, duration, elapsed)
    return ContactSegments(
        accepted.material,
        pair_history,
        other_history,
        tuple((p, q, active) for (p, q), active in sorted(boundary_modes.items())),
        tuple(segments),
        tuple(events),
        released_total,
    )
