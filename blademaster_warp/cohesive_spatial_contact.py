"""Current particle contacts, indexed history and storage in affine material stages."""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ._cohesive_expansion import E, assign, divide, norm, positive
from .cohesive_contact import predict_motion
from .cohesive_contact_batch import CombinedContacts, join_states, split_state
from .cohesive_pair_contact import predict_pair_contacts
from .cohesive_pairing import PairHistory, pairs_at
from .cohesive_stages import AffineMaterialStep, advance_affine
from .compensated_state import CompensatedArray, as_compensated, add_gradient


def pair_storage(position, history, parameters, diameter):
    """Return normal and tangential elastic energy in J for each indexed pair."""
    position = as_compensated(position)
    count = len(history.pairs)
    if position.high.shape != (history.particle_count,) or position.high.dtype != wp.vec3d:
        raise ValueError("Pair energy requires matching float64 particle positions")
    if parameters.dtype != wp.float64 or parameters.shape not in [(1, 5), (count, 5)]:
        raise ValueError("Pair energy requires five coefficients per pair or one shared row")
    if diameter.dtype != wp.float64 or diameter.shape != (1,):
        raise ValueError("Spatial pair energy requires one shared float64 diameter")
    sources = [position.high, history.values.high, parameters, diameter]
    if any(a.device != position.high.device or not np.isfinite(a.numpy()).all() for a in sources):
        raise ValueError("Pair energy inputs must be finite and share a device")
    if np.any(parameters.numpy()[:, :2] <= 0) or diameter.numpy()[0] <= 0:
        raise ValueError("Pair energy requires positive stiffness and diameter")
    x, h = position.cpu(), history.values.cpu()
    constants, distance_limit = E(parameters.numpy()), E(diameter.numpy()[0])
    values = E(np.zeros((count, 2)))
    geometry = []
    for i, (p, q) in enumerate(history.pairs):
        delta = x[p] - x[q]
        distance = norm(delta)
        if not positive(distance):
            raise ValueError("Coincident particle centres do not define a contact normal")
        gap = distance - distance_limit
        compression = gap if positive(-gap) else E(0.0)
        coefficients = constants[0 if len(constants.high) == 1 else i]
        square = (h[i] * h[i]).sum()
        assign(values, (i, 0), 0.5 * coefficients[0] * compression * compression)
        assign(values, (i, 1), 0.5 * coefficients[1] * square)
        geometry.append((divide(delta, distance), compression, square, coefficients))
    result = wp.array(values.numpy(), dtype=wp.float64, device=position.high.device, requires_grad=True)
    tape = wp._src.context.runtime.tape
    if tape is not None and count:

        def backward():
            seed = E(result.grad.numpy())
            px, ph, pp = [E(np.zeros_like(a.numpy())) for a in sources[:3]]
            pd = E(np.zeros(1))
            for i, ((p, q), (normal, compression, square, coefficients)) in enumerate(
                zip(history.pairs, geometry)
            ):
                load = seed[i, 0] * coefficients[0] * compression
                force = load * normal
                assign(px, p, px[p] + force)
                assign(px, q, px[q] - force)
                assign(ph, i, seed[i, 1] * coefficients[1] * h[i])
                row = 0 if len(constants.high) == 1 else i
                assign(pp, (row, 0), pp[row, 0] + 0.5 * seed[i, 0] * compression * compression)
                assign(pp, (row, 1), pp[row, 1] + 0.5 * seed[i, 1] * square)
                pd = pd - load
            for source, gradient in zip(sources, [px, ph, pp, pd]):
                add_gradient(source, gradient)

        tape.record_func(backward, [result, *[a for a in sources if a.requires_grad and a.size]])
    return result


def _position(state):
    return CompensatedArray(state.x, state.rounding_remainders.get("x"))


def _union(first, second):
    return np.unique(np.concatenate([first, second]), axis=0).reshape(-1, 2)


class _NewPairs(Exception):
    def __init__(self, pairs):
        self.pairs = pairs


@dataclass(frozen=True)
class SpatialPairStep:
    material: AffineMaterialStep
    history: PairHistory
    retired: PairHistory
    pairs: np.ndarray
    initial_storage: wp.array
    final_storage: wp.array
    other_history: CompensatedArray | None
    rebuilds: int


def advance_spatial_pairs(
    *,
    transfer,
    state,
    pair_history,
    codes,
    parameters,
    diameter,
    other_contacts=None,
    other_history=None,
    **stage_options,
):
    """Assemble pair contacts at both free stages and recheck corrected stage positions.

    parameters is a shared (1, 5) array or a callable mapping sorted particle
    pairs to their coefficients. diameter is the common particle contact
    diameter in m. Existing rows survive the whole segment; departed rows and
    their final history/storage are returned with the result.
    """
    model = transfer.model
    if pair_history.particle_count != model.n or pair_history.values.high.device != model.device:
        raise ValueError("Pair history must match the material particles and device")
    if (other_contacts is None) != (other_history is None):
        raise ValueError("Additional contact stages require a provider and its history")
    if "contacts" in stage_options or "contact_history" in stage_options:
        raise ValueError("Use other_contacts and other_history for additional contact geometries")
    if diameter.dtype != wp.float64 or diameter.shape != (1,) or diameter.device != model.device:
        raise ValueError("Spatial contact requires one matching float64 diameter")
    reach = float(diameter.numpy()[0])
    initial_position = _position(state)

    def current(position):
        return pairs_at(position, model.reference, codes, reach, model.config.grid_spacing)

    selected = _union(pair_history.pairs, current(initial_position))
    point_map = model.array(np.eye(model.n), float, False)
    moment_map = transfer.flow_map
    extra = None if other_history is None else as_compensated(other_history)
    extra_count = 0 if extra is None else len(extra.high)
    rebuilds = 0
    while True:
        keys = set(map(tuple, selected))
        retained, _ = pair_history.reindex(selected)
        coefficients = parameters(selected) if callable(parameters) else parameters
        if not callable(parameters) and parameters.shape != (1, 5):
            raise ValueError("Spatial pair coefficients require one shared row or a particle-pair callback")
        initial_history = retained.values if extra is None else join_states([retained.values, extra])

        def require_pairs(position):
            found = current(position)
            if any(tuple(pair) not in keys for pair in found):
                raise _NewPairs(found)

        def provider(x, moments, duration, elapsed, contact_history):
            predicted, _ = predict_motion(x, moments, duration, point_map, moment_map)
            require_pairs(x)
            require_pairs(predicted)
            if extra is None:
                local_history = contact_history
            else:
                local_history, additional_history = split_state(contact_history, [len(selected), extra_count])
            batch = predict_pair_contacts(
                x, moments, duration, local_history, point_map, moment_map, selected, coefficients, diameter
            )
            if extra is None:
                return batch
            additional = other_contacts(x, moments, duration, elapsed, additional_history)
            return CombinedContacts([batch, additional])

        try:
            result = advance_affine(
                transfer=transfer,
                state=state,
                **stage_options,
                contacts=provider,
                contact_history=initial_history,
            )
            require_pairs(_position(result.stage))
            require_pairs(_position(result.state))
        except _NewPairs as change:
            selected = _union(selected, change.pairs)
            rebuilds += 1
            continue
        break
    if extra is None:
        final_values, final_extra = result.contact_history, None
    else:
        final_values, final_extra = split_state(result.contact_history, [len(selected), extra_count])
    full_history = PairHistory(model.n, selected, final_values)
    final_position = _position(result.state)
    final_history, retired = full_history.reindex(current(final_position))
    return SpatialPairStep(
        result,
        final_history,
        retired,
        selected,
        pair_storage(initial_position, retained, coefficients, diameter),
        pair_storage(final_position, full_history, coefficients, diameter),
        final_extra,
        rebuilds,
    )
