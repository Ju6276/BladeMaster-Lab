"""Complete and segmented reverse passes retain material and friction state."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_contact import predict_contacts
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_stages import AffineState, advance_affine
from blademaster_warp.compensated_state import CompensatedArray
from test_cohesive_stages import setup


def prepared(plastic=False, mode="opened", dt=2e-6):
    m, state, geometry, history, data = setup(mode, dt)
    state.v.assign(np.tile([0.4 if mode == "opened" else 0.002, 0.001, -0.1], (m.n, 1)))
    state.affine.zero_()
    state.stress = None
    if plastic:
        m.yield_stress.assign([20.0])
        A = np.diag([2000.0, -400.0, -200.0])
        state.affine.assign(np.tile(A, (m.n, 1, 1)))
        state.v.assign(state.v.numpy() + m.reference @ A.T)
    indices = np.flatnonzero(m.reference[:, 2] == np.min(m.reference[:, 2]))
    W = m.array(np.eye(m.n)[indices], float, False)
    C = m.array(data["transfer"].flow_map.numpy()[indices], float, False)
    normal = m.array(np.tile([0.0, 0.0, 1.0], (len(indices), 1)), wp.vec3, False)
    parameters = m.array(np.tile([1000.0, 500.0, 0.2, 0.3, 0.2], (len(indices), 1)), float)
    plane = m.array([[0.0, 0.0, 0.0016]], wp.vec3)
    velocity = m.zeros(1, wp.vec3, False)
    rate = m.array([2000.0], float)
    contact = CompensatedArray(m.array(np.tile([1e-7, -2e-7, 0.0], (len(indices), 1)), wp.vec3))

    def provider(x, z, tau, elapsed, h):
        return predict_contacts(x, z, tau, h, W, C, plane, velocity, normal, parameters)

    def advance(s, g, h, c):
        return advance_affine(
            state=s,
            geometry=g,
            history=h,
            **{k: v for k, v in data.items() if k != "force"},
            force=lambda s, v, t: data["force"],
            damping_rate=rate,
            contacts=provider,
            contact_history=c
        )

    return (
        m,
        state,
        geometry,
        history,
        contact,
        data,
        [data["duration"], m.young, m.yield_stress, parameters, plane, rate],
        advance,
    )


def state_arrays(state, geometry, history, contact):
    arrays = [state.x, state.v, state.F, state.Fp, state.affine]
    if state.stress is not None:
        arrays.append(state.stress)
    return [*arrays, geometry.gap, geometry.frame, history, contact.high]


def arrays(step):
    return state_arrays(step.state, step.geometry, step.history, step.contact_history)


@pytest.mark.parametrize("plastic", [False, True])
def test_segmented_contact_gradient_matches_full_reverse(plastic):
    check_segmented_contact_gradient(prepared(plastic), plastic)


def check_segmented_contact_gradient(fixture, plastic):
    m, state, geometry, history, contact, data, parameters, advance = fixture
    initial_arrays = state_arrays(state, geometry, history, contact)
    with wp.Tape() as full:
        first = advance(state, geometry, history, contact)
        final = advance(first.state, first.geometry, first.history, first.contact_history)
    if plastic:
        assert np.count_nonzero(final.state.yielded.numpy()) > 0
    rng = np.random.default_rng(2744)
    terminal = [*arrays(final), final.interface.force]
    seeds = [rng.normal(size=a.numpy().shape) for a in terminal]
    full.backward(grads={a: m.array(s, a.dtype, False) for a, s in zip(terminal, seeds)})
    expected_initial = [a.grad.numpy().copy() for a in initial_arrays]
    expected_parameters = [a.grad.numpy().copy() for a in parameters]
    full.zero()
    restored = AffineState.restore(first.state.snapshot(), device=m.device)
    restored_geometry = InterfaceGeometry.restore(first.geometry.snapshot(), device=m.device)
    restored_history = m.array(first.history.numpy(), float)
    restored_contact = CompensatedArray.restore(
        first.contact_history.snapshot(), dtype=wp.vec3d, device=m.device
    )
    boundary = state_arrays(restored, restored_geometry, restored_history, restored_contact)
    with wp.Tape() as second_tape:
        second = advance(restored, restored_geometry, restored_history, restored_contact)
    for actual, expected in zip([*arrays(second), second.interface.force], terminal):
        np.testing.assert_array_equal(actual.numpy(), expected.numpy())
    for name in final.state.rounding_remainders:
        np.testing.assert_array_equal(
            second.state.rounding_remainders[name], final.state.rounding_remainders[name]
        )
    np.testing.assert_array_equal(second.contact_history.low, final.contact_history.low)
    np.testing.assert_array_equal(second.geometry.gap_remainder, final.geometry.gap_remainder)
    np.testing.assert_array_equal(second.geometry.frame_remainder, final.geometry.frame_remainder)
    second_tape.backward(
        grads={
            a: m.array(s, a.dtype, False) for a, s in zip([*arrays(second), second.interface.force], seeds)
        }
    )
    boundary_seeds = [a.grad.numpy().copy() for a in boundary]
    totals = [a.grad.numpy().copy() for a in parameters]
    second_tape.zero()
    with wp.Tape() as first_tape:
        repeated = advance(state, geometry, history, contact)
    first_tape.backward(
        grads={a: m.array(s, a.dtype, False) for a, s in zip(arrays(repeated), boundary_seeds)}
    )
    for source, expected in zip(initial_arrays, expected_initial):
        np.testing.assert_allclose(source.grad.numpy(), expected, rtol=1e-10, atol=1e-10)
    for source, second_value, expected in zip(parameters, totals, expected_parameters):
        np.testing.assert_allclose(source.grad.numpy() + second_value, expected, rtol=1e-10, atol=1e-10)
