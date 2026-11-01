"""Simultaneous plane/blade constraints and partitioned friction state."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._cohesive_expansion import E
from blademaster_warp.cohesive_contact import MappedContacts, solve
from blademaster_warp.cohesive_contact_batch import CombinedContacts, join_states, split_state
from blademaster_warp.cohesive_rigid_contact import RigidContacts
from blademaster_warp.compensated_state import CompensatedArray
from test_mapped_contact import fixture
from test_rigid_joint_contact import rigid_fixture


def test_contact_history_partition_preserves_remainders_and_adjoint():
    rng = np.random.default_rng(2756)
    high, low = rng.normal(size=(5, 3)), rng.normal(size=(5, 3)) * 2**-60
    state = CompensatedArray.from_cpu(E(high, low), dtype=wp.vec3d, device="cuda:0")
    with wp.Tape() as tape:
        parts = split_state(state, [0, 2, 3, 0])
        result = join_states(parts)
    np.testing.assert_array_equal(result.high.numpy(), state.high.numpy())
    np.testing.assert_array_equal(result.low, state.low)
    seed = rng.normal(size=(5, 3))
    tape.backward(grads={result.high: wp.array(seed, dtype=wp.vec3d, device="cuda:0")})
    np.testing.assert_array_equal(state.high.grad.numpy(), seed)


@pytest.mark.parametrize("interface", [False, True])
def test_empty_mixed_contacts_keep_free_or_interface_solution(interface):
    inputs, geometry, _, array = fixture(interface, 0.001)
    plane = MappedContacts.from_rows(np.empty((0, 2)), [], "cuda:0")
    blade = RigidContacts.from_rows(np.empty((0, 2)), [], "cuda:0")
    combined = CombinedContacts([plane, blade])
    with wp.Tape() as tape:
        result = solve(inputs, geometry, None, combined)
    expected = solve(inputs, geometry, None, plane)
    np.testing.assert_array_equal(result[1].numpy(), expected[1].numpy())
    assert combined.count == 0 and result[3].high.shape == result[4].high.shape == (0,)
    tape.backward(grads={result[1]: array(np.ones((2, 3)), wp.vec3d, False)})


@pytest.mark.parametrize("interface", [False, True])
def test_simultaneous_plane_blade_joint_root_gradient(interface):
    inputs, geometry, plane, array = fixture(interface, 0.001)
    _, _, blade, _ = rigid_fixture(interface, 0.004, 0.4)
    combined = CombinedContacts([plane, blade])

    def run():
        root, u, force, contacts, history = solve(inputs, geometry, None, combined)
        return [u, force, contacts.high, history.high]

    with wp.Tape() as tape:
        outputs = run()
    rng = np.random.default_rng(2757)
    seeds = [rng.normal(size=a.numpy().shape) * scale for a, scale in zip(outputs, [1e4, 1, 1, 1e4])]
    tape.backward(grads={a: array(seed, a.dtype, False) for a, seed in zip(outputs, seeds) if a.size})
    for source, scale in [
        (plane.mapping.high, 0.1),
        (blade.mapping.high, 0.1),
        (plane.history.high, 1e-5),
        (blade.history.high, 1e-5),
        (plane.parameters, [100, 50, 0.02, 0.01, 0.01]),
        (blade.position, 1e-4),
        (blade.omega, 0.1),
        (inputs[9], 2e-5),
    ]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                values.append([a.numpy() for a in run()])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*values, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source.assign(baseline)
