import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_contact import MappedContacts, solve
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.reference_inertia import ReferenceInertia
from test_differentiable_cohesive import arrays, closed_pair
from test_reference_inertia import transfer


@pytest.mark.parametrize("constrain_normal", [False, True])
@pytest.mark.parametrize("theta", [None, 2.0])
@pytest.mark.parametrize("exponent", [-35, -60])
def test_joint_closed_root_preserves_weak_constraint_gradient(exponent, theta, constrain_normal):
    delta = 2.0**exponent
    values = closed_pair(4e-6, [0.0, 0.0, 0.0])
    values = [np.repeat(a, 2, axis=0) if i < 8 else a for i, a in enumerate(values)]
    values[5] = np.ones((2, 2))
    values[7][:] = 2e-6
    sources = arrays(values)
    contacts = MappedContacts.from_rows(np.zeros((0, 2)), [], sources[0].device)
    inertia = None
    if theta is not None:
        tr = transfer(2, sources[0].device)
        inertia = ReferenceInertia(tr, sources[9], tr.model.array([theta / values[9][0]], float))
    remainders = (np.array([[0.0, 0.0], [0.0, delta]]), np.zeros_like(values[6]))

    def run():
        return solve(
            sources, InterfaceGeometry(*sources[:2]), inertia, contacts,
            map_remainders=remainders, constrain_normal=constrain_normal,
        )[1]

    with wp.Tape() as tape:
        displacement = run()
    seed = np.array([[0.0, 0.0, 0.0], [0.0, delta, 0.0]])
    tape.backward(grads={displacement: wp.array(seed, dtype=wp.vec3d, device=displacement.device)})
    expected = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(sources[0].grad.numpy(), expected, rtol=2e-13, atol=0.0)
    for epsilon in [1e-10, 1e-11]:
        step = epsilon * delta**2
        losses = []
        for sign in [-1.0, 1.0]:
            sources[0].assign(sign * step * expected)
            losses.append(float(np.sum(run().numpy() * seed)))
        np.testing.assert_allclose((losses[1] - losses[0]) / (2 * step), 2.0, rtol=2e-5, atol=0.0)
