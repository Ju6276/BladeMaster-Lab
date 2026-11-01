import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_geometry import InterfaceGeometry, solve_geometry
from blademaster_warp.reference_inertia import ReferenceInertia
from test_differentiable_cohesive import arrays, fixture
from test_reference_inertia import transfer


@pytest.mark.parametrize("scale", [0.0, 1e-14, 1e14])
@pytest.mark.parametrize("metric", [False, True])
def test_implicit_adjoint_is_linear_in_the_loss_seed(scale, metric):
    rng = np.random.default_rng(75)
    values = fixture("opening")
    seed_u = rng.normal(size=(values[5].shape[1], 3))
    seed_f = rng.normal(size=values[0].shape)
    results = []
    for magnitude in [1.0, scale]:
        sources = arrays(values)
        inertia = None
        parameters = []
        if metric:
            tr = transfer(values[5].shape[1], sources[0].device)
            inertia = ReferenceInertia(tr, sources[9], tr.model.array([2000.0], float))
            parameters = inertia.parameters
        with wp.Tape() as tape:
            result, _ = solve_geometry(
                InterfaceGeometry(*sources[:2]), *sources[2:], integrated_force=True, inertia=inertia
            )
        tape.backward(
            grads={
                result.displacement: wp.array(
                    magnitude * seed_u, dtype=wp.vec3d, device=result.displacement.device
                ),
                result.force: wp.array(magnitude * seed_f, dtype=wp.vec3d, device=result.force.device),
            }
        )
        results.append([a.grad.numpy().copy() for a in [*sources, *parameters]])
    for expected, actual in zip(*results):
        if scale == 0.0:
            np.testing.assert_array_equal(actual, 0.0)
        else:
            np.testing.assert_allclose(actual / scale, expected, rtol=1e-8, atol=1e-12)
