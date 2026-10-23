"""Near-dependent compressive constraints retain equilibrium under solver scaling."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp._cohesive_solve import solve_primal
from test_cohesive_stages import setup


def near_closed_values(seed, maximum):
    m, state, geometry, history, data = setup("rigid", 2e-6)
    transfer = data["transfer"]
    B, G = data["separation_map"].numpy(), data["frame_map"].numpy()
    _, singular, right = np.linalg.svd(B, full_matrices=False)
    rank = np.count_nonzero(singular > singular[0] * max(B.shape) * np.finfo(float).eps)
    z = transfer.pack(state.v, state.affine).numpy()
    z += np.random.default_rng(seed).normal(scale=0.03, size=z.shape)
    z -= right[:rank].T @ (right[:rank] @ z)
    v, affine = transfer.unpack(m.array(z, wp.vec3))
    z = transfer.pack(v, affine).numpy()
    tau = (1 - 1 / np.sqrt(2)) * 2e-6
    P = transfer.grid_map.numpy()
    L = np.eye(len(z)) - P.T @ (transfer.grid_mass[:, None] / m.mass * P)
    eigen, vectors = np.linalg.eigh(L)
    S = (vectors / np.sqrt(1 + tau * 2000 * eigen)) @ vectors.T
    mapped = B @ S
    gradient = np.einsum("qic,ij->qjc", G, S)
    whitened = S @ z
    gap = tau * (mapped.astype(np.longdouble) @ whitened.astype(np.longdouble))
    frame = geometry.frame.numpy().astype(np.longdouble) + tau * np.einsum(
        "pa,qpb->qab", whitened.astype(np.longdouble), gradient.astype(np.longdouble)
    )
    return [
        gap.astype(float),
        frame.astype(float),
        data["reference_normal"].numpy(),
        np.full(len(history), maximum),
        data["material"].numpy(),
        mapped,
        gradient,
        data["area"].numpy(),
        np.array([m.mass]),
        np.array([tau]),
    ]


@pytest.mark.parametrize("seed", [218, 306])
@pytest.mark.parametrize("maximum", [1e-13, 1e-15])
def test_near_closed_equilibrium_is_independent_of_proximal_scale(seed, maximum):
    values = near_closed_values(seed, maximum)
    u, force, _, _ = solve_primal(values, return_traction=True, integrated_force=True)
    other_u, other_force, _, _ = solve_primal(values, 0.25, return_traction=True, integrated_force=True)
    np.testing.assert_allclose(other_u, u, rtol=2e-8, atol=1e-20)
    np.testing.assert_allclose(other_force, force, rtol=2e-8, atol=1e-12)
