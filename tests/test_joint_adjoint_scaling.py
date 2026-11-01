import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_contact import MappedContacts
from test_separated_adhesion import array, evaluate, fixture


@pytest.mark.parametrize("scale", [0.0, 1e-14, 1e14])
@pytest.mark.parametrize("contact", [False, True])
def test_separated_contact_adjoint_is_linear_in_the_loss_seed(scale, contact):
    gradients = []
    for magnitude in [1.0, scale]:
        inputs, geometry, contacts = fixture()
        if not contact:
            contacts = MappedContacts.from_rows(np.zeros((0, 1)), [], "cuda:0")
        with wp.Tape() as tape:
            outputs = evaluate(inputs, geometry, contacts)
        rng = np.random.default_rng(426)
        seeds = {
            output: array(magnitude * rng.normal(size=output.numpy().shape), output.dtype, False)
            for output in outputs
            if output.size
        }
        tape.backward(grads=seeds)
        gradients.append([
            a.grad.numpy().copy() if a.size else np.zeros_like(a.numpy())
            for a in [*inputs, *contacts.arrays]
        ])
    for expected, actual in zip(*gradients):
        if scale == 0.0:
            np.testing.assert_array_equal(actual, 0.0)
        else:
            np.testing.assert_allclose(actual / scale, expected, rtol=1e-8, atol=1e-12)
