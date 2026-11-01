"""Nested contact groups retain leaf laws, states and parameter pullbacks."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_contact import MappedContacts, solve
from blademaster_warp.cohesive_contact_batch import CombinedContacts
from test_mapped_contact import fixture
from test_rigid_joint_contact import rigid_fixture


@pytest.mark.parametrize("interface", [False, True])
def test_nested_contacts_preserve_force_history_and_pullback(interface):
    results = []
    for nested in [False, True]:
        inputs, geometry, plane, array = fixture(interface, 0.001)
        _, _, blade, _ = rigid_fixture(interface, 0.004, 0.4)
        batches = [plane, blade]
        if nested:
            empty = MappedContacts.from_rows(np.empty((0, 2)), [], "cuda:0")
            batches = [empty, CombinedContacts([plane, CombinedContacts([blade])])]
        combined = CombinedContacts(batches)
        assert combined.count == plane.count + blade.count
        with wp.Tape() as tape:
            _, displacement, force, contact, history = solve(inputs, geometry, None, combined)
        outputs = [displacement, force, contact.high, history.high]
        rng = np.random.default_rng(2761)
        seeds = [rng.normal(size=a.numpy().shape) * scale
                 for a, scale in zip(outputs, [1e4, 1., 1., 1e4])]
        tape.backward(grads={a: array(seed, a.dtype, False) for a, seed in zip(outputs, seeds) if a.size})
        sources = [*inputs, *plane.arrays, *blade.arrays]
        results.append([*[a.numpy() for a in outputs], contact.low, history.low,
                        *[a.grad.numpy() for a in sources if a.requires_grad]])
    for flat, nested in zip(*results):
        np.testing.assert_array_equal(flat, nested)
