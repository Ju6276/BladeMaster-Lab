"""Sampling provenance and common material-cell boundaries."""

from fractions import Fraction

import numpy as np
import pytest

from blademaster_warp.geometry import BANANA_DEMO_SIZE_M, points
from blademaster_warp.reference_coordinates import ReferenceCoordinates, rational


@pytest.mark.parametrize("shape", ["block", "cylinder", "banana"])
@pytest.mark.parametrize("seed", [0, 17])
@pytest.mark.parametrize("spacing", [0.006, 0.004])
def test_reference_metadata_preserves_sampling_and_shape_filtering(shape, seed, spacing):
    size = BANANA_DEMO_SIZE_M if shape == "banana" else (0.012, 0.012, 0.012)
    reference, _, _ = points(size, spacing, shape, seed)
    coordinates = ReferenceCoordinates.sampled(size, spacing, shape, seed)
    np.testing.assert_array_equal(coordinates.head, reference)
    np.testing.assert_allclose(
        np.asarray(coordinates.exact, dtype=float),
        reference,
        rtol=0,
        atol=8 * np.finfo(float).eps * max(size),
    )
    np.testing.assert_array_equal(coordinates.exact - rational(reference), coordinates.remainder)
    assert not coordinates.exact.flags.writeable
    assert not coordinates.head.flags.writeable


@pytest.mark.parametrize("spacing", [0.006, 0.004, 0.0024])
def test_adjacent_reference_cells_share_the_same_exact_boundary(spacing):
    coordinates = ReferenceCoordinates.sampled((0.012,) * 3, spacing, "block")
    half = Fraction(float(spacing)) / 4
    for axis in range(3):
        centres = sorted(set(coordinates.exact[:, axis]))
        assert len(centres) > 1
        for lower, upper in zip(centres, centres[1:]):
            assert lower + half == upper - half


def test_arbitrary_reference_coordinates_are_preserved():
    values = np.random.default_rng(53).normal(size=(7, 3))
    coordinates = ReferenceCoordinates.from_values(values)
    np.testing.assert_array_equal(coordinates.head, values)
    np.testing.assert_array_equal(coordinates.remainder, 0)


def test_material_reference_metadata_uses_the_configured_sampling_seed():
    from blademaster_warp.differentiable_material import MaterialTrajectory
    from blademaster_warp.solver import Config

    model = MaterialTrajectory(
        Config(
            size=(0.012,) * 3,
            grid_spacing=0.006,
            shape="cylinder",
            sampling_seed=17,
            young_modulus=20000.0,
            yield_stress=0.0,
        ),
        precision="float64",
    )
    np.testing.assert_array_equal(model.reference_coordinates.head, model.reference)
    assert model.reference_coordinates is model.reference_coordinates
