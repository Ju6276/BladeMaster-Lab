"""Current contact partners, multi-cut deduplication and indexed friction cotangents."""

from itertools import combinations

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from blademaster_warp.cohesive_pairing import PairHistory, pairs_at
from blademaster_warp.compensated_state import CompensatedArray


def array(value, low=None):
    return CompensatedArray(
        wp.array(np.asarray(value).reshape(-1, 3), dtype=wp.vec3d, device="cuda:0", requires_grad=True), low
    )


def test_multiple_cut_disagreements_produce_one_physical_contact():
    position = array([[0, 0, 0], [0.0028, 0, 0]])
    result = pairs_at(position, position.high.numpy(), np.array([4, 8]), 0.003, 0.006)
    np.testing.assert_array_equal(result, [[0, 1]])


def test_sliding_reselects_the_current_opposing_particle():
    position = array([[0, 0, 0], [0.0028, 0, 0], [0.0028, 0.01, 0]])
    reference = position.high.numpy().copy()
    codes = np.array([1, 2, 2])
    np.testing.assert_array_equal(pairs_at(position, reference, codes, 0.003, 0.006), [[0, 1]])
    moved = position.high.numpy()
    moved[0, 1] = 0.01
    position.high.assign(moved)
    np.testing.assert_array_equal(pairs_at(position, reference, codes, 0.003, 0.006), [[0, 2]])


def test_reference_neighbors_are_excluded_but_distant_material_can_self_contact():
    position = array([[0, 0, 0], [0.001, 0, 0], [0.002, 0, 0]])
    reference = np.array([[0, 0, 0], [0.003, 0, 0], [0.2, 0, 0]])
    np.testing.assert_array_equal(
        pairs_at(position, reference, np.zeros(3, dtype=int), 0.003, 0.006), [[0, 2], [1, 2]]
    )


@pytest.mark.parametrize("sign", [-1, 1])
def test_current_contact_distance_uses_the_rounding_remainder(sign):
    low = np.zeros((2, 3))
    low[1, 0] = sign * 2.0**-80
    position = array([[0, 0, 0], [0.003, 0, 0]], low)
    result = pairs_at(position, position.high.numpy(), np.array([1, 2]), 0.003, 0.006)
    np.testing.assert_array_equal(result, np.array([[0, 1]] if sign < 0 else [], dtype=int).reshape(-1, 2))


@pytest.mark.parametrize("seed", [12, 39, 81])
def test_spatial_candidates_match_all_pairs_after_rigid_rotation(seed):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(40, 3)) * 0.01
    reference = rng.normal(size=(40, 3)) * 0.02
    codes = rng.integers(0, 9, size=40)
    expected = []
    for p, q in combinations(range(40), 2):
        incompatible = any(
            a and b and a != b for a, b in zip([codes[p] % 3, codes[p] // 3], [codes[q] % 3, codes[q] // 3])
        )
        if np.linalg.norm(points[p] - points[q]) <= 0.01 and (
            incompatible or np.linalg.norm(reference[p] - reference[q]) >= 0.012
        ):
            expected.append([p, q])
    expected = np.asarray(expected, dtype=int).reshape(-1, 2)
    for rotation in [np.eye(3), Rotation.from_rotvec([0.3, -0.7, 1.2]).as_matrix()]:
        result = pairs_at(array(points @ rotation.T), reference @ rotation.T, codes, 0.01, 0.006)
        np.testing.assert_array_equal(result, expected)


def test_reindexed_history_preserves_existing_and_retired_values_and_gradients():
    rng = np.random.default_rng(4093)
    values = rng.normal(size=(3, 3)) * 1e-5
    low = rng.normal(size=(3, 3)) * 1e-24
    original = PairHistory(4, np.array([[0, 1], [0, 2], [1, 3]]), array(values, low))
    with wp.Tape() as tape:
        current, retired = original.reindex(np.array([[0, 2], [2, 3]]))
    np.testing.assert_array_equal(current.values.high.numpy(), [values[1], np.zeros(3)])
    np.testing.assert_array_equal(current.values.low, [low[1], np.zeros(3)])
    np.testing.assert_array_equal(retired.pairs, [[0, 1], [1, 3]])
    np.testing.assert_array_equal(retired.values.high.numpy(), values[[0, 2]])
    seeds = [rng.normal(size=(2, 3)) for _ in range(2)]
    tape.backward(grads={a.values.high: array(seed).high for a, seed in zip([current, retired], seeds)})
    np.testing.assert_array_equal(original.values.high.grad.numpy(), [seeds[1][0], seeds[0][0], seeds[1][1]])
    restored = PairHistory.restore(current.snapshot(), device="cuda:0")
    np.testing.assert_array_equal(restored.pairs, current.pairs)
    np.testing.assert_array_equal(restored.values.high.numpy(), current.values.high.numpy())
    np.testing.assert_array_equal(restored.values.low, current.values.low)


@pytest.mark.parametrize("next_pairs", [[], [[0, 1]]])
def test_history_reindex_handles_empty_and_unchanged_sets(next_pairs):
    pairs = np.asarray(next_pairs, dtype=int).reshape(-1, 2)
    empty = PairHistory.empty(2, device="cuda:0")
    born, retired = empty.reindex(pairs)
    assert len(retired.pairs) == 0
    np.testing.assert_array_equal(born.values.high.numpy(), np.zeros((len(pairs), 3)))
    with wp.Tape() as tape:
        current, retired = born.reindex(pairs)
    tape.backward(grads={current.values.high: array(np.ones((len(pairs), 3))).high})
    np.testing.assert_array_equal(born.values.high.grad.numpy(), np.ones((len(pairs), 3)))
    assert len(retired.pairs) == 0
