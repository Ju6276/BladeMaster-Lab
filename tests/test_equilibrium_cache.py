import numpy as np

from blademaster_warp._cohesive_expansion import E
from blademaster_warp._cohesive_refinement import _cached_response


def test_reused_evaluation_is_isolated_from_state_and_result_mutation():
    calls = []

    def equations(state):
        calls.append(state.high.copy())
        return state + 1.0, E(np.eye(len(state.high))), [state * 2.0]

    evaluate = _cached_response(equations)
    state = E([1.0, 2.0])
    residual, jacobian, gaps = evaluate(state)
    residual.high[:] = 99.0
    jacobian.high[:] = 99.0
    gaps[0].high[:] = 99.0
    actual = evaluate(state)
    assert len(calls) == 1
    np.testing.assert_array_equal(actual[0].numpy(), [2.0, 3.0])
    np.testing.assert_array_equal(actual[1].numpy(), np.eye(2))
    np.testing.assert_array_equal(actual[2][0].numpy(), [2.0, 4.0])
    state.high[0] = 3.0
    evaluate(state)
    assert len(calls) == 2


def test_remainder_changes_and_new_equation_scopes_recompute():
    calls = []

    def equations(state):
        calls.append(None)
        return state, E(np.eye(len(state.high))), []

    state = E([1.0], [2.0**-60])
    evaluate = _cached_response(equations)
    evaluate(state)
    changed = E([1.0], [2.0**-61])
    result = evaluate(changed)
    assert len(calls) == 2
    np.testing.assert_array_equal(result[0].low, changed.low)
    _cached_response(equations)(changed)
    assert len(calls) == 3
