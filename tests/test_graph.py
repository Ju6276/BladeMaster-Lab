from dataclasses import replace
import numpy as np
from blademaster_warp.solver import Config, Solver


def test_graph_matches_eager_across_cut_and_history_parity():
    cfg = replace(Config(), yield_stress=2000.0)
    eager, graph = Solver(cfg), Solver(cfg)
    for s in (eager, graph):
        s.tool_pos.assign([[0.0, 0.0, 0.047]])
    # Odd/even batches exercise both contact-history buffers, target updates,
    # damage reset on a new cut, and a graph-cache hit after a target change.
    for n, cut, on, x in [
        (5, 0, True, 0.0),
        (6, 0, True, 0.001),
        (5, 0, False, 0.0),
        (6, 1, True, -0.001),
        (6, 1, True, 0.002),
    ]:
        commands = np.column_stack([np.full(n, x), np.zeros(n), np.linspace(0.02, 0.019, n)])
        eager.advance(commands, on, cut, graph=False)
        graph.advance(commands, on, cut, graph=True)
        a, b = eager.state(), graph.state()
        assert eager.step_count == graph.step_count
        np.testing.assert_allclose(a["position"], b["position"], atol=2e-7)
        np.testing.assert_allclose(a["velocity"], b["velocity"], atol=2e-5)
        np.testing.assert_allclose(a["deformation"], b["deformation"], atol=2e-5)
        np.testing.assert_allclose(a["plastic_deformation"], b["plastic_deformation"], atol=2e-5)
        np.testing.assert_allclose(a["blade_position"], b["blade_position"], atol=2e-7)
        np.testing.assert_array_equal(a["sides"], b["sides"])
