import numpy as np
from blademaster_warp.solver import Solver
from blademaster_warp.diagnostics import Metrics


def test_recorded_energy_matches_uniform_analytic_states():
    s = Solver()
    meter = Metrics(s)
    v = np.tile([0.2, -0.1, 0.3], (s.n, 1))
    s.v.assign(v)
    values = meter.read()
    np.testing.assert_allclose(values[0], 0.5 * s.n * s.mass * 0.14, rtol=2.0e-5)
    assert abs(values[2]) < 1.0e-10
    # Pure rigid rotation has no elastic energy, uniform dilatation does.
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    s.F.assign(np.tile(R, (s.n, 1, 1)))
    assert abs(meter.read()[2]) < 1.0e-10
    s.F.assign(np.tile(1.01 * np.eye(3), (s.n, 1, 1)))
    expected = s.n * s.volume * (3 * s.mu * 0.01**2 + 0.5 * s.lam * (1.01**3 - 1) ** 2)
    np.testing.assert_allclose(meter.read()[2], expected, rtol=5.0e-5)
