import numpy as np
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.solver import Config, Solver


def settled_bottom(board_normal_scale):
    cfg = Config(
        size=(0.06, 0.06, 0.06),
        grid_spacing=0.01,
        dt=0.002,
        young_modulus=10000.0,
        poisson_ratio=0.4,
        external_tool=True,
        board_normal_scale=board_normal_scale,
    )
    s = Solver(cfg)
    port = ExternalToolPort(s)
    away = ToolState([0.5, 0.0, 0.5], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0])
    for _ in range(500):
        port.advance(away, cfg.dt, cutting=False)
    x = s.x.numpy()
    assert np.isfinite(x).all()
    return x[:, 2].min() - s.radius


def test_static_board_takes_the_whole_predicted_penetration_at_large_steps():
    # 12 particle layers: about N g dt^2 / s_n, 4.7 mm with s_n = 0.1 and 0.5 mm with 1.
    assert settled_bottom(0.0) < -0.003
    assert settled_bottom(1.0) > -0.001
