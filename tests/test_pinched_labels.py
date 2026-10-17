import numpy as np
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.solver import Config, Solver
from blademaster_warp.topology import components


def test_bottom_layer_beside_a_pitched_edge_resting_on_the_board_is_cut():
    # A blade pitched about its length axis: bottom particles a few millimetres to the
    # side lie below its edge plane although the edge itself rests on the board.
    cfg = Config(
        size=(0.06, 0.03, 0.03),
        grid_spacing=0.006,
        dt=0.00025,
        sampling_seed=1,
        external_tool=True,
        board_normal_scale=1.0,
        blade_half_size=(0.0005, 0.025, 0.02),
    )
    s = Solver(cfg)
    port = ExternalToolPort(s)
    pitch = np.deg2rad(10.0)
    q = [0.0, np.sin(pitch / 2), 0.0, np.cos(pitch / 2)]
    reach = cfg.blade_half_size[2] * np.cos(pitch)
    lean = cfg.blade_half_size[2] * np.sin(pitch)
    interval, speed, start, stop = 0.001, 0.05, 0.04, 0.0002
    for i in range(round(1.0 / interval)):
        edge = max(start - speed * i * interval, stop)
        moving = edge > stop
        state = ToolState([lean, 0.0, edge + reach], q, [0.0, 0.0, -speed if moving else 0.0], [0, 0, 0])
        port.advance(state, interval, cutting=True)
    codes = s.codes.numpy()
    x = s.x.numpy()
    assert not ((codes == 0) & (np.abs(x[:, 0]) < 2 * cfg.grid_spacing)).any()
    _, sizes = components(s.node_numpy, s.weight_numpy, codes)
    assert len(sizes) == 2
