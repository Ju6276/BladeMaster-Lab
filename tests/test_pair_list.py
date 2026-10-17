from dataclasses import replace
import numpy as np
import warp as wp
import pytest
from blademaster_warp.coupling import ExternalToolPort, ToolState
from blademaster_warp.presets import banana_config
from blademaster_warp.solver import Solver
from blademaster_warp.solver import Config


@pytest.mark.parametrize(
    "dt,speed,skin", [(0.001, 1.0, 0.0), (0.002, 1.0, 0.0), (0.0000625, 16.0, 0.0), (0.001, 1.0, 0.0001)]
)
@pytest.mark.parametrize("graph", [False, True])
def test_rebuild_covers_this_steps_predictive_contacts(dt, speed, skin, graph):
    cfg = Config(
        size=(0.018, 0.018, 0.018), grid_spacing=0.006, dt=dt, grid_storage="compact", pair_skin=skin
    )
    results = []
    for mode in ("all", "hash", "list"):
        s = Solver(replace(cfg, pair_search=mode))
        x = np.zeros((s.n, 3), np.float32)
        x[:, 0] = np.arange(s.n) * 0.1 + 2
        x[:, 2] = 0.1
        x[:2] = [[0, 0, 0.1], [0.0048, 0, 0.1]]
        v = np.zeros_like(x)
        v[:2] = [[speed, 0, 0], [-speed, 0, 0]]
        codes = np.zeros(s.n, np.int64)
        codes[:2] = [1, 2]
        s.x.assign(x)
        s.v.assign(v)
        s.codes.assign(codes)
        s.speed.fill_(speed)
        if graph:
            import blademaster_warp.contact as contact

            wp.load_module(module=contact, device=s.device)
            with wp.ScopedCapture(device=s.device) as capture:
                s.particle_contacts()
            wp.capture_launch(capture.graph)
        else:
            s.particle_contacts()
        assert s.pair_count[1].numpy()[0] == 1
        results.append(s.impulses.numpy())
    for result in results[1:]:
        np.testing.assert_allclose(result, results[0], rtol=1e-6, atol=1e-12)


def pair_impulses(s, listed):
    if not hasattr(s, "pair_grid"):
        s.pair_grid = wp.HashGrid(*s.hash_grid_shape, device=s.device)
        s.pair_grid.reserve(s.n)
    s.pair_list = listed
    s.impulses.zero_()
    s.particle_contacts()
    count = int(s.pair_count[(s.step_count + 1) % 2].numpy()[0])
    return s.impulses.numpy().astype(float), count


def test_neighbour_list_matches_the_per_step_grid_query_during_a_cut():
    cfg = replace(banana_config(), external_tool=True, grid_storage="compact")
    s = Solver(cfg)
    port = ExternalToolPort(s)
    half, interval, speed = cfg.blade_half_size[2], 0.001, 0.02
    builds, checked = [], 0
    for i in range(1500):
        edge = max(0.036 - speed * i * interval, 0.0)
        state = ToolState(
            [0.0, 0.0, edge + half], [0, 0, 0, 1], [0.0, 0.0, -speed if edge > 0 else 0.0], [0, 0, 0]
        )
        port.advance(state, interval, cutting=True, graph=True)
        builds.append(int(s.pair_list_builds.numpy()[0]))
        if i % 50 == 49:
            listed, n_listed = pair_impulses(s, True)
            grid, n_grid = pair_impulses(s, False)
            s.pair_list = True
            assert n_listed == n_grid
            np.testing.assert_allclose(listed, grid, rtol=1e-4, atol=1e-9 * s.mass)
            checked += n_grid > 0
    assert checked >= 3
    # Rebuilt while side codes change, but far less often than every substep.
    substeps = 1500 * round(interval / cfg.dt)
    assert 1 < builds[-1] < substeps
