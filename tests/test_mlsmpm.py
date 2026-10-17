import numpy as np
from blademaster_warp.coupling import ToolState
from blademaster_warp.mlsmpm import MlsSolver
from blademaster_warp.solver import Config

SIZE = (0.04, 0.02, 0.024)


def config(**kw):
    return Config(
        size=SIZE,
        grid_spacing=0.006,
        dt=0.0001,
        young_modulus=400000.0,
        poisson_ratio=0.4,
        yield_stress=40000.0,
        blade_half_size=(0.0005, 0.015, 0.02),
        sampling_seed=1,
        **kw,
    )


def parked():
    return ToolState([0, 0, 0.2], [0, 0, 0, 1], [0, 0, 0], [0, 0, 0])


def test_block_rests_on_the_board_without_damage():
    s = MlsSolver(config())
    for _ in range(50):
        s.advance(parked(), 0.002)
    st = s.state()
    assert np.abs(st["velocity"]).max() < 0.05
    assert st["position"][:, 2].min() > 0.9 * s.radius
    assert not st["damage"].any()


def test_soft_block_settling_under_gravity_is_not_damage():
    # Fig. 4 material: rho g H / E is about 6 % strain at the bottom, beyond Eq. 6's 2.5 %.
    soft = Config(
        size=(0.06, 0.06, 0.056),
        grid_spacing=0.01,
        dt=0.002,
        young_modulus=10000.0,
        poisson_ratio=0.4,
        density=1100.0,
        normal_scale=1.0,
        board_normal_scale=1.0,
        blade_half_size=(0.0005, 0.05, 0.04),
    )
    literal = MlsSolver(soft)
    settled = MlsSolver(soft, damage_from_start=False)
    for _ in range(250):
        literal.advance(parked(), 0.002)
        settled.advance(parked(), 0.002)
    settled.start_damage()
    for _ in range(250):
        literal.advance(parked(), 0.002)
        settled.advance(parked(), 0.002)
    assert literal.damage.numpy().sum() > 0.05 * literal.n
    assert not settled.damage.numpy().any()
    assert settled.x.numpy()[:, 2].max() < 0.056


def test_inverted_particle_is_clamped_like_plasticinelab_not_fatal():
    s = MlsSolver(config())
    F = s.F.numpy()
    F[0] = np.diag([1.0, 1.0, -0.5])
    s.F.assign(F)
    s.advance(parked(), 0.0002)
    assert s.diagnostics() == dict(left_domain=0, clamped_inversions=1)
    assert np.linalg.det(s.F.numpy()[0]) > 0


def test_diverged_particle_is_reported_without_out_of_bounds_access():
    import pytest

    s = MlsSolver(config())
    x = s.x.numpy()
    x[0] = 1.0e30
    x[1] = np.nan
    s.x.assign(x)
    with pytest.raises(RuntimeError, match="2 particles non-finite or outside the grid"):
        s.advance(parked(), 0.002)
    assert np.isfinite(s.x.numpy()[2:]).all()


def test_knife_damages_material_at_its_plane_and_feels_the_reaction():
    s = MlsSolver(config(), damage_from_start=False)
    for _ in range(25):
        s.advance(parked(), 0.002)
    s.start_damage()
    half, speed, interval = 0.02, 0.05, 0.002
    lift = []
    for i in range(int(0.024 / speed / interval)):
        edge = SIZE[2] + 0.002 - speed * i * interval
        r = s.advance(ToolState([0, 0, edge + half], [0, 0, 0, 1], [0, 0, -speed], [0, 0, 0]), interval)
        lift.append(r.force_n[2])
    damaged = s.state()["damage"] > 0
    assert damaged.sum() >= 3
    assert np.abs(s.reference[damaged, 0]).mean() < 2 * s.config.grid_spacing
    assert np.mean(lift[-10:]) > 0.5
