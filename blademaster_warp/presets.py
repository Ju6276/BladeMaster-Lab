"""Shared material and blade configurations."""

import math
from dataclasses import replace
from .geometry import BANANA_DEMO_SIZE_M, DEMO_KNIFE_HALF_SIZE_M, DEMO_KNIFE_BEVEL_HEIGHT_M
from .solver import Config

DEFAULT_BANANA_PRESET = "paper-material"


def banana_config(*, paper_material=True):
    """Paper §IV-C banana material on the synthetic banana geometry."""
    cfg = Config(
        size=BANANA_DEMO_SIZE_M,
        grid_spacing=0.006,
        dt=0.00025,
        blade_half_size=DEMO_KNIFE_HALF_SIZE_M,
        blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M,
        shape="banana",
        # A regular lattice puts the cut plane between two particle columns, so the
        # blade wedges them apart and the cutting-work resistance never engages.
        sampling_seed=1,
        # Fixed knife contact distance: with the particle radius h/4 the effective
        # blade thickness, and the steady cutting force, grow linearly with h.
        # On this 6 mm grid the solver raises it to h/4 minus the half thickness, 1 mm.
        blade_contact_radius=0.0005,
    )
    if not paper_material:
        return replace(cfg, contact_reference_dt=cfg.dt)
    # Contact scales are the values at 25 us; 62.5 us (16 substeps per 1 ms
    # robot step) keeps that stiffness, see docs/RESULTS.md.
    return replace(
        cfg,
        young_modulus=400000.0,
        poisson_ratio=0.45,
        yield_stress=30000.0,
        blade_mass=0.1,
        friction_static=0.2,
        friction_dynamic=0.2,
        cutting_energy=700.0,
        dt=0.0000625,
        contact_reference_dt=0.000025,
    )


def banana_dt(cfg, spacing, robot_step=0.001):
    """Material step for a grid spacing: at most 62.5 us and half the grid CFL limit.

    At 3 mm, 62.5 us (CFL 0.77) leaves kinetic energy growing after the cut while
    41.7 us decays; see docs/RESULTS.md. The step divides the robot step exactly.
    """
    nu = cfg.poisson_ratio
    p_modulus = cfg.young_modulus * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    wave_speed = math.sqrt(p_modulus / cfg.density)
    limit = min(0.0000625, 0.5 * spacing / wave_speed)
    return robot_step / math.ceil(robot_step / limit - 1e-9)
