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
        # Full-cell jitter amplifies the release kick in coarse banana cuts.
        # Keep regular sub-cell samples; seeds > 0 remain available
        # for sampling sensitivity studies. The particle-radius contact envelope
        # also reaches a blade placed between the regular particle columns.
        sampling_seed=0,
        blade_contact_radius=0.0,
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
        # The paper gives no guided drive. At the DexLab 250 N/m the ~5 N peak cutting
        # force leaves a 20 mm lag and the blade never reaches the board; 1000 N/m
        # (the Franka Cartesian stiffness), critically damped for the 0.1 kg blade.
        drive_stiffness=1000.0,
        drive_damping=20.0,
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


def tangent_history_dt(cfg, maximum):
    """Halve a step until fixed-normal sticking history has eigenvalues inside the unit circle.

    Contact stiffness stays fixed through contact_reference_dt. For state
    (tangent velocity, history / dt), the update is [[1-d, -k], [1, 1]],
    where k=gamma_k and d=2*gamma_c*sqrt(k).
    """
    if maximum <= 0:
        raise ValueError("Maximum time step must be positive")
    if cfg.tangent_stiffness == 0:
        return maximum
    if cfg.tangent_damping <= 0:
        raise ValueError("Explicit sticking history requires positive tangential damping")
    step = maximum
    while True:
        ratio = 1.0 if cfg.contact_reference_dt == 0 else (step / cfg.contact_reference_dt) ** 2
        stiffness = cfg.tangent_stiffness * ratio
        damping = 2 * cfg.tangent_damping * math.sqrt(stiffness)
        if stiffness < damping and 4 - 2 * damping + stiffness > 0:
            return step
        if cfg.contact_reference_dt == 0:
            raise ValueError("Reducing dt requires contact_reference_dt to preserve contact stiffness")
        step *= 0.5
