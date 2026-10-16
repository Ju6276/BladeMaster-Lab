"""Shared material and blade configurations."""

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
