"""Shared material/asset configuration, independent of who drives the knife."""
from dataclasses import replace
from .geometry import BANANA_DEMO_SIZE_M, DEMO_KNIFE_HALF_SIZE_M, DEMO_KNIFE_BEVEL_HEIGHT_M
from .solver import Config

DEFAULT_BANANA_PRESET = 'paper-material'


def banana_config(*, paper_material=True):
    """Published material values on the original synthetic demo geometry.

    Contact discretization, step size and synthetic assets are declared
    numerical/test choices, not author data. Drivers only choose external_tool.
    """
    cfg=Config(size=BANANA_DEMO_SIZE_M,grid_spacing=.006,dt=.00025,
               blade_half_size=DEMO_KNIFE_HALF_SIZE_M,
               blade_bevel_height=DEMO_KNIFE_BEVEL_HEIGHT_M,shape='banana')
    if paper_material:
        cfg=replace(cfg,young_modulus=400000.,poisson_ratio=.45,yield_stress=30000.,
                    blade_mass=.1,friction_static=.2,friction_dynamic=.2,cutting_energy=700.,dt=.000025)
    # Contact stiffness is fixed at its value for the preset dt; other dt keep it.
    return replace(cfg,contact_reference_dt=cfg.dt)
