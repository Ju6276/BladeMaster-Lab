"""Material colors sampled from the project-page videos (https://jango6324.github.io/blademaster/).

Shared by the offline renderers and the browser viewer. A colormap maps reference
particle positions (material frame, m) to RGB in [0, 1]; the cut is the opening,
not a change of material.
"""

import numpy as np


MATERIALS = ("elastic-block", "paper-banana", "fig4-cake", "fig7-tofu", "fig8-luncheon")
# Recorded paper tasks and the material they cut.
TASK_MATERIAL = {
    "fig4-slice-lift": "fig4-cake",
    "fig7-shallow": "fig7-tofu",
    "fig7-deep": "fig7-tofu",
    "fig8": "fig8-luncheon",
}
ELASTIC = (0xE9 / 255, 0xBC / 255, 0x58 / 255)
BANANA = (0.93, 0.91, 0.85)
TOFU = (0.90, 0.87, 0.82)
LUNCHEON = (0.85, 0.69, 0.66)
# Matcha cake of Fig. 4: dark top layer over a light sponge, split at 60 % of
# the reference height.
CAKE_LAYERS = dict(fraction=0.6, below=(0.74, 0.86, 0.52), above=(0.12, 0.38, 0.14))


def uniform(rgb):
    base = np.asarray(rgb, float)
    return lambda reference: np.tile(base, (len(reference), 1))


def layers(boundary_z, below_rgb, above_rgb):
    """Two layers split at a reference height, e.g. a cake."""
    below, above = np.asarray(below_rgb, float), np.asarray(above_rgb, float)
    return lambda reference: np.where((reference[:, 2] > boundary_z)[:, None], above, below)


def mottled(rgb, scale=0.006, amplitude=0.08, seed=7):
    """Smooth random brightness field over reference positions (luncheon-meat mottling).

    A sum of random plane waves with wavelength around `scale` (m); deterministic per seed.
    """
    rng = np.random.default_rng(seed)
    count = 24
    waves = rng.normal(size=(count, 3)) * (2 * np.pi / scale)
    phases = rng.uniform(0, 2 * np.pi, count)
    base = np.asarray(rgb, float)

    def color(reference):
        field = np.cos(np.asarray(reference) @ waves.T + phases).sum(1) * np.sqrt(2.0 / count)
        return np.clip(base * (1.0 + amplitude * field)[:, None], 0, 1)

    return color


def material_colormap(material, reference):
    """Colormap of a material; the cake layer height comes from its reference particles."""
    if material == "fig4-cake":
        z = np.asarray(reference)[:, 2]
        boundary = z.min() + CAKE_LAYERS["fraction"] * np.ptp(z)
        return layers(boundary, CAKE_LAYERS["below"], CAKE_LAYERS["above"])
    if material == "fig8-luncheon":
        return mottled(LUNCHEON)
    colors = {"elastic-block": ELASTIC, "paper-banana": BANANA, "fig7-tofu": TOFU}
    if material not in colors:
        raise ValueError(f"Unknown material {material}")
    return uniform(colors[material])
