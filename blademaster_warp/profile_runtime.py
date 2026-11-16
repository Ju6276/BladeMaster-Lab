"""Select explicit tapered geometry without changing legacy kernel interfaces."""

import importlib
from functools import lru_cache


@lru_cache(maxsize=None)
def replacement_kernel(kernel, batch):
    name = kernel.module.name.rsplit(".", 1)[-1]
    if name not in ("contact", "tool", "blade_constraint"):
        return None
    if "bevel" not in [v.label for v in kernel.adj.args]:
        return None
    prefix = "batch_profile_" if batch else "profile_"
    module = importlib.import_module("." + prefix + name, __package__)
    return getattr(module, kernel.key)


def profile_kernel(kernel, args, config, batch=False):
    width = config.blade_edge_half_width
    if width is None:
        return None, args
    target = replacement_kernel(kernel, batch)
    return (None, args) if target is None else (target, [*args, width])
