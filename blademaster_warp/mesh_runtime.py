"""Explicit mesh dispatch, preserving the legacy wedge paths."""
import importlib
from functools import lru_cache
@lru_cache(None)
def replacement(kernel,batch):
    name=kernel.module.name.rsplit(".",1)[-1]
    if name not in ("contact","blade_constraint"):
        return None
    module=importlib.import_module("."+("batch_mesh_" if batch else "mesh_")+name,__package__)
    target=getattr(module,kernel.key)
    return target if "geometry" in [a.label for a in target.adj.args] else None
def mesh_kernel(kernel,args,geometry,batch=False):
    if geometry is None:
        return None,args
    target=replacement(kernel,batch)
    return (None,args) if target is None else (target,[*args,geometry.data])
