"""Reference sampling/stencil adapted from DexLab b955d85 (Apache-2.0).

Synthetic banana, NOT the BladeMaster author's scan or calibration asset.
See vendor/DexLab-LICENSE and NOTICE.md.
"""
from itertools import product
import numpy as np

# Original DexLab banana demo domain and length, shared by both runners.
# This is a synthetic asset, not a measured or author-supplied banana mesh.
BANANA_DEMO_SIZE_M = (.12, .06, .03)
# The original DexLab banana demo's synthetic wedge, not an author knife CAD.
DEMO_KNIFE_HALF_SIZE_M = (.0005, .04, .02)
DEMO_KNIFE_BEVEL_HEIGHT_M = .005


def levelset(x, size, shape):
    """Analytic original boundary, not a cut mesh. Negative inside."""
    x=np.asarray(x);size=np.asarray(size)
    if shape == 'block':
        q=np.abs(x-[0,0,size[2]/2])-size/2
        return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(np.max(q,axis=-1),0)
    if shape == 'cylinder':
        return np.maximum(np.linalg.norm(x[...,:2],axis=-1)-min(size[:2])/2,
                          np.abs(x[...,2]-size[2]/2)-size[2]/2)
    if shape == 'banana':
        u=x[...,0]/(size[0]/2)
        cy=.035*(u*u-.5)
        radius=.003+.009*np.maximum(0,1-u*u)**.55
        return np.maximum(np.abs(x[...,0])-size[0]/2,
            np.sqrt((x[...,1]-cy)**2+(x[...,2]-.012)**2)-radius)
    raise ValueError(shape)


def check_domain(size, shape):
    if shape == 'banana':
        u=np.linspace(-1,1,10001)
        cy=.035*(u*u-.5);r=.003+.009*np.maximum(0,1-u*u)**.55
        if max(np.max(cy+r),-np.min(cy-r)) > size[1]/2+1.e-10 or .024 > size[2]+1.e-10:
            raise ValueError('Banana analytic geometry is clipped by the sampling domain; use y size >= 0.059 m and z >= 0.024 m')


def points(size, spacing, shape):
    size = np.asarray(size)
    check_domain(size,shape)
    cells = np.rint(size / spacing).astype(int)
    origin = -np.array([size[0] / 2, size[1] / 2, 0.])
    coords = [origin[i] + (np.arange(2*cells[i])+.5)*spacing/2 for i in range(3)]
    x = np.array(list(product(*coords)))
    if shape in ('banana','cylinder'):
        x=x[levelset(x,size,shape)<=0]
    elif shape != 'block':
        raise ValueError(shape)
    return x, origin, cells


def stencil(reference, origin, spacing, cells):
    fractional = (reference-origin)/spacing
    base = np.floor(fractional).astype(int)
    local = fractional-base
    offsets = np.array(list(product((0, 1), repeat=3)))
    factors = np.where(offsets[None] == 1, local[:, None], 1-local[:, None])
    w = factors.prod(axis=2)
    g = np.empty((*w.shape, 3))
    for a in range(3):
        other = [b for b in range(3) if b != a]
        g[:, :, a] = (2*offsets[:, a]-1)[None]*factors[:, :, other].prod(axis=2)/spacing
    ijk = base[:, None]+offsets[None]
    nodes = np.ravel_multi_index(ijk.transpose(2, 0, 1), tuple(cells+1))
    return nodes, w, g
