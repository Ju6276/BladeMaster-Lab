"""Closed CAD blade queries and its measured piecewise-linear cutting edge."""
import json, hashlib
from pathlib import Path
import numpy as np
import warp as wp

@wp.struct
class MeshTool:
    mesh: wp.uint64
    edge: wp.array(dtype=wp.vec3)
    edge_count: int
    radius: float

@wp.func
def mesh_query(local: wp.vec3, geometry: MeshTool):
    hit = wp.mesh_query_point_sign_winding_number(geometry.mesh, local, 10.0)
    closest = wp.mesh_eval_position(geometry.mesh, hit.face, hit.u, hit.v)
    normal = wp.mesh_eval_face_normal(geometry.mesh, hit.face)
    delta = local - closest
    length = wp.length(delta)
    if length > 1.0e-10:
        normal = hit.sign * delta / length
    return closest, normal, hit.sign * length

@wp.func
def edge_at(y: float, geometry: MeshTool):
    lo = int(0)
    hi = geometry.edge_count - 1
    inside = y >= geometry.edge[lo][1] and y <= geometry.edge[hi][1]
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if geometry.edge[mid][1] <= y:
            lo = mid
        else:
            hi = mid
    a = geometry.edge[lo]
    b = geometry.edge[hi]
    u = wp.clamp((y - a[1]) / (b[1] - a[1]), 0.0, 1.0)
    return a + u * (b - a), inside

@wp.func
def side_surface(local: wp.vec3, side: float, geometry: MeshTool):
    start = wp.vec3(side * (geometry.radius + 0.01), local[1], local[2])
    direction = wp.vec3(-side, 0.0, 0.0)
    hit = wp.mesh_query_ray(geometry.mesh, start, direction, 2.0 * geometry.radius + 0.02)
    near = start + hit.t * direction
    return near, hit.normal, hit.result

class MeshGeometry:
    def __init__(self, manifest, device):
        meta=json.loads(Path(manifest).read_text())
        path=Path(meta["geometry"])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=meta["geometry_sha256"]:
            raise ValueError("Blade geometry hash mismatch")
        data=np.load(path)
        vertices=data["vertices_blademaster"].astype(np.float32)
        faces=data["faces"].astype(np.int32).ravel()
        edge=data["edge_blademaster"].astype(np.float32)
        if len(edge)<2 or not np.all(np.diff(edge[:,1])>0):
            raise ValueError("Mesh edge must be ordered, nondegenerate and monotone")
        self.mesh=wp.Mesh(points=wp.array(vertices,dtype=wp.vec3,device=device),
                          indices=wp.array(faces,dtype=int,device=device),support_winding_number=True)
        self.edge=wp.array(edge,dtype=wp.vec3,device=device)
        self.data=MeshTool()
        self.data.mesh=self.mesh.id
        self.data.edge=self.edge
        self.data.edge_count=len(edge)
        self.data.radius=float(np.linalg.norm(vertices,axis=1).max())
        self.metadata=meta
