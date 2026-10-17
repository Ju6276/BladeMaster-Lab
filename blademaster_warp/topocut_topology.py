"""TopoCut topology discovery (Wang et al., CoRL 2025, Sec. 3.1 and App. 7.6).

After a cut action: SDF(x) = min_p |x - x_p| - r_p on a voxel grid above the
board, with the knife trajectories of the cuts since the last new piece forced
outside the object; Warp Marching Cubes on the zero isosurface; Laplacian smoothing; connected components of the faces;
each particle joins the cluster i with SDF_i(x_p) < tau (else -1); and cluster
IDs are combined with the previous ones so earlier pieces stay distinct.
"""

from dataclasses import dataclass
import numpy as np
import warp as wp


@dataclass(frozen=True)
class TopologyParams:
    # Voxel edge and particle influence radius r_p, in particle spacings (h / 2).
    voxel: float = 0.5
    influence: float = 1.0
    # The carved trajectory is at least this many voxels thick, so the gap it
    # forces survives sampling even where the blade is thinner than a voxel.
    gap_voxels: float = 2.0
    smoothing_alpha: float = 0.5
    smoothing_steps: int = 3
    # tau of Eq. 11, in particle spacings.
    tau: float = 0.5


@wp.func
def box_sdf(x: wp.vec3, pos: wp.vec3, rot: wp.quat, half: wp.vec3):
    q = wp.quat_rotate_inv(rot, x - pos)
    d = wp.vec3(wp.abs(q[0]) - half[0], wp.abs(q[1]) - half[1], wp.abs(q[2]) - half[2])
    outside = wp.length(wp.vec3(wp.max(d[0], 0.0), wp.max(d[1], 0.0), wp.max(d[2], 0.0)))
    return outside + wp.min(wp.max(d[0], wp.max(d[1], d[2])), 0.0)


@wp.kernel
def particle_sdf(
    grid: wp.uint64,
    x: wp.array(dtype=wp.vec3),
    lower: wp.vec3,
    voxel: float,
    radius: float,
    reach: float,
    poses: wp.array(dtype=wp.transform),
    half: wp.vec3,
    field: wp.array3d(dtype=float),
):
    i, j, k = wp.tid()
    v = lower + voxel * wp.vec3(float(i), float(j), float(k))
    nearest = reach
    query = wp.hash_grid_query(grid, v, reach)
    p = int(0)
    while wp.hash_grid_query_next(query, p):
        nearest = wp.min(nearest, wp.length(v - x[p]))
    # The object lies on the board: intersect with z >= 0.
    value = wp.max(nearest - radius, -v[2])
    for t in range(poses.shape[0]):
        pose = poses[t]
        # Edge lowered by one voxel: a cut ending within a voxel of the board carves through.
        centre = wp.transform_get_translation(pose)
        rot = wp.transform_get_rotation(pose)
        centre = centre - wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 0.5 * voxel))
        if box_sdf(v, centre, rot, half + wp.vec3(0.0, 0.0, 0.5 * voxel)) <= 0.0:
            value = wp.max(value, voxel)
    field[i, j, k] = value


@wp.kernel
def nearest_cluster(
    meshes: wp.array(dtype=wp.uint64),
    x: wp.array(dtype=wp.vec3),
    tau: float,
    reach: float,
    labels: wp.array(dtype=int),
):
    p = wp.tid()
    best = tau
    label = int(-1)
    for i in range(meshes.shape[0]):
        q = wp.mesh_query_point_sign_winding_number(meshes[i], x[p], reach)
        if q.result:
            closest = wp.mesh_eval_position(meshes[i], q.face, q.u, q.v)
            d = q.sign * wp.length(x[p] - closest)
            if d < best:
                best = d
                label = i
    labels[p] = label


def smooth(vertices, faces, alpha, steps):
    """Eq. 10: v_i <- v_i + alpha * sum_j (v_j - v_i), normalised by the vertex degree."""
    from scipy.sparse import coo_matrix

    n = len(vertices)
    a = np.concatenate([faces[:, [0, 1, 2]].ravel(), faces[:, [1, 2, 0]].ravel()])
    b = np.concatenate([faces[:, [1, 2, 0]].ravel(), faces[:, [0, 1, 2]].ravel()])
    adjacency = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    adjacency.data[:] = 1.0
    degree = np.maximum(np.asarray(adjacency.sum(1)).ravel(), 1.0)
    for _ in range(steps):
        vertices = vertices + alpha * (adjacency @ vertices / degree[:, None] - vertices)
    return vertices


def face_components(faces, n_vertices):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    a = faces[:, [0, 1, 2]].ravel()
    b = faces[:, [1, 2, 0]].ravel()
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n_vertices, n_vertices))
    _, vertex_label = connected_components(graph, directed=False)
    return vertex_label[faces[:, 0]]


class TopologyTracker:
    """Cluster IDs of the particles, updated after each cut action."""

    def __init__(self, n, spacing, half_size, params=TopologyParams(), device=None):
        self.params = params
        self.device = wp.get_device(device)
        self.particle_spacing = spacing / 2
        self.half = np.asarray(half_size, float)
        self.labels = np.zeros(n, int)
        self.meshes = []
        # Trajectories of cut actions that have not yet split off a piece; a piece
        # that needs several cuts (a wedge) is carved by all of them together.
        self.pending = []

    def reconstruct(self, positions, knife_poses=()):
        """One topology discovery; knife_poses are (position, quaternion xyzw) of the cut's trajectory."""
        prm, s = self.params, self.particle_spacing
        voxel, radius = prm.voxel * s, prm.influence * s
        x = np.asarray(positions, np.float32)
        lower = x.min(0) - radius - 2 * voxel
        upper = x.max(0) + radius + 2 * voxel
        dims = np.ceil((upper - lower) / voxel).astype(int) + 1
        upper = lower + (dims - 1) * voxel
        half = self.half.copy()
        half[0] = max(half[0], prm.gap_voxels * voxel / 2)
        before = len(self.pieces())
        self.pending += list(knife_poses)
        poses = [wp.transform(wp.vec3(*p), wp.quat(*q)) for p, q in self.pending]
        with wp.ScopedDevice(self.device):
            gx = wp.array(x, dtype=wp.vec3)
            reach = radius + 2 * voxel
            grid = wp.HashGrid(*(int(d) for d in np.minimum(np.ceil((upper - lower) / reach), 128)))
            grid.build(gx, reach)
            field = wp.empty(tuple(int(d) for d in dims), dtype=float)
            wp.launch(
                particle_sdf,
                field.shape,
                [
                    grid.id,
                    gx,
                    wp.vec3(*lower),
                    voxel,
                    radius,
                    reach,
                    wp.array(poses, dtype=wp.transform),
                    wp.vec3(*half),
                    field,
                ],
            )
            mc = wp.MarchingCubes(
                *(int(d) for d in dims),
                domain_bounds_lower_corner=wp.vec3(*lower),
                domain_bounds_upper_corner=wp.vec3(*upper),
            )
            mc.surface(field, 0.0)
            vertices = mc.verts.numpy().astype(float)
            faces = mc.indices.numpy().reshape(-1, 3)
            if not len(faces):
                self.labels[:] = -1
                self.meshes = []
                return self.labels
            vertices = smooth(vertices, faces, prm.smoothing_alpha, prm.smoothing_steps)
            face_label = face_components(faces, len(vertices))
            meshes = []
            for c in np.unique(face_label):
                used = faces[face_label == c]
                # Voids between particles extract as inward-facing closed surfaces.
                a, b, d = (vertices[used[:, k]] for k in range(3))
                if np.einsum("ij,ij->i", a, np.cross(b, d)).sum() <= 0.0:
                    continue
                keep, local = np.unique(used, return_inverse=True)
                meshes.append(
                    wp.Mesh(
                        points=wp.array(vertices[keep], dtype=wp.vec3),
                        indices=wp.array(local.ravel().astype(np.int32), dtype=wp.int32),
                        support_winding_number=True,
                    )
                )
            # All particles and smoothed surface vertices lie within these bounds.
            query_reach = float(np.linalg.norm(upper - lower)) + prm.tau * s
            labels = wp.empty(len(x), dtype=int)
            wp.launch(
                nearest_cluster,
                len(x),
                [wp.array([m.id for m in meshes], dtype=wp.uint64), gx, prm.tau * s, query_reach, labels],
            )
            cluster = labels.numpy()
        self.meshes = [(m.points.numpy(), m.indices.numpy().reshape(-1, 3)) for m in meshes]
        # A new ID for every (previous ID, cluster) pair; Eq. 11 leaves -1 unassigned.
        assigned = cluster >= 0
        previous = self.labels.copy()
        # A particle left unassigned by an earlier cut (inside its gap) takes the
        # previous ID of the nearest particle in its new cluster that had one.
        orphan = assigned & (previous < 0)
        if orphan.any():
            from scipy.spatial import cKDTree

            for c in np.unique(cluster[orphan]):
                known = np.flatnonzero((cluster == c) & (previous >= 0))
                lost = np.flatnonzero(orphan & (cluster == c))
                if len(known):
                    previous[lost] = previous[known[cKDTree(x[known]).query(x[lost])[1]]]
        pair = previous * (len(meshes) + 1) + cluster
        self.labels = np.full(len(x), -1)
        self.labels[assigned] = np.unique(pair[assigned], return_inverse=True)[1]
        self.unassigned = int((~assigned).sum())
        if len(self.pieces()) > before:
            self.pending = []
        return self.labels

    def pieces(self):
        labels = self.labels[self.labels >= 0]
        return np.bincount(labels) if len(labels) else np.zeros(0, int)
