"""Incremental surface updates from the current simulation state."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import grid_cells
from .surface import LinkedSurface, MaterialEmbedding, moving_segment_hits


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2))


@dataclass
class ToolSample:
    time_s: float
    pose: np.ndarray  # position, quaternion xyzw in the material frame
    edge: np.ndarray
    normal: np.ndarray
    cutting: bool
    cut_index: int

    def same_pose_and_cut(self, other):
        return (
            self.cutting == other.cutting
            and self.cut_index == other.cut_index
            and np.array_equal(self.pose, other.pose)
        )


class SurfaceRecorder:
    def __init__(
        self, solver, directory, spacing=None, embedding_backend="cpu", ccd_backend="cpu", precision="float64"
    ):
        if embedding_backend not in ("cpu", "cuda"):
            raise ValueError("Surface embedding backend must be cpu or cuda")
        if ccd_backend not in ("cpu", "cuda"):
            raise ValueError("Surface CCD backend must be cpu or cuda")
        self.ccd = moving_segment_hits
        if ccd_backend == "cuda":
            from .surface_ccd_gpu import CudaSegmentCCD

            self.ccd = CudaSegmentCCD(solver.device)
        self.directory = Path(directory) if directory is not None else None
        # zlib releases the GIL: frames compress on one writer thread while the
        # simulation continues. close() waits for them and re-raises write errors.
        self._writer = None
        self._pending = deque()
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)
            self._writer = ThreadPoolExecutor(1, thread_name_prefix="surface-writer")
        self.current = None
        cfg = solver.config
        # Independent visualization resolution. At h=6 mm, the 120 mm demo's
        # narrow tips are disconnected by node sampling before any cut occurs.
        # A finer reference surface must still embed in the ORIGINAL MPM grid.
        surface_spacing = cfg.grid_spacing / 2 if spacing is None else float(spacing)
        if not np.isfinite(surface_spacing) or surface_spacing <= 0:
            raise ValueError("Surface spacing must be finite and positive")
        self.surface = LinkedSurface(cfg.size, surface_spacing, cfg.shape)
        self.embedding = MaterialEmbedding(
            self.surface.reference,
            solver.reference,
            cfg.grid_spacing,
            solver.origin,
            grid_cells(cfg.size, cfg.grid_spacing),
        )
        if embedding_backend == "cuda":
            from .surface_gpu import CudaMaterialEmbedding

            self.embedding = CudaMaterialEmbedding.from_reference(self.embedding, solver.device, precision)
        self.half = np.array(cfg.blade_half_size)
        # Same band as the material's side assignment (contact.assign): paper h_k = 6 dx.
        self.band = (6.0 * cfg.grid_spacing, 2.0 * cfg.grid_spacing)
        self.previous = None
        self.previous_cut = None
        self.history = []
        self.previous_nodes = None
        self.previous_deformation = None
        self._tool_samples = []
        self.last_path = np.empty((0, 10))
        self.wall_s = 0.0
        self.timing_s = dict(
            node_embedding=0.0, link_sweep=0.0, mesh_construction=0.0, vertex_embedding=0.0, recording=0.0
        )
        method = {
            "method": "reference-grid linked volume, Hermite intersections, local-region dual contouring",
            "embedding_backend": embedding_backend,
            "ccd_backend": ccd_backend,
            "gpu_scope": (
                ["compatible nodal position/deformation fields"] if embedding_backend == "cuda" else []
            )
            + (
                ["finite moving-segment CCD, FP64 polynomial roots and segment bounds"]
                if ccd_backend == "cuda"
                else []
            ),
            "cpu_scope": (["link CCD"] if ccd_backend == "cpu" else [])
            + [
                "Hermite updates",
                "local regions and bounded QEF",
                "triangle construction",
                "final trilinear interpolation",
                "recording",
            ],
            "topology_cache": "reference mesh reused only while no cut-dependent cell is dirty; world-space vertex deformation still recomputed every observation",
            "surface_spacing_m": surface_spacing,
            "material_grid_spacing_m": cfg.grid_spacing,
            "tool_path_columns": [
                "time_s",
                "x_m",
                "y_m",
                "z_m",
                "qx",
                "qy",
                "qz",
                "qw",
                "cutting",
                "cut_index",
            ],
            "embedding": "fixed trilinear weights on original MPM grid, compatible affine-extrapolated position fields",
            "empty_node_extension": "one reference-grid support ring, same compatible cut side, affine",
            "sampling": "all supplied tool substeps; material-link motion interpolated between observations; path retained in each recorded frame",
        }
        if self.directory is not None:
            (self.directory / "method.json").write_text(json.dumps(method, indent=2))

    def record_trajectory(self, times, poses, cutting, cut_index):
        """Retain actual tool substeps independently of the display/recording cadence.

        times are simulation seconds; poses are material-frame [x,y,z,qx,qy,qz,qw].
        The flag/index of a sample governs the interval starting at that sample.
        """
        times, poses = np.asarray(times, float), np.asarray(poses, float)
        if (
            not len(times)
            or poses.shape != (len(times), 7)
            or not np.isfinite(poses).all()
            or not np.isfinite(times).all()
        ):
            raise ValueError("Expected finite tool times and (samples, 7) poses")
        if np.any(np.diff(times) < 0) or (
            self._tool_samples and times[0] < self._tool_samples[-1].time_s - 1e-10
        ):
            raise ValueError("Tool sample times must be monotonic")
        rotations = Rotation.from_quat(poses[:, 3:]).as_matrix()
        local_edge = np.array([[0, -self.half[1], -self.half[2]], [0, self.half[1], -self.half[2]]])
        edges = np.einsum("nij,kj->nki", rotations, local_edge) + poses[:, None, :3]
        for time_s, pose, edge, rotation in zip(times, poses, edges, rotations):
            sample = ToolSample(
                float(time_s), pose.copy(), edge, rotation[:, 0], bool(cutting), int(cut_index)
            )
            # Hold mode supplies the same pose every material step. Keep both ends
            # of a stationary interval, including its material motion, without
            # retaining thousands of identical interior samples.
            if (
                len(self._tool_samples) >= 2
                and sample.same_pose_and_cut(self._tool_samples[-1])
                and sample.same_pose_and_cut(self._tool_samples[-2])
            ):
                self._tool_samples[-1] = sample
            else:
                self._tool_samples.append(sample)

    def update(self, t, x, F, codes, tool, cutting=True, cut_index=0, record_frame=True):
        tick = time.perf_counter()
        s = self.surface
        pose = np.concatenate((tool.position, tool.orientation_xyzw))
        self.record_trajectory([t], pose[None], cutting, cut_index)
        samples = self._tool_samples
        self.last_path = np.array([[p.time_s, *p.pose, int(p.cutting), p.cut_index] for p in samples])
        phase = time.perf_counter()
        nodes, nodeF = self.embedding.node_state(x, F, codes, s.codes)
        self.timing_s["node_embedding"] += time.perf_counter() - phase
        before = s.codes.copy()
        phase = time.perf_counter()
        if self.previous_nodes is not None:
            start = samples[0].time_s
            span = t - start
            blade_points = [p.edge for a, b in zip(samples, samples[1:]) if a.cutting for p in (a, b)]
            candidates = (
                s.swept_candidates(self.previous_nodes, nodes, np.concatenate(blade_points))
                if blade_points
                else np.empty(0, int)
            )

            def interpolate(previous, current, time_s):
                a = np.clip((time_s - start) / span, 0.0, 1.0) if span > 0 else 1.0
                return previous if a == 0 else current if a == 1 else (1 - a) * previous + a * current

            previous_time, interpolated_nodes, interpolated_F = None, None, None
            for a, b in zip(samples, samples[1:]):
                if not a.cutting:
                    continue
                # Adjacent intervals and observations may supply different poses
                # at one timestamp. That reset is not a material substep.
                if b.time_s <= a.time_s:
                    continue
                a_nodes = (
                    interpolated_nodes
                    if previous_time == a.time_s
                    else interpolate(self.previous_nodes, nodes, a.time_s)
                )
                a_F = (
                    interpolated_F
                    if previous_time == a.time_s
                    else interpolate(self.previous_deformation, nodeF, a.time_s)
                )
                b_nodes = (
                    a_nodes if a.time_s == b.time_s else interpolate(self.previous_nodes, nodes, b.time_s)
                )
                b_F = a_F if a.time_s == b.time_s else interpolate(self.previous_deformation, nodeF, b.time_s)
                s.sweep(
                    a.edge,
                    b.edge,
                    b_nodes,
                    a.cut_index,
                    b.normal if b.cut_index == a.cut_index else a.normal,
                    b_F,
                    previous_nodes=a_nodes,
                    previous_deformation=a_F,
                    ccd=self.ccd,
                    band=self.band,
                    candidate_edges=candidates,
                )
                previous_time, interpolated_nodes, interpolated_F = b.time_s, b_nodes, b_F
        self.timing_s["link_sweep"] += time.perf_counter() - phase
        phase = time.perf_counter()
        self.previous_nodes, self.previous_deformation = (
            (nodes, nodeF)
            if np.array_equal(before, s.codes)
            else self.embedding.node_state(x, F, codes, s.codes)
        )
        self.timing_s["node_embedding"] += time.perf_counter() - phase
        self._tool_samples = [samples[-1]]
        self.previous = samples[-1].edge.copy()
        self.previous_cut = cut_index if cutting else None
        phase = time.perf_counter()
        mesh = s.mesh()
        self.timing_s["mesh_construction"] += time.perf_counter() - phase
        phase = time.perf_counter()
        self.embedding.material_grid.extension_nodes = 0
        vertices = self.embedding.vertices(mesh, x, F, codes)
        self.timing_s["vertex_embedding"] += time.perf_counter() - phase
        phase = time.perf_counter()
        count = len(self.history)
        self.current = dict(
            vertices=vertices,
            triangles=mesh.triangles,
            reference=mesh.reference,
            codes=mesh.codes,
            tool_position=tool.position,
            tool_orientation=tool.orientation_xyzw,
            tool_path=self.last_path,
        )
        if self.directory is not None and record_frame:
            frame = {k: np.array(v, copy=True) for k, v in self.current.items()}
            self._submit(np.savez_compressed, self.directory / f"{count:05d}.npz", **frame)
        if record_frame:
            self.history.append(
                dict(
                    time_s=float(t),
                    broken_links=int(s.broken.sum()),
                    updated_cells=s.updated_cells,
                    vertices=len(vertices),
                    triangles=len(mesh.triangles),
                    affine_extended_node_queries=self.embedding.material_grid.extension_nodes,
                )
            )
            if self.directory is not None:
                self._submit(_write_json, self.directory / "frames.json", list(self.history))
        self.timing_s["recording"] += time.perf_counter() - phase
        self.wall_s += time.perf_counter() - tick

    def _submit(self, function, *args, **kwargs):
        while len(self._pending) >= 8:
            self._pending.popleft().result()
        self._pending.append(self._writer.submit(function, *args, **kwargs))

    def close(self):
        """Wait for every recorded frame to be on disk."""
        if self._writer is None:
            return
        tick = time.perf_counter()
        try:
            while self._pending:
                self._pending.popleft().result()
        finally:
            self._writer.shutdown(wait=True)
            self._writer = None
            waited = time.perf_counter() - tick
            self.timing_s["recording"] += waited
            self.wall_s += waited
