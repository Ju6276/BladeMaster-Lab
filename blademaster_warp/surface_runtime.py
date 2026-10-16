"""Incremental surface updates from the current simulation state."""

from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .surface import LinkedSurface, MaterialEmbedding, moving_segment_hits


class SurfaceRecorder:
    def __init__(self, solver, directory, spacing=None, embedding_backend="cpu", ccd_backend="cpu"):
        if embedding_backend not in ("cpu", "cuda"):
            raise ValueError("Surface embedding backend must be cpu or cuda")
        if ccd_backend not in ("cpu", "cuda"):
            raise ValueError("Surface CCD backend must be cpu or cuda")
        self.ccd = moving_segment_hits
        if ccd_backend == "cuda":
            from .surface_ccd_gpu import CudaSegmentCCD

            self.ccd = CudaSegmentCCD(solver.device)
        self.directory = Path(directory) if directory is not None else None
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=False)
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
            np.rint(np.asarray(cfg.size) / cfg.grid_spacing).astype(int),
        )
        if embedding_backend == "cuda":
            from .surface_gpu import CudaMaterialEmbedding

            self.embedding = CudaMaterialEmbedding.from_reference(self.embedding, solver.device)
        self.half = np.array(cfg.blade_half_size)
        self.previous = None
        self.previous_cut = None
        self.history = []
        self.previous_nodes = None
        self.previous_deformation = None
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
            "embedding": "fixed trilinear weights on original MPM grid, compatible affine-extrapolated position fields",
            "empty_node_extension": "one reference-grid support ring, same compatible cut side, affine",
            "sampling": "continuous segment/segment detection with linearly interpolated actual blade AND material-link endpoints; interval recorded per frame",
        }
        if self.directory is not None:
            (self.directory / "method.json").write_text(json.dumps(method, indent=2))

    def update(self, t, x, F, codes, tool, cutting=True, cut_index=0, record_frame=True):
        tick = time.perf_counter()
        s = self.surface
        r = Rotation.from_quat(tool.orientation_xyzw).as_matrix()
        edge = (
            np.array([[0, -self.half[1], -self.half[2]], [0, self.half[1], -self.half[2]]]) @ r.T
            + tool.position
        )
        if cutting and self.previous_cut == cut_index and self.previous is not None:
            phase = time.perf_counter()
            nodes, nodeF = self.embedding.node_state(x, F, codes, s.codes)
            self.timing_s["node_embedding"] += time.perf_counter() - phase
            phase = time.perf_counter()
            s.sweep(
                self.previous,
                edge,
                nodes,
                cut_index,
                r[:, 0],
                nodeF,
                previous_nodes=self.previous_nodes,
                previous_deformation=self.previous_deformation,
                ccd=self.ccd,
            )
            self.timing_s["link_sweep"] += time.perf_counter() - phase
        self.previous = edge.copy()
        self.previous_cut = cut_index if cutting else None
        phase = time.perf_counter()
        self.previous_nodes, self.previous_deformation = self.embedding.node_state(x, F, codes, s.codes)
        self.timing_s["node_embedding"] += time.perf_counter() - phase
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
        )
        if self.directory is not None and record_frame:
            np.savez_compressed(self.directory / f"{count:05d}.npz", **self.current)
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
                (self.directory / "frames.json").write_text(json.dumps(self.history, indent=2))
        self.timing_s["recording"] += time.perf_counter() - phase
        self.wall_s += time.perf_counter() - tick
