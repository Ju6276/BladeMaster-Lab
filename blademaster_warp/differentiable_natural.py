"""Rebuild cut fields and contact pairs from each trajectory's evolving state."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial import cKDTree
import warp as wp

from . import contact
from .differentiable_cutting import FixedCutContact


@wp.kernel
def select_pairs(
    candidates: wp.array2d(dtype=int),
    x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
    reference: wp.array(dtype=wp.vec3), codes: wp.array(dtype=wp.int64),
    radius: float, spacing: float, dt: float, activity: wp.array(dtype=int),
):
    index = wp.tid()
    p, q = candidates[index, 0], candidates[index, 1]
    activity[index] = contact.pair_activity(
        x[p], x[q], v[p], v[q], reference[p], reference[q], codes[p], codes[q],
        radius, spacing, dt,
    )


@dataclass(frozen=True)
class CutInstruction:
    cutting: bool = True
    cut_index: int = 0


class NaturalCutContact(FixedCutContact):
    """External hold-mode knife, with side assignment at material-step boundaries.

    Discrete codes and pair IDs follow each forward trajectory. Continuous
    states use Warp's executed-branch derivative. Release is either the original
    boundary switch or the within-step impulse integral used by EventCutContact.
    """

    def __init__(self, trajectory, *, release="integrated", friction_update='lagged'):
        super().__init__(trajectory, friction_update=friction_update)
        cfg = trajectory.config
        if cfg.tool_count != 1 or cfg.free_blade:
            raise ValueError("Natural cutting requires one externally prescribed knife")
        if release not in ("boundary", "integrated"):
            raise ValueError("Release must be boundary or integrated")
        if release == "integrated" and cfg.cut_hold:
            raise ValueError("Integrated release requires cut_hold=False")
        self.integrate_release = release == "integrated"
        self.reference = trajectory.array(trajectory.reference, wp.vec3, False)
        self.max_cut_index = 0
        while 3 ** (self.max_cut_index + 2) * trajectory.node_count <= 2**63 - 1:
            self.max_cut_index += 1

    def history(self, rigid=None, pairs=None, damage=None, side=None, *, codes=None,
                pair_keys=None, cut_index=0):
        m = self.model
        self._check_cut_index(cut_index, 0)
        result = super().history(rigid, pairs, damage, side)
        codes = np.zeros(m.n, dtype=np.int64) if codes is None else np.asarray(codes, dtype=np.int64)
        if (codes.shape != (m.n,) or np.any(codes < 0)
                or np.any(codes // 3 ** (int(cut_index) + 1))):
            raise ValueError("Side codes must fit the current cut index and particle count")
        keys = np.asarray([] if pair_keys is None else pair_keys, dtype=np.int64)
        if (keys.ndim != 1 or len(keys) != result.pairs.shape[0]
                or np.any(keys < 0) or np.any(keys >= m.n * m.n)
                or np.any(keys // m.n >= keys % m.n)
                or np.any(np.diff(keys) <= 0)):
            raise ValueError("Pair keys must be sorted unique p*n+q with p<q and matching histories")
        result.codes = m.array(codes, wp.int64, False)
        result.pair_keys = keys.copy()
        result.cut_index = int(cut_index)
        return result

    def _check_cut_index(self, index, previous):
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or index < previous or index > self.max_cut_index):
            raise ValueError(f"Cut indices must be monotonic integers within [0, {self.max_cut_index}]")

    def contact_keys(self, state, codes, dt=None):
        m, cfg = self.model, self.model.config
        dt = cfg.dt if dt is None else dt
        x, v = state.x.numpy(), state.v.numpy()
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise FloatingPointError("Nonfinite particle positions or velocities")
        # ||vp-vq|| <= 2 max||v|| bounds the original predictive contact reach.
        speed = np.max(np.linalg.norm(v.astype(np.float64), axis=1))
        reach = cfg.grid_spacing / 2 + 2 * dt * speed
        reach = float(np.nextafter(np.asarray(reach, dtype=x.dtype), np.asarray(np.inf, dtype=x.dtype)))
        candidates = cKDTree(x).query_pairs(reach, output_type="ndarray")
        if not len(candidates):
            return np.empty(0, dtype=np.int64)
        flags = m.zeros(len(candidates), int, False)
        m.launch(select_pairs, len(candidates),
                 inputs=[m.array(candidates, int, False), state.x, state.v, self.reference, codes,
                         cfg.grid_spacing / 4, cfg.grid_spacing, dt],
                 outputs=[flags], device=m.device)
        flags = flags.numpy()
        if np.any(flags < 0):
            raise ValueError("Coincident contact particle centres")
        pairs = candidates[flags == 1]
        return np.sort(pairs[:, 0].astype(np.int64) * m.n + pairs[:, 1])

    def step(self, state, history, tool, instruction=CutInstruction()):
        m, cfg = self.model, self.model.config
        if history.codes is None or history.pair_keys is None:
            raise ValueError("Natural cutting history requires side codes and pair keys")
        self._check_cut_index(instruction.cut_index, history.cut_index)
        if instruction.cut_index != history.cut_index:
            history = replace(history, damage=m.zeros(m.n, float), cut_index=int(instruction.cut_index))
        keys = self.contact_keys(state, history.codes)
        branch = self.freeze(history.codes.numpy(), keys, history.pair_keys,
                             history.damage.numpy() >= 1.0, cutting=instruction.cutting)
        result, updated, reaction = super().step(state, history, tool, branch)
        updated.codes = wp.clone(history.codes, requires_grad=False)
        released = m.zeros(1, int, False)
        dirty = m.zeros(1, int, False)
        m.launch(contact.mark_released, m.n, inputs=[updated.damage], outputs=[released], device=m.device)
        m.launch(contact.assign, m.n,
                 inputs=[result.x, result.v, updated.codes, tool.position, tool.velocity, tool.rotation,
                         tool.omega, cfg.blade_half_size, cfg.grid_spacing, cfg.grid_spacing / 4,
                         cfg.board_half_extent, cfg.dt, 3 ** instruction.cut_index,
                         int(instruction.cutting), updated.damage, released, 0, int(cfg.label_board_layer)],
                 outputs=[dirty], device=m.device)
        updated.pair_keys = keys
        updated.cut_index = int(instruction.cut_index)
        return result, updated, reaction
