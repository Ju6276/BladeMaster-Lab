"""World-indexed batch execution of the existing BladeMaster solver equations.

Each object group has its own dt/material/particle layout; worlds share launches,
never reference nodes, pair tables, histories or reaction accumulators.
"""

from typing import Any
import types
import numpy as np
import warp as wp
from .solver import Solver
from .blade_constraint import BladeConstraint
from . import batch_material, batch_contact, batch_tool, batch_blade_constraint


@wp.kernel
def replicate(src: wp.array(dtype=Any), dst: wp.array(dtype=Any), count: int):
    i = wp.tid()
    dst[i] = src[i % count]


@wp.kernel
def masked_fill(dst: wp.array(dtype=Any), enabled: wp.array(dtype=int), count: int, value: Any):
    i = wp.tid()
    if enabled[i // count] != 0:
        dst[i] = value


@wp.kernel
def masked_copy(src: wp.array(dtype=Any), dst: wp.array(dtype=Any), enabled: wp.array(dtype=int), count: int):
    i = wp.tid()
    if enabled[i // count] != 0:
        dst[i] = src[i]


@wp.kernel
def align_history(
    a: wp.array(dtype=Any),
    b: wp.array(dtype=Any),
    active: wp.array(dtype=int),
    parity: wp.array(dtype=int),
    expected: int,
    count: int,
):
    i = wp.tid()
    world = i // count
    if active[world] != 0 and parity[world] != expected:
        value = a[i]
        a[i] = b[i]
        b[i] = value


@wp.kernel
def needs_alignment(
    active: wp.array(dtype=int), parity: wp.array(dtype=int), expected: int, flag: wp.array(dtype=int)
):
    w = wp.tid()
    if active[w] != 0 and parity[w] != expected:
        wp.atomic_max(flag, 0, 1)


@wp.kernel
def store_parity(active: wp.array(dtype=int), parity: wp.array(dtype=int), value: int):
    world = wp.tid()
    if active[world] != 0:
        parity[world] = value


@wp.kernel
def mask_condition(
    parent: wp.array(dtype=int),
    condition: wp.array2d(dtype=int),
    selected: wp.array(dtype=int),
    flag: wp.array(dtype=int),
):
    w = wp.tid()
    selected[w] = int(parent[w] != 0 and condition[w, 0] != 0)
    if selected[w] != 0:
        wp.atomic_max(flag, 0, 1)


class BatchArray:
    def __init__(self, owner, data):
        self.owner, self.data = owner, data

    def __getattr__(self, name):
        return getattr(self.data, name)

    def __getitem__(self, key):
        if isinstance(key, slice) and key.start in (None, 0) and key.stop == 1 and self.data.shape[1] == 1:
            return self
        raise ValueError("Only the single-tool full slice is supported on batch arrays")

    def numpy(self):
        return self.data.numpy()

    def zero_(self):
        self.fill_(0)

    def fill_(self, value):
        a = self.data.flatten()
        count = self.data.size // self.owner.worlds
        v = self.data.dtype(value)
        wp.launch(masked_fill, a.size, [a, self.owner.enabled, count, v], device=self.owner.device)


class WarpProxy:
    def __getattr__(self, name):
        return getattr(wp, name)

    def copy(self, dst, src, *args, **kwargs):
        if isinstance(dst, BatchArray) and isinstance(src, BatchArray):
            if args or kwargs:
                raise ValueError("Partial masked copies unsupported")
            owner = dst.owner
            wp.launch(
                masked_copy,
                dst.data.size,
                [src.data.flatten(), dst.data.flatten(), owner.enabled, dst.data.size // owner.worlds],
                device=owner.device,
            )
            return
        return wp.copy(dst, src, *args, **kwargs)

    def capture_if(self, condition, body):
        return condition.owner.conditional(condition, body, False)

    def capture_while(self, condition, body):
        return condition.owner.conditional(condition, body, True)


def cloned(method):
    glob = dict(method.__globals__)
    glob["wp"] = WarpProxy()
    return types.FunctionType(method.__code__, glob, method.__name__, method.__defaults__, method.__closure__)


class BatchedConstraint:
    pass


for _name, _method in BladeConstraint.__dict__.items():
    if isinstance(_method, types.FunctionType):
        setattr(BatchedConstraint, _name, cloned(_method))


class BatchedSolver:
    """Equal-geometry/material worlds, vectorized inside each original kernel."""

    def __init__(self, config, worlds, device="cuda:0", *, reference_points=None, precision="float32"):
        if precision != "float32":
            raise ValueError("World-indexed solver currently supports float32")
        if worlds < 1:
            raise ValueError("worlds must be positive")
        if config.tool_count != 1 or not config.external_tool or config.tool_substep != "hold":
            raise ValueError("Batched integration supports one externally held tool per world")
        if config.pair_search not in ("auto", "list") or config.grid_storage != "compact":
            raise ValueError("Batched integration requires unchanged list search / compact grid")
        self.worlds = worlds
        prototype = Solver(config, device, reference_points=reference_points)
        self.device = prototype.device
        self.enabled = wp.ones(worlds, dtype=int, device=device)
        self.active = wp.ones(worlds, dtype=int, device=device)
        self._condition_buffers = {}
        arrays = {}

        def batch(value):
            if isinstance(value, wp.array):
                if id(value) not in arrays:
                    dst = wp.empty((worlds, *value.shape), dtype=value.dtype, device=device)
                    wp.launch(
                        replicate, dst.size, [value.flatten(), dst.flatten(), value.size], device=device
                    )
                    arrays[id(value)] = BatchArray(self, dst)
                return arrays[id(value)]
            if isinstance(value, list):
                return [batch(v) for v in value]
            if isinstance(value, dict):
                return {k: batch(v) for k, v in value.items()}
            return value

        for name, value in vars(prototype).items():
            if name != "blade_constraint":
                setattr(self, name, batch(value))
        # Source single-tool views alias the full tool arrays; preserve that
        # identity rather than broadcasting sliced Python objects separately.
        self.tool_views = [
            dict(
                pos=self.tool_pos,
                vel=self.tool_vel,
                rot=self.tool_rot,
                omega=self.tool_omega,
                reaction=self.reaction,
                torque=self.torque,
                table=self.tool_table_impulse,
            )
        ]
        self.blade_constraint = None
        if prototype.blade_constraint is not None:
            bc = object.__new__(BatchedConstraint)
            for name, value in vars(prototype.blade_constraint).items():
                setattr(bc, name, self if value is prototype else batch(value))
            self.blade_constraint = bc
        self._arrays = list(arrays.values())
        for a in self._arrays:
            if a.data.dtype == wp.int32 and a.data.shape == (worlds, 1):
                self._condition_buffers[id(a)] = (
                    wp.zeros(worlds, dtype=int, device=device),
                    wp.zeros(1, dtype=int, device=device),
                )
        self._graphs = {}
        self.world_parity = wp.zeros(worlds, dtype=int, device=device)
        self.parity_dirty = wp.zeros(1, dtype=int, device=device)
        self.packed = BatchArray(self, wp.zeros((worlds, 49), dtype=float, device=device))
        self.result = BatchArray(self, wp.zeros((worlds, 10), dtype=float, device=device))
        self._modules = {
            "material": batch_material,
            "contact": batch_contact,
            "tool": batch_tool,
            "blade_constraint": batch_blade_constraint,
        }
        self._reset_snapshots = None
        self.enabled = wp.zeros(worlds, dtype=int, device=device)
        seen = set()
        for array in self._arrays:
            if array.dtype in seen:
                continue
            seen.add(array.dtype)
            array.fill_(0)
            flat = array.data.flatten()
            wp.launch(masked_copy, flat.size, [flat, flat, self.enabled, flat.size // worlds], device=device)
        for buffers in (self.pair_keys, self.pair_history, self.pair_used, self.pair_count):
            left, right = [a.data.flatten() for a in buffers]
            wp.launch(
                align_history,
                left.size,
                [left, right, self.enabled, self.world_parity, 0, left.size // worlds],
                device=device,
            )
        self.enabled = self.active

    def launch(self, kernel, dim, args):
        module = self._modules[kernel.module.name.rsplit(".", 1)[-1]]
        target = getattr(module, kernel.key)
        from .profile_runtime import profile_kernel
        replacement, args = profile_kernel(kernel, args, self.config, batch=True)
        if replacement is not None:
            target = replacement
        dims = (dim,) if isinstance(dim, int) else tuple(dim)
        values = [a.data if isinstance(a, BatchArray) else a for a in args]
        return wp.launch(target, (self.worlds, *dims), [self.enabled, *values], device=self.device)

    def conditional(self, condition, body, loop):
        selected, flag = self._condition_buffers[id(condition)]
        parent = self.enabled

        def update():
            flag.zero_()
            wp.launch(
                mask_condition, self.worlds, [parent, condition.data, selected, flag], device=self.device
            )

        def execute():
            previous = self.enabled
            self.enabled = selected
            try:
                body()
            finally:
                self.enabled = previous
            if loop:
                update()

        update()
        if loop:
            wp.capture_while(flag, execute)
        else:
            wp.capture_if(flag, execute)

    def advance(self, packed, enabled, duration=0.002, cutting=True):
        steps = round(duration / self.config.dt)
        if abs(steps * self.config.dt - duration) > 1e-12:
            raise ValueError("dt must divide exchange interval")
        packed = np.asarray(packed, dtype=np.float32)
        enabled = np.asarray(enabled)
        if packed.shape != (self.worlds, 49) or not np.isfinite(packed).all():
            raise ValueError("Expected finite [worlds,49] external tool states")
        if enabled.shape != (self.worlds,) or not np.isin(enabled, [0, 1]).all():
            raise ValueError("Expected one boolean active flag per world")
        if (np.abs(np.linalg.norm(packed[:, 3:7], axis=1) - 1) > 1e-5).any():
            raise ValueError("External tool quaternion must be unit length")
        mobility = packed[:, 13:].reshape(self.worlds, 6, 6)
        if np.any(mobility) and (
            np.max(np.abs(mobility - mobility.swapaxes(-1, -2))) > 1e-10
            or np.linalg.eigvalsh(mobility).min() < -1e-10
        ):
            raise ValueError("External tool mobility must be symmetric positive semidefinite")
        self.active.assign(enabled.astype(np.int32))
        self.enabled = self.active
        self.packed.data.assign(packed)
        key = (steps, bool(cutting), self.step_count % 2)
        if key not in self._graphs:
            for module in self._modules.values():
                wp.load_module(module=module, device=self.device)
            count = self.step_count
            try:
                with wp.ScopedCapture(device=self.device) as capture:
                    from . import tool, material

                    self.parity_dirty.zero_()
                    wp.launch(
                        needs_alignment,
                        self.worlds,
                        [self.active, self.world_parity, count % 2, self.parity_dirty],
                        device=self.device,
                    )

                    def align():
                        for buffers in (self.pair_keys, self.pair_history, self.pair_used, self.pair_count):
                            left, right = [a.data.flatten() for a in buffers]
                            wp.launch(
                                align_history,
                                left.size,
                                [
                                    left,
                                    right,
                                    self.active,
                                    self.world_parity,
                                    count % 2,
                                    left.size // self.worlds,
                                ],
                                device=self.device,
                            )

                    wp.capture_if(self.parity_dirty, align)
                    self.launch(
                        tool.load_external_state,
                        1,
                        [
                            self.packed,
                            self.tool_pos,
                            self.tool_rot,
                            self.tool_vel,
                            self.tool_omega,
                            self.tool_mobility,
                            self.interval_impulse,
                            self.interval_angular_impulse,
                        ],
                    )
                    for _ in range(steps):
                        self._step_kernels((0.0, 0.0, 0.0), cutting, 0)
                    wp.launch(
                        store_parity,
                        self.worlds,
                        [self.active, self.world_parity, (count + steps) % 2],
                        device=self.device,
                    )
                    self.launch(material.validate_interval, self.n, [self.x, self.v, self.F, self.errors])
                    self.launch(
                        tool.store_interval,
                        1,
                        [self.interval_impulse, self.interval_angular_impulse, self.errors, self.result],
                    )
                    if self.blade_constraint is not None:
                        from .blade_constraint import store_residual

                        self.launch(store_residual, 1, [self.blade_constraint.max_violation, self.result])
                self._graphs[key] = capture.graph
            finally:
                self.step_count = count
        wp.capture_launch(self._graphs[key])
        self.step_count += steps
        return self.result.data

    def save_reset(self):
        # Rebuild grid/pair caches, while retaining the actual board-friction
        # history established during settling. Reset snapshots precede cutting.
        if np.any(self.codes.numpy()) or np.any(self.damage.numpy()):
            raise ValueError("Reset snapshots must be taken before cutting")
        names = (
            "x",
            "v",
            "F",
            "Fp",
            "viscous",
            "stress",
            "codes",
            "damage",
            "tool_pos",
            "tool_rot",
            "tool_vel",
            "tool_omega",
            "tool_mobility",
            "histories",
            "impulses",
        )
        self._reset_snapshots = {name: wp.clone(getattr(self, name).data) for name in names}

    def reset(self, mask):
        if self._reset_snapshots is None:
            raise RuntimeError("No settled reset state")
        self.active.assign(np.asarray(mask, dtype=np.int32))
        self.enabled = self.active
        for name, snapshot in self._reset_snapshots.items():
            dst = getattr(self, name).data
            wp.launch(
                masked_copy,
                dst.size,
                [snapshot.flatten(), dst.flatten(), self.active, dst.size // self.worlds],
                device=self.device,
            )
        for name in (
            "interval_impulse",
            "interval_angular_impulse",
            "reaction",
            "torque",
            "board",
            "work",
            "contact_parts",
            "errors",
            "speed",
            "cut_released",
            "single_field_codes",
        ):
            getattr(self, name).zero_()
        for key in self.pair_keys:
            key.fill_(-1)
        for count in self.pair_count:
            count.zero_()
        for history in self.pair_history:
            history.zero_()
        self.topology_dirty.fill_(1)
        self.pair_list_dirty.fill_(1)
        wp.launch(
            store_parity,
            self.worlds,
            [self.active, self.world_parity, self.step_count % 2],
            device=self.device,
        )
        if self.blade_constraint is not None:
            for value in vars(self.blade_constraint).values():
                if isinstance(value, BatchArray):
                    value.zero_()


# Reuse host scheduling of physical kernels, with masked fills and per-world
# conditional branches; no particle/material/contact expression is rewritten.
for _name, _method in Solver.__dict__.items():
    if _name in BatchedSolver.__dict__ or _name in ("__dict__", "__weakref__"):
        continue
    if isinstance(_method, types.FunctionType):
        setattr(BatchedSolver, _name, cloned(_method))
    elif isinstance(_method, property):
        setattr(BatchedSolver, _name, _method)
