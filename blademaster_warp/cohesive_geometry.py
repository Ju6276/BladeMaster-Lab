"""Compensated reference-interface geometry and total-state pullbacks.

A high value and a rounding remainder encode one physical variable. Its
cotangent is carried by the high array; both components survive checkpoints.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from ._compensated import Expansion
from .compensated_state import as_compensated, add_gradient as _add_gradient
from ._cohesive_solve import refine_geometry
from .differentiable_cohesive import (
    CohesiveStep,
    CohesiveForceStep,
    _evaluate,
    _implicit_pullback,
    _validated_inputs,
    _canonical_output,
)


EXTENDED = np.longdouble


@dataclass(frozen=True)
class InterfaceGeometry:
    gap: wp.array
    frame: wp.array
    gap_remainder: np.ndarray | None = None
    frame_remainder: np.ndarray | None = None

    def __post_init__(self):
        if self.gap.dtype != wp.vec3d or self.frame.dtype != wp.mat33d:
            raise ValueError("Interface geometry requires float64 vectors and matrices")
        if self.gap.shape != self.frame.shape or self.gap.device != self.frame.device:
            raise ValueError("Gap and frame must have matching shapes and devices")
        for name, shape in [("gap", (len(self.gap), 3)), ("frame", (len(self.frame), 3, 3))]:
            remainder = getattr(self, name + "_remainder")
            remainder = np.zeros(shape) if remainder is None else np.asarray(remainder, dtype=float).copy()
            if remainder.shape != shape or not np.isfinite(remainder).all():
                raise ValueError("Geometry remainder must be finite and match its high array")
            object.__setattr__(self, name + "_remainder", remainder)

    def snapshot(self):
        return dict(
            gap=self.gap.numpy().copy(),
            frame=self.frame.numpy().copy(),
            gap_remainder=self.gap_remainder.copy(),
            frame_remainder=self.frame_remainder.copy(),
        )

    @classmethod
    def restore(cls, snapshot, *, device=None, requires_grad=True):
        return cls(
            wp.array(snapshot["gap"], dtype=wp.vec3d, device=device, requires_grad=requires_grad),
            wp.array(snapshot["frame"], dtype=wp.mat33d, device=device, requires_grad=requires_grad),
            snapshot["gap_remainder"],
            snapshot["frame_remainder"],
        )

    def advance(self, separation_map, frame_map, velocity, duration):
        velocity, duration = as_compensated(velocity), as_compensated(duration)
        sources = [self.gap, self.frame, separation_map, frame_map, velocity.high, duration.high]
        B = Expansion(separation_map.numpy())
        G = [Expansion(frame_map.numpy()[:, :, c]) for c in range(3)]
        v, dt = velocity.cpu(), duration.cpu()[0]
        gap_rate = B @ v
        frame_rate = [g @ v for g in G]
        gap = Expansion(self.gap.numpy(), self.gap_remainder) + dt * gap_rate
        frames = [
            Expansion(self.frame.numpy()[:, :, c], self.frame_remainder[:, :, c]) + dt * rate
            for c, rate in enumerate(frame_rate)
        ]
        high_gap = wp.array(gap.high, dtype=wp.vec3d, device=self.gap.device, requires_grad=True)
        high_frame = wp.array(
            np.stack([f.high for f in frames], axis=-1),
            dtype=wp.mat33d,
            device=self.frame.device,
            requires_grad=True,
        )
        result = InterfaceGeometry(high_gap, high_frame, gap.low, np.stack([f.low for f in frames], axis=-1))
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                sg = Expansion(high_gap.grad.numpy())
                sF = [Expansion(high_frame.grad.numpy()[:, :, c]) for c in range(3)]
                _add_gradient(self.gap, sg)
                _add_gradient(self.frame, high_frame.grad.numpy())
                _add_gradient(separation_map, dt * (sg @ v.T))
                map_gradients = [dt * (seed @ v.T) for seed in sF]
                _add_gradient(
                    frame_map,
                    Expansion(
                        np.stack([a.high for a in map_gradients], axis=-1),
                        np.stack([a.low for a in map_gradients], axis=-1),
                    ),
                )
                dv = B.T @ sg
                time = (sg * gap_rate).sum()
                for g, seed, rate in zip(G, sF, frame_rate):
                    dv = dv + g.T @ seed
                    time = time + (seed * rate).sum()
                _add_gradient(velocity.high, dt * dv)
                _add_gradient(duration.high, time.reshape(1))

            tape.record_func(backward, [high_gap, high_frame, *[a for a in sources if a.requires_grad]])
        return result

    def extrapolate(self, other, fraction):
        """Form self + fraction * (other - self), including rounding remainders."""
        if self.gap.shape != other.gap.shape or self.gap.device != other.gap.device:
            raise ValueError("Geometry stages must have matching shapes and devices")
        if not np.isfinite(fraction):
            raise ValueError("Geometry stage fraction must be finite")
        outputs, remainders = [], []
        for name in ("gap", "frame"):
            a, b = getattr(self, name), getattr(other, name)
            av = Expansion(a.numpy(), getattr(self, name + "_remainder"))
            bv = Expansion(b.numpy(), getattr(other, name + "_remainder"))
            value = av + fraction * (bv - av)
            out = wp.array(value.high, dtype=a.dtype, device=a.device, requires_grad=True)
            outputs.append(out)
            remainders.append(value.low)
            tape = wp._src.context.runtime.tape
            if tape is not None:

                def backward(a=a, b=b, out=out):
                    seed = Expansion(out.grad.numpy())
                    _add_gradient(a, (Expansion(1.0) - fraction) * seed)
                    _add_gradient(b, fraction * seed)

                tape.record_func(backward, [out, *[a for a in (a, b) if a.requires_grad]])
        return InterfaceGeometry(*outputs, *remainders)


def solve_geometry(
    geometry,
    reference_normal,
    history,
    material,
    separation_map,
    frame_map,
    area,
    particle_mass,
    duration,
    *,
    compliance_scale=1.0,
    integrated_force=False,
    closed_motion=None,
    inertia=None,
):
    """Solve the interface response, retaining compensated geometry.

    closed_motion optionally supplies d with g=B d on positive-area rows;
    it is a factored evaluation of the same free gap, in mass-normalized metres.
    The VJP follows g and B, whose chain rule includes that nodal motion.
    """
    inertia = None if inertia is None else inertia.frozen()
    inputs, values = _validated_inputs(
        geometry.gap,
        geometry.frame,
        reference_normal,
        history,
        material,
        separation_map,
        frame_map,
        area,
        particle_mass,
        duration,
        compliance_scale=compliance_scale,
        inertia=inertia,
        integrated_force=integrated_force,
    )
    precise_u, t_numpy, compliance, info, gap, frame = refine_geometry(
        values,
        geometry.gap_remainder,
        geometry.frame_remainder,
        compliance_scale,
        integrated_force=integrated_force,
        closed_motion=closed_motion,
        inertia=inertia,
    )
    u_numpy = precise_u.astype(float)
    displacement_remainder = (precise_u - u_numpy).astype(float)
    device = geometry.gap.device
    u = wp.array(u_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
    traction = wp.array(t_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
    outer = wp._src.context.runtime.tape
    wp._src.context.runtime.tape = None
    try:
        raw, _, _ = _evaluate(inputs, u, traction, compliance, integrated_force=integrated_force)
        result = [wp.array(a.numpy(), dtype=a.dtype, device=device, requires_grad=True) for a in raw]
    finally:
        wp._src.context.runtime.tape = outer
    result[0].assign(gap.astype(float))
    result[5].assign(frame.astype(float))
    updated = InterfaceGeometry(
        result[0],
        result[5],
        gap - result[0].numpy().astype(EXTENDED),
        frame - result[5].numpy().astype(EXTENDED),
    )
    sources = inputs + ([] if inertia is None else inertia.parameters)
    if outer is not None:

        def backward():
            clones = [
                wp.array(value, dtype=a.dtype, device=device, requires_grad=True)
                for value, a in zip(values, inputs)
            ]
            fixed_u = wp.array(u_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
            fixed_t = wp.array(t_numpy, dtype=wp.vec3d, device=device, requires_grad=True)
            with wp.Tape() as local:
                selected_t = _canonical_output(
                    clones,
                    fixed_u,
                    fixed_t,
                    compliance,
                    info if inertia is None else dict(info, closed_linear_solution=False),
                    integrated_force=integrated_force,
                )
                output, _, _ = _evaluate(
                    clones, fixed_u, selected_t, compliance, integrated_force=integrated_force
                )
            local.backward(
                grads={
                    a: wp.array(b.grad.numpy(), dtype=a.dtype, device=device) for a, b in zip(output, result)
                }
            )
            explicit = [a.grad.numpy().copy() for a in clones]
            gradients = _implicit_pullback(
                inputs,
                values,
                u_numpy,
                t_numpy,
                compliance,
                (fixed_u.grad.numpy() + u.grad.numpy()).ravel(),
                fixed_t.grad.numpy().ravel() / compliance,
                closed=info.get("closed_linear_solution", False),
                inertia=inertia,
                integrated_force=integrated_force,
            )
            explicit.extend(np.zeros_like(a.numpy()) for a in sources[len(inputs) :])
            for source, direct, indirect in zip(sources, explicit, gradients):
                _add_gradient(source, direct + indirect)

        outer.record_func(backward, [*result, u, *[a for a in sources if a.requires_grad]])
    result_type = CohesiveForceStep if integrated_force else CohesiveStep
    return result_type(*result, u, info, displacement_remainder), updated
