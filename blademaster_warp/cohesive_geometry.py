"""Compensated reference-interface geometry and total-state pullbacks.

A high value and a rounding remainder encode one physical variable. Its
cotangent is carried by the high array; both components survive checkpoints.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

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


def _add_gradient(source, value):
    if source.requires_grad:
        source.grad.assign(source.grad.numpy() + value)


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
        if np.finfo(EXTENDED).nmant <= np.finfo(np.float64).nmant:
            raise RuntimeError("Interface geometry requires extended-precision CPU arithmetic")
        sources = [self.gap, self.frame, separation_map, frame_map, velocity, duration]
        values = [a.numpy().copy() for a in sources]
        g, F, B, G, v, dt = [a.astype(EXTENDED) for a in values]
        gap = g + self.gap_remainder + dt[0] * (B @ v)
        frame = F + self.frame_remainder + dt[0] * np.einsum("pa,qpb->qab", v, G)
        high_gap = wp.array(gap.astype(float), dtype=wp.vec3d, device=self.gap.device, requires_grad=True)
        high_frame = wp.array(
            frame.astype(float), dtype=wp.mat33d, device=self.frame.device, requires_grad=True
        )
        result = InterfaceGeometry(
            high_gap,
            high_frame,
            gap - high_gap.numpy().astype(EXTENDED),
            frame - high_frame.numpy().astype(EXTENDED),
        )
        tape = wp._src.context.runtime.tape
        if tape is not None:

            def backward():
                g, F, B, G, v, dt = [a.astype(EXTENDED) for a in values]
                sg = high_gap.grad.numpy().astype(EXTENDED)
                sF = high_frame.grad.numpy().astype(EXTENDED)
                _add_gradient(self.gap, sg)
                _add_gradient(self.frame, sF)
                _add_gradient(separation_map, dt[0] * (sg @ v.T))
                _add_gradient(frame_map, dt[0] * np.einsum("qab,pa->qpb", sF, v))
                dv = dt[0] * (B.T @ sg + np.einsum("qab,qpb->pa", sF, G))
                _add_gradient(velocity, dv)
                rate = np.einsum("pa,qpb->qab", v, G)
                _add_gradient(duration, np.array([np.sum(sg * (B @ v)) + np.sum(sF * rate)]))

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
            av = a.numpy().astype(EXTENDED) + getattr(self, name + "_remainder")
            bv = b.numpy().astype(EXTENDED) + getattr(other, name + "_remainder")
            value = av + EXTENDED(fraction) * (bv - av)
            out = wp.array(value.astype(float), dtype=a.dtype, device=a.device, requires_grad=True)
            outputs.append(out)
            remainders.append(value - out.numpy().astype(EXTENDED))
            tape = wp._src.context.runtime.tape
            if tape is not None:

                def backward(a=a, b=b, out=out):
                    seed = out.grad.numpy()
                    _add_gradient(a, (1.0 - fraction) * seed)
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
    u_numpy, t_numpy, compliance, info, gap, frame = refine_geometry(
        values,
        geometry.gap_remainder,
        geometry.frame_remainder,
        compliance_scale,
        integrated_force=integrated_force,
        closed_motion=closed_motion,
        inertia=inertia,
    )
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
    return result_type(*result, u, info), updated
