"""Separated adhesive traction preserves fracture work and moving-face virtual work."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive import surface_normal
from blademaster_warp.cohesive_adhesion import adhesive_proximal_response, adhesive_surface_response
from blademaster_warp.precision import launch_double


def array(value, dtype=wp.float64, grad=True):
    return wp.array(np.asarray(value), dtype=dtype, device="cuda:0", requires_grad=grad)


def evaluate(inputs):
    outputs = [
        wp.zeros(len(inputs[0]), dtype=wp.vec3d if i < 2 else wp.float64, device="cuda:0", requires_grad=True)
        for i in range(5)
    ]
    launch_double(adhesive_proximal_response, len(inputs[0]), inputs=inputs, outputs=outputs)
    return outputs


@wp.kernel
def mapped_normals(
    frame: wp.array(dtype=wp.mat33), reference: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.vec3)
):
    i = wp.tid()
    out[i] = surface_normal(frame[i], reference[i])


@wp.kernel
def incremental_energy(
    free_gap: wp.array(dtype=wp.vec3),
    gap: wp.array(dtype=wp.vec3),
    parameters: wp.array(dtype=wp.vec3),
    area: wp.array(dtype=float),
    storage: wp.array(dtype=float),
    fracture: wp.array(dtype=float),
    total: wp.array(dtype=float),
):
    i = wp.tid()
    wp.atomic_add(
        total,
        0,
        0.5 * wp.length_sq(gap[i] - free_gap[i]) / parameters[i][2] + area[i] * (storage[i] + fracture[i]),
    )


@pytest.mark.parametrize("gc", [7.0, 70.0, 700.0, 1400.0])
def test_compressed_adhesion_has_exact_shear_fracture_work(gc):
    strength, area = 5e4, 4e-6
    failure = 2 * gc / strength
    compliance = 0.1 * failure / (area * strength)
    opening = np.array([0, 0.3, 0.6, 0.1, 0.6, 1, 1.2, 0.3, 0]) * failure
    history = np.maximum.accumulate(opening)
    traction = np.array(
        [
            strength if high == 0 else strength * (1 - high / failure) * r / high if high < failure else 0
            for r, high in zip(opening, history)
        ]
    )
    q = np.column_stack(
        [np.full(len(opening), -3 * failure), opening + compliance * area * traction, np.zeros(len(opening))]
    )
    inputs = [
        array(q, wp.vec3d),
        array(np.tile([1, 0, 0], (len(q), 1)), wp.vec3d),
        array(history),
        array(np.tile([strength, gc, compliance], (len(q), 1)), wp.vec3d),
        array(np.full(len(q), area)),
    ]
    gap, force, maximum, storage, fracture = [a.numpy() for a in evaluate(inputs)]
    np.testing.assert_array_equal(gap[:, 0], q[:, 0])
    np.testing.assert_array_equal(force[:, 0], 0.0)
    np.testing.assert_allclose(gap[:, 1], opening, atol=failure * 1e-14)
    np.testing.assert_allclose(force[:, 1], area * traction, atol=area * strength * 1e-14)
    work = 0.5 * (force[1:, 1] + force[:-1, 1]) * np.diff(opening)
    np.testing.assert_allclose(work, area * (np.diff(storage) + np.diff(fracture)), atol=area * gc * 1e-14)
    np.testing.assert_allclose(work.sum(), area * gc, rtol=1e-14)
    np.testing.assert_allclose(maximum, history, atol=failure * 1e-14)


def fixture():
    rng = np.random.default_rng(12519)
    count = 48
    strength = rng.uniform(2e4, 8e4, count)
    gc = rng.choice([7.0, 70.0, 700.0, 1400.0], count)
    area = rng.uniform(1e-6, 8e-6, count)
    failure = 2 * gc / strength
    compliance = rng.uniform(0.02, 0.7, count) * failure / (area * strength)
    q = rng.normal(size=(count, 3)) * 0.7 * failure[:, None]
    history = rng.uniform(0.05, 1.3, count) * failure
    history[::3] = 0
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    inputs = [
        array(q, wp.vec3d),
        array(normals, wp.vec3d),
        array(history),
        array(np.column_stack([strength, gc, compliance]), wp.vec3d),
        array(area),
    ]
    return inputs, failure, rng


def test_adhesion_all_input_gradients():
    inputs, failure, rng = fixture()
    with wp.Tape() as tape:
        outputs = evaluate(inputs)
    gap, force, *_ = [a.numpy() for a in outputs]
    q, n, h, p, area = [a.numpy() for a in inputs]
    np.testing.assert_allclose(q - gap, p[:, 2, None] * force, rtol=2e-13, atol=1e-14)
    compressed = np.sum(q * n, axis=1) < 0
    np.testing.assert_allclose(np.sum(force[compressed] * n[compressed], axis=1), 0, atol=1e-15)
    scales = [failure[:, None], (p[:, 0] * area)[:, None], failure, p[:, 1], p[:, 1]]
    seeds = [rng.normal(size=a.numpy().shape) / scale for a, scale in zip(outputs, scales)]
    tape.backward(grads={a: array(s, a.dtype, False) for a, s in zip(outputs, seeds)})
    for index, source in enumerate(inputs):
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * (failure[:, None] if index == 0 else baseline)
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-6, 1e-7]:
            values = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                values.append([a.numpy() for a in evaluate(inputs)])
            fd = sum(np.sum((b - a) * s) for a, b, s in zip(*values, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=1e-6, atol=1e-8)
        source.assign(baseline)


def test_adhesive_force_and_face_moment_are_incremental_energy_derivatives():
    inputs, _, rng = fixture()
    frame = array(
        np.tile(np.eye(3), (len(inputs[0]), 1, 1)) + rng.normal(scale=0.1, size=(len(inputs[0]), 3, 3)),
        wp.mat33d,
    )
    reference = inputs[1]
    normal = wp.zeros_like(reference, requires_grad=True)
    inputs[1] = normal
    total = wp.zeros(1, dtype=wp.float64, device="cuda:0", requires_grad=True)
    with wp.Tape() as tape:
        launch_double(mapped_normals, len(normal), inputs=[frame, reference], outputs=[normal])
        result = evaluate(inputs)
        launch_double(
            incremental_energy,
            len(normal),
            inputs=[inputs[0], result[0], inputs[3], inputs[4], result[3], result[4]],
            outputs=[total],
        )
    tape.backward(total)
    moment = wp.empty_like(frame)
    launch_double(
        adhesive_surface_response,
        len(normal),
        inputs=[result[0], frame, reference, result[1]],
        outputs=[moment],
    )
    np.testing.assert_allclose(inputs[0].grad.numpy(), result[1].numpy(), rtol=2e-12, atol=1e-14)
    np.testing.assert_allclose(frame.grad.numpy(), moment.numpy(), rtol=2e-12, atol=1e-14)
