"""Natural rigid-geometry perturbations through the joint material/interface root."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_contact import solve
from blademaster_warp.cohesive_rigid_contact import RigidContacts
from test_mapped_contact import fixture


def rotated(q, value):
    return value + 2 * np.cross(q[:3], np.cross(q[:3], value) + q[3] * value)


@pytest.mark.parametrize("interface", [False, True])
def test_finite_blade_flat_face_matches_plane_equilibrium(interface):
    inputs, geometry, plane, array = fixture(interface, 0.001)
    row = plane.rows()[0]
    row.pop("normal")
    row.update(
        point=[0.0, 0.0, 0.1 - 2e-4],
        position=[0, 0, 0],
        rotation=[0, 0, 0, 1],
        tool_velocity=[0, 0, 0],
        omega=[0, 0, 0],
        half=[0.1, 0.1, 0.1],
        bevel=0.0,
        offset=0.0,
    )
    blade = RigidContacts.from_rows(plane.mapping.cpu(), [row], "cuda:0")
    a, b = [solve(inputs, geometry, None, contacts) for contacts in [plane, blade]]
    for first, second in zip(a[1:], b[1:]):
        first = first.numpy() if isinstance(first, wp.array) else first.cpu().numpy()
        second = second.numpy() if isinstance(second, wp.array) else second.cpu().numpy()
        np.testing.assert_allclose(first, second, rtol=2e-12, atol=1e-15)


def rigid_fixture(interface, bevel, speed):
    inputs, geometry, _, array = fixture(interface, speed)
    axis = np.array([0.3, -0.7, 0.2])
    axis /= np.linalg.norm(axis)
    q = np.r_[axis * np.sin(0.37 / 2), np.cos(0.37 / 2)]
    origin = np.array([0.001, -0.001, 0.002])
    if bevel:
        n = np.array([bevel, 0, -0.001]) / np.hypot(bevel, 0.001)
        points = [np.array([0.0005, 0.0, -0.01 + 0.5 * bevel]) + 2e-4 * n, [0.0003, -0.0049, 0.002]]
    else:
        points = [[0.0011, 0.00515, 0.002], [0.0003, -0.0051, 0.01015]]
    rows = [
        dict(
            point=origin + rotated(q, np.asarray(p)),
            velocity=[speed, 0.1 * speed, -0.01],
            history=[1e-5, -1e-5, 2e-6],
            parameters=[1000, 500, 0.2, 0.3, 0.2],
            position=origin,
            rotation=q,
            tool_velocity=[0.0001, -0.0002, 0.001],
            omega=[0.3, -0.2, 0.7],
            half=[0.001, 0.005, 0.01],
            bevel=bevel,
            offset=5e-4,
        )
        for p in points
    ]
    blade = RigidContacts.from_rows([[0.8, 0.2], [0.1, 0.9]], rows, "cuda:0")
    fields = ["parameters", "position", "rotation", "tool_velocity", "omega", "half", "bevel", "offset"]
    blade = replace(
        blade,
        **{name: array(getattr(blade, name).numpy()[:1], getattr(blade, name).dtype) for name in fields}
    )
    return inputs, geometry, blade, array


@pytest.mark.parametrize(
    "interface,bevel,speed",
    [(False, 0.0, 0.001), (False, 0.004, 0.4), (True, 0.0, 0.4), (True, 0.004, 0.001)],
)
def test_joint_rigid_contact_all_inputs_against_natural_differences(interface, bevel, speed):
    inputs, geometry, blade, array = rigid_fixture(interface, bevel, speed)

    def run():
        root, u, force, contact, history = solve(inputs, geometry, None, blade)
        return [u, force, contact.high, history.high]

    with wp.Tape() as tape:
        outputs = run()
    rng = np.random.default_rng(2754)
    seeds = [rng.normal(size=a.numpy().shape) * s for a, s in zip(outputs, [1e4, 1.0, 1.0, 1e4])]
    tape.backward(grads={a: array(s, a.dtype, False) for a, s in zip(outputs, seeds) if a.size})
    sources = [
        (blade.mapping.high, 0.1),
        (blade.point.high, 1e-4),
        (blade.velocity.high, 0.1),
        (blade.history.high, 1e-5),
        (blade.parameters, [100, 50, 0.02, 0.01, 0.01]),
        (blade.position, 1e-4),
        (blade.rotation, 0.03),
        (blade.tool_velocity, 0.01),
        (blade.omega, 0.1),
        (blade.half, 2e-4),
        (blade.offset, 1e-4),
        (inputs[8], 0.001),
        (inputs[9], 2e-5),
    ]
    if bevel:
        sources.append((blade.bevel, 2e-4))
    if interface:
        sources += [(inputs[0], 1e-5), (inputs[3], 1e-5), (inputs[4], [100, 0.1, 0.0])]
    for source, scale in sources:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        if source is blade.rotation:
            direction -= np.sum(direction * baseline, axis=1, keepdims=True) * baseline
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                perturbed = baseline + sign * epsilon * direction
                if source is blade.rotation:
                    perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
                source.assign(perturbed)
                values.append([a.numpy() for a in run()])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*values, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)
        source.assign(baseline)
