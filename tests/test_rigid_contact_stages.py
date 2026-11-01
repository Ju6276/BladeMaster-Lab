"""Moving finite blade in the two-stage material and fracture update."""

import numpy as np
import pytest
import warp as wp

from blademaster_warp.cohesive_rigid_contact import predict_rigid_contacts
from blademaster_warp.cohesive_stages import advance_affine
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.differentiable_geometry import advance_pose
from test_cohesive_stages import setup, targets


def prepared_rigid(plastic=False, dt=2e-6):
    m, state, geometry, history, data = setup("opened", dt)
    state.v.assign(np.tile([0.4, 0.001, -0.1], (m.n, 1)))
    state.affine.zero_()
    state.stress = None
    if plastic:
        m.yield_stress.assign([20.0])
        A = np.diag([2000.0, -400.0, -200.0])
        state.affine.assign(np.tile(A, (m.n, 1, 1)))
        state.v.assign(state.v.numpy() + m.reference @ A.T)
    edge = (m.reference[:, 2] == np.min(m.reference[:, 2])) & (m.reference[:, 0] == np.min(m.reference[:, 0]))
    indices = np.flatnonzero(edge)[[0, -1]]
    W = m.array(np.eye(m.n)[indices], float, False)
    C = m.array(data["transfer"].flow_map.numpy()[indices], float, False)
    outward = np.array([0.004, 0.0, -0.0006]) / np.hypot(0.004, 0.0006)
    local = np.array([0.0003, 0.0, -0.008]) + 1e-4 * outward
    origin = m.reference[indices[0]] - local
    origin[1] = 0.0
    fields = dict(
        parameters=m.array([[1000, 500, 0.2, 0.3, 0.2]], float),
        position=m.array([origin], wp.vec3),
        rotation=m.array([[0, 0, 0, 1]], wp.quat),
        tool_velocity=m.array([[-0.03, 0.01, 0.02]], wp.vec3),
        omega=m.array([[3.0, -2.0, 1.0]], wp.vec3),
        half=m.array([[0.0006, 0.02, 0.01]], wp.vec3),
        bevel=m.array([0.004], float),
        offset=m.array([4e-4], float),
    )
    rate = m.array([2000.0], float)
    contact = CompensatedArray(m.array(np.tile([1e-7, -2e-7, 0.0], (len(indices), 1)), wp.vec3))

    def provider(x, z, tau, elapsed, h):
        position, rotation = m.zeros(1, wp.vec3), m.zeros(1, wp.quat)
        m.launch(
            advance_pose,
            1,
            inputs=[
                fields["position"],
                fields["tool_velocity"],
                fields["rotation"],
                fields["omega"],
                elapsed,
            ],
            outputs=[position, rotation],
        )
        return predict_rigid_contacts(
            x, z, tau, h, W, C, **dict(fields, position=position, rotation=rotation)
        )

    def advance(s=state, g=geometry, h=history, c=contact):
        return advance_affine(
            state=s,
            geometry=g,
            history=h,
            **{key: value for key, value in data.items() if key != "force"},
            force=lambda s, v, t: data["force"],
            damping_rate=rate,
            contacts=provider,
            contact_history=c,
        )

    return m, state, geometry, history, contact, data, fields, advance


@pytest.mark.parametrize("plastic", [False, True])
def test_moving_finite_blade_stage_time_and_control_derivatives(plastic):
    m, state, geometry, history, contact, data, fields, advance = prepared_rigid(plastic)

    def outputs(result):
        return [*targets(result), result.contact_history.high, result.contact_stages[-1][1].high]

    with wp.Tape() as tape:
        result = advance()
    if plastic:
        assert np.count_nonzero(result.state.yielded.numpy()) > 0
    rng = np.random.default_rng(2755)
    values = outputs(result)
    seeds = [
        rng.normal(size=a.numpy().shape) / scale
        for a, scale in zip(values, [0.01, 1, 100, 1, 1, 2e4, 0.001, 1, 0.001, 0.001, 1])
    ]
    tape.backward(grads={a: m.array(seed, a.dtype, False) for a, seed in zip(values, seeds)})
    motions = [
        (fields[name], scale)
        for name, scale in [("position", 1e-4), ("rotation", 0.01), ("tool_velocity", 0.03), ("omega", 0.3)]
    ]
    for sources in [[(data["duration"], 2e-7)], motions]:
        original = [a.numpy().copy() for a, _ in sources]
        directions = [rng.normal(size=a.shape) * scale for a, (_, scale) in zip(original, sources)]
        for i, (source, _) in enumerate(sources):
            if source is fields["rotation"]:
                directions[i] -= np.sum(directions[i] * original[i], axis=1, keepdims=True) * original[i]
        ad = sum(np.sum(source.grad.numpy() * d) for (source, _), d in zip(sources, directions))
        for epsilon in [1e-4, 1e-5]:
            trials = []
            for sign in [-1, 1]:
                for (source, _), baseline, direction in zip(sources, original, directions):
                    value = baseline + sign * epsilon * direction
                    if source is fields["rotation"]:
                        value /= np.linalg.norm(value, axis=1, keepdims=True)
                    source.assign(value)
                trials.append([a.numpy() for a in outputs(advance())])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*trials, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
        for (source, _), baseline in zip(sources, original):
            source.assign(baseline)
