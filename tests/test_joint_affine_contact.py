"""Physical geometry perturbations and uncut affine contact stages."""

import numpy as np
import warp as wp

from blademaster_warp.cohesive_contact import predict_contacts, solve
from blademaster_warp.cohesive_history import CohesiveHistory
from blademaster_warp.cohesive_geometry import InterfaceGeometry
from blademaster_warp.cohesive_stages import AffineState, advance_affine, GAMMA
from blademaster_warp.reference_affine import build_transfer
from test_mapped_contact import fixture
from test_cohesive_stages import setup


def test_empty_contact_provider_preserves_material_step_and_duration_gradient():
    from blademaster_warp.cohesive_contact import MappedContacts
    from blademaster_warp.compensated_state import CompensatedArray

    m, state, geometry, history, data = setup("opened")
    parameters = {key: value for key, value in data.items() if key != "force"}
    kwargs = dict(state=state, geometry=geometry, history=history, **parameters, gravity=(0, 0, 0))
    empty = CompensatedArray(m.zeros(0, wp.vec3))

    def contacts(x, moments, tau, elapsed, h):
        return MappedContacts.from_rows(np.empty((0, 4 * m.n)), [], m.device)

    with wp.Tape() as tape:
        result = advance_affine(**kwargs, contacts=contacts, contact_history=empty)
    seed = m.array(np.ones((m.n, 3)), wp.vec3, False)
    tape.backward(grads={result.state.x: seed})
    actual = data["duration"].grad.numpy().copy()
    tape.zero()
    with wp.Tape() as original:
        expected = advance_affine(**kwargs)
    original.backward(grads={expected.state.x: seed})
    for name in ["x", "v", "F", "Fp", "affine", "stress"]:
        np.testing.assert_allclose(
            getattr(result.state, name).numpy(), getattr(expected.state, name).numpy(), rtol=2e-11, atol=1e-13
        )
    np.testing.assert_allclose(actual, data["duration"].grad.numpy(), rtol=2e-10, atol=1e-12)


def test_repeated_closed_trace_differentiates_particle_motion():
    inputs, _, contacts, array = fixture(True, 0.001)
    count = 3
    gap = array(np.zeros((count, 3)), wp.vec3d)
    frame = array(np.tile(np.eye(3), (count, 1, 1)), wp.mat33d)
    geometry = InterfaceGeometry(gap, frame)
    inputs[:8] = [
        gap,
        frame,
        array(np.tile([1.0, 0, 0], (count, 1)), wp.vec3d),
        array(np.zeros(count)),
        array(np.tile([1000.0, 1.0, 0], (count, 1)), wp.vec3d),
        array(np.tile([-1.0, 1.0], (count, 1))),
        array(np.zeros((count, 2, 3)), wp.vec3d),
        array(np.full(count, 4e-4 / count)),
    ]
    motion = array(np.zeros((2, 3)), wp.vec3d)
    one = array([1.0], grad=False)

    def run():
        free = geometry.advance(inputs[5], inputs[6], motion, one)
        local = [free.gap, free.frame, *inputs[2:]]
        root, u, force, contact, history = solve(local, free, None, contacts)
        assert root.branches.closed.all()
        return [u, force, contact.high, history.high]

    with wp.Tape() as tape:
        outputs = run()
    rng = np.random.default_rng(2743)
    seeds = [rng.normal(size=a.numpy().shape) * s for a, s in zip(outputs, [1e4, 1.0, 1.0, 1e4])]
    tape.backward(grads={a: array(s, a.dtype, False) for a, s in zip(outputs, seeds)})
    direction = rng.normal(size=motion.numpy().shape) * 1e-5
    ad = np.sum(motion.grad.numpy() * direction)
    for eps in [1e-4, 1e-5]:
        values = []
        for sign in [-1.0, 1.0]:
            motion.assign(sign * eps * direction)
            values.append([a.numpy() for a in run()])
        fd = sum(np.sum((b - a) * s) for a, b, s in zip(*values, seeds)) / (2 * eps)
        np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=1e-8)


def test_uncut_affine_stages_match_uniform_contact_solution():
    m, _, _, _, _ = setup()
    transfer = build_transfer(m, m.uncut)
    retained = CohesiveHistory.empty(m.n, device=m.device)
    initial_v = np.array([0.001, 0.0002, 0.0])
    state = AffineState.from_material(m.state(v=np.tile(initial_v, (m.n, 1))), m.zeros(m.n, wp.mat33))
    h0 = np.array([1e-7, -2e-7, 0.0])
    history = m.array(np.tile(h0, (m.n, 1)), wp.vec3)
    origin = m.array(m.reference + [0, 0, 2e-4], wp.vec3)
    normal = m.array(np.tile([0, 0, 1.0], (m.n, 1)), wp.vec3, False)
    parameters = m.array(np.tile([1000, 500, 0.2, 0.3, 0.2], (m.n, 1)), float)
    W = m.array(np.eye(m.n), float, False)
    C = transfer.flow_map
    velocity = m.array([[0, 0, 0]], wp.vec3, False)
    duration = m.array([2e-6], float)

    def contacts(x, moments, tau, elapsed, h):
        return predict_contacts(x, moments, tau, h, W, C, origin, velocity, normal, parameters)

    step = advance_affine(
        transfer,
        state,
        retained.geometry,
        retained.history,
        retained.quadrature.normals,
        retained.material,
        m.zeros((0, 4 * m.n), float),
        m.zeros((0, 4 * m.n), wp.vec3),
        retained.quadrature.area,
        duration=duration,
        gravity=(0, 0, 0),
        damping_rate=m.array([0.0], float, False),
        contacts=contacts,
        contact_history=history,
    )
    dt = duration.numpy()[0]
    tau = GAMMA * dt
    gap0 = -2e-4

    def stage(v, h, gap):
        vn = (v[2] - tau * 1000 * gap / m.mass) / (1 + tau * tau * 1000 / m.mass)
        tangent = (v[:2] - tau * 500 * h[:2] / m.mass) / (1 + tau * (0.2 + tau * 500) / m.mass)
        result = np.r_[tangent, vn]
        return result, h + tau * np.r_[tangent, 0.0], gap + tau * vn

    v1, h1, g1 = stage(initial_v, h0, gap0)
    a1 = (v1 - initial_v) / tau
    r1 = (h1 - h0) / tau
    vb = initial_v + (1 - GAMMA) * dt * a1
    hb = h0 + (1 - GAMMA) * dt * r1
    gb = gap0 + (1 - GAMMA) * dt * v1[2]
    v2, h2, g2 = stage(vb, hb, gb)
    np.testing.assert_allclose(step.state.v.numpy(), np.tile(v2, (m.n, 1)), rtol=2e-10, atol=2e-14)
    np.testing.assert_allclose(
        step.contact_history.high.numpy(), np.tile(h2, (m.n, 1)), rtol=2e-10, atol=1e-17
    )
    np.testing.assert_allclose(
        step.state.x.numpy()[:, 2] - m.reference[:, 2], g2 - gap0, rtol=2e-9, atol=1e-17
    )
