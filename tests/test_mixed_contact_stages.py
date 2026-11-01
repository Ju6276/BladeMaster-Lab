"""Board and moving finite blade share both material stages and friction history."""

import numpy as np
import warp as wp

from blademaster_warp.cohesive_contact import predict_contacts
from blademaster_warp.cohesive_contact_batch import CombinedContacts, join_states, split_state
from blademaster_warp.cohesive_rigid_contact import predict_rigid_contacts
from blademaster_warp.cohesive_stages import advance_affine
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.differentiable_geometry import advance_pose
from test_cohesive_stages import targets
from test_rigid_contact_stages import prepared_rigid


def prepared_mixed():
    m, state, geometry, history, knife_history, data, fields, _ = prepared_rigid()
    board = np.flatnonzero(m.reference[:, 2] == np.min(m.reference[:, 2]))
    edge = board[m.reference[board, 0] == np.min(m.reference[board, 0])]
    knife = edge[[0, -1]]
    W = [m.array(np.eye(m.n)[indices], float, False) for indices in [board, knife]]
    C = [m.array(data["transfer"].flow_map.numpy()[indices], float, False) for indices in [board, knife]]
    normal = m.array(np.tile([0, 0, 1], (len(board), 1)), wp.vec3, False)
    parameters = m.array(np.tile([1000, 500, 0.2, 0.3, 0.2], (len(board), 1)), float)
    plane = m.array([[0, 0, 0.0016]], wp.vec3)
    velocity = m.zeros(1, wp.vec3, False)
    board_history = CompensatedArray(m.array(np.tile([1e-7, -2e-7, 0], (len(board), 1)), wp.vec3))
    contact = join_states([board_history, knife_history])
    rate = m.array([2000.0], float)

    def provider(x, z, tau, elapsed, h):
        board_h, knife_h = split_state(h, [len(board), len(knife)])
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
        return CombinedContacts(
            [
                predict_contacts(x, z, tau, board_h, W[0], C[0], plane, velocity, normal, parameters),
                predict_rigid_contacts(
                    x, z, tau, knife_h, W[1], C[1], **dict(fields, position=position, rotation=rotation)
                ),
            ]
        )

    def advance():
        return advance_affine(
            state=state,
            geometry=geometry,
            history=history,
            **{key: value for key, value in data.items() if key != "force"},
            force=lambda s, v, t: data["force"],
            damping_rate=rate,
            contacts=provider,
            contact_history=contact,
        )

    return m, contact, data, fields, plane, advance


def test_board_blade_stage_duration_controls_and_history_gradient():
    m, contact, data, fields, plane, advance = prepared_mixed()

    def outputs(step):
        return [*targets(step), step.contact_history.high, step.contact_stages[-1][1].high]

    with wp.Tape() as tape:
        step = advance()
    values = outputs(step)
    rng = np.random.default_rng(2758)
    scales = [0.01, 1, 100, 1, 1, 2e4, 0.001, 1, 0.001, 0.001, 1]
    seeds = [rng.normal(size=a.numpy().shape) / s for a, s in zip(values, scales)]
    tape.backward(grads={a: m.array(seed, a.dtype, False) for a, seed in zip(values, seeds)})
    for source, scale in [
        (data["duration"], 2e-7),
        (contact.high, 1e-7),
        (plane, 1e-4),
        (fields["omega"], 0.3),
    ]:
        baseline = source.numpy().copy()
        direction = rng.normal(size=baseline.shape) * scale
        ad = np.sum(source.grad.numpy() * direction)
        for epsilon in [1e-4, 1e-5]:
            values = []
            for sign in [-1, 1]:
                source.assign(baseline + sign * epsilon * direction)
                values.append([a.numpy() for a in outputs(advance())])
            fd = sum(np.sum((b - a) * seed) for a, b, seed in zip(*values, seeds)) / (2 * epsilon)
            np.testing.assert_allclose(ad, fd, rtol=2e-5, atol=2e-8)
        source.assign(baseline)
