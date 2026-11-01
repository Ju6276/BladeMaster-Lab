"""Accepted-stage energy accounting for a horizontal board and finite rigid blade."""

import numpy as np

from blademaster_warp import cohesive_contact
from blademaster_warp._blade_contact_geometry import query
from blademaster_warp._cohesive_expansion import E, divide, norm
from blademaster_warp.cohesive_contact_batch import CombinedContacts
from blademaster_warp.cohesive_pair_contact import PairContacts
from blademaster_warp.cohesive_rigid_contact import RigidContacts
from blademaster_warp.cohesive_spatial_contact import pair_storage
from blademaster_warp.cohesive_stages import GAMMA
from blademaster_warp.compensated_state import CompensatedArray
from blademaster_warp.energy import elastic_density, energy_snapshot
from blademaster_warp.reference_affine import build_transfer


def leaves(batch):
    if isinstance(batch, CombinedContacts):
        for child in batch.batches:
            yield from leaves(child)
    else:
        yield batch


def cohesive_storage(retained):
    gap = retained.geometry.gap.numpy() + retained.geometry.gap_remainder
    frame = retained.geometry.frame.numpy() + retained.geometry.frame_remainder
    normal = retained.quadrature.normals.numpy()
    if len(gap):
        normal = np.einsum("nij,nj->ni", np.linalg.inv(frame).transpose(0, 2, 1), normal)
        normal /= np.linalg.norm(normal, axis=1)[:, None]
    gap = gap - np.minimum(np.sum(gap * normal, axis=1), 0.0)[:, None] * normal
    maximum = retained.history.numpy()
    strength, gc = retained.material.numpy()[:, :2].T
    failure = 2 * gc / strength
    active = (maximum > 0) & (maximum < failure)
    storage = np.zeros(len(gap))
    storage[active] = (
        0.5
        * strength[active]
        * (1 - maximum[active] / failure[active])
        / maximum[active]
        * np.sum(gap[active] ** 2, axis=1)
    )
    area = retained.quadrature.area.numpy()
    return dict(
        cohesive_j=float(area @ storage),
        fracture_j=float(area @ (0.5 * strength * np.minimum(maximum, failure))),
    )


def snapshot(driver, state, history, tool, env):
    m = driver.model
    transfer = build_transfer(m, m.topology(history.codes.numpy()))
    result = energy_snapshot(m, state, driver.gravity, affine_transfer=transfer)
    result.update(cohesive_storage(history.cohesive))
    p = env["parameters"].numpy()[0]
    x = state.x.numpy() + state.rounding_remainders.get("x", 0.0)
    rigid_history = history.contact.cpu().numpy()
    board_gap = x[:, 2] - env["plane"].numpy()[0, 2]
    gaps = []
    for point in x:
        geo = query(
            point,
            np.zeros(3),
            tool.position.numpy()[0],
            tool.rotation.numpy()[0],
            tool.velocity.numpy()[0],
            tool.omega.numpy()[0],
            env["half"].numpy()[0],
            env["bevel"].numpy()[0],
            env["offset"].numpy()[0],
        )
        gaps.append(float(geo.gap.numpy()))
    result["rigid_normal_j"] = float(
        0.5 * p[0] * (np.sum(np.minimum(board_gap, 0.0) ** 2) + np.sum(np.minimum(gaps, 0.0) ** 2))
    )
    result["rigid_tangent_j"] = float(0.5 * p[1] * np.sum(rigid_history**2))
    pair = pair_storage(
        CompensatedArray(state.x, state.rounding_remainders.get("x")),
        history.pairs,
        driver.pair_parameters,
        driver.diameter,
    ).numpy()
    result["pair_normal_j"] = float(np.sum(pair[:, 0]))
    result["pair_tangent_j"] = float(np.sum(pair[:, 1]))
    return result


def stage_power(batch, duration, groups):
    totals = dict(
        external_w=0.0,
        viscous_w=0.0,
        sliding_w=0.0,
        history_projection_w=0.0,
        friction_identity_abs_w=0.0,
        response_error_n=0.0,
    )
    for leaf, points, velocities, forces, histories in groups:
        for i, row in enumerate(leaf.rows()):
            point, velocity = points[i], velocities[i]
            value = leaf.evaluate(row, point, velocity, duration)
            totals["response_error_n"] = max(
                totals["response_error_n"], float(np.max(np.abs(value.force.numpy() - forces[i].numpy())))
            )
            if isinstance(leaf, RigidContacts):
                geo = query(
                    point,
                    velocity,
                    *[
                        row[k]
                        for k in ("position", "rotation", "tool_velocity", "omega", "half", "bevel", "offset")
                    ]
                )
                normal, relative = geo.normal, geo.relative
                external = velocity - relative
            elif isinstance(leaf, PairContacts):
                normal = divide(point, norm(point))
                relative, external = velocity, E(np.zeros(3))
            else:
                normal = row["normal"]
                relative, external = velocity, E(np.zeros(3))
            force, updated = forces[i], histories[i]
            totals["external_w"] += float((force * external).sum().numpy())
            kt, damping = row["parameters"][1], row["parameters"][2]
            previous = row["history"]
            projected = previous - (previous * normal).sum() * normal
            if value.mode == "open":
                released = divide(0.5 * kt * (previous * previous).sum(), duration)
                totals["history_projection_w"] += float(released.numpy())
                continue
            tangent = force - (force * normal).sum() * normal
            speed = relative - (relative * normal).sum() * normal
            rate = divide(updated - projected, duration)
            viscous = damping * (rate * rate).sum()
            sliding = -(tangent * (speed - rate)).sum()
            identity = -(tangent * speed).sum() - kt * (updated * rate).sum() - viscous - sliding
            totals["viscous_w"] += float(viscous.numpy())
            totals["sliding_w"] += float(sliding.numpy())
            projection = divide(
                0.5 * kt * ((previous * previous).sum() - (projected * projected).sum()), duration
            )
            totals["history_projection_w"] += float(projection.numpy())
            totals["friction_identity_abs_w"] = max(
                totals["friction_identity_abs_w"], abs(float(identity.numpy()))
            )
    return totals


def measure_step(driver, state, history, tool, result, *, parameters, plane, half, bevel, offset):
    env = dict(parameters=parameters, plane=plane, half=half, bevel=bevel, offset=offset)
    initial = snapshot(driver, state, history, tool, env)
    final = snapshot(driver, result.state, result.history, result.tool, env)
    totals = dict(
        external_j=0.0,
        viscous_j=0.0,
        sliding_j=0.0,
        history_projection_j=0.0,
        damping_j=0.0,
        plastic_j=0.0,
        friction_identity_abs_w=0.0,
        response_error_n=0.0,
    )
    segment_rows = []
    m = driver.model
    for segment in result.segments:
        material = segment.step.material
        dt = float(segment.duration.numpy()[0])
        powers = []
        for weight, stage in zip((1 - GAMMA, GAMMA), material.contact_stages, strict=True):
            groups = []
            offset = 0
            for batch in leaves(stage.batch):
                points, velocities = cohesive_contact.contact_kinematics(
                    batch, stage.correction.high, stage.correction.low, stage.duration
                )
                stop = offset + batch.count
                groups.append(
                    (
                        batch,
                        points.cpu(),
                        velocities.cpu(),
                        stage.force.cpu()[offset:stop],
                        stage.history.cpu()[offset:stop],
                    )
                )
                offset = stop
            if offset != stage.batch.count:
                raise ValueError("Contact leaf counts do not match the accepted stage")
            values = stage_power(stage.batch, float(stage.duration.numpy()[0]), groups)
            powers.append(values)
            for name in ("external", "viscous", "sliding", "history_projection"):
                totals[name + "_j"] += dt * weight * values[name + "_w"]
            for name in ("friction_identity_abs_w", "response_error_n"):
                totals[name] = max(totals[name], values[name])
        totals["damping_j"] += float(np.sum(material.damping_energy.numpy()))
        F = material.state.F.numpy()
        young, poisson = float(m.young.numpy()[0]), m.config.poisson_ratio
        totals["plastic_j"] += float(
            m.volume
            * np.sum(
                elastic_density(F, segment.before.Fp.numpy(), young, poisson)
                - elastic_density(F, material.state.Fp.numpy(), young, poisson)
            )
        )
        segment_rows.append(dict(duration_s=dt, powers=powers))
    totals["contact_retirement_j"] = float(sum(e.released_energy.numpy()[0] for e in result.contact_events))
    totals["energy_change_j"] = sum(final.values()) - sum(initial.values())
    totals["balance_residual_j"] = (
        totals["energy_change_j"]
        - totals["external_j"]
        + sum(totals[k] for k in ("viscous_j", "sliding_j", "damping_j", "plastic_j", "contact_retirement_j"))
    )
    row = dict(initial=initial, final=final, totals=totals, segments=segment_rows)
    return row
