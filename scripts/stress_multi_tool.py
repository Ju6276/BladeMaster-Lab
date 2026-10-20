"""Seeded shared-material knife checks; CPU contacts or full CUDA trajectories.

python -m scripts.stress_multi_tool --device cpu --seed 17
python -m scripts.stress_multi_tool --device cuda:0 --seed 17 --steps 120
"""

import argparse
from dataclasses import asdict, replace
from itertools import permutations
import json
from pathlib import Path

import numpy as np
import warp as wp

from blademaster_warp.provenance import git_revision
from blademaster_warp.solver import Config, Solver
from blademaster_warp.teleop_input import quaternion_product, rotation_increment


def cut_code_capacity(config=None, solver=None):
    """Budget complete ternary digits for code * node_count + node int64 keys."""
    from blademaster_warp.geometry import grid_cells

    config = solver.config if solver is not None else (Config() if config is None else config)
    if solver is None:
        node_count = int(np.prod(grid_cells(config.size, config.grid_spacing) + 1))
        digits, key_range = 0, node_count
        while key_range * 3 <= np.iinfo(np.int64).max:
            digits += 1
            key_range *= 3
    else:
        node_count = solver.node_count
        digits = solver.max_cut_index + 1
        key_range = node_count * 3**digits
    assert key_range - 1 <= np.iinfo(np.int64).max < key_range * 3 - 1
    return dict(
        node_count=node_count,
        digits=digits,
        max_cut_index=digits - 1,
        complete_strokes=digits // config.tool_count,
        max_key=key_range - 1,
    )


class ContactProbe(Solver):
    """Allocate only contact state, using Solver's actual multi-knife dispatch.

    The production material solver requires CUDA. This probe also runs those
    contact kernels on CPU, with fixed particle positions between contacts.
    """

    def __init__(self, config, positions, device="cpu"):
        self.config, self.device = config, wp.get_device(device)
        self.n = len(positions)
        self.mass = config.density * (config.grid_spacing / 2) ** 3
        self.radius = config.grid_spacing / 4
        self.blade_offset = max(
            config.blade_contact_radius or self.radius, self.radius - config.blade_half_size[0]
        )
        self.half = wp.vec3(*config.blade_half_size)
        self.board_half = wp.vec2(*config.board_half_extent)
        self.board_scale = config.board_normal_scale or config.normal_scale
        self.normal_scale, self.tangent_stiffness = config.contact_scales()
        hx, hy, hz = config.blade_half_size
        self.inertia_inv = wp.vec3(
            *(3 / (config.blade_mass * np.array([hy * hy + hz * hz, hx * hx + hz * hz, hx * hx + hy * hy])))
        )
        self.threshold = (
            config.cutting_energy * (config.grid_spacing / 2) ** 2 * (1 - config.sharpness) / config.sharpness
        )

        def zeros(shape, dtype=wp.vec3):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        k = config.tool_count
        self.x = wp.array(positions, dtype=wp.vec3, device=self.device)
        self.v = zeros(self.n)
        self.tool_pos, self.tool_vel, self.tool_omega = (zeros(k) for _ in range(3))
        self.tool_rot = wp.array([[0, 0, 0, 1]] * k, dtype=wp.quat, device=self.device)
        self.tool_mobility = zeros((6, 6), float)
        self.reaction, self.torque = zeros(k), zeros(k)
        self.contact_parts = zeros(4)
        self.tool_views = [
            dict(
                pos=self.tool_pos[i : i + 1],
                vel=self.tool_vel[i : i + 1],
                rot=self.tool_rot[i : i + 1],
                omega=self.tool_omega[i : i + 1],
                reaction=self.reaction[i : i + 1],
                torque=self.torque[i : i + 1],
            )
            for i in range(k)
        ]
        self.tool_damage = [zeros(self.n, float) for _ in range(k)]
        self.damage = self.tool_damage[0]
        self.histories = zeros((2, self.n))
        self.extra_histories = zeros((k, self.n)) if k > 1 else None
        self.impulses, self.board = zeros(self.n), zeros(1)
        self.speed, self.work = zeros(1, float), zeros(1, float)
        self.codes = zeros(self.n, wp.int64)
        self.cut_released = zeros(k, int)
        self.topology_dirty = zeros(1, int)
        self.cut_index = 0
        self.max_cut_index = cut_code_capacity(config)["max_cut_index"]

    def contact(self, poses, velocities, rotations, omegas, cutting=True, cut_index=0):
        self._set_cut(cut_index)
        for array, values in (
            (self.tool_pos, poses),
            (self.tool_vel, velocities),
            (self.tool_rot, rotations),
            (self.tool_omega, omegas),
        ):
            array.assign(values)
        for array in (self.reaction, self.torque, self.board, self.work, self.speed):
            array.zero_()
        self.rigid_contacts(cutting)
        return contact_state(self)


def contact_state(solver):
    histories = [solver.histories.numpy()[1]]
    if solver.config.tool_count > 1:
        histories.extend(solver.extra_histories.numpy()[1:])
    return dict(
        damage=np.stack([d.numpy() for d in solver.tool_damage]),
        friction=np.stack(histories),
        reaction=solver.reaction.numpy().copy(),
        torque=solver.torque.numpy().copy(),
        impulses=solver.impulses.numpy().copy(),
        board=solver.board.numpy().copy(),
        work=solver.work.numpy().copy(),
    )


def canonical_codes(codes, order, strokes):
    """order[slot] is the physical knife id; each stroke owns len(order) digits."""
    codes = np.asarray(codes, dtype=np.int64)
    result = np.zeros_like(codes)
    count = len(order)
    for stroke in range(strokes):
        for slot, identity in enumerate(order):
            digit = (codes // 3 ** (stroke * count + slot)) % 3
            result += digit * 3 ** (stroke * count + identity)
    if np.any(codes // 3 ** (strokes * count)):
        raise AssertionError("Unexpected cut digit outside the replay history")
    return result


def capacity_stress(seed=17, device="cpu", tool_count=3, config=None):
    """Exercise production side assignment and hash registration through the final digit."""
    from blademaster_warp import material
    from blademaster_warp.topology import components

    cfg = Config(free_blade=True, tool_count=tool_count) if config is None else config
    tool_count = cfg.tool_count
    production = Solver(cfg, device=device) if wp.get_device(device).is_cuda else None
    capacity = cut_code_capacity(cfg, solver=production)
    last = capacity["digits"] - tool_count
    if last < 0:
        raise ValueError("Not enough cut digits for the tool count")
    rng = np.random.default_rng(seed)
    x = np.array([[-0.0007, 0, 0.041], [-0.0006, 0, 0.041], [0.0006, 0, 0.041], [0.0007, 0, 0.041]])
    x[:, 1] = rng.uniform(-0.0003, 0.0003, len(x))
    poses = np.tile([0, 0, 0.06], (tool_count, 1))
    rotations = np.array([rotation_increment(v) for v in rng.uniform(-0.003, 0.003, (tool_count, 3))])
    velocities = np.tile([0, 0, -0.2], (tool_count, 1))
    indices = list(range(0, last + 1, tool_count))
    # A partially filled final group still must allow the highest legal window;
    # its overlapping digits must retain their original labels.
    if indices[-1] != last:
        indices.append(last)
    finals = []
    for _ in range(2):
        probe = ContactProbe(cfg, x, device)
        assert probe.max_cut_index == capacity["max_cut_index"]
        assert not probe.codes.numpy().any()
        assert all(not damage.numpy().any() for damage in probe.tool_damage)
        old = np.zeros(len(x), dtype=np.int64)
        for index in indices:
            for damage in probe.tool_damage:
                damage.fill_(0.75)
            previous_index = probe.cut_index
            probe._set_cut(index)
            if index != previous_index:
                assert all(not damage.numpy().any() for damage in probe.tool_damage)
                assert not probe.cut_released.numpy().any()
            else:
                assert all(np.all(damage.numpy() == 0.75) for damage in probe.tool_damage)
            # A free knife opens its band only after the cut is paid; pay it here.
            for damage in probe.tool_damage:
                damage.fill_(1.0)
            probe.contact(poses, velocities, rotations, np.zeros((tool_count, 3)), cut_index=index)
            probe.assign_sides(index, cutting=False)
            np.testing.assert_array_equal(probe.codes.numpy(), old)
            probe.assign_sides(index)
            codes = probe.codes.numpy()
            for digit in range(capacity["digits"]):
                before, after = (old // 3**digit) % 3, (codes // 3**digit) % 3
                np.testing.assert_array_equal(after[before != 0], before[before != 0])
            expected = np.array([1, 1, 2, 2], dtype=np.int64) * ((3 ** (index + tool_count) - 1) // 2)
            np.testing.assert_array_equal(codes, expected)
            # Relabeling from the opposite side cannot rewrite an existing digit.
            probe.tool_rot.assign(np.tile([0, 0, 1, 0], (tool_count, 1)))
            probe.assign_sides(index)
            np.testing.assert_array_equal(probe.codes.numpy(), codes)
            old = codes
        for slot, damage in enumerate(probe.tool_damage):
            damage.fill_(0.125 * (slot + 1))
        for invalid in (-1, last + 1, last - 1):
            before = contact_state(probe)
            try:
                probe._set_cut(invalid)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid cut index accepted: {invalid}")
            assert probe.cut_index == last
            np.testing.assert_array_equal(probe.codes.numpy(), old)
            after = contact_state(probe)
            for key, value in before.items():
                np.testing.assert_array_equal(after[key], value)
        # Include the highest grid node, where an overflow would first appear.
        nodes = np.tile([0, 1, 2, 3, 4, 5, 6, capacity["node_count"] - 1], (len(x), 1))
        keys = wp.full(128, -1, dtype=wp.int64, device=device)
        slots = wp.zeros(nodes.shape, dtype=int, device=device)
        errors = wp.zeros(1, dtype=int, device=device)
        wp.launch(
            material.register,
            nodes.shape,
            [
                wp.array(nodes, dtype=int, device=device),
                probe.codes,
                keys,
                slots,
                capacity["node_count"],
                128,
                errors,
            ],
            device=device,
        )
        assert not errors.numpy().any()
        expected_keys = np.array(
            [
                [int(code) * capacity["node_count"] + int(node) for node in row]
                for code, row in zip(old, nodes)
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(keys.numpy()[slots.numpy()], expected_keys)
        assert int(expected_keys.max()) == capacity["max_key"]
        _, sizes = components(nodes, np.ones(nodes.shape), old)
        np.testing.assert_array_equal(sizes, [2, 2])
        finals.append(old)
    np.testing.assert_array_equal(*finals)
    return dict(
        capacity=capacity,
        capacity_source="solver" if production is not None else "reference_grid",
        sessions=2,
        cut_indices=indices,
        labeled_digits=capacity["digits"],
        highest_code=int(finals[-1].max()),
        highest_registered_key=int(expected_keys.max()),
        fixture_component_sizes=sizes.tolist(),
        rejected_indices=[-1, last + 1, last - 1],
    )


def trajectory_tolerances(config):
    """Absolute bounds in SI units, except strain, quaternion and damage fractions."""
    spacing, dt = config.grid_spacing, config.dt
    radius = float(np.linalg.norm(config.blade_half_size))
    # Position/friction: 10 ppm of a grid cell. Speed: 5 ppm of a cell per step.
    velocity = 5e-6 * spacing / dt
    # 10 micro-newtons of force, converted to the stored linear/angular impulses.
    impulse = 1e-5 * dt
    return dict(
        position=1e-5 * spacing,
        velocity=velocity,
        deformation=5e-6,
        plastic_deformation=1e-7,
        blade_position=1e-5 * spacing,
        blade_velocity=velocity,
        # Quaternion error dq moves the blade tip by about 2 dq r: same bound as blade_position.
        blade_orientation=1e-5 * spacing / radius,
        blade_omega=velocity / radius,
        tool_damage=1e-4,  # 0.01% of the work required for failure, including near-zero damage.
        time=0.0,
        friction=1e-5 * spacing,
        reaction=impulse,
        torque=impulse * radius,
    )


def contact_stress(seed=17, device="cpu", tool_count=3, motion="fixed"):
    if tool_count not in (2, 3):
        raise ValueError("Use two or three simultaneous knives")
    if motion not in ("fixed", "crossing"):
        raise ValueError("motion must be fixed or crossing")
    rng = np.random.default_rng(seed)
    cfg = Config(free_blade=True, tool_count=tool_count, cutting_energy=80.0)
    # All knives contact these same particles, with different sliding velocities.
    x = np.array(
        [[0.0007, -0.004, 0.039], [-0.0007, 0.003, 0.039], [0.0004, 0.006, 0.0388], [0.0002, -0.006, 0.0392]]
    )
    poses = np.tile([0.0, 0.0, 0.06], (tool_count, 1))
    poses[:, 0] = np.linspace(-0.0003, 0.0003, tool_count)
    velocities = rng.uniform(-0.08, 0.08, (tool_count, 3))
    velocities[:, 2] = -np.linspace(0.25, 0.65, tool_count)
    omegas = rng.uniform(-0.3, 0.3, (tool_count, 3))
    rotations = np.array([rotation_increment(v) for v in rng.uniform(-0.004, 0.004, (tool_count, 3))])
    if motion == "crossing":
        velocities[:, 0] = -2 * poses[:, 0] / (9 * cfg.dt)
        omegas[:, 2] += np.linspace(8.0, -8.0, tool_count)
        rotations = np.array([rotation_increment([0, 0, a]) for a in np.linspace(-0.08, 0.08, tool_count)])
    baseline = None
    worst_balance = 0.0
    independent_errors = {key: np.zeros(tool_count) for key in ("damage", "friction", "reaction", "torque")}
    common = 0
    for permutation in permutations(range(tool_count)):
        order = np.asarray(permutation)
        inverse = np.argsort(order)
        combined = ContactProbe(cfg, x, device)
        independent = [ContactProbe(replace(cfg, tool_count=1), x, device) for _ in order]
        trace = []
        for tick in range(12):
            p, v, w = poses.copy(), velocities.copy(), omegas.copy()
            q = rotations.copy()
            if motion == "crossing":
                elapsed = (tick - min(2, max(0, tick - 5))) * cfg.dt
                p[:, 0] += velocities[:, 0] * elapsed
                q = np.array(
                    [
                        quaternion_product(rotation_increment(omega * elapsed), rotation)
                        for omega, rotation in zip(omegas, rotations)
                    ]
                )
            if tick in (3, 4):
                p[0, 0] += 0.2  # One knife leaves; its neighbors keep their histories.
            if tick in (6, 7):
                v[:] = w[:] = 0.0
            if tick >= 8:
                v[:, 1] *= -1
                if motion == "fixed":
                    v[:, 0] *= -1
            cutting = tick not in (6, 7)
            cut_index = tool_count if tick >= 9 else 0
            if tick == 9:
                assert any(d.numpy().any() for d in combined.tool_damage)
                combined._set_cut(cut_index)
                assert all(not d.numpy().any() for d in combined.tool_damage)
            state = combined.contact(p[order], v[order], q[order], w[order], cutting, cut_index)
            singles = [
                probe.contact(p[i : i + 1], v[i : i + 1], q[i : i + 1], w[i : i + 1], cutting, cut_index)
                for i, probe in enumerate(independent)
            ]
            for key in ("damage", "friction", "reaction", "torque"):
                state[key] = state[key][inverse]
                expected = np.concatenate([s[key] for s in singles])
                difference = abs(state[key] - expected).reshape(tool_count, -1).max(axis=1)
                independent_errors[key] = np.maximum(independent_errors[key], difference)
                np.testing.assert_allclose(
                    state[key],
                    expected,
                    rtol=2e-5,
                    atol=1e-9,
                    err_msg=f"independent {key}, tick {tick}",
                )
            gravity = np.array([0, 0, -9.81 * combined.mass * cfg.dt])
            expected = sum(s["impulses"] for s in singles) - (tool_count - 1) * gravity
            np.testing.assert_allclose(state["impulses"], expected, rtol=2e-5, atol=1e-9)
            np.testing.assert_allclose(state["work"], sum(s["work"] for s in singles), rtol=2e-5, atol=1e-9)
            residual = (
                state["impulses"].astype(float).sum(0)
                + state["reaction"].astype(float).sum(0)
                - len(x) * gravity
                - state["board"].sum(0)
            )
            np.testing.assert_allclose(residual, 0, atol=1e-9)
            worst_balance = max(worst_balance, float(np.max(abs(residual))))
            if tick == 2:
                common = int(np.count_nonzero(np.all(state["damage"] > 0, axis=0)))
                assert common == len(x), "The knives must damage the same particles"
                assert np.all(np.linalg.norm(state["friction"], axis=(1, 2)) > 0)
                assert np.all(np.linalg.norm(state["torque"], axis=1) > 0)
                assert not np.allclose(state["friction"][0], state["friction"][1], atol=1e-9)
            if tick == 4:
                np.testing.assert_array_equal(state["friction"][0], 0)
                np.testing.assert_array_equal(state["reaction"][0], 0)
                assert np.linalg.norm(state["friction"][1:]) > 0
            if tick in (6, 7):
                np.testing.assert_array_equal(state["damage"], trace[-1]["damage"])
            trace.append(state)
        if baseline is None:
            baseline = trace
        else:
            for tick, (actual, reference) in enumerate(zip(trace, baseline)):
                for key in reference:
                    np.testing.assert_allclose(
                        actual[key],
                        reference[key],
                        rtol=2e-5,
                        atol=1e-9,
                        err_msg=f"permutation {permutation}, tick {tick}, {key}",
                    )
    crossed_pairs = sum(
        (poses[a, 0] - poses[b, 0]) * (p[a, 0] - p[b, 0]) < 0
        for a in range(tool_count)
        for b in range(a + 1, tool_count)
    )
    rotation_change = float(np.max(2 * np.arccos(np.clip(abs(np.sum(q * rotations, axis=1)), 0, 1))))
    if motion == "crossing":
        assert crossed_pairs == tool_count * (tool_count - 1) // 2
        assert rotation_change > 0.03
    return dict(
        config=asdict(cfg),
        motion=motion,
        crossed_tool_pairs=int(crossed_pairs),
        max_rotation_change_rad=rotation_change,
        permutations=len(list(permutations(range(tool_count)))),
        contact_steps=12,
        shared_damaged_particles=common,
        max_momentum_residual_kg_m_s=worst_balance,
        independent_max_abs_error_by_tool={key: value.tolist() for key, value in independent_errors.items()},
    )


def trajectory_stress(seed=17, device="cuda:0", steps=120, tool_count=3):
    """Full shared-material trajectories compared after physical tool relabeling."""
    from blademaster_warp.topology import components

    if steps < 60 or tool_count not in (2, 3):
        raise ValueError("Use at least 60 steps and two or three tools")
    cfg = Config(
        free_blade=True, tool_count=tool_count, cache_topology=True, grid_storage="compact", dt=0.0005
    )
    rng = np.random.default_rng(seed)
    poses = np.tile([0.0, 0.0, 0.052], (tool_count, 1))
    poses[:, 0] = np.linspace(-0.001, 0.001, tool_count)
    velocities = rng.uniform(-0.015, 0.015, (tool_count, 3))
    velocities[:, 2] = -np.linspace(0.4, 0.6, tool_count)
    omegas = rng.uniform(-0.1, 0.1, (tool_count, 3))
    rotations = np.array([rotation_increment(v) for v in rng.uniform(-0.005, 0.005, (tool_count, 3))])
    baseline = None
    errors, repeat_errors = {}, {}
    tolerances = trajectory_tolerances(cfg)
    max_balance = 0.0
    shared = 0
    orders = list(permutations(range(tool_count)))
    for trial, permutation in enumerate([orders[0], *orders]):
        order = np.asarray(permutation)
        inverse = np.argsort(order)
        s = Solver(cfg, device=device)
        s.tool_pos.assign(poses[order])
        s.tool_vel.assign(velocities[order])
        s.tool_rot.assign(rotations[order])
        s.tool_omega.assign(omegas[order])
        trace = []
        previous = s.state()
        old_codes = np.zeros(s.n, dtype=np.int64)
        for tick in range(steps):
            stroke = int(tick >= steps // 2)
            s.step(cutting=True, cut_index=stroke * tool_count)
            state = s.state()
            contact = contact_state(s)
            total_mass = s.mass * s.n + cfg.blade_mass * tool_count
            delta = s.mass * (state["velocity"].astype(float) - previous["velocity"]).sum(
                0
            ) + cfg.blade_mass * (state["blade_velocity"].astype(float) - previous["blade_velocity"]).sum(0)
            external = (
                np.array([0, 0, -9.81 * total_mass * cfg.dt])
                + contact["board"].sum(0)
                + s.tool_table_impulse.numpy().sum(0)
            )
            residual = float(np.max(abs(delta - external)))
            max_balance = max(max_balance, residual)
            # External impulse balance within 50 ppm of one step's gravity impulse.
            balance_atol = 5e-5 * 9.81 * total_mass * cfg.dt
            np.testing.assert_allclose(delta, external, atol=balance_atol, rtol=0)
            previous = state.copy()
            state["sides"] = canonical_codes(state["sides"], order, 2)
            state["component_labels"], state["component_sizes"] = components(
                s.node_numpy, s.weight_numpy, state["sides"]
            )
            for digit in range(2 * tool_count):
                old = (old_codes // 3**digit) % 3
                new = (state["sides"] // 3**digit) % 3
                np.testing.assert_array_equal(new[old != 0], old[old != 0])
            old_codes = state["sides"].copy()
            for key in (
                "blade_position",
                "blade_velocity",
                "blade_orientation",
                "blade_omega",
                "tool_damage",
            ):
                state[key] = state[key][inverse]
            # damage is slot zero's alias; tool_damage contains the physical identities.
            del state["damage"]
            for key in ("friction", "reaction", "torque"):
                state[key] = contact[key][inverse]
            shared = max(shared, int(np.count_nonzero(np.sum(state["tool_damage"] > 0, axis=0) >= 2)))
            trace.append(state)
        assert shared > 0, "No particle was cut by multiple knives"
        for stroke in range(2):
            for identity in range(tool_count):
                assert np.any((old_codes // 3 ** (stroke * tool_count + identity)) % 3), "Missing cut history"
        if baseline is None:
            baseline = trace
        else:
            for tick, (actual, reference) in enumerate(zip(trace, baseline)):
                for key in reference:
                    if key in ("sides", "component_labels", "component_sizes"):
                        np.testing.assert_array_equal(
                            actual[key], reference[key], err_msg=f"tick {tick}, {key}"
                        )
                        continue
                    measured = repeat_errors if trial == 1 else errors
                    measured[key] = max(
                        measured.get(key, 0.0), float(np.max(abs(np.asarray(actual[key]) - reference[key])))
                    )
                    np.testing.assert_allclose(
                        actual[key],
                        reference[key],
                        rtol=0,
                        atol=tolerances[key],
                        err_msg=f"permutation {permutation}, tick {tick}, {key}",
                    )
    return dict(
        config=asdict(cfg),
        particles=s.n,
        steps=steps,
        strokes=2,
        permutations=len(orders),
        identical_order_repeats=1,
        shared_damaged_particles=shared,
        max_momentum_residual_kg_m_s=max_balance,
        permutation_max_abs_error=errors,
        identical_order_max_abs_error=repeat_errors,
        comparison_atol=tolerances,
        momentum_balance_atol_kg_m_s=balance_atol,
        final_component_sizes=trace[-1]["component_sizes"].tolist(),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=17)
    seeds.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--tools", type=int, choices=(2, 3), default=3)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    reports = []
    for seed in args.seeds if args.seeds is not None else [args.seed]:
        report = dict(
            revision=git_revision(),
            seed=seed,
            device=args.device,
            contacts=contact_stress(seed, args.device, args.tools),
            crossing_contacts=contact_stress(seed, args.device, args.tools, motion="crossing"),
            capacity=capacity_stress(seed, args.device, args.tools),
        )
        if wp.get_device(args.device).is_cuda:
            report["trajectories"] = trajectory_stress(seed, args.device, args.steps, args.tools)
        report["status"] = "passed"
        reports.append(report)
    report = reports[0] if args.seeds is None else dict(status="passed", runs=reports)
    result = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
