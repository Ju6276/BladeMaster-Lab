"""Energy terms for accepted natural-cutting segments, in joules."""

import numpy as np
import warp as wp

from .differentiable_midpoint import blade_contact, board_gap
from .energy import elastic_density, energy_snapshot, friction_increment_energy


@wp.kernel
def rigid_geometry(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                   damage: wp.array(dtype=float), departed: wp.array(dtype=int),
                   position: wp.array(dtype=wp.vec3), velocity: wp.array(dtype=wp.vec3),
                   rotation: wp.array(dtype=wp.quat), omega: wp.array(dtype=wp.vec3),
                   half: wp.vec3, bevel: float, radius: float, offset: float, board_half: wp.vec2,
                   hold: int, gaps: wp.array2d(dtype=float), normals: wp.array2d(dtype=wp.vec3),
                   relative: wp.array2d(dtype=wp.vec3)):
    p = wp.tid()
    gaps[0, p] = board_gap(x[p], radius, board_half)
    normals[0, p] = wp.vec3(0., 0., 1.)
    relative[0, p] = v[p]
    gap, normal, speed = blade_contact(x[p], v[p], position[0], velocity[0], rotation[0], omega[0],
                                       half, bevel, offset, hold, damage[p], departed[p])
    gaps[1, p] = gap
    normals[1, p] = normal
    relative[1, p] = speed


class EnergyLedger:
    def __init__(self, contact):
        self.contact = contact
        self.model = contact.model
        self.initial = None
        self.final = None
        self.rows = []
        self.events = []
        self.elapsed = 0.
        self._trial = None
        contact.energy_ledger = self

    def geometry(self, state, history, tool, cutting):
        m, cfg = self.model, self.model.config
        radius = cfg.grid_spacing / 4
        gaps = m.zeros((2, m.n), float, False)
        normals = m.zeros((2, m.n), wp.vec3, False)
        relative = m.zeros((2, m.n), wp.vec3, False)
        departed = self.contact._no_edge_departure if history.edge_departed is None else history.edge_departed
        m.launch(rigid_geometry, m.n,
                 inputs=[state.x, state.v, history.damage, departed, tool.position, tool.velocity,
                         tool.rotation, tool.omega, cfg.blade_half_size, cfg.blade_bevel_height,
                         radius, max(cfg.blade_contact_radius or radius, radius - cfg.blade_half_size[0]),
                         cfg.board_half_extent, int(cutting and cfg.cut_hold)],
                 outputs=[gaps, normals, relative], device=m.device)
        return gaps.numpy(), normals.numpy(), relative.numpy()

    def constants(self):
        m, cfg = self.model, self.model.config
        sn, gk = cfg.contact_scales()
        kn = sn * m.mass / cfg.dt**2
        board_kn = (cfg.board_normal_scale or sn) * m.mass / cfg.dt**2
        kt = gk * m.mass / cfg.dt**2
        damping = 2 * cfg.tangent_damping * np.sqrt(m.mass * kt)
        return kn, board_kn, kt, damping

    def snapshot(self, state, history, tool, cutting):
        m = self.model
        kn, board_kn, kt, _ = self.constants()
        gaps, _, _ = self.geometry(state, history, tool, cutting)
        keys = self.contact.contact_keys(state, history.codes, 0.)
        x = state.x.numpy()
        delta = x[keys // m.n] - x[keys % m.n]
        pair_gap = np.linalg.norm(delta, axis=-1) - m.config.grid_spacing / 2
        result = energy_snapshot(m, state)
        result['contact_normal_j'] = float(
            .5 * board_kn * np.sum(np.minimum(gaps[0], 0.)**2)
            + .5 * kn * np.sum(np.minimum(gaps[1], 0.)**2)
            + .25 * kn * np.sum(np.minimum(pair_gap, 0.)**2))
        result['contact_tangent_j'] = float(.5 * kt * np.sum(history.rigid.numpy()**2)
                                             + .25 * kt * np.sum(history.pairs.numpy()**2))
        if not np.isfinite(list(result.values())).all():
            raise FloatingPointError('Nonfinite contact energy')
        return result

    def begin(self, state, history, tool, cutting):
        current = self.snapshot(state, history, tool, cutting)
        if self.initial is None:
            self.initial = current.copy()
        else:
            self._event(current, 'step_boundary')
        self.final = current

    def _event(self, current, kind):
        changes = {key: current[key] - self.final[key] for key in current}
        if any(value != 0. for value in changes.values()):
            self.events.append(dict(time_s=self.elapsed, kind=kind, changes=changes))
        self.final = current

    def event(self, state, history, tool, cutting):
        self._event(self.snapshot(state, history, tool, cutting), 'cut_side')

    def capture(self, stage, base_history, tool, branch, duration, impulse, result, raw_history,
                constraint, trace, *, midpoint):
        self._trial = dict(stage=stage, base_history=base_history, stage_tool=tool, branch=branch,
                           dt=float(duration.numpy()[0]), impulse=impulse, result=result,
                           raw_history=raw_history, constraint=constraint, trace=trace,
                           midpoint=midpoint)

    def finish_trial(self, start, history, tool, result, updated, cutting):
        if self._trial is None or self._trial['result'] is not result:
            raise RuntimeError('Missing material trace for contact segment')
        self._trial.update(start=start, start_history=history, start_tool=tool,
                           updated=updated, cutting=cutting)

    def _friction_terms(self, gap, normal, velocity, old, updated, kn, kt, damping, dt, theta, midpoint):
        vn = np.sum(normal * velocity, axis=-1)
        predicted = gap if midpoint else gap + dt * np.minimum(vn, 0.)
        active = predicted < 0
        h0 = old - np.sum(old * normal, axis=-1)[..., None] * normal
        vt = velocity - vn[..., None] * normal
        dh = updated[active] - h0[active]
        force = -kt * (h0[active] + theta * dh) - damping * dh / dt
        terms = friction_increment_energy(h0[active], updated[active], dt * vt[active], force,
                                           kt, damping, dt, theta)
        residual = (terms['input_work_j'] - terms['storage_change_j'] - terms['viscous_j']
                    - terms['sliding_j'] - terms['integration_j'])
        values = {key: float(np.sum(value)) for key, value in terms.items()}
        values['identity_max_abs_j'] = float(np.max(np.abs(residual), initial=0.))
        values['normal_relative_work_j'] = float(dt * np.sum(kn * (-predicted[active]) * vn[active]))
        normal_impulse = np.zeros_like(velocity)
        tangent_impulse = np.zeros_like(velocity)
        normal_impulse[active] = dt * kn * (-predicted[active, None]) * normal[active]
        tangent_impulse[active] = dt * force
        return values, normal_impulse, tangent_impulse

    def accept(self, result, history, end_tool):
        r = self._trial
        if r is None or r['result'] is not result or r['updated'] is not history:
            raise RuntimeError('Accepted state does not match the recorded contact segment')
        m, cfg, dt = self.model, self.model.config, r['dt']
        if self.initial is None:
            self.begin(r['start'], r['start_history'], r['start_tool'], r['cutting'])
        start = self.snapshot(r['start'], r['start_history'], r['start_tool'], r['cutting'])
        end = self.snapshot(result, history, end_tool, r['cutting'])
        stage = r['stage']
        raw, base, branch = r['raw_history'], r['base_history'], r['branch']
        kn, board_kn, kt, damping = self.constants()
        gaps, normals, relative = self.geometry(stage, base, r['stage_tool'], r['cutting'])
        theta = .5 if self.contact.friction_update == 'midpoint' else 1.
        groups = [self._friction_terms(gaps[i], normals[i], relative[i], base.rigid.numpy()[i],
                                      raw.rigid.numpy()[i], stiffness, kt, damping, dt, theta, r['midpoint'])
                 for i, stiffness in enumerate((board_kn, kn))]
        normal_impulse = groups[0][1] + groups[1][1]
        tangent_impulse = groups[0][2] + groups[1][2]
        pairs = branch.pairs.numpy()
        if len(pairs):
            x, v = stage.x.numpy(), stage.v.numpy()
            delta = x[pairs[:, 0]] - x[pairs[:, 1]]
            length = np.linalg.norm(delta, axis=-1)
            normal = delta / length[:, None]
            previous = branch.previous.numpy()
            old = np.zeros((len(pairs), 3))
            valid = previous >= 0
            old[valid] = base.pairs.numpy()[previous[valid]]
            groups.append(self._friction_terms(length - cfg.grid_spacing / 2, normal,
                         v[pairs[:, 0]] - v[pairs[:, 1]], old, raw.pairs.numpy(),
                         kn / 2, kt / 2, damping / 2, dt, theta, r['midpoint']))
            for particle_impulses, contact_impulses in ((normal_impulse, groups[-1][1]),
                                                         (tangent_impulse, groups[-1][2])):
                np.add.at(particle_impulses, pairs[:, 0], contact_impulses)
                np.add.at(particle_impulses, pairs[:, 1], -contact_impulses)
        terms = [group[0] for group in groups]
        contact = {name: sum(item[name] for item in terms) for name in terms[0]}
        trace = r['trace']
        slots = trace['grid'].slots.numpy()
        weights = m.weights.numpy()[..., None]
        flow = np.sum(weights * trace['old_v'].numpy()[slots], axis=1)
        constraint_dv = np.sum(weights * (trace['new_v'].numpy() - trace['unconstrained_v'].numpy())[slots], axis=1)
        stage_v = stage.v.numpy()
        gravity = np.array([0., 0., -9.81])
        particle_impulse = r['impulse'].numpy() - dt * m.mass * gravity
        plastic = m.volume * np.sum(elastic_density(result.F.numpy(), r['start'].Fp.numpy(),
                                    float(m.young.numpy()[0]), cfg.poisson_ratio)
                                    - elastic_density(result.F.numpy(), result.Fp.numpy(),
                                    float(m.young.numpy()[0]), cfg.poisson_ratio))
        grid_work = 0.
        if r['constraint'] is not None:
            constraint = r['constraint']
            grid_work = float(np.sum(constraint.linear_impulse.numpy() * r['stage_tool'].velocity.numpy())
                              + np.sum(constraint.angular_impulse.numpy() * r['stage_tool'].omega.numpy()))
        knife_work = float(raw.tool_work.numpy()[0])
        actual_relative_work = float(np.sum(particle_impulse * stage_v)) - (knife_work - grid_work)
        raw_relative_work = contact['normal_relative_work_j'] - contact['input_work_j']
        normal_transfer = float(np.sum(normal_impulse * (flow - stage_v)))
        friction_transfer = float(np.sum(tangent_impulse * (flow - stage_v)))
        release_transfer = float(np.sum((particle_impulse - normal_impulse - tangent_impulse) * (flow - stage_v)))
        row = dict(
            duration_s=dt,
            kinetic_change_j=end['kinetic_j'] - start['kinetic_j'],
            elastic_change_j=end['elastic_j'] - start['elastic_j'],
            normal_change_j=end['contact_normal_j'] - start['contact_normal_j'],
            tangent_change_j=end['contact_tangent_j'] - start['contact_tangent_j'],
            knife_work_j=knife_work,
            gravity_work_j=float(m.mass * np.sum((result.x.numpy() - r['start'].x.numpy()) * gravity)),
            plastic_j=float(plastic),
            friction_viscous_j=contact['viscous_j'],
            friction_sliding_j=contact['sliding_j'],
            friction_integration_j=contact['integration_j'],
            history_transport_j=contact['storage_change_j'] - (end['contact_tangent_j'] - start['contact_tangent_j']),
            pic_j=float(dt * (1 - cfg.flip_fraction) / (cfg.flip_reference_dt or cfg.dt)
                        * m.mass * np.sum(stage_v * (stage_v - flow))),
            normal_transfer_work_j=normal_transfer,
            friction_transfer_work_j=friction_transfer,
            release_transfer_work_j=release_transfer,
            release_correction_work_j=actual_relative_work - raw_relative_work,
            grid_constraint_work_gap_j=grid_work - float(m.mass * np.sum(stage_v * constraint_dv)),
            friction_identity_max_abs_j=contact['identity_max_abs_j'],
            normal_work_residual_j=(end['contact_normal_j'] - start['contact_normal_j']) + contact['normal_relative_work_j'] + normal_transfer,
            damage_work_j=float(np.sum(raw.damage_work.numpy())),
        )
        dissipation = sum(row[key] for key in ('plastic_j', 'friction_viscous_j', 'friction_sliding_j',
                                              'friction_integration_j', 'history_transport_j', 'pic_j'))
        row['balance_residual_j'] = (row['kinetic_change_j'] + row['elastic_change_j'] + row['normal_change_j']
                                     + row['tangent_change_j'] + dissipation - knife_work - row['gravity_work_j'])
        row['integration_residual_j'] = (row['balance_residual_j'] - row['friction_transfer_work_j']
                                         - row['release_transfer_work_j'] - row['release_correction_work_j'] + row['grid_constraint_work_gap_j'])
        if not np.isfinite(list(row.values())).all():
            raise FloatingPointError('Nonfinite segment energy ledger')
        self.rows.append(row)
        self.elapsed += dt
        self.final = end
        self._trial = None

    def report(self):
        if self.initial is None:
            raise RuntimeError('Energy ledger has no accepted trajectory')
        totals = {name: float(sum(row[name] for row in self.rows)) for name in self.rows[0]} if self.rows else {}
        if self.rows:
            totals['friction_identity_max_abs_j'] = max(row['friction_identity_max_abs_j'] for row in self.rows)
        event_changes = {name: sum(event['changes'][name] for event in self.events) for name in self.initial}
        totals['event_energy_change_j'] = sum(value for name, value in event_changes.items() if name != 'gravity_j')
        totals['global_balance_residual_j'] = totals.get('balance_residual_j', 0.) + totals['event_energy_change_j']
        return dict(initial=self.initial, final=self.final, duration_s=self.elapsed, segments=len(self.rows),
                    totals=totals, event_changes=event_changes, events=self.events)
