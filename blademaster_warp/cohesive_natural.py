"""Natural geometric cuts and particle contacts in one affine material flow."""

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from .cohesive_event_stages import advance_contact_events, apply_contact_event, ContactEvent
from .cohesive_history import CohesiveHistory
from .cohesive_pair_events import guard as pair_guard
from .cohesive_pairing import PairHistory
from .cohesive_spatial_contact import advance_spatial_pairs
from .compensated_state import CompensatedArray, as_compensated
from .differentiable_geometry import GeometryEvents, assign_event
from .reference_affine import build_transfer
from .reference_surface import swept_edge


@wp.kernel
def _add_time(a: wp.array(dtype=wp.float64), b: wp.array(dtype=wp.float64),
              out: wp.array(dtype=wp.float64)):
    out[0] = a[0] + b[0]


@dataclass(frozen=True)
class NaturalHistory:
    codes: wp.array
    cohesive: CohesiveHistory
    pairs: PairHistory
    contact: CompensatedArray
    modes: tuple = ()
    cut_index: int = 0

    def snapshot(self):
        return dict(codes=self.codes.numpy().copy(), cohesive=self.cohesive.snapshot(),
                    pairs=self.pairs.snapshot(), contact=self.contact.snapshot(),
                    modes=self.modes, cut_index=self.cut_index)

    @classmethod
    def restore(cls, snapshot, *, device):
        return cls(
            wp.array(snapshot["codes"], dtype=wp.int64, device=device),
            CohesiveHistory.restore(snapshot["cohesive"], device=device),
            PairHistory.restore(snapshot["pairs"], device=device),
            CompensatedArray.restore(snapshot["contact"], dtype=wp.vec3d, device=device),
            snapshot["modes"], snapshot["cut_index"],
        )

    def differentiable_arrays(self):
        result = {"cohesive_" + key: value
                  for key, value in self.cohesive.differentiable_arrays().items()}
        result.update(pair_history=self.pairs.values.high, contact_history=self.contact.high)
        return result


@dataclass(frozen=True)
class NaturalSegment:
    start: wp.array
    duration: wp.array
    before: object
    step: object
    history_before: NaturalHistory
    history_after: NaturalHistory


@dataclass(frozen=True)
class NaturalStep:
    state: object
    history: NaturalHistory
    tool: object
    segments: tuple
    geometry_events: tuple
    contact_events: tuple


@dataclass(frozen=True)
class _TrialHistory:
    history: NaturalHistory
    path_midpoint: wp.array


class NaturalCohesive(GeometryEvents):
    """Prescribed finite-knife cuts with permanent reference-surface history.

    contact_provider(transfer, tool, x, moments, tau, elapsed, history) builds
    the independent knife/board contacts at an implicit stage. The tool is at
    the segment start; elapsed is the differentiable stage-end time.
    """

    def __init__(self, model, *, material, contact_provider, pair_parameters, diameter,
                 gravity=(0.0, 0.0, -9.81), damping_rate=None):
        if model.precision != "float64":
            raise ValueError("Natural cohesive integration requires float64 material state")
        if material.dtype != wp.vec3d or material.shape != (model.n,):
            raise ValueError("Cohesive material requires particle strength and fracture energy")
        self.model = model
        self.material = material
        self.contact_provider = contact_provider
        self.pair_parameters = pair_parameters
        self.diameter = diameter
        self.gravity = gravity
        self.damping_rate = damping_rate
        self.reference = model.array(model.reference, wp.vec3, False)
        self.particle_half = model.array(
            np.full((model.n, 3), model.config.grid_spacing / 4), wp.vec3, False
        )
        self.half = model.array([model.config.blade_half_size], wp.vec3, False)
        self.max_cut_index = 0
        while 3 ** (self.max_cut_index + 2) * model.node_count <= 2**63 - 1:
            self.max_cut_index += 1

    def history(self, contact_history):
        m = self.model
        return NaturalHistory(
            m.zeros(m.n, wp.int64, False), CohesiveHistory.empty(m.n, device=m.device),
            PairHistory.empty(m.n, device=m.device), as_compensated(contact_history),
        )

    def _add(self, first, second):
        result = self.model.zeros(1, float)
        wp.launch(_add_time, 1, inputs=[first, second], outputs=[result], device=self.model.device)
        return result

    def _arguments(self, state, history, tool, duration):
        m = self.model
        topology = m.topology(history.codes.numpy())
        transfer = build_transfer(m, topology)
        retained = history.cohesive
        if len(retained.history):
            trace = retained.reference_trace(m, topology, equivalent_slots=transfer.equivalent_slots)
            separation, frame = transfer.compose_trace(trace)
        else:
            separation = m.zeros((0, 4 * m.n), float)
            frame = m.zeros((0, 4 * m.n), wp.vec3)

        def provider(x, moments, tau, elapsed, contact_history):
            return self.contact_provider(transfer, tool, x, moments, tau, elapsed, contact_history)

        return dict(
            transfer=transfer, state=state, pair_history=history.pairs, codes=history.codes.numpy(),
            parameters=self.pair_parameters, diameter=self.diameter,
            other_contacts=provider, other_history=history.contact,
            geometry=retained.geometry, history=retained.history,
            reference_normal=retained.quadrature.normals, material=retained.material,
            separation_map=separation, frame_map=frame, area=retained.quadrature.area,
            duration=duration, gravity=self.gravity, damping_rate=self.damping_rate,
            constrain_normal=False,
        )

    @staticmethod
    def _updated(history, spatial):
        material = spatial.material
        return replace(history, cohesive=history.cohesive.updated(material.geometry, material.history),
                       pairs=spatial.history, contact=spatial.other_history)

    def advance_segment(self, state, history, tool, duration, cutting):
        spatial = advance_spatial_pairs(**self._arguments(state, history, tool, duration))
        updated = self._updated(history, spatial)
        return spatial.material.state, _TrialHistory(updated, spatial.material.path_midpoint), spatial

    def _assign(self, state, history, tool, particle):
        m = self.model
        codes = wp.clone(history.codes, requires_grad=False)
        departed = m.zeros(m.n, int, False)
        m.launch(assign_event, 1,
                 inputs=[state.x, tool.position, tool.rotation, particle, 3 ** history.cut_index, 0],
                 outputs=[codes, departed], device=m.device)
        return replace(history, codes=codes)

    def step(self, state, history, tool, *, duration=None, cutting=True, cut_index=None):
        m = self.model
        end = m.duration if duration is None else duration
        if end.dtype != wp.float64 or end.shape != (1,) or end.device != m.device:
            raise ValueError("Natural cut duration must be a matching float64 scalar")
        if not np.isfinite(end.numpy()).all() or end.numpy()[0] <= 0:
            raise ValueError("Natural cut duration must be positive and finite")
        index = history.cut_index if cut_index is None else cut_index
        if (isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer))
                or not history.cut_index <= index <= self.max_cut_index):
            raise ValueError("Cut indices must be monotonic and fit the side-code representation")
        history = replace(history, cut_index=int(index))
        elapsed = m.array([0.0], float, False)
        segments, geometry_events, contact_events = [], [], []
        while elapsed.numpy()[0] < end.numpy()[0]:
            start_tool = self.pose(tool, elapsed)
            if cutting:
                immediate = self.candidates(state, state, history, start_tool, start_tool, index)
                if immediate[1] == 0:
                    history = self._assign(state, history, start_tool, immediate[0])
                    geometry_events.append((immediate[0], elapsed))
                    continue
            remaining = self.interval(end, elapsed)
            trial = advance_contact_events(
                **self._arguments(state, history, start_tool, remaining),
                modes=history.modes, stop_at_first=True,
            )
            spatial = trial.segments[0].step
            accepted_time = trial.segments[0].duration
            accepted_state = spatial.material.state
            updated = replace(self._updated(history, spatial), pairs=trial.pair_history,
                              contact=trial.other_history, modes=trial.modes)
            trial_history = _TrialHistory(updated, spatial.material.path_midpoint)
            accepted_tool = self.pose(start_tool, accepted_time)
            selected = (-1, 2.0, -1, -1, 0.0, 1.0)
            if cutting:
                selected = self.candidates(state, accepted_state, history, start_tool, accepted_tool,
                                           index, trial_history, accepted_time)
            particle, fraction, kind, plane, lower, upper = selected
            if fraction == 0:
                history = self._assign(state, history, start_tool, particle)
                geometry_events.append((particle, elapsed))
                continue
            chosen_contacts = trial.events
            if fraction <= 1:
                zero = m.array([0.0], float, False)
                bound, bound_state, bound_tool = accepted_time, accepted_state, accepted_tool
                for _ in range(m.n + 1):
                    located = self.locate(state, history, start_tool, start_tool, zero, bound,
                                          bound_state, bound_tool, particle, kind, plane, lower, upper)
                    accepted_time, accepted_state, root_history, spatial, accepted_tool, tolerance = located
                    earlier = self.candidates(state, accepted_state, history, start_tool, accepted_tool,
                                              index, root_history, accepted_time)
                    if earlier[0] == particle or earlier[1] >= 1.0:
                        break
                    particle, _, kind, plane, lower, upper = earlier
                    bound, bound_state, bound_tool = accepted_time, accepted_state, accepted_tool
                else:
                    raise RuntimeError("Natural cut event ordering did not converge")
                updated = root_history.history
                chosen_contacts = ()
                if trial.events:
                    position = CompensatedArray(accepted_state.x, accepted_state.rounding_remainders.get("x"))
                    changes = tuple(change for change in trial.events[0].changes
                                    if abs(pair_guard(position, change[:2], self.diameter).high.numpy()[0])
                                    <= tolerance)
                    if changes:
                        pairs, released = apply_contact_event(
                            m, accepted_state, updated.pairs, self.pair_parameters, self.diameter, changes
                        )
                        modes = {(p, q): active for p, q, active in history.modes}
                        modes.update({(p, q): active for p, q, active in changes})
                        updated = replace(updated, pairs=pairs,
                                          modes=tuple((p, q, active) for (p, q), active in sorted(modes.items())))
                        chosen_contacts = (ContactEvent(accepted_time, changes, released),)
            next_elapsed = self._add(elapsed, accepted_time)
            if next_elapsed.numpy()[0] <= elapsed.numpy()[0]:
                raise RuntimeError("Natural cut segment did not advance time")
            if cutting:
                candidate = swept_edge(
                    state.x, state.F, accepted_state.x, accepted_state.F, self.particle_half,
                    start_tool.position, start_tool.rotation, accepted_tool.position, accepted_tool.rotation,
                    self.half, reference_coordinates=m.reference_coordinates,
                )
                retained = updated.cohesive.append(
                    candidate, self.reference, accepted_state.F, self.material,
                    cut_index=index, tool_rotation=accepted_tool.rotation,
                )
                updated = replace(updated, cohesive=retained)
            if fraction <= 1:
                updated = self._assign(accepted_state, updated, accepted_tool, particle)
                geometry_events.append((particle, next_elapsed))
            contact_events.extend(ContactEvent(self._add(elapsed, event.time), event.changes,
                                                event.released_energy) for event in chosen_contacts)
            segments.append(NaturalSegment(elapsed, accepted_time, state, spatial, history, updated))
            state, history, elapsed = accepted_state, updated, next_elapsed
        return NaturalStep(state, history, self.pose(tool, end), tuple(segments),
                           tuple(geometry_events), tuple(contact_events))
