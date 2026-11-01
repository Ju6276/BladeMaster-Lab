"""CPU boundary snapshots and reverse recomputation of a natural material flow."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import warp as wp

from blademaster_warp.cohesive_natural import NaturalHistory
from blademaster_warp.cohesive_stages import AffineState
from blademaster_warp.differentiable_contact import Tool


def state_arrays(state, history, tool):
    result = {
        "state_" + name: getattr(state, name)
        for name in ["x", "v", "F", "Fp", "affine", "stress"]
        if getattr(state, name) is not None
    }
    result.update({"history_" + name: value for name, value in history.differentiable_arrays().items()})
    result.update(
        {"tool_" + name: getattr(tool, name) for name in ["position", "velocity", "rotation", "omega"]}
    )
    return result


def snapshot(state, history, tool):
    return dict(
        state=state.snapshot(),
        history=history.snapshot(),
        tool={
            name: getattr(tool, name).numpy().copy() for name in ["position", "velocity", "rotation", "omega"]
        },
    )


def restore(value, device):
    return (
        AffineState.restore(value["state"], device=device),
        NaturalHistory.restore(value["history"], device=device),
        Tool(
            *[
                wp.array(value["tool"][name], dtype=kind, device=device, requires_grad=True)
                for name, kind in [
                    ("position", wp.vec3d),
                    ("velocity", wp.vec3d),
                    ("rotation", wp.quatd),
                    ("omega", wp.vec3d),
                ]
            ]
        ),
    )


def assert_snapshot_equal(actual, expected):
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for name in expected:
            assert_snapshot_equal(actual[name], expected[name])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for first, second in zip(actual, expected):
            assert_snapshot_equal(first, second)
    else:
        assert actual == expected


def event_signature(result):
    return (
        tuple((int(particle), float(time.numpy()[0])) for particle, time in result.geometry_events),
        tuple((float(event.time.numpy()[0]), event.changes) for event in result.contact_events),
    )


def gradient_value(array):
    return np.zeros_like(array.numpy()) if array.grad is None else array.grad.numpy().copy()


@dataclass(frozen=True)
class NaturalGradient:
    terminal: dict
    loss: float
    initial_gradients: dict
    parameter_gradients: dict
    timings: dict


def checkpoint_vjp(
    initial,
    history,
    tool,
    *,
    advance,
    objective,
    steps,
    segment_steps,
    parameters,
    expected_terminal=None,
    progress=None
):
    """Callbacks use global step indices and reproduce their forward computations.

    advance(index, state, history, tool) returns a NaturalStep. objective(index,
    result, loss) adds to a float64 scalar. Parameters are independent of the
    initial boundary arrays. Snapshots contain numerical remainders and discrete
    geometry; adjoints carry every continuous boundary field.
    """
    if steps < 1 or segment_steps < 1:
        raise ValueError("Step and segment counts must be positive")
    device = initial.x.device
    original_arrays = state_arrays(initial, history, tool)
    if len({id(value) for value in parameters.values()}) != len(parameters):
        raise ValueError("Parameter names must identify distinct arrays")
    if {id(value) for value in parameters.values()} & {id(value) for value in original_arrays.values()}:
        raise ValueError("Boundary arrays must be differentiated as state, not repeated parameters")
    for array in [*original_arrays.values(), *parameters.values()]:
        if array.size and array.grad is None:
            raise ValueError("Continuous state and parameters require gradient buffers")
    boundaries = [*range(0, steps, segment_steps), steps]
    checkpoints = {0: snapshot(initial, history, tool)}
    events = {}
    timings = dict(forward_s=0.0, recompute_s=0.0, backward_s=0.0, segments=[])
    totals = {name: np.zeros_like(value.numpy()) for name, value in parameters.items()}

    def now():
        wp.synchronize_device(device)
        return perf_counter()

    def report(phase, first, last):
        if progress is not None:
            progress(dict(phase=phase, first=first, last=last))

    started = now()
    prepass_loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
    state, retained, current_tool = initial, history, tool
    for index in range(boundaries[-2]):
        result = advance(index, state, retained, current_tool)
        objective(index, result, prepass_loss)
        state, retained, current_tool = result.state, result.history, result.tool
        events[index] = event_signature(result)
        if index + 1 in boundaries:
            checkpoints[index + 1] = snapshot(state, retained, current_tool)
            report("checkpoint", index + 1, index + 1)
    timings["forward_s"] = now() - started
    loss_value = float(prepass_loss.numpy()[0])
    carried = None
    terminal = None
    for first, last in reversed(list(zip(boundaries[:-1], boundaries[1:]))):
        report("recompute", first, last)
        started = now()
        state, retained, current_tool = restore(checkpoints[first], device)
        inputs = state_arrays(state, retained, current_tool)
        for value in [*inputs.values(), *parameters.values()]:
            if value.grad is not None:
                value.grad.zero_()
        loss = wp.zeros(1, dtype=wp.float64, device=device, requires_grad=True)
        with wp.Tape() as tape:
            for index in range(first, last):
                result = advance(index, state, retained, current_tool)
                objective(index, result, loss)
                state, retained, current_tool = result.state, result.history, result.tool
                if index in events:
                    assert_snapshot_equal(event_signature(result), events[index])
        actual = snapshot(state, retained, current_tool)
        if last in checkpoints:
            assert_snapshot_equal(actual, checkpoints[last])
        if last == steps:
            terminal = actual
            loss_value += float(loss.numpy()[0])
            if expected_terminal is not None:
                assert_snapshot_equal(actual, expected_terminal)
        timings["recompute_s"] += now() - started
        outputs = state_arrays(state, retained, current_tool)
        seeds = {loss: wp.ones(1, dtype=wp.float64, device=device)}
        if carried is not None:
            assert outputs.keys() == carried.keys()
            grouped = {}
            for name, value in outputs.items():
                if value in grouped:
                    grouped[value] += carried[name]
                else:
                    grouped[value] = carried[name].copy()
            seeds.update(
                {
                    value: wp.array(seed, dtype=value.dtype, device=device)
                    for value, seed in grouped.items()
                    if value.size
                }
            )
        report("backward", first, last)
        started = now()
        tape.backward(grads=seeds)
        carried = {name: gradient_value(value) for name, value in inputs.items()}
        for name, value in parameters.items():
            totals[name] += gradient_value(value)
        tape.zero()
        elapsed = now() - started
        timings["backward_s"] += elapsed
        timings["segments"].append(dict(first=first, last=last, backward_s=elapsed))
        del tape
    return NaturalGradient(terminal, loss_value, carried, totals, timings)
