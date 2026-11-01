import json
import pytest

import numpy as np

from blademaster_warp.natural_task import NaturalTask
from blademaster_warp.natural_checkpoint import assert_snapshot_equal
from blademaster_warp.solver import Config


def make_task(cutting):
    config = Config(
        size=(0.012, 0.012, 0.012),
        grid_spacing=0.006,
        dt=0.0002,
        young_modulus=20000.0,
        yield_stress=0.0,
        flip_fraction=1.0,
        external_tool=True,
        blade_grid_constraint=False,
        label_board_layer=False,
        blade_half_size=(0.0005, 0.0077, 0.006),
        blade_bevel_height=0.002,
    )
    return NaturalTask(
        config,
        durations=[0.001 if cutting else 0.0002, 0.0002, 0.0002],
        controls=[
            [0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        position=[0.0002, 0.0001, 0.01801 if cutting else 0.01721],
        rotation=[0.0, 0.0, 0.0, 1.0],
        strength=50000.0,
        fracture_energy=70.0,
        contact_parameters=[1000.0, 500.0, 0.2, 0.3, 0.2],
        diameter=0.003,
        offset=0.0015,
        force_weight=0.1,
        cutting=cutting,
    )


@pytest.mark.parametrize("cutting", [False, True])
def test_task_force_and_position_objective_full_checkpoint_and_natural_difference(cutting, tmp_path):
    task = make_task(cutting)
    full = task.evaluate(gradient=True)
    if cutting:
        assert len(full.terminal["history"]["cohesive"]["history"]) > 0
    position_loss = np.sum((full.terminal["state"]["x"] - task.model.reference) ** 2) / (
        task.model.n * task.position_scale**2
    )
    assert full.loss > position_loss
    checkpoint = task.evaluate(gradient=True, segment_steps=1)
    assert_snapshot_equal(checkpoint.terminal, full.terminal)
    np.testing.assert_allclose(checkpoint.loss, full.loss, rtol=1e-13, atol=1e-15)
    errors = {}
    for name in full.parameter_gradients:
        a, b = full.parameter_gradients[name], checkpoint.parameter_gradients[name]
        np.testing.assert_allclose(a, b, rtol=2e-10, atol=1e-12, err_msg=name)
        errors[name] = float(np.max(abs(a - b), initial=0.0))
    base = task.controls.numpy().copy()
    direction = np.random.default_rng(921).normal(size=base.shape)
    derivative = float(np.sum(full.parameter_gradients["controls"] * direction))
    assert derivative != 0.0
    differences = []
    for epsilon in [1e-4, 1e-5]:
        values = []
        for sign in [-1.0, 1.0]:
            task.controls.assign(base + sign * epsilon * direction)
            values.append(task.evaluate().loss)
        fd = (values[1] - values[0]) / (2 * epsilon)
        np.testing.assert_allclose(derivative, fd, rtol=2e-5, atol=1e-10)
        differences.append(
            dict(
                epsilon=epsilon,
                ad=derivative,
                fd=fd,
                relative_error=(
                    abs(fd - derivative) / max(abs(fd), abs(derivative))
                    if max(abs(fd), abs(derivative))
                    else 0.0
                ),
            )
        )
    task.controls.assign(base)
    (tmp_path / "report.json").write_text(
        json.dumps(
            dict(
                loss=full.loss,
                checkpoint_max_abs=errors,
                differences=differences,
                full_timings=full.timings,
                checkpoint_timings=checkpoint.timings,
            ),
            indent=2,
        )
    )
