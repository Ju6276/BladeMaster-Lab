"""Fig. 5 runner and persisted comparison artifacts."""

from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np

from .fig5 import (
    Fig5Config,
    PAPER_RESULTS,
    PAPER_URL,
    commands,
    evaluate,
    initial_left_velocity,
    middle_tetrahedra,
    piece_ids,
    protocol,
    reference_particles,
    half_depth_gap,
    moving_piece_ids,
)
from .provenance import git_revision


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_blademaster(cfg, output, device="cuda:0", graph=True):
    from .solver import Config
    from .fig5_warp import Fig5Solver
    from .coupling import ToolState, ExternalToolPort
    from .topology import components

    solver_config = Config(
        size=tuple(cfg.size),
        grid_spacing=cfg.grid_spacing,
        dt=cfg.dt,
        shape="block",
        sampling_seed=0,
        young_modulus=cfg.young_modulus,
        poisson_ratio=cfg.poisson_ratio,
        density=cfg.density,
        yield_stress=0,
        dynamic_viscosity=0,
        external_tool=True,
        friction_static=cfg.friction,
        friction_dynamic=cfg.friction,
        blade_half_size=(cfg.blade_thickness / 2, cfg.blade_half_length, cfg.blade_half_height),
        board_normal_scale=1.0,
    )
    solver = Fig5Solver(solver_config, cfg.cut_x, device=device)
    port = ExternalToolPort(solver)
    ref = solver.reference.copy()
    pieces = piece_ids(ref, cfg.cut_x)
    cmds = commands(cfg)
    withdrawn_index = round(protocol(cfg)[1]["withdrawn"] / cfg.interval)
    positions = np.empty((len(cmds) + 1, len(ref), 3), dtype=np.float32)
    velocities = np.empty_like(positions)
    impulses = np.zeros((len(cmds) + 1, 3), dtype=float)
    board_impulses = np.zeros_like(impulses)
    pair_impulses = np.zeros((len(cmds) + 1, 3, 3), dtype=float)
    positions[0], velocities[0] = solver.x.numpy(), solver.v.numpy()
    before = after = None
    current_cut = 0
    connectivity = {}
    event_labels, event_codes = [], []
    event_indices = {round(t / cfg.interval): name for name, t in protocol(cfg)[1].items()}
    start = time.perf_counter()
    for i, (_, p, v, cut) in enumerate(cmds):
        if i == withdrawn_index:
            pieces = moving_piece_ids(cfg, positions)
            solver.fig5_pieces.assign(pieces.astype(np.int32))
            before = solver.v.numpy().copy()
            solver.v.assign(initial_left_velocity(before, pieces, cfg.left_velocity))
            after = solver.v.numpy().copy()
        state = ToolState(p, [0, 0, 0, 1], v, [0, 0, 0])
        current_cut = max(current_cut, cut)
        solver.reset_contact_measurement()
        reaction = port.advance(state, cfg.interval, cutting=cut >= 0, cut_index=current_cut, graph=graph)
        positions[i + 1], velocities[i + 1] = solver.x.numpy(), solver.v.numpy()
        impulses[i + 1] = reaction.impulse_ns
        board_impulses[i + 1] = solver.fig5_board_sum.numpy()[0]
        pair_impulses[i + 1] = solver.fig5_pair_sum.numpy()
        if i + 1 in event_indices:
            codes = solver.codes.numpy().copy()
            labels, sizes = components(solver.node_numpy, solver.weight_numpy, codes)
            connectivity[event_indices[i + 1]] = sizes.tolist()
            event_labels.append(labels)
            event_codes.append(codes)
    return {
        "reference": ref,
        "time": np.arange(len(cmds) + 1) * cfg.interval,
        "position": positions,
        "velocity": velocities,
        "blade_impulse_ns": impulses,
        "board_impulse_ns": board_impulses,
        "pair_impulse_by_piece_ns": pair_impulses,
        "velocity_before_kick": before,
        "velocity_after_kick": after,
        "measurement_piece_ids": pieces,
        "topology_event_names": np.array(list(protocol(cfg)[1])),
        "component_labels_by_event": np.array(event_labels),
        "cut_codes_by_event": np.array(event_codes),
    }, {
        "backend": "blademaster",
        "solver_config": asdict(solver_config),
        "device": solver.device.name,
        "graph": graph,
        "wall_s": time.perf_counter() - start,
        "component_sizes_by_event": connectivity,
    }


def run(cfg, output, backend, *, build_dir=None, device="cuda:0", graph=True):
    if backend not in ("blademaster", "cressim_cpu"):
        raise ValueError(f"Unknown Fig5 backend: {backend}")
    ref = reference_particles(cfg)
    middle_tetrahedra(ref, cfg.cut_x)
    for cut in cfg.cut_x:
        half_depth_gap(ref, ref, cut, cfg.size[2] / 2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "experiment": "fig5_parallel_cuts",
        "status": "running",
        "backend": backend,
        "fixture": asdict(cfg),
        "protocol": protocol(cfg)[0],
        "revision": git_revision(),
        "paper": PAPER_URL,
        "source_parameters": {
            "blade_thickness": "paper Fig. 5",
            "other_fixture_values": "local controlled fixture",
        },
    }
    write_json(output / "manifest.json", manifest)
    try:
        if backend == "cressim_cpu":
            if build_dir is None:
                raise ValueError("CRESSim requires --cressim-build")
            from .fig5_cressim import run as run_cressim
            from .fig5_cressim import validate_build

            manifest["source_build"] = validate_build(build_dir)
            data, meta = run_cressim(cfg, output, build_dir)
        else:
            data, meta = run_blademaster(cfg, output, device, graph)
        np.savez_compressed(output / "states.npz", **data)
        result = evaluate(cfg, data, component_sizes_by_event=meta.get("component_sizes_by_event"))
        write_json(output / "result.json", result)
        manifest.update(meta, status="complete")
    except Exception as exc:
        manifest.update(status="failed", error=repr(exc))
        raise
    finally:
        write_json(output / "manifest.json", manifest)
    return result


def load_run(path):
    path = Path(path)
    meta = json.loads((path / "manifest.json").read_text())
    if meta.get("status") != "complete" or meta.get("experiment") != "fig5_parallel_cuts":
        raise ValueError(f"Incomplete Fig5 run: {path}")
    cfg = Fig5Config(**meta["fixture"])
    with np.load(path / "states.npz", allow_pickle=False) as archive:
        result = evaluate(cfg, archive, component_sizes_by_event=meta.get("component_sizes_by_event"))
    return meta, result


def compare(blademaster, cressim):
    bm, bm_result = load_run(blademaster)
    cr, cr_result = load_run(cressim)
    if bm["backend"] != "blademaster" or cr["backend"] != "cressim_cpu":
        raise ValueError("Expected BladeMaster and independent CRESSim-MPM runs")
    from .fig5_cressim import CRESSIM_REVISION

    if cr.get("source_build", {}).get("commit") != CRESSIM_REVISION:
        raise ValueError("CRESSim source revision is missing or mismatched")
    bm_fixture = asdict(Fig5Config(**bm["fixture"]))
    cr_fixture = asdict(Fig5Config(**cr["fixture"]))
    bm_fixture.pop("cressim_mass_unit_kg")
    mass_unit = cr_fixture.pop("cressim_mass_unit_kg")
    bm_fixture.pop("cressim_domain_margin")
    domain_margin = cr_fixture.pop("cressim_domain_margin")
    if bm_fixture != cr_fixture or bm["protocol"] != cr["protocol"]:
        raise ValueError("Comparison requires identical fixture and trajectory configurations")
    statuses = [result["fixture_validity"]["status"] for result in (bm_result, cr_result)]
    comparison_status = "unverified"
    if "mismatched" in statuses:
        comparison_status = "invalid"
    elif statuses == ["matched", "matched"]:
        comparison_status = "matched"
    material_status = "invalid"
    if (
        bm_result["actual_partition"]["status"] == "matched"
        and cr_result["actual_partition"]["status"] in ("matched", "geometric")
        and all(
            result["middle_volume"]["valid_for_material_region_volume"] for result in (bm_result, cr_result)
        )
        and all(
            volume["valid"]
            for result in (bm_result, cr_result)
            for volume in result["full_material_volume"]["after_cuts"].values()
        )
    ):
        material_status = "matched"
    return {
        "fixture": bm_fixture,
        "cressim_mass_unit_kg": mass_unit,
        "cressim_domain_margin_m": domain_margin,
        "cressim_fix_corotated_dispatch": cr["source_build"].get("fix_corotated_dispatch", False),
        "fixture_comparison_status": comparison_status,
        "material_measurement_comparison_status": material_status,
        "blademaster": bm_result,
        "cressim_cpu": cr_result,
        "published_fig5": PAPER_RESULTS,
        "paper": PAPER_URL,
        "runs": {
            "blademaster": str(Path(blademaster).resolve()),
            "cressim_cpu": str(Path(cressim).resolve()),
        },
    }
