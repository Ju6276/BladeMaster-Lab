"""Build and run the pinned upstream CRESSim-MPM CPU MLS solver in a subprocess."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np

from .fig5 import commands, integer_steps, protocol, reference_particles
from .provenance import git_revision


CRESSIM_URL = "https://github.com/yafei-ou/CRESSim-MPM.git"
CRESSIM_REVISION = "09aa5009b8580351f516b6df7660e87821fc5eb6"


def material_in_solver_units(cfg):
    """Distance/time units are metre/second; one solver mass unit is u kilograms."""
    unit = cfg.cressim_mass_unit_kg
    return {
        "particle_mass": cfg.density * (cfg.grid_spacing / 2) ** 3 / unit,
        "young_modulus": cfg.young_modulus / unit,
        "density": cfg.density / unit,
    }


def impulses_to_si(impulse, mass_unit_kg):
    return np.asarray(impulse, dtype=np.float64) * mass_unit_kg


def source_revision(source):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()

    revision = git("rev-parse", "HEAD")
    if revision != CRESSIM_REVISION or git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"Expected clean CRESSim-MPM source at {CRESSIM_REVISION}")
    return revision


def build(source, output, compiler="c++"):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():
        subprocess.run(["git", "clone", CRESSIM_URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", CRESSIM_REVISION], check=True)
    source_revision(source)
    output.mkdir(parents=True, exist_ok=False)
    driver = Path(__file__).with_name("fig5_cressim_runner.cpp").resolve()
    driver_revision = git_revision()
    driver_snapshot = output / driver.name
    shutil.copyfile(driver, driver_snapshot)
    includes = [
        "include/core",
        "include/math",
        "include/utils",
        "include/solver",
        "include/engine",
        "include/geometry_utils",
        "src/core",
        "src/solver",
    ]
    executable = output / "fig5_cressim_cpu"
    compiler = shutil.which(compiler) or compiler
    command = [
        compiler,
        "-std=c++17",
        "-O2",
        "-DCRMPM_ENGINE_STATIC",
        "-include",
        "climits",
        "-include",
        "cstdint",
        "-include",
        "cstring",
    ]
    command += [
        f'-DFIG5_SOURCE_REVISION="{CRESSIM_REVISION}"',
        f'-DFIG5_DRIVER_REVISION="{driver_revision["commit"]}"',
    ]
    command += [f"-I{source / include}" for include in includes]
    command += [
        str(driver_snapshot),
        str(source / "src/solver/cpu_mls_mpm_solver.cpp"),
        "-o",
        str(executable),
    ]
    meta = {
        "repository": CRESSIM_URL,
        "commit": CRESSIM_REVISION,
        "dirty": False,
        "solver": "crmpm::CpuMlsMpmSolver",
        "source": str(source),
        "command": command,
        "driver_revision": driver_revision,
        "compiler": subprocess.check_output([compiler, "--version"], text=True),
        "status": "building",
    }
    try:
        subprocess.run(command, check=True)
        meta["status"] = "complete"
    except Exception as exc:
        meta.update(status="failed", error=str(exc))
        raise
    finally:
        (output / "build.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def validate_build(build_dir):
    build_dir = Path(build_dir).resolve()
    meta = json.loads((build_dir / "build.json").read_text())
    if meta["status"] != "complete" or meta["commit"] != CRESSIM_REVISION:
        raise ValueError("CRESSim build is incomplete or has a different revision")
    source_revision(meta["source"])
    driver = Path(__file__).with_name("fig5_cressim_runner.cpp")
    if driver.read_bytes() != (build_dir / driver.name).read_bytes():
        raise ValueError("CRESSim driver changed since compilation; rebuild in a new directory")
    identity = subprocess.check_output(
        [str(build_dir / "fig5_cressim_cpu"), "--identity"], text=True
    ).splitlines()
    if identity != [CRESSIM_REVISION, str(meta["driver_revision"]["commit"])]:
        raise ValueError("CRESSim executable source/driver identity differs from build metadata")
    return meta


def run(cfg, output, build_dir):
    output, build_dir = Path(output), Path(build_dir).resolve()
    build_meta = validate_build(build_dir)
    material = material_in_solver_units(cfg)
    mass_unit = cfg.cressim_mass_unit_kg
    ref = reference_particles(cfg)
    cmds = commands(cfg)
    events = protocol(cfg)[1]
    withdrawn_index = round(events["withdrawn"] / cfg.interval)
    # Include expected leftward travel plus four grid cells around the body.
    margin = max(4 * cfg.grid_spacing, abs(cfg.left_velocity) * cfg.manipulation_s + 4 * cfg.grid_spacing)
    lower = np.floor((ref.min(0) - margin) / cfg.grid_spacing) * cfg.grid_spacing
    upper = np.ceil((ref.max(0) + margin) / cfg.grid_spacing) * cfg.grid_spacing
    input_path, raw_path, kick_path = [
        output / name for name in ("cressim_input.txt", "cressim_states.bin", "cressim_kick.bin")
    ]
    with input_path.open("w") as stream:
        header = [
            len(ref),
            len(cmds),
            integer_steps(cfg.interval, cfg.dt),
            cfg.grid_spacing,
            cfg.dt,
            material["particle_mass"],
            material["young_modulus"],
            cfg.poisson_ratio,
            cfg.blade_half_length,
            cfg.blade_half_height,
            cfg.blade_thickness / 2,
            cfg.friction,
            cfg.cressim_smooth_distance,
            cfg.cressim_drag,
            cfg.cut_x[0],
            cfg.left_velocity,
            *lower,
            *upper,
        ]
        stream.write(" ".join(map(str, header)) + "\n")
        np.savetxt(stream, ref, fmt="%.17g")
        for i, (_, p, v, _) in enumerate(cmds):
            stream.write(" ".join(map(str, [*p, *v, int(i == withdrawn_index)])) + "\n")
    command = [str(build_dir / "fig5_cressim_cpu"), str(input_path), str(raw_path), str(kick_path)]
    start = time.perf_counter()
    with (output / "cressim_stdout.log").open("w") as stdout, (output / "cressim_stderr.log").open(
        "w"
    ) as stderr:
        subprocess.run(command, stdout=stdout, stderr=stderr, check=True)
    frames = np.fromfile(raw_path, dtype=np.float32).reshape(len(cmds) + 1, 6 * len(ref) + 6)
    particles = frames[:, : 6 * len(ref)].reshape(len(cmds) + 1, len(ref), 6)
    data = {
        "reference": ref,
        "time": np.arange(len(cmds) + 1) * cfg.interval,
        "position": particles[..., :3],
        "velocity": particles[..., 3:],
        "blade_impulse_ns": impulses_to_si(frames[:, -6:-3], mass_unit),
        "board_impulse_ns": impulses_to_si(frames[:, -3:], mass_unit),
        "pair_impulse_by_piece_ns": np.zeros((len(cmds) + 1, 3, 3)),
        "velocity_before_kick": particles[withdrawn_index, :, 3:].copy(),
        "velocity_after_kick": np.fromfile(kick_path, dtype=np.float32).reshape(len(ref), 3),
    }
    meta = {
        "backend": "cressim_cpu",
        "source_build": build_meta,
        "command": command,
        "solver_config": {
            "material_selection": "eCoRotational",
            "stress_function": "computeLinearElasticStress",
            "stress_dispatch": "upstream computeStress falls through from eCoRotational to eLinearElastic",
            "grid_lower_m": lower.tolist(),
            "grid_upper_m": upper.tolist(),
            "contact_model": "eKinematic",
            "blade_geometry": "eQuadSlicer",
            "shape_type": "eDynamic",
            "sdf_fatten_m": cfg.blade_thickness / 2,
            "initial_volume": "upstream grid-density estimate",
            "mass_unit_kg": mass_unit,
            "length_unit_m": 1.0,
            "time_unit_s": 1.0,
            "material_in_solver_units": material,
            "raw_impulse_unit_kg_m_s": mass_unit,
            "grid_mass_cutoff_solver_units": 1e-4,
            "grid_mass_cutoff_kg": 1e-4 * mass_unit,
            "initial_density_floor_kg_m3": 1e-8 * mass_unit,
            "blade_impulse": "upstream nodal momentum exchange on prescribed blade",
            "pair_contact": "upstream MLS uses shared-grid transfer without an explicit pair-contact kernel",
            "fixture": asdict(cfg),
        },
        "wall_s": time.perf_counter() - start,
    }
    return data, meta
