"""Replay recorded material/tool paths at different surface observation intervals.

    python -m scripts.surface_observations runs/example --strides 1 2 4 --output /tmp/intervals.json

Every recorded tool substep is supplied at every stride. Only material observations
are decimated; their intervening motion is interpolated by SurfaceRecorder. Input
surface frames must contain tool_path, including cutting flags and cut indices.
CUDA runs embedding and CCD by default; --backend cpu selects the reference path.
Contact and manifold checks run on the final frame by default. --check-frames
selects additional frames, and --check-all-observations checks every observation.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from blademaster_warp.geometry import points
from blademaster_warp.provenance import git_revision
from blademaster_warp.surface_runtime import SurfaceRecorder
from scripts.surface_deformed import geometry_report


@dataclass
class Observation:
    time_s: float
    particles: np.ndarray
    deformation: np.ndarray
    codes: np.ndarray
    pose: np.ndarray
    tool_path: np.ndarray


def load_recording(directory, limit=None):
    if limit is not None and limit < 2:
        raise ValueError("The frame limit must be at least two")
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    config = SimpleNamespace(**manifest["config"])
    reference, origin, _ = points(
        config.size, config.grid_spacing, config.shape, getattr(config, "sampling_seed", 0)
    )
    solver = SimpleNamespace(config=config, reference=reference, origin=origin, device="cpu")
    frames = json.loads((directory / "surface/frames.json").read_text())
    spacing = json.loads((directory / "surface/method.json").read_text())["surface_spacing_m"]
    observations = []
    with np.load(directory / "trajectory.npz") as data:
        x, F, codes, times = data["particles"], data["deformation"], data["codes"], data["control_tcp"][:, 0]
        if not (len(frames) == len(x) == len(F) == len(codes) == len(times)):
            raise ValueError("Surface and material recordings must have the same frame count")
        for i, frame in enumerate(frames[:limit]):
            if abs(frame["time_s"] - times[i]) > 1e-9:
                raise ValueError("Surface and material timestamps do not match")
            with np.load(directory / "surface" / f"{i:05d}.npz") as mesh:
                if "tool_path" not in mesh:
                    raise ValueError(f"Frame {i} has no tool_path; substep cadence cannot be reconstructed")
                observations.append(
                    Observation(
                        float(times[i]),
                        x[i],
                        F[i],
                        codes[i],
                        np.r_[mesh["tool_position"], mesh["tool_orientation"]],
                        mesh["tool_path"],
                    )
                )
    if len(observations) < 2:
        raise ValueError("At least two material observations are required")
    return solver, spacing, observations


def compare_observations(
    solver,
    spacing,
    observations,
    strides=(1, 2, 4),
    backend="cpu",
    check_frames=(),
    check_all=False,
    progress=False,
):
    """Compare final geometry and all common checkpoints to the undecimated replay."""
    strides = sorted(set([1, *strides]))
    if strides[0] < 1 or any(int(s) != s for s in strides):
        raise ValueError("Observation strides must be positive integers")
    if len(observations) < 2:
        raise ValueError("At least two material observations are required")
    if backend not in ("cpu", "cuda"):
        raise ValueError("Backend must be cpu or cuda")
    solver = SimpleNamespace(
        config=solver.config,
        reference=solver.reference,
        origin=solver.origin,
        device="cuda:0" if backend == "cuda" else "cpu",
    )
    check_frames = set(check_frames) | {len(observations) - 1}
    if any(i < 0 or i >= len(observations) for i in check_frames):
        raise ValueError("Geometry check frame is outside the recording")
    times = np.array([o.time_s for o in observations])
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Material observation times must be finite and strictly increasing")
    baseline, results = {}, []
    for stride in strides:
        recorder = SurfaceRecorder(
            solver, None, spacing=spacing, embedding_backend=backend, ccd_backend=backend
        )
        observed = sorted(set([*range(0, len(observations), stride), len(observations) - 1]))
        selected = set(observed)
        mismatch_frames, max_displacement = [], 0.0
        matching_frames = 0
        broken_difference, code_difference = 0, 0
        geometry_checks = []
        last_tool_time = times[0]
        for i, sample in enumerate(observations):
            path = np.asarray(sample.tool_path)
            if path.ndim != 2 or path.shape[1] != 10 or not len(path) or not np.isfinite(path).all():
                raise ValueError(f"Frame {i} requires a finite nonempty (samples, 10) tool_path")
            if abs(path[-1, 0] - sample.time_s) > 1e-9:
                raise ValueError(f"Frame {i} tool path must end at its material observation")
            for row in path:
                if row[0] >= last_tool_time - 1e-10:
                    recorder.record_trajectory([row[0]], row[None, 1:8], bool(row[8]), int(row[9]))
                    last_tool_time = row[0]
            if i not in selected:
                continue
            pose = SimpleNamespace(position=sample.pose[:3], orientation_xyzw=sample.pose[3:])
            recorder.update(
                sample.time_s,
                sample.particles,
                sample.deformation,
                sample.codes,
                pose,
                bool(path[-1, 8]),
                int(path[-1, 9]),
            )
            current = recorder.current
            if check_all or i in check_frames:
                geometry_checks.append(
                    dict(
                        frame=i,
                        time_s=sample.time_s,
                        geometry=geometry_report(current["vertices"], current["triangles"]),
                    )
                )
            if stride == 1:
                baseline[i] = (current, recorder.surface.broken.copy(), recorder.surface.codes.copy())
            else:
                reference, broken, codes = baseline[i]
                broken_difference = max(
                    broken_difference, int(np.count_nonzero(recorder.surface.broken != broken))
                )
                code_difference = max(code_difference, int(np.count_nonzero(recorder.surface.codes != codes)))
                same = (
                    np.array_equal(current["triangles"], reference["triangles"])
                    and np.array_equal(current["codes"], reference["codes"])
                    and current["vertices"].shape == reference["vertices"].shape
                )
                if same:
                    matching_frames += 1
                    max_displacement = max(
                        max_displacement,
                        float(np.linalg.norm(current["vertices"] - reference["vertices"], axis=1).max()),
                    )
                else:
                    mismatch_frames.append(i)
        final = recorder.current
        result = dict(
            stride=stride,
            observations=len(observed),
            max_observation_interval_s=float(np.diff(times[observed]).max()),
            broken_links=int(recorder.surface.broken.sum()),
            max_broken_link_difference=broken_difference,
            max_node_code_difference=code_difference,
            topology_mismatch_frames=mismatch_frames,
            matching_topology_frames=len(observed) if stride == 1 else matching_frames,
            max_matching_vertex_displacement_m=max_displacement,
            geometry=geometry_checks[-1]["geometry"],
            geometry_checks=geometry_checks,
            reference_geometry=geometry_report(final["reference"], final["triangles"], intersections=False),
        )
        if results:
            result["relative_volume_difference"] = (
                result["geometry"]["signed_volume_m3"] / results[0]["geometry"]["signed_volume_m3"] - 1
            )
        results.append(result)
        recorder.close()
        if progress:
            print(
                f"stride {stride}: {len(observed)} observations, {len(geometry_checks)} geometry checks, "
                f"{len(mismatch_frames)} topology mismatches",
                file=sys.stderr,
                flush=True,
            )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("directory", type=Path)
    parser.add_argument("--strides", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument(
        "--backend", choices=("cuda", "cpu"), default="cuda", help="Embedding and CCD backend (default: cuda)"
    )
    parser.add_argument(
        "--surface-spacing", type=float, help="Override the recorded surface spacing in metres"
    )
    parser.add_argument("--limit", type=int, help="Read only the first N recorded observations")
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument(
        "--check-frames",
        type=int,
        nargs="+",
        default=[],
        help="Check contacts at these frames if observed, plus the final frame",
    )
    checks.add_argument(
        "--check-all-observations",
        action="store_true",
        help="Check contacts at every replay observation; default final frame only",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    solver, spacing, observations = load_recording(args.directory, args.limit)
    spacing = args.surface_spacing if args.surface_spacing is not None else spacing
    results = compare_observations(
        solver,
        spacing,
        observations,
        args.strides,
        backend=args.backend,
        check_frames=args.check_frames,
        check_all=args.check_all_observations,
        progress=True,
    )
    manifest = json.loads((args.directory / "manifest.json").read_text())
    report = dict(
        revision=git_revision(),
        input_directory=str(args.directory.resolve()),
        input_revision=manifest.get("revision"),
        input_manifest=manifest,
        settings=dict(
            backend=args.backend,
            surface_spacing_m=spacing,
            strides=sorted(set([1, *args.strides])),
            limit=args.limit,
            check_frames=args.check_frames,
            check_all_observations=args.check_all_observations,
        ),
        input_observations=len(observations),
        input_time_range_s=[observations[0].time_s, observations[-1].time_s],
        results=results,
    )
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
