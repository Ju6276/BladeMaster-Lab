"""Offline whole-arm + surface video from a recorded Franka run."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .replay_style import ReplayScene
from .robot_replay import RobotVisuals
from .provenance import git_revision


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_camera(scene, position, focal, scale):
    camera = scene.renderer.GetActiveCamera()
    camera.SetPosition(*position)
    camera.SetFocalPoint(*focal)
    camera.SetParallelScale(scale)
    scene.renderer.ResetCameraClippingRange()


def render(directory, output=None, preview=False, layout="overview"):
    if layout not in ("overview", "paper"):
        raise ValueError("Unknown replay layout")
    directory = Path(directory)
    output = (
        Path(output)
        if output
        else directory / ("franka_paper_pair.mp4" if layout == "paper" else "franka_paper_replay.mp4")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    report = json.loads((directory / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Cannot show a failed run as a completed replay")
    frames = json.loads((directory / "surface/frames.json").read_text())
    if len(frames) < 2:
        raise ValueError("At least two saved surface frames are required")
    input_paths = [
        directory / name
        for name in ("trajectory.npz", "manifest.json", "scene.xml", "report.json", "surface/frames.json")
    ]
    hashes = {str(p.relative_to(directory)): sha256(p) for p in input_paths}
    z = np.load(directory / "trajectory.npz")
    # Energy diagnostics use float32; control_tcp retains the recorder's
    # float64 simulation timestamps alongside these exact qpos frames.
    times = z["control_tcp"][:, 0]
    frame_times = np.array([f["time_s"] for f in frames])
    fps = float(1 / np.median(np.diff(frame_times)))
    # Runs that end on --duration also save the last step, usually less than one
    # frame interval after the previous frame; a fixed-rate video leaves it out.
    if len(frame_times) > 2 and frame_times[-1] - frame_times[-2] < 1 / fps - 1.0e-8:
        frames, frame_times = frames[:-1], frame_times[:-1]
    if not np.allclose(np.diff(frame_times), 1 / fps, atol=1.0e-8):
        raise ValueError("Irregular sample times: require explicit variable-frame-rate encoding")
    geom = ET.parse(directory / "scene.xml").find(".//geom[@name='board']")
    board = dict(
        size_mm=(np.fromstring(geom.get("size"), sep=" ") * 2000).tolist(),
        center_mm=((np.fromstring(geom.get("pos"), sep=" ") - z["origin"]) * 1000).tolist(),
    )
    scenes = [
        ReplayScene(manifest["config"], board, 640, 720, style="paper"),
        ReplayScene(manifest["config"], board, 640, 720 if layout == "paper" else 360, style="paper"),
    ]
    robots = [RobotVisuals(scene, directory / "scene.xml", z["origin"]) for scene in scenes]
    # Camera selection changes neither geometry nor the recorded knife motion.
    cameras = [
        dict(position=[950, -1400, 850], focal=[-170, 0, -45], scale=480),
        dict(position=[120, -270, 210], focal=[0, 0, 29], scale=69),
    ]
    if layout == "paper":
        # Fig. 1 uses the same camera for particles/surface and views the
        # banana length horizontally. This is an explicitly chosen view,
        # not claimed to be the unpublished author camera calibration.
        cameras = [dict(position=[300, -600, 470], focal=[0, 0, 80], scale=140) for _ in scenes]
    for scene, camera in zip(scenes, cameras):
        set_camera(scene, **camera)
    cycle_path = directory / "cycle.json"
    cycle = json.loads(cycle_path.read_text()) if cycle_path.exists() else dict(phases=[], events=[])
    phase_labels = [{"descend": "Cut"}.get(name, name.capitalize()) for name in cycle["phases"]]
    cuts = sum(event["phase"] == "descend" for event in cycle["events"])
    title = "Franka | Single cut" if cuts <= 1 else f"Franka | {cuts} cuts + push"
    if "teleop" in report:
        title = f"Franka | Teleoperation, {report['teleop']['cut_strokes']} cut strokes"
    font = ImageFont.truetype("DejaVuSans.ttf", 22)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    selected = sorted({0, len(frames) // 2, len(frames) - 1}) if preview else range(len(frames))
    proc = None
    surface_hash = hashlib.sha256()
    max_position_error = max_rotation_error = max_time_error = 0.0
    try:
        for fi in selected:
            now = frame_times[fi]
            idx = int(np.argmin(abs(times - now)))
            time_error = float(abs(times[idx] - now))
            if time_error > 1.0e-8:
                raise ValueError(f"Robot/material timestamp mismatch: {time_error} s")
            max_time_error = max(max_time_error, time_error)
            path = directory / "surface" / f"{fi:05d}.npz"
            surface_hash.update(path.read_bytes())
            with np.load(path) as snapshot:
                for scene, robot in zip(scenes, robots):
                    scene.update(snapshot, z["particles"][idx], z["codes"][idx])
                    pe, re = robot.update(z["qpos"][idx], snapshot)
                    max_position_error, max_rotation_error = max(max_position_error, pe), max(
                        max_rotation_error, re
                    )
                    scene.renderer.ResetCameraClippingRange()
            frame = Image.new("RGB", (1280, 800), (241, 244, 246))
            if layout == "paper":
                frame.paste(Image.fromarray(scenes[0].frame(particles=True)), (0, 48))
                frame.paste(Image.fromarray(scenes[1].frame()), (640, 48))
            else:
                frame.paste(Image.fromarray(scenes[0].frame()), (0, 48))
                frame.paste(Image.fromarray(scenes[1].frame()), (640, 48))
                frame.paste(Image.fromarray(scenes[1].frame(particles=True)), (640, 408))
            draw = ImageDraw.Draw(frame)
            ink = (28, 39, 47)
            phase = ""
            trace = z["cycle_trace"]
            if trace.size:
                phase = phase_labels[int(trace[np.argmin(abs(trace[:, 0] - now)), 1])]
            draw.text((20, 12), title, font=font, fill=ink)
            draw.text((1020, 12), f"{phase}  {now:.2f} s", font=font, fill=ink)
            if layout == "paper":
                draw.rectangle((12, 56, 242, 82), fill=(241, 244, 246))
                draw.rectangle((652, 56, 730, 82), fill=(241, 244, 246))
            draw.text(
                (18, 60),
                "Particles | cut-history codes" if layout == "paper" else "Robot + surface",
                font=small,
                fill=ink,
            )
            draw.text((658, 60), "Surface", font=small, fill=ink)
            if layout != "paper":
                draw.text((658, 420), "Particles | cut-history codes", font=small, fill=ink)
                draw.line((640, 408, 1280, 408), fill=(255, 255, 255), width=3)
            # Colors are cut-history codes (one ternary digit per cut), not connected components.
            legend_x = 18 if layout == "paper" else 658
            draw.rectangle((legend_x - 6, 739, legend_x + 300, 763), fill=(241, 244, 246))
            draw.text((legend_x, 742), "Colors: cut-history codes", font=small, fill=ink)
            draw.line((640, 48, 640, 768), fill=(255, 255, 255), width=3)
            force = z["log"][np.argmin(abs(z["log"][:, 0] - now)), 7:10]
            draw.text(
                (20, 777),
                f"Material reaction: {np.linalg.norm(force):.3f} N",
                font=small,
                fill=ink,
            )
            if fi in (0, len(frames) // 2, len(frames) - 1):
                frame.save(output.with_name(f"{output.stem}_{fi:04d}.png"))
            if not preview:
                if proc is None:
                    proc = subprocess.Popen(
                        [
                            "ffmpeg",
                            "-y",
                            "-v",
                            "error",
                            "-f",
                            "rawvideo",
                            "-pixel_format",
                            "rgb24",
                            "-video_size",
                            "1280x800",
                            "-framerate",
                            str(fps),
                            "-i",
                            "-",
                            "-c:v",
                            "libx264",
                            "-crf",
                            "18",
                            "-pix_fmt",
                            "yuv420p",
                            "-movflags",
                            "+faststart",
                            str(output),
                        ],
                        stdin=subprocess.PIPE,
                    )
                proc.stdin.write(np.asarray(frame).tobytes())
        if proc:
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError("ffmpeg failed")
    finally:
        for scene in scenes:
            scene.window.Finalize()
        z.close()
        if proc and proc.poll() is None:
            proc.kill()
    for path in input_paths:
        if hashes[str(path.relative_to(directory))] != sha256(path):
            raise RuntimeError("Recorded input changed during replay")
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                kind="preview" if preview else "read-only replay",
                source=str(directory.resolve()),
                frames=len(selected),
                fps=fps,
                input_sha256=hashes,
                surface_sha256_in_render_order=surface_hash.hexdigest(),
                max_robot_surface_position_error_m=max_position_error,
                max_robot_surface_rotation_matrix_error=max_rotation_error,
                max_robot_surface_time_error_s=max_time_error,
                cameras=cameras,
                board=board,
                layout=layout,
                style_reference="BladeMaster Fig. 1 / https://arxiv.org/html/2609.27342v2#S1.F1",
                presentation_changes=[
                    "pale background",
                    "green board color",
                    "lighting",
                    "camera framing",
                    "labels",
                ],
                particle_colors="saved cut-history codes, not component IDs; single cut: 0 yellow unmarked, 1 blue, 2 orange",
                geometry="recorded mesh vertices and triangles; compiled Menagerie visuals transformed by saved qpos; no smoothing or synthetic motion",
                playback="recorded simulation time, not wall-clock computation time",
                revision=git_revision(),
            ),
            indent=2,
        )
    )
    print(output if not preview else output.with_name(f"{output.stem}_{selected[-1]:04d}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument(
        "--layout",
        choices=["overview", "paper"],
        default="overview",
        help="Whole arm with close-ups, or matched particle/surface views inspired by Fig. 1",
    )
    args = parser.parse_args()
    render(args.directory, args.output, args.preview, args.layout)
