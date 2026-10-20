"""Particle videos of the robot-free paper experiments (Fig. 3, Fig. 5, Fig. 6).

Same view direction as the Franka replays (browser viewer's default detail
camera, in front of the arm); the distance frames each scene. Colors are
sampled from the project-page videos. Presentation only: recorded particle and
knife states are drawn as saved.

python -m blademaster_warp.render_free fig3 RUN --output out.mp4 --title "..."
python -m blademaster_warp.render_free fig6 RUN --output out.mp4
python -m blademaster_warp.render_free fig5 BM_RUN CRESSIM_RUN --output out.mp4
"""

import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PIL import Image, ImageDraw, ImageFont
from .render import actor, blade_mesh
from .replay_style import mottled, place_lights, wood_texture
from .topology import piece_rgb, recorded_pieces
from .provenance import git_revision

JELLY = (0.50, 0.04, 0.13)
LUNCHEON = (0.85, 0.69, 0.66)
CREAM = (0.88, 0.85, 0.78)
INK = (28, 39, 47)


def front_view():
    """Unit view direction and elevation of the default detail camera of the Franka scene."""
    from .franka import make_model, LAYOUT_BASE_YAW
    from .render_franka import web_cameras

    model, origin, _ = make_model(base_yaw=LAYOUT_BASE_YAW["paper"])
    detail = web_cameras(model.body("link0").pos, origin)[1]
    offset = np.subtract(detail["position"], detail["focal"])
    return offset / np.linalg.norm(offset), detail["view_angle"]


def particle_radius(reference):
    """Half the mean particle spacing, so the sampled block reads as solid."""
    extent = np.ptp(reference, axis=0)
    return 0.5 * (np.prod(extent) / len(reference)) ** (1 / 3)


class FreeScene:
    def __init__(self, width, height, radius_m, board=None):
        self.width, self.height = width, height
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.87, 0.90, 0.92)
        self.renderer.SetBackground2(0.97, 0.975, 0.98)
        self.renderer.GradientBackgroundOn()
        self.window = vtk.vtkEGLRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(width, height)
        self.window.SetMultiSamples(4)
        self.window.AddRenderer(self.renderer)
        if board is not None:
            source = vtk.vtkCubeSource()
            source.SetXLength(board["size_mm"][0])
            source.SetYLength(board["size_mm"][1])
            source.SetZLength(board["size_mm"][2])
            source.SetCenter(*board["center_mm"])
            uv = vtk.vtkTextureMapToPlane()
            uv.SetInputConnection(source.GetOutputPort())
            uv.SetOrigin(-board["size_mm"][0] / 2, -board["size_mm"][1] / 2, 0)
            uv.SetPoint1(board["size_mm"][0] / 2, -board["size_mm"][1] / 2, 0)
            uv.SetPoint2(-board["size_mm"][0] / 2, board["size_mm"][1] / 2, 0)
            wood = actor(uv, self.renderer, (1.0, 1.0, 1.0))
            wood.SetTexture(wood_texture())
            wood.GetProperty().SetAmbient(0.35)
            wood.GetProperty().SetDiffuse(0.6)
        self.cloud = vtk.vtkPolyData()
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius_m * 1000)
        sphere.SetThetaResolution(12)
        sphere.SetPhiResolution(8)
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(self.cloud)
        mapper.SetSourceConnection(sphere.GetOutputPort())
        mapper.ScalingOff()
        self.particles = vtk.vtkActor()
        self.particles.SetMapper(mapper)
        self.particles.GetProperty().SetAmbient(0.45)
        self.particles.GetProperty().SetDiffuse(0.55)
        self.renderer.AddActor(self.particles)
        self.blades = []
        self.capture = vtk.vtkWindowToImageFilter()
        self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGB()
        self.capture.ReadFrontBufferOff()

    def enable_surface(self, color, opacity):
        """Draw the recorded embedded surface instead of particles; opacity < 1 is a translucent gel."""
        self.surface = vtk.vtkPolyData()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(self.surface)
        normals.SetFeatureAngle(45)
        normals.SplittingOn()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        body = actor(normals, self.renderer, color)
        body.GetProperty().SetOpacity(opacity)
        body.GetProperty().SetAmbient(0.25)
        body.GetProperty().SetDiffuse(0.6)
        body.GetProperty().SetSpecular(0.6)
        body.GetProperty().SetSpecularPower(60)
        self.particles.SetVisibility(False)
        if opacity < 1.0:
            self.renderer.SetUseDepthPeeling(1)
            self.renderer.SetMaximumNumberOfPeels(8)
            self.window.SetAlphaBitPlanes(1)
            self.window.SetMultiSamples(0)

    def set_surface(self, snapshot):
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(snapshot["vertices"] * 1000, deep=True))
        cells = vtk.vtkCellArray()
        for tri in snapshot["triangles"]:
            cells.InsertNextCell(3)
            for v in tri:
                cells.InsertCellPoint(int(v))
        self.surface.SetPoints(points)
        self.surface.SetPolys(cells)
        self.surface.Modified()

    def add_blade(self, half, bevel, color=(0.35, 0.37, 0.40)):
        blade = actor(blade_mesh(half, bevel), self.renderer, color)
        blade.GetProperty().SetAmbient(0.3)
        blade.GetProperty().SetSpecular(0.8)
        blade.GetProperty().SetSpecularPower(60)
        self.blades.append(blade)

    def frame_view(self, points_m, direction, view_angle):
        """Look along the shared front direction; distance encloses the given points."""
        centre = 0.5 * (points_m.min(0) + points_m.max(0))
        radius = np.linalg.norm(points_m - centre, axis=1).max()
        aspect = min(1.0, self.width / self.height)
        half = np.radians(view_angle) / 2
        distance = 1.1 * radius / np.sin(np.arctan(np.tan(half) * aspect))
        camera = self.renderer.GetActiveCamera()
        camera.SetViewAngle(view_angle)
        camera.SetFocalPoint(*(centre * 1000))
        camera.SetPosition(*((centre + direction * distance) * 1000))
        camera.SetViewUp(0, 0, 1)
        place_lights(self.renderer, np.arctan2(direction[1], direction[0]))
        self.renderer.ResetCameraClippingRange()

    def update(self, positions, colors, blade_poses):
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(np.asarray(positions, float) * 1000, deep=True))
        self.cloud.SetPoints(pts)
        self.cloud.GetPointData().SetScalars(numpy_to_vtk((np.asarray(colors) * 255).astype(np.uint8), deep=True))
        self.cloud.Modified()
        for blade, (position, quat_xyzw) in zip(self.blades, blade_poses):
            r = Rotation.from_quat(quat_xyzw).as_matrix()
            matrix = vtk.vtkMatrix4x4()
            for row in range(3):
                for col in range(3):
                    matrix.SetElement(row, col, float(r[row, col]))
                matrix.SetElement(row, 3, float(position[row] * 1000))
            blade.SetUserMatrix(matrix)
        self.renderer.ResetCameraClippingRange()

    def frame(self):
        self.window.Render()
        self.capture.Modified()
        self.capture.Update()
        return vtk_to_numpy(self.capture.GetOutput().GetPointData().GetScalars()).reshape(
            self.height, self.width, 3
        )[::-1].copy()


def encoder(output, fps, size):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pixel_format", "rgb24",
         "-video_size", f"{size[0]}x{size[1]}", "-framerate", str(fps), "-i", "-",
         "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)],
        stdin=subprocess.PIPE,
    )


def free_knife_video(kind, directory, output, title, fps=25):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    cfg = manifest["config"]
    z = np.load(directory / "states.npz")
    reference, positions, times = z["reference"], z["position"], z["time"]
    blade_p, blade_q = z["blade_position"], z["blade_orientation"]
    direction, angle = front_view()
    board = None
    color = JELLY
    if kind == "fig6":
        color = LUNCHEON
        extent = np.ptp(reference, axis=0) * 1000
        lo = reference.min(0) * 1000
        size = [extent[0] * 2.2, extent[1] * 4.0, 8.0]
        board = dict(size_mm=size, center_mm=[lo[0] + extent[0] / 2, lo[1] + extent[1] / 2, lo[2] - 4.0])
        # No arm in this scene: keep the elevation; turn halfway from the front
        # direction toward the side the knives enter from (oblique, as in the video).
        approach = blade_p[0].mean(0) - reference.mean(0)
        horizontal = np.linalg.norm(direction[:2])
        bisector = direction[:2] / horizontal + approach[:2] / np.linalg.norm(approach[:2])
        if np.linalg.norm(bisector) < 1e-6:
            bisector = approach[:2]
        direction = np.array([*(bisector / np.linalg.norm(bisector) * horizontal), direction[2]])
    scene = FreeScene(1280, 672, particle_radius(reference), board)
    surface_dir = directory / "surface"
    surface = (surface_dir / "frames.json").exists()
    if surface:
        frames = json.loads((surface_dir / "frames.json").read_text())
        if not np.allclose([f["time_s"] for f in frames], times[: len(frames)]) or len(frames) != len(times):
            raise ValueError("Surface frames must match the saved states")
        # Fig. 3's jelly is translucent in the project-page video.
        scene.enable_surface(color, 0.8 if kind == "fig3" else 1.0)
    blade_color = (0.86, 0.87, 0.88) if kind == "fig3" else (0.35, 0.37, 0.40)
    for _ in range(blade_p.shape[1]):
        scene.add_blade(cfg["blade_half_size"], cfg["blade_bevel_height"], blade_color)
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    blade_extent = [p + corners * np.asarray(cfg["blade_half_size"]).max() for p in blade_p[0]]
    scene.frame_view(np.vstack([reference, *blade_extent]), direction, angle)
    base = np.clip(mottled(color)(reference), 0, 1) if kind == "fig6" else np.tile(color, (len(reference), 1))
    pieces = recorded_pieces(cfg, z["sides"])
    font = ImageFont.truetype("DejaVuSans.ttf", 26)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    slow = fps * float(np.median(np.diff(times)))
    proc = encoder(output, fps, (1280, 720))
    try:
        for i, t in enumerate(times):
            scene.update(positions[i], piece_rgb(pieces[i], base), list(zip(blade_p[i], blade_q[i])))
            if surface:
                with np.load(surface_dir / f"{i:05d}.npz") as snapshot:
                    scene.set_surface(snapshot)
            image = Image.new("RGB", (1280, 720), (241, 244, 246))
            image.paste(Image.fromarray(scene.frame()), (0, 48))
            draw = ImageDraw.Draw(image)
            draw.text((20, 10), title, font=font, fill=INK)
            draw.text((1060, 12), f"{t:.3f} s", font=font, fill=INK)
            count = len(np.unique(pieces[i]))
            view = "Surface" if surface else f"Particles | {count} piece{'s' if count > 1 else ''}"
            draw.text((20, 690), f"{view}  |  {1 / slow:.0f}x slow motion" if slow < 1 else view, font=small, fill=INK)
            if kind == "fig3":
                draw.line((1180, 90, 1180, 160), fill=INK, width=3)
                draw.polygon([(1172, 150), (1188, 150), (1180, 166)], fill=INK)
                draw.text((1194, 110), "g", font=font, fill=INK)
            if i in (0, len(times) // 2, len(times) - 1):
                image.save(Path(output).with_name(f"{Path(output).stem}_{i:04d}.png"))
            proc.stdin.write(np.asarray(image).tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    Path(output).with_suffix(".json").write_text(
        json.dumps(dict(source=str(directory.resolve()), kind=kind, frames=len(times), fps=fps,
                        particle_color=color, revision=git_revision()), indent=2)
    )
    print(output)


def fig5_video(runs, output, fps=25, stride=4):
    from .fig5 import Fig5Config, commands

    labels = ["BladeMaster", "CRESSim-MPM"]
    loaded = []
    for directory in runs:
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        loaded.append((Fig5Config(**manifest["fixture"]), np.load(Path(directory) / "states.npz")))
    direction, angle = front_view()
    font = ImageFont.truetype("DejaVuSans.ttf", 26)
    small = ImageFont.truetype("DejaVuSans.ttf", 18)
    scenes, paths = [], []
    for cfg, z in loaded:
        scene = FreeScene(640, 624, particle_radius(z["reference"]))
        half = (cfg.blade_thickness / 2, cfg.blade_half_length, cfg.blade_half_height)
        scene.add_blade(half, 0.0, (0.55, 0.62, 0.72))
        rows = commands(cfg)
        path_t = np.array([0.0] + [r[0] for r in rows])
        path_p = np.vstack([rows[0][1] - rows[0][2] * cfg.interval] + [r[1] for r in rows])
        paths.append((path_t, path_p))
        top = np.array([0, 0, cfg.size[2] + cfg.clearance + 2 * cfg.blade_half_height])
        scene.frame_view(np.vstack([z["reference"], z["reference"] + top * [0, 0, 1]]), direction, angle)
        scenes.append(scene)
    times = loaded[0][1]["time"]
    if any(len(z["time"]) != len(times) or not np.allclose(z["time"], times) for _, z in loaded):
        raise ValueError("Fig. 5 runs must share sample times")
    frames = range(0, len(times), stride)
    slow = fps * stride * float(np.median(np.diff(times)))
    proc = encoder(output, fps, (1280, 720))
    try:
        for n, i in enumerate(frames):
            image = Image.new("RGB", (1280, 720), (241, 244, 246))
            for k, ((cfg, z), scene, (path_t, path_p)) in enumerate(zip(loaded, scenes, paths)):
                p = np.array([np.interp(times[i], path_t, path_p[:, a]) for a in range(3)])
                scene.update(z["position"][i], np.tile(CREAM, (len(z["reference"]), 1)), [(p, (0, 0, 0, 1))])
                image.paste(Image.fromarray(scene.frame()), (640 * k, 48))
            draw = ImageDraw.Draw(image)
            draw.text((20, 10), "Fig. 5 | Two parallel cuts, then the left block is pushed", font=font, fill=INK)
            draw.text((1100, 12), f"{times[i]:.2f} s", font=font, fill=INK)
            for k, label in enumerate(labels):
                draw.text((640 * k + 18, 58), label, font=font, fill=INK)
            draw.line((640, 48, 640, 672), fill=(255, 255, 255), width=3)
            note = f"Particles  |  {1 / slow:.1f}x slow motion" if slow < 1 else "Particles"
            draw.text((20, 690), note, font=small, fill=INK)
            if n in (0, len(frames) // 2, len(frames) - 1):
                image.save(Path(output).with_name(f"{Path(output).stem}_{i:04d}.png"))
            proc.stdin.write(np.asarray(image).tobytes())
    finally:
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    Path(output).with_suffix(".json").write_text(
        json.dumps(dict(sources=[str(Path(r).resolve()) for r in runs], frames=len(frames), fps=fps,
                        knife="prescribed Fig. 5 protocol path", revision=git_revision()), indent=2)
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["fig3", "fig5", "fig6"])
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="")
    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "fig5":
        fig5_video(args.runs, args.output)
    else:
        free_knife_video(args.kind, args.runs[0], args.output, args.title)
