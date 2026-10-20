"""Light-wood scene style for offline rendering (adapted from DexLab cutting_showcase, Apache-2.0)."""

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from PIL import Image, ImageDraw, ImageFont
from .render import actor, blade_mesh


def wood_texture():
    y, x = np.mgrid[:512, :1024]
    rng = np.random.default_rng(197)
    grain = (
        0.012 * np.sin(0.13 * y + 2.8 * np.sin(x / 240))
        + 0.009 * np.sin(1.7 * y + 4 * np.sin(x / 330))
        + 0.008 * rng.normal(size=x.shape)
    )
    planks = 0.008 * (y // 128 % 3) - 0.012 * (y % 128 < 1)
    rgb = np.clip(np.array([0.81, 0.71, 0.55]) + (grain + planks)[:, :, None], 0, 1)
    data = vtk.vtkImageData()
    data.SetDimensions(1024, 512, 1)
    data.GetPointData().SetScalars(numpy_to_vtk((rgb * 255).astype(np.uint8).reshape(-1, 3), deep=True))
    texture = vtk.vtkTexture()
    texture.SetInputData(data)
    texture.InterpolateOn()
    return texture


# Key, fill and back light, designed for a camera at azimuth atan2(-220, 110)
# about the material origin.
LIGHTS = [((-80, -90, 220), 0.8), ((180, -30, 120), 0.5), ((-60, 130, 90), 0.35)]
LIGHT_CAMERA_AZIMUTH = np.arctan2(-220, 110)


def place_lights(renderer, camera_azimuth):
    """Turn the light rig about the vertical with the camera, so the faces it sees stay lit."""
    c, s = np.cos(camera_azimuth - LIGHT_CAMERA_AZIMUTH), np.sin(camera_azimuth - LIGHT_CAMERA_AZIMUTH)
    renderer.RemoveAllLights()
    for (x, y, z), intensity in LIGHTS:
        light = vtk.vtkLight()
        light.SetLightTypeToSceneLight()
        light.SetPosition(c * x - s * y, s * x + c * y, z)
        light.SetFocalPoint(0, 0, 0)
        light.SetIntensity(intensity)
        renderer.AddLight(light)
    # The rig is mostly overhead; a headlight keeps the side faces toward the camera lit.
    headlight = vtk.vtkLight()
    headlight.SetLightTypeToHeadlight()
    headlight.SetIntensity(0.45)
    renderer.AddLight(headlight)


class ReplayScene:
    def __init__(self, cfg, board, width=960, height=480, style="wood"):
        if style not in ("wood", "paper"):
            raise ValueError("Unknown replay style")
        self.width, self.height = width, height
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.91, 0.93, 0.94)
        self.renderer.SetBackground2(0.985, 0.985, 0.98)
        self.renderer.GradientBackgroundOn()
        self.window = vtk.vtkEGLRenderWindow()
        self.window.SetOffScreenRendering(1)
        self.window.SetSize(width, height)
        self.window.SetMultiSamples(0)
        self.window.AddRenderer(self.renderer)
        ao = vtk.vtkSSAOPass()
        ao.SetDelegatePass(vtk.vtkRenderStepsPass())
        ao.SetRadius(2.5)
        ao.SetBias(0.15)
        ao.SetKernelSize(64)
        ao.BlurOn()
        self.renderer.SetPass(ao)
        place_lights(self.renderer, LIGHT_CAMERA_AZIMUTH)
        source = vtk.vtkCubeSource()
        source.SetXLength(board["size_mm"][0])
        source.SetYLength(board["size_mm"][1])
        source.SetZLength(board["size_mm"][2])
        source.SetCenter(*board["center_mm"])
        wood = actor(source, self.renderer, (1.0, 1.0, 1.0))
        uv = vtk.vtkTextureMapToPlane()
        uv.SetInputConnection(source.GetOutputPort())
        uv.SetOrigin(-85, -70, 0)
        uv.SetPoint1(85, -70, 0)
        uv.SetPoint2(-85, 70, 0)
        wood.GetMapper().SetInputConnection(uv.GetOutputPort())
        wood.SetTexture(wood_texture())
        wood.GetProperty().SetAmbient(0.35)
        wood.GetProperty().SetDiffuse(0.6)
        wood.GetProperty().SetSpecular(0.08)
        wood.GetProperty().SetSpecularPower(20)
        if style == "paper":
            # Presentation only: Fig. 1's pale background and green cutting
            # board. The board retains its recorded dimensions and pose.
            wood.SetTexture(None)
            wood.GetProperty().SetColor(0.66, 0.74, 0.62)
            self.renderer.SetBackground(0.87, 0.90, 0.92)
            self.renderer.SetBackground2(0.96, 0.97, 0.98)
            # Avoid the grain/striping of close-range SSAO on the metre-scale
            # robot overview. Lighting is shared by both replay panels.
            self.renderer.SetPass(None)
            self.window.SetMultiSamples(4)
        self.surface = vtk.vtkPolyData()
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(self.surface)
        normals.SetFeatureAngle(45)
        normals.SplittingOn()
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        self.body = actor(normals, self.renderer, (0.985, 0.95, 0.78))
        self.material_color = None
        self.body.GetProperty().SetAmbient(0.38)
        self.body.GetProperty().SetDiffuse(0.62)
        self.body.GetProperty().SetSpecular(0.12)
        self.body.GetProperty().SetSpecularPower(35)
        self.cloud = vtk.vtkPolyData()
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(cfg["grid_spacing"] * 250)
        sphere.SetThetaResolution(16)
        sphere.SetPhiResolution(12)
        mapper = vtk.vtkGlyph3DMapper()
        mapper.SetInputData(self.cloud)
        mapper.SetSourceConnection(sphere.GetOutputPort())
        mapper.ScalingOff()
        self.particles = vtk.vtkActor()
        self.particles.SetMapper(mapper)
        self.particles.GetProperty().SetAmbient(0.52)
        self.particles.GetProperty().SetDiffuse(0.48)
        self.renderer.AddActor(self.particles)
        self.blade = actor(
            blade_mesh(cfg["blade_half_size"], cfg["blade_bevel_height"]), self.renderer, (0.66, 0.72, 0.76)
        )
        self.blade.GetProperty().SetAmbient(0.3)
        self.blade.GetProperty().SetSpecular(0.95)
        self.blade.GetProperty().SetSpecularPower(90)
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(110, -220, 175)
        camera.SetFocalPoint(0, 0, 24)
        camera.SetViewUp(0, 0, 1)
        camera.ParallelProjectionOn()
        camera.SetParallelScale(57)
        self.capture = vtk.vtkWindowToImageFilter()
        self.capture.SetInput(self.window)
        self.capture.SetInputBufferTypeToRGB()
        self.capture.ReadFrontBufferOff()

    def set_material_color(self, rgb):
        """Paint the surface and every particle one color. The cut stays the opening."""
        self.material_color = tuple(float(channel) for channel in rgb)
        self.body.GetProperty().SetColor(*self.material_color)

    def set_material_layers(self, reference_particles, boundary_z, below_rgb, above_rgb):
        """Color by reference height (m), e.g. a two-layer cake. Follows material, not world height."""
        self.layers = (float(boundary_z), np.asarray(below_rgb, float), np.asarray(above_rgb, float))
        self.particle_reference = np.asarray(reference_particles)
        self.material_color = tuple(self.layers[1])

    def _layer_rgb(self, reference):
        boundary, below, above = self.layers
        return np.where((reference[:, 2] > boundary)[:, None], above, below)

    def set_camera(self, position, focal, view_angle):
        camera = self.renderer.GetActiveCamera()
        camera.ParallelProjectionOff()
        camera.SetViewAngle(view_angle)
        camera.SetPosition(*position)
        camera.SetFocalPoint(*focal)
        camera.SetViewUp(0, 0, 1)
        place_lights(self.renderer, np.arctan2(*(np.subtract(position, focal)[1::-1])))
        self.renderer.ResetCameraClippingRange()

    def update(self, snapshot, positions, codes):
        from scipy.spatial.transform import Rotation

        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(snapshot["vertices"] * 1000, deep=True))
        cells = vtk.vtkCellArray()
        for tri in snapshot["triangles"]:
            cells.InsertNextCell(3)
            for v in tri:
                cells.InsertCellPoint(int(v))
        self.surface.SetPoints(points)
        self.surface.SetPolys(cells)
        if getattr(self, "layers", None) is not None:
            self.surface.GetPointData().SetScalars(
                numpy_to_vtk((self._layer_rgb(snapshot["reference"]) * 255).astype(np.uint8), deep=True)
            )
        self.surface.Modified()
        pts = vtk.vtkPoints()
        pts.SetData(numpy_to_vtk(positions * 1000, deep=True))
        self.cloud.SetPoints(pts)
        if self.material_color is None:
            palette = np.array(
                [
                    [0.90, 0.76, 0.47],
                    [0.24, 0.58, 0.70],
                    [0.95, 0.58, 0.31],
                    [0.51, 0.70, 0.47],
                    [0.68, 0.52, 0.76],
                ]
            )
            rgb = palette[codes % len(palette)]
        elif getattr(self, "layers", None) is not None:
            rgb = self._layer_rgb(self.particle_reference)
        else:
            rgb = np.tile(self.material_color, (len(codes), 1))
        self.cloud.GetPointData().SetScalars(
            numpy_to_vtk((rgb * 255).astype(np.uint8), deep=True)
        )
        self.cloud.Modified()
        r = Rotation.from_quat(snapshot["tool_orientation"]).as_matrix()
        matrix = vtk.vtkMatrix4x4()
        for row in range(3):
            for col in range(3):
                matrix.SetElement(row, col, float(r[row, col]))
            matrix.SetElement(row, 3, float(snapshot["tool_position"][row] * 1000))
        self.blade.SetUserMatrix(matrix)
        self.renderer.ResetCameraClippingRange()

    def frame(self, particles=False):
        self.body.SetVisibility(not particles)
        self.particles.SetVisibility(particles)
        self.window.Render()
        self.capture.Modified()
        self.capture.Update()
        return (
            vtk_to_numpy(self.capture.GetOutput().GetPointData().GetScalars())
            .reshape(self.height, self.width, 3)[::-1]
            .copy()
        )


def caption(rgb, title, view, phase, now, duration, force, audit):
    header, footer = 76, 114
    image = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + header + footer), (248, 250, 248))
    image.paste(Image.fromarray(rgb), (0, header))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 28)
    label = ImageFont.truetype("DejaVuSans.ttf", 19)
    small = ImageFont.truetype("DejaVuSans.ttf", 16)
    ink, secondary = (31, 49, 48), (83, 104, 101)
    draw.text((24, 9), title, font=font, fill=ink)
    draw.text((24, 45), f"{view}  |  {phase}", font=label, fill=secondary)
    draw.text((image.width - 150, 9), f"{now:.2f} s", font=font, fill=ink)
    top = header + rgb.shape[0]
    draw.line((24, top, image.width - 24, top), fill=(188, 205, 198), width=2)
    values = (
        ["--"] * 3
        if force is None
        else [f"{np.linalg.norm(force):.3f} N", f"{-force[0]:+.3f} N", f"{-force[2]:+.3f} N"]
    )
    for column, (name, value) in enumerate(
        zip(("Knife contact |F|", "On material Fx", "On material Fz"), values)
    ):
        x = 24 + column * (image.width - 48) // 3
        draw.text((x, top + 9), name, font=label, fill=secondary)
        draw.text((x, top + 33), value, font=font, fill=ink)
    draw.text((24, top + 76), audit, font=small, fill=(165, 57, 40) if "FOLDED" in audit else secondary)
    draw.line((24, image.height - 6, image.width - 24, image.height - 6), fill=(188, 205, 198), width=3)
    draw.line(
        (24, image.height - 6, 24 + (image.width - 48) * now / duration, image.height - 6),
        fill=(46, 101, 87),
        width=3,
    )
    return np.asarray(image)
