"""Render the recorded MuJoCo visuals in the material replay's mm coordinates.

Only forward kinematics is evaluated. No physics steps or authored robot motion.
Compiled mesh vertices already include MuJoCo's mesh centering; geom_xmat/xpos
provide the matching transforms. Collision proxies are not drawn twice.
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
import vtk
from vtk.util.numpy_support import numpy_to_vtk
from .franka import SOURCE
from .render import actor


def mesh_polydata(model, mesh_id):
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    # Keep upstream corner normals. Recomputing them on MuJoCo's expanded
    # mesh would turn smooth robot shells into separately shaded triangles.
    na, nn = model.mesh_normaladr[mesh_id], model.mesh_normalnum[mesh_id]
    triangle_vertices = model.mesh_face[fa:fa+fn].ravel()
    triangle_normals = model.mesh_facenormal[fa:fa+fn].ravel()
    points = vtk.vtkPoints()
    points.SetData(numpy_to_vtk(model.mesh_vert[va:va+vn][triangle_vertices]*1000, deep=True))
    faces = vtk.vtkCellArray()
    for tri in np.arange(fn*3).reshape(-1,3):
        faces.InsertNextCell(3)
        for vertex in tri:
            faces.InsertCellPoint(int(vertex))
    poly = vtk.vtkPolyData()
    poly.SetPoints(points)
    poly.SetPolys(faces)
    poly.GetPointData().SetNormals(numpy_to_vtk(model.mesh_normal[na:na+nn][triangle_normals],deep=True))
    return poly


class RobotVisuals:
    def __init__(self, scene, xml_path, origin):
        root = ET.parse(xml_path).getroot()
        root.find('compiler').set('meshdir', str(SOURCE.parent/'assets'))
        self.model = mj.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mj.MjData(self.model)
        self.origin = np.asarray(origin)
        self.actors = {}
        self.meshes = {}
        for gid in range(self.model.ngeom):
            name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_GEOM, gid)
            # Board and blade are already drawn from recorded geometry by the
            # shared material scene; floor remains the neutral background.
            if self.model.geom_group[gid] == 3 or name in ('board', 'floor', 'knife'):
                continue
            kind = self.model.geom_type[gid]
            if kind == mj.mjtGeom.mjGEOM_MESH:
                mid = self.model.geom_dataid[gid]
                if mid not in self.meshes:
                    self.meshes[mid] = mesh_polydata(self.model, mid)
                src = self.meshes[mid]
            elif kind == mj.mjtGeom.mjGEOM_CYLINDER:
                cylinder = vtk.vtkCylinderSource()
                cylinder.SetRadius(self.model.geom_size[gid,0]*1000)
                cylinder.SetHeight(self.model.geom_size[gid,1]*2000)
                cylinder.SetResolution(48)
                rotation = vtk.vtkTransform()
                rotation.RotateX(90)  # VTK cylinder Y -> MuJoCo cylinder Z.
                src = vtk.vtkTransformPolyDataFilter()
                src.SetInputConnection(cylinder.GetOutputPort())
                src.SetTransform(rotation)
            elif kind == mj.mjtGeom.mjGEOM_BOX:
                src = vtk.vtkCubeSource()
                src.SetXLength(self.model.geom_size[gid,0]*2000)
                src.SetYLength(self.model.geom_size[gid,1]*2000)
                src.SetZLength(self.model.geom_size[gid,2]*2000)
            else:
                raise ValueError(f'Unsupported recorded visual geometry: {name}, type {kind}')
            mat = self.model.geom_matid[gid]
            color = self.model.mat_rgba[mat,:3] if mat >= 0 else self.model.geom_rgba[gid,:3]
            visual = actor(src, scene.renderer, color)
            visual.GetProperty().SetAmbient(.40)
            visual.GetProperty().SetDiffuse(.60)
            visual.GetProperty().SetSpecular(.12)
            visual.GetProperty().SetSpecularPower(32)
            self.actors[gid] = visual

    def update(self, qpos, snapshot):
        self.data.qpos[:] = qpos
        mj.mj_forward(self.model, self.data)
        for gid, visual in self.actors.items():
            matrix = vtk.vtkMatrix4x4()
            rotation = self.data.geom_xmat[gid].reshape(3,3)
            position = (self.data.geom_xpos[gid]-self.origin)*1000
            for row in range(3):
                for col in range(3):
                    matrix.SetElement(row,col,float(rotation[row,col]))
                matrix.SetElement(row,3,float(position[row]))
            visual.SetUserMatrix(matrix)
        # Refuse a visually detached knife instead of hiding timestamp/frame
        # mismatches with an invented mount or independently animated blade.
        from scipy.spatial.transform import Rotation
        blade = self.model.body('knife').id
        position_error = np.linalg.norm(self.data.xpos[blade]-self.origin-snapshot['tool_position'])
        rotation_error = np.linalg.norm(self.data.xmat[blade].reshape(3,3)-Rotation.from_quat(snapshot['tool_orientation']).as_matrix())
        if position_error > 1.e-6 or rotation_error > 1.e-5:
            raise ValueError(f'Robot/surface pose mismatch: {position_error} m, {rotation_error} rotation norm')
        return float(position_error),float(rotation_error)
