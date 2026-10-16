"""Render recorded particle positions only, without a pre-cut decorative mesh.

Uses VTK (offscreen EGL) and ffmpeg. This is replay, not a live physics engine.
"""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from scipy.spatial.transform import Rotation
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def actor(data, renderer, color):
    mapper=vtk.vtkPolyDataMapper()
    if isinstance(data,vtk.vtkPolyData):mapper.SetInputData(data)
    else:mapper.SetInputConnection(data.GetOutputPort())
    a=vtk.vtkActor();a.SetMapper(mapper);a.GetProperty().SetColor(*color)
    renderer.AddActor(a);return a


def blade_mesh(half,bevel):
    x,y,z=np.asarray(half)*1000;bevel*=1000
    if bevel==0:
        src=vtk.vtkCubeSource();src.SetXLength(2*x);src.SetYLength(2*y);src.SetZLength(2*z)
        return src
    poly=np.array([[0,-z],[x,-z+bevel],[x,z],[-x,z],[-x,-z+bevel]])
    vertices=np.array([[u,v,w] for v in [-y,y] for u,w in poly])
    points=vtk.vtkPoints();points.SetData(numpy_to_vtk(vertices,deep=True))
    faces=vtk.vtkCellArray()
    for face in [list(range(4,-1,-1)),list(range(5,10))]+[[i,(i+1)%5,(i+1)%5+5,i+5] for i in range(5)]:
        faces.InsertNextCell(len(face))
        for i in face:faces.InsertCellPoint(i)
    mesh=vtk.vtkPolyData();mesh.SetPoints(points);mesh.SetPolys(faces);return mesh


def render(directory,output,cpu_directory=None):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Refuse completed replay of failed run')
    states=np.load(directory/'states.npz');cfg=manifest['config']
    sources=[('Warp GPU | independent implementation',states,None)]
    if cpu_directory:
        path=Path(cpu_directory);cs=np.load(path/'states.npz');ct=np.load(path/'trace.npz')
        sources.insert(0,('DexLab CPU | third-party baseline',cs,ct))
    width,height=640*len(sources),600
    window=vtk.vtkEGLRenderWindow();window.SetOffScreenRendering(1);window.SetSize(width,height)
    panels=[]
    for i,(title,data,trace) in enumerate(sources):
        renderer=vtk.vtkRenderer();renderer.SetViewport(i/len(sources),0,(i+1)/len(sources),1)
        renderer.SetBackground(.96,.97,.98);window.AddRenderer(renderer)
        board=vtk.vtkCubeSource();board.SetXLength(170);board.SetYLength(105);board.SetZLength(3);board.SetCenter(0,0,-1.5)
        actor(board,renderer,(.66,.48,.29))
        pts=vtk.vtkPoints();pts.SetData(numpy_to_vtk(data['position'][0]*1000,deep=True))
        cloud=vtk.vtkPolyData();cloud.SetPoints(pts)
        # Color derives from reference x, not future cut labels or assumed pieces.
        ref=data['reference'];u=(ref[:,0]-ref[:,0].min())/np.ptp(ref[:,0])
        rgb=np.column_stack([.15+.75*u,.65-.12*u,.8-.65*u])
        cloud.GetPointData().SetScalars(numpy_to_vtk((rgb*255).astype(np.uint8),deep=True))
        sphere=vtk.vtkSphereSource();sphere.SetRadius(cfg['grid_spacing']*250);sphere.SetThetaResolution(9);sphere.SetPhiResolution(8)
        mapper=vtk.vtkGlyph3DMapper();mapper.SetInputData(cloud);mapper.SetSourceConnection(sphere.GetOutputPort());mapper.ScalingOff()
        a=vtk.vtkActor();a.SetMapper(mapper);a.GetProperty().SetAmbient(.3);renderer.AddActor(a)
        blade=actor(blade_mesh(cfg['blade_half_size'],cfg['blade_bevel_height']),renderer,(.42,.46,.52))
        camera=renderer.GetActiveCamera();camera.SetPosition(115,-220,165);camera.SetFocalPoint(0,0,20);camera.SetViewUp(0,0,1)
        camera.ParallelProjectionOn();camera.SetParallelScale(70)
        label=vtk.vtkTextActor();label.SetPosition(16,555);label.GetTextProperty().SetFontSize(18);label.GetTextProperty().SetColor(.1,.15,.2);renderer.AddViewProp(label)
        detail=vtk.vtkTextActor();detail.SetPosition(16,20);detail.GetTextProperty().SetFontSize(15);detail.GetTextProperty().SetColor(.1,.15,.2);renderer.AddViewProp(detail)
        panels.append((title,data,trace,pts,blade,label,detail,renderer))
    capture=vtk.vtkWindowToImageFilter();capture.SetInput(window);capture.SetInputBufferTypeToRGB();capture.ReadFrontBufferOff()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
             '-framerate','25','-i','-','-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    proc=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for frame in range(0,len(states['time']),4):
            t=float(states['time'][frame])
            for title,data,trace,pts,blade,label,detail,renderer in panels:
                idx=int(np.argmin(abs(data['time']-t)))
                pts.SetData(numpy_to_vtk(data['position'][idx]*1000,deep=True));pts.Modified()
                if trace is None:
                    pos=data['blade_position'][idx,0]
                else:
                    ti=int(np.argmin(abs(trace['time_s']-t)));pos=trace['blade_position_m'][ti]
                matrix=vtk.vtkMatrix4x4()
                rotation=Rotation.from_quat(data['blade_orientation'][idx,0]).as_matrix() if trace is None else np.eye(3)
                for row in range(3):
                    for col in range(3):matrix.SetElement(row,col,rotation[row,col])
                    matrix.SetElement(row,3,float(pos[row]*1000))
                blade.SetUserMatrix(matrix)
                label.SetInput(f'{title}\n{manifest["preset"]} | {t:.2f} s | {len(data["reference"]):,} particles')
                detail.SetInput(f'E={cfg["young_modulus"]/1000:g} kPa | Gc={cfg["cutting_energy"]:g} J/m2 | dt={cfg["dt"]*1e6:g} us\n'
                                'Physical particles; color = reference x | no pre-cut mesh')
                renderer.ResetCameraClippingRange()
            window.Render();capture.Modified();capture.Update()
            pixels=vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars()).reshape(height,width,3)[::-1].copy()
            proc.stdin.write(pixels.tobytes())
        proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError('ffmpeg encoding failed')
    finally:
        window.Finalize()
        if proc.poll() is None:proc.kill()
    print(output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');p.add_argument('--output',required=True);p.add_argument('--cpu-directory')
    a=p.parse_args();render(a.directory,a.output,a.cpu_directory)
