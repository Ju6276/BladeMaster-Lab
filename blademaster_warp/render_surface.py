"""Read-only saved-surface replay in the earlier light-wood demo presentation.

No subdivision, smoothing, pre-cut skin, generated motion, or resimulation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from .replay_style import ReplayScene, caption
from .provenance import git_revision


def render(directory, output=None, view='compare', preview=False,surface_directory=None):
    directory = Path(directory)
    output = Path(output) if output else directory/'surface_replay.mp4'
    output.parent.mkdir(parents=True,exist_ok=True)
    surface_directory=Path(surface_directory) if surface_directory else directory/'surface'
    meta = json.loads((directory/'manifest.json').read_text())
    cfg = meta['config']
    report = json.loads((directory/'report.json').read_text()) if (directory/'report.json').exists() else meta
    if report['status'] != 'completed':
        raise ValueError('Failed run: refuse an unlabelled completed replay')
    frames = json.loads((surface_directory/'frames.json').read_text())
    state_path = directory/('states.npz' if (directory/'states.npz').exists() else 'trajectory.npz')
    z = np.load(state_path)
    is_robot = state_path.name == 'trajectory.npz'
    positions = z['particles' if is_robot else 'position']
    codes = z['codes' if is_robot else 'sides']
    times = z['energy'][:,0] if is_robot else z['time']
    forces = z['log'][:,[0,7,8,9]] if is_robot else z['forces'][:,:4] if 'forces' in z else None
    audit_path = directory/'material_geometry_validation.json'
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else None
    geometry = ('FOLDED' if audit['status']=='sampled_material_map_folded' else 'no sampled fold') if audit else 'not checked'
    surface_check=directory/'surface_validation.json' if surface_directory==directory/'surface' else surface_directory/'validation.json'
    surface_ok=json.loads(surface_check.read_text()).get('final_component_counts_match') if surface_check.exists() else None
    audit_line = (f'Final geometry audit: {geometry}  |  Synthetic geometry; forces not calibrated' if surface_ok is not False
                  else 'SURFACE TOPOLOGY MISMATCH  |  Mechanical and display pieces differ')
    board = dict(size_mm=[170,140,3],center_mm=[0,0,-1.5],source='display tile for infinite MPM plane')
    if is_robot and (directory/'scene.xml').exists():
        geom = ET.parse(directory/'scene.xml').find(".//geom[@name='board']")
        board = dict(size_mm=(np.fromstring(geom.get('size'),sep=' ')*2000).tolist(),
                     center_mm=((np.fromstring(geom.get('pos'),sep=' ')-z['origin'])*1000).tolist(),
                     source='recorded MuJoCo scene.xml board geometry')
    scene = ReplayScene(cfg,board)
    names = ['Surface','Particles'] if view=='compare' else [view.title()]
    title = ('Banana' if cfg['shape']=='banana' else 'Material') + (' / Franka single cut' if is_robot else ' / Cutting replay')
    duration = float(frames[-1]['time_s'])
    fps = 1/np.median(np.diff([f['time_s'] for f in frames])) if len(frames)>1 else 25.
    selected = sorted({0,len(frames)//2,len(frames)-1}) if preview else range(len(frames))
    hashes = {state_path.name:hashlib.sha256(state_path.read_bytes()).hexdigest()}
    surface_hash = hashlib.sha256()
    proc = None
    rendered = 0
    try:
        for fi in selected:
            info = frames[fi]
            snap_path = surface_directory/f'{fi:05d}.npz'
            surface_hash.update(snap_path.read_bytes())
            with np.load(snap_path) as snap:
                idx = int(np.argmin(abs(times-info['time_s'])))
                scene.update(snap,positions[idx],codes[idx])
            now = float(info['time_s'])
            force = forces[np.argmin(abs(forces[:,0]-now)),1:] if forces is not None else None
            phase = 'Recorded state'
            if is_robot and z['cycle_trace'].size:
                trace = z['cycle_trace']
                phase = ('Settle','Descend','Withdraw','Verify','Done')[int(trace[np.argmin(abs(trace[:,0]-now)),1])]
            panels = [caption(scene.frame(name=='Particles'),title,name,phase,now,duration,force,audit_line) for name in names]
            image = np.concatenate(panels,axis=0)
            if fi in (0,len(frames)//2,len(frames)-1):
                Image.fromarray(image).save(output.with_name(f'{output.stem}_{fi:04d}.png'))
            if not preview:
                if proc is None:
                    height,width = image.shape[:2]
                    proc = subprocess.Popen(['ffmpeg','-y','-v','error','-f','rawvideo','-pixel_format','rgb24',
                        '-video_size',f'{width}x{height}','-framerate',str(fps),'-i','-',
                        '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)],stdin=subprocess.PIPE)
                proc.stdin.write(image.tobytes())
            rendered += 1
        if proc:
            proc.stdin.close()
            if proc.wait()!=0:
                raise RuntimeError('Video encoding failed')
    finally:
        scene.window.Finalize()
        z.close()
        if proc and proc.poll() is None:
            proc.kill()
    hashes['surface_snapshots_in_render_order'] = surface_hash.hexdigest()
    if hashes[state_path.name] != hashlib.sha256(state_path.read_bytes()).hexdigest():
        raise RuntimeError('Physical state changed during rendering')
    output.with_suffix('.json').write_text(json.dumps(dict(
        kind='read-only replay preview' if preview else 'read-only replay',source=str(directory.resolve()),
        input_sha256=hashes,frames=rendered,fps=float(fps),view=view,board=board,
        surface_directory=str(surface_directory.resolve()),
        style_reference='DexLab b955d85 cutting_showcase: palette, wood texture, lighting and caption styling only',
        geometry='saved linked-volume vertices and triangles; no smoothing, subdivision or pre-cut skin',
        normals='45-degree feature-angle vertex normals for lighting only',
        particle_colors='saved persistent cut-history codes; not inferred piece count',
        force='recorded interval-average reaction on tool; material Fx/Fz have opposite sign',
        limits='Normal shading and material color cannot repair coarse geometry; final geometry audit is not a force validation',
        revision=git_revision()
    ),indent=2))
    print(output if not preview else output.with_name(f'{output.stem}_{selected[-1]:04d}.png'))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory')
    p.add_argument('--output',type=Path)
    p.add_argument('--view',choices=['compare','surface','particles'],default='compare')
    p.add_argument('--preview',action='store_true',help='Render first, middle and final PNGs only')
    p.add_argument('--surface-directory',type=Path,help='Use a separately reconstructed surface; original physical states stay unchanged')
    args = p.parse_args()
    render(args.directory,args.output,args.view,args.preview,args.surface_directory)
