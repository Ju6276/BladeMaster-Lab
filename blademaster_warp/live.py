"""Live browser display: serves the current robot, surface and particle state over HTTP.

Binds localhost; use an SSH tunnel for remote access."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Event, Lock, Thread
from urllib.parse import urlparse, parse_qs
import gzip
import json
import mimetypes
import time
import numpy as np

WEB = Path(__file__).with_name("web")


def scene_asset(robot):
    import mujoco as mj

    m = robot.model
    meshes, geoms = {}, []
    for gid in range(m.ngeom):
        name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, gid)
        if m.geom_group[gid] == 3 or name == "floor":
            continue  # Hidden collision proxies, as in the upstream visuals.
        kind = int(m.geom_type[gid])
        entry = dict(id=gid, name=name, kind=kind, size=m.geom_size[gid].tolist())
        mat = m.geom_matid[gid]
        entry["rgba"] = (m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[gid]).tolist()
        if kind == int(mj.mjtGeom.mjGEOM_MESH):
            mid = int(m.geom_dataid[gid])
            entry["mesh"] = mid
            if mid not in meshes:
                va, vn = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
                fa, fn = m.mesh_faceadr[mid], m.mesh_facenum[mid]
                na, nn = m.mesh_normaladr[mid], m.mesh_normalnum[mid]
                # Preserve upstream face-corner normals, rather than inventing CAD.
                vertices = m.mesh_vert[va : va + vn][m.mesh_face[fa : fa + fn].ravel()]
                normals = m.mesh_normal[na : na + nn][m.mesh_facenormal[fa : fa + fn].ravel()]
                meshes[mid] = dict(vertices=vertices.ravel().tolist(), normals=normals.ravel().tolist())
        elif kind not in (2, 3, 4, 5, 6):
            raise ValueError(f"Unsupported live visual {name}: {kind}")
        geoms.append(entry)
    return dict(
        meshes=meshes,
        geoms=geoms,
        origin=robot.material_origin.tolist(),
        geometry_source="Actual compiled MuJoCo visuals; material surface reconstructed from current particles",
    )


class LiveState:
    def __init__(self):
        self.lock = Lock()
        self.play, self.restart = Event(), Event()
        self.stop_requested = False
        self.sequence, self.generation = 0, int(time.time() * 1.0e6)
        self.asset = None
        self.snapshot = dict(state="loading", sequence=0)
        self.autostart = False
        self.can_restart = True

    def initialize(self, robot, solver, surface):
        setup_begin = time.perf_counter()
        asset = scene_asset(robot)
        self.ids = [g["id"] for g in asset["geoms"]]
        self.generation += 1
        asset["generation"] = self.generation
        self.asset = asset
        self.stop_requested = False
        self.play.clear()
        if solver:
            x, F, codes = solver.x.numpy(), solver.F.numpy(), solver.codes.numpy()
            surface.update(0.0, x, F, codes, robot.state(), cutting=False, record_frame=False)
        else:
            x = codes = None
        self.publish(robot, solver, surface, x, codes, 0.0, "ready", np.zeros(6), state="ready")
        if self.autostart:
            self.play.set()
            self.autostart = False
        self.setup_s = time.perf_counter() - setup_begin
        self.play.wait()  # Startup wait is outside the physics timing.

    def before_step(self):
        if self.play.is_set():
            return 0.0
        begin = time.perf_counter()
        self.play.wait()
        return time.perf_counter() - begin

    def publish(self, robot, solver, surface, x, codes, wall_s, phase, wrench, state="running"):
        ids = self.ids
        if state == "running" and not self.play.is_set() and not self.stop_requested:
            state = "paused"
        packet = dict(
            state=state,
            generation=self.generation,
            sim_s=float(robot.data.time),
            wall_s=float(wall_s),
            ratio=float(robot.data.time / wall_s) if wall_s > 0 else 0.0,
            phase=phase,
            wrench=np.asarray(wrench).tolist(),
            geom_positions=robot.data.geom_xpos[ids].tolist(),
            geom_rotations=robot.data.geom_xmat[ids].tolist(),
            particles=(x + robot.material_origin).ravel().tolist() if x is not None else [],
            codes=codes.tolist() if codes is not None else [],
            radius=float(solver.radius) if solver else 0.0,
        )
        if surface and surface.current:
            packet["vertices"] = (surface.current["vertices"] + robot.material_origin).ravel().tolist()
            packet["triangles"] = surface.current["triangles"].ravel().tolist()
        self._set(packet)

    def _set(self, packet):
        with self.lock:
            self.sequence += 1
            self.snapshot = dict(packet, sequence=self.sequence)

    def finish(self, report):
        self._set(dict(self.snapshot, state=report["status"], report=report, can_restart=self.can_restart))

    def command(self, command):
        if command == "start":
            state = self.snapshot["state"]
            if state in ("completed", "stopped", "failed"):
                if not self.can_restart:
                    raise ValueError(
                        "Choose a new --output directory in a new process before recording again"
                    )
                self.autostart = True
                self._set(dict(self.snapshot, state="loading"))
                self.restart.set()
            else:
                self.play.set()
                self._set(dict(self.snapshot, state="running"))
        elif command == "pause":
            self.play.clear()
            self._set(dict(self.snapshot, state="paused"))
        elif command == "stop":
            self.stop_requested = True
            self.play.set()
        else:
            raise ValueError("Unknown command")


def make_server(live, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body, content_type, status=200):
            body = gzip.compress(body, compresslevel=1) if body else b""
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            url = urlparse(self.path)
            if url.path == "/state":
                with live.lock:
                    packet = live.snapshot
                if str(packet["sequence"]) == parse_qs(url.query).get("after", [""])[0]:
                    self.send(b"", "application/json", 204)
                else:
                    self.send(json.dumps(packet, allow_nan=False).encode(), "application/json")
            elif url.path == "/scene":
                self.send(json.dumps(live.asset).encode(), "application/json")
            else:
                relative = "index.html" if url.path == "/" else url.path.lstrip("/")
                file = (WEB / relative).resolve()
                if not file.is_relative_to(WEB.resolve()) or not file.is_file():
                    self.send(b"Not found", "text/plain", 404)
                    return
                self.send(file.read_bytes(), mimetypes.guess_type(file)[0] or "application/octet-stream")

        def do_POST(self):
            # Reject cross-origin browser commands. The server is localhost-only.
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc != self.headers.get("Host"):
                self.send(b"Origin rejected", "text/plain", 403)
                return
            try:
                if not self.path.startswith("/command/"):
                    raise ValueError("Unknown endpoint")
                live.command(self.path.removeprefix("/command/"))
                self.send(b"{}", "application/json")
            except ValueError as error:
                self.send(str(error).encode(), "text/plain", 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    return server


def serve(args, run):
    live = LiveState()
    live.can_restart = not bool(args.output)
    server = make_server(live, args.port)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        f"LIVE http://127.0.0.1:{server.server_port} — open browser, then Start. No video file is used.",
        flush=True,
    )
    try:
        while True:
            try:
                run(args, observer=live)
            except Exception as error:
                live.finish(dict(status="failed", exception=repr(error)))
                print(repr(error), flush=True)
            live.restart.wait()
            live.restart.clear()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
