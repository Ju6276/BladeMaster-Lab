"""BladeMaster vs a TopoCut-style single-field MLS-MPM under the same prescribed knife.

Both solvers take the scene's Config and the same ToolState every coupling step.
BladeMaster pieces are its mechanical components; TopoCut pieces come from SDF
topology discovery after each cut action (App. 7.6). Both are scored with the
spectral reward against the intended pieces in the reference configuration.

python -m scripts.topocut_compare --scenario fig8 --output runs/topocut_fig8
python -m scripts.topocut_compare --plot runs/topocut_fig8 --output fig.png
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time
import numpy as np
from blademaster_warp import spectral
from blademaster_warp.coupling import ToolState
from blademaster_warp.provenance import git_revision
from blademaster_warp.scenarios import FIG4_CUT_ANGLES_DEG, FIG8_CUTS, FIG8_SIZE, FIG8_SLICE
from blademaster_warp.scenarios import command_at, keyframes, scene_config


def goal_labels(name, reference):
    """Intended pieces in the reference configuration."""
    if name == "fig8":
        planes = FIG8_SIZE[0] / 2 - FIG8_SLICE * np.arange(1, FIG8_CUTS + 1)
        return np.searchsorted(-planes, -reference[:, 0])
    if name == "fig4":
        angle = np.degrees(np.arctan2(reference[:, 1], reference[:, 0]))
        return (np.abs(angle) < max(FIG4_CUT_ANGLES_DEG)).astype(int)
    raise ValueError(name)


def tilt_deg(reference, current):
    """Angle between the reference vertical and its image under the best-fit rotation."""
    a, b = reference - reference.mean(0), current - current.mean(0)
    u, _, vt = np.linalg.svd(a.T @ b)
    r = (u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt).T
    return float(np.degrees(np.arccos(np.clip(r[2, 2], -1, 1))))


class BladeMasterRun:
    def __init__(self, cfg):
        from blademaster_warp.coupling import ExternalToolPort
        from blademaster_warp.solver import Solver

        self.solver = Solver(cfg)
        self.port = ExternalToolPort(self.solver)
        self.reference = self.solver.reference

    def advance(self, tool, interval, active, index):
        return self.port.advance(tool, interval, cutting=active, cut_index=index)

    def positions(self):
        return self.solver.x.numpy()

    def end_of_cut(self, poses):
        pass

    def labels(self):
        from blademaster_warp.topology import components

        return components(self.solver.node_numpy, self.solver.weight_numpy, self.solver.codes.numpy())[0]


    def settled(self):
        pass


class TopoCutRun:
    def __init__(self, cfg, damaged_shear=0.0):
        from blademaster_warp.mlsmpm import MlsSolver, TopoCutParams
        from blademaster_warp.topocut_topology import TopologyTracker

        self.solver = MlsSolver(cfg, TopoCutParams(damaged_shear=damaged_shear), damage_from_start=False)
        self.reference = self.solver.reference
        self.tracker = TopologyTracker(self.solver.n, cfg.grid_spacing, cfg.blade_half_size)
        self.damaged = 0
        self.reconstruct_s = 0.0

    def advance(self, tool, interval, active, index):
        return self.solver.advance(tool, interval)

    def positions(self):
        return self.solver.x.numpy()

    def damage_count(self):
        return int((self.solver.damage.numpy() > 0).sum())

    def settled(self):
        self.solver.start_damage()

    def end_of_cut(self, poses):
        tick = time.perf_counter()
        self.tracker.reconstruct(self.positions(), poses)
        self.reconstruct_s += time.perf_counter() - tick

    def labels(self):
        return self.tracker.labels


def run(name, method, interval, dt=None, damaged_shear=0.0, settle_s=1.0):
    """Both solvers first settle under gravity for settle_s with the knife parked at its
    first pose and not cutting; TopoCut's Eq. 6 then measures J from the settled state."""
    cfg = scene_config(name, dt)
    keys = keyframes(name)
    sim = BladeMasterRun(cfg) if method == "blademaster" else TopoCutRun(cfg, damaged_shear)
    steps = round(keys[-1][0] / interval)
    saved_t, saved_x, poses, was_active = [], [], [], False
    damage_at_start = 0
    tracking = False
    status, error = "completed", None
    begin = time.perf_counter()
    try:
        parked = command_at(0.0, keys)[0]
        parked = ToolState(parked.position, parked.orientation_xyzw, [0, 0, 0], [0, 0, 0])
        for _ in range(round(settle_s / interval)):
            sim.advance(parked, interval, False, 0)
        sim.settled()
        for i in range(steps):
            t = i * interval
            tool, active, index = command_at(t, keys)
            if active and not was_active and method == "topocut":
                poses, tracking, damage_at_start = [], False, sim.damage_count()
            sim.advance(tool, interval, active, index)
            if active and method == "topocut":
                # App. 7.6: the trajectory is tracked from the first damaged particle on.
                tracking = tracking or sim.damage_count() > damage_at_start
                if tracking:
                    poses.append((np.asarray(tool.position), np.asarray(tool.orientation_xyzw)))
            if was_active and not active and method == "topocut" and tracking:
                sim.end_of_cut(poses)
            was_active = active
            if i % 25 == 0 or i == steps - 1:
                saved_t.append(t + interval)
                saved_x.append(sim.positions())
    except Exception as exc:
        status, error = "failed", repr(exc)
    wall = time.perf_counter() - begin
    labels = sim.labels()
    return dict(
        status=status,
        error=error,
        wall_s=wall,
        times=np.array(saved_t),
        positions=np.array(saved_x),
        labels=labels,
        reference=sim.reference,
        reconstruct_s=getattr(sim, "reconstruct_s", 0.0),
        damaged=sim.damage_count() if method == "topocut" else None,
        diagnostics=sim.solver.diagnostics() if method == "topocut" else None,
    )


def evaluate(name, result):
    ref, x, labels = result["reference"], result["positions"][-1], result["labels"]
    goal = goal_labels(name, ref)
    pieces = [np.flatnonzero(labels == k) for k in np.unique(labels[labels >= 0])]
    pieces.sort(key=len, reverse=True)
    goals = [ref[goal == g] for g in np.unique(goal)]
    total, _, pairs = spectral.match([x[p] for p in pieces], goals, jitter=0.0)
    # The same pieces in the reference configuration: partition only, no bending of tipped slices.
    partition, _, _ = spectral.match([ref[p] for p in pieces], goals, jitter=0.0)
    out = dict(
        pieces=[len(p) for p in pieces],
        unassigned=int((labels < 0).sum()),
        goal_pieces=[len(g) for g in goals],
        spectral_reward=float(total),
        spectral_reward_reference=float(partition),
        spectral_reward_max=float(len(goals)),
    )
    # Purity: share of each intended piece's particles in its dominant found piece.
    out["purity"] = [
        float(np.bincount(labels[(goal == g) & (labels >= 0)]).max() / (goal == g).sum())
        for g in np.unique(goal)
    ]
    if name == "fig8":
        # Physical separation: tilt of each intended slice (paper: slices tip over).
        out["slice_tilt_deg"] = [tilt_deg(ref[goal == g], x[goal == g]) for g in range(FIG8_CUTS)]
    if name == "fig4":
        z0 = result["positions"][0][:, 2]
        lift = x[:, 2] - z0
        out["slice_lift_mm"] = float(np.median(lift[goal == 1]) * 1000)
        out["rest_lift_mm_max_p95"] = [
            float(lift[goal == 0].max() * 1000),
            float(np.percentile(lift[goal == 0], 95) * 1000),
        ]
    return out


def plot(directory, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    summary = json.loads((directory / "summary.json").read_text())
    name = summary["scenario"]
    colors = np.array(["#9aa5b1", "#e07a1f", "#2f7fc1", "#3a9d5d", "#c23b5a", "#7d5ba6", "#c9a227"])
    runs = summary["runs"]
    finals = [np.load(directory / f"{r}.npz")["positions"][-1] * 1000 for r in runs]
    lo = np.min([f.min(0) for f in finals], axis=0)
    hi = np.max([f.max(0) for f in finals], axis=0)
    fig = plt.figure(figsize=(5.2 * len(runs), 5.4))
    for c, run_name in enumerate(runs):
        d = np.load(directory / f"{run_name}.npz")
        x, labels = d["positions"][-1] * 1000, d["labels"]
        ax = fig.add_subplot(1, len(runs), c + 1, projection="3d")
        rank = {k: r for r, k in enumerate(np.argsort(-np.bincount(labels[labels >= 0])))}
        col = np.array([colors[rank[k] % len(colors)] if k >= 0 else "#000000" for k in labels])
        ax.scatter(*x.T, s=1.5, c=col, depthshade=False, linewidths=0)
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect(np.maximum(hi - lo, 1))
        ax.view_init(elev=20, azim=-70)
        ax.set_axis_off()
        e = summary[run_name]["evaluation"]
        if e is None:
            ax.set_title(f"{run_name}: {summary[run_name]['error']}", fontsize=9)
            continue
        if run_name == "blademaster":
            title = "BladeMaster (multi-field TLMPM)"
        else:
            title = f"TopoCut single-field MLS-MPM\ndamaged shear x{summary[run_name]['damaged_shear']:g}"
        big = sum(p >= 0.01 * len(labels) for p in e["pieces"])
        ax.set_title(
            f"{title}\n{big} pieces (>1%), spectral reward {e['spectral_reward']:.1f} / {e['spectral_reward_max']:.0f}",
            fontsize=10,
        )
    fig.suptitle(f"{name}: same knife path, final state, colour = piece", fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=110)
    print(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scenario", choices=["fig8", "fig4"], default="fig8")
    p.add_argument("--output", required=True)
    p.add_argument("--dt", type=float, default=None, help="Material step of both solvers; default per scene")
    p.add_argument(
        "--damaged-shear",
        type=float,
        nargs="+",
        default=[0.0],
        help="TopoCut runs: shear modulus factor of damaged particles (the paper leaves the change open)",
    )
    p.add_argument(
        "--settle", type=float, default=1.0, help="Gravity settling before the knife path, s (both solvers)"
    )
    p.add_argument("--plot", default=None, help="Plot a finished run directory to --output")
    a = p.parse_args()
    if a.plot:
        plot(a.plot, a.output)
        return
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=False)
    cfg = scene_config(a.scenario, a.dt)
    interval = max(0.002, cfg.dt)
    runs = [("blademaster", "blademaster", None)] + [
        (f"topocut_shear{v:g}", "topocut", v) for v in a.damaged_shear
    ]
    summary = dict(
        scenario=a.scenario,
        dt=cfg.dt,
        interval=interval,
        settle_s=a.settle,
        runs=[r[0] for r in runs],
        config=asdict(cfg),
        revision=git_revision(),
        command=" ".join(sys.argv),
    )
    for run_name, method, shear in runs:
        result = run(a.scenario, method, interval, a.dt, shear or 0.0, a.settle)
        np.savez_compressed(
            out / f"{run_name}.npz",
            **{k: result[k] for k in ("times", "positions", "labels", "reference")},
        )
        summary[run_name] = dict(
            method=method,
            damaged_shear=shear,
            status=result["status"],
            error=result["error"],
            wall_s=result["wall_s"],
            reconstruct_s=result["reconstruct_s"],
            damaged=result["damaged"],
            diagnostics=result["diagnostics"],
            evaluation=evaluate(a.scenario, result) if result["status"] == "completed" else None,
        )
        print(run_name, json.dumps(summary[run_name]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
