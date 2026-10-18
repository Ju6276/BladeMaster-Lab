"""Reproducible Fig. 5 fixture, upstream CRESSim CPU build, and common evaluation.

python -m scripts.fig5_compare build-cressim --source /tmp/cressim --output /tmp/cressim-build
python -m scripts.fig5_compare run --backend cressim_cpu --cressim-build /tmp/cressim-build --output runs/fig5-cr
python -m scripts.fig5_compare run --backend blademaster --output runs/fig5-bm
python -m scripts.fig5_compare compare runs/fig5-bm runs/fig5-cr --output runs/fig5-comparison.json
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from blademaster_warp.fig5 import Fig5Config
from blademaster_warp.fig5_run import compare, load_run, run, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-cressim", help="Compile pinned upstream CPU MLS solver")
    build.add_argument("--source", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--compiler", default="c++")
    build.add_argument(
        "--fix-corotated-dispatch",
        action="store_true",
        help="Insert the missing eCoRotational break in a build-local header snapshot",
    )
    runner = commands.add_parser("run")
    runner.add_argument("--backend", choices=("blademaster", "cressim_cpu"), required=True)
    runner.add_argument("--output", required=True)
    runner.add_argument("--config", help="JSON object with Fig5Config fields")
    runner.add_argument("--spacing", type=float)
    runner.add_argument("--dt", type=float)
    runner.add_argument("--cressim-build")
    runner.add_argument(
        "--cressim-mass-unit-kg",
        type=float,
        help="Kilograms per CRESSim internal mass unit (default 1); metre/second unchanged",
    )
    runner.add_argument("--device", default="cuda:0")
    runner.add_argument("--no-graph", action="store_true")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("run")
    comparison = commands.add_parser("compare")
    comparison.add_argument("blademaster")
    comparison.add_argument("cressim")
    comparison.add_argument("--output", required=True)
    fixture = commands.add_parser("config", help="Write explicit default fixture parameters")
    fixture.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "build-cressim":
        from blademaster_warp.fig5_cressim import build

        result = build(
            args.source,
            args.output,
            args.compiler,
            fix_corotated_dispatch=args.fix_corotated_dispatch,
        )
    elif args.command == "run":
        values = json.loads(Path(args.config).read_text()) if args.config else {}
        if args.spacing is not None:
            values["grid_spacing"] = args.spacing
        if args.dt is not None:
            values["dt"] = args.dt
        if args.cressim_mass_unit_kg is not None:
            values["cressim_mass_unit_kg"] = args.cressim_mass_unit_kg
        result = run(
            Fig5Config(**values),
            args.output,
            args.backend,
            build_dir=args.cressim_build,
            device=args.device,
            graph=not args.no_graph,
        )
    elif args.command == "evaluate":
        _, result = load_run(args.run)
    elif args.command == "compare":
        result = compare(args.blademaster, args.cressim)
        write_json(args.output, result)
    else:
        result = asdict(Fig5Config())
        write_json(args.output, result)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
