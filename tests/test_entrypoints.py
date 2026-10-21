from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_compile():
    for directory in ("blademaster_warp", "scripts"):
        for path in sorted((ROOT / directory).rglob("*.py")):
            compile(path.read_bytes(), str(path.relative_to(ROOT)), "exec")


@pytest.mark.parametrize(
    "module",
    [
        "blademaster_warp.paper_freefall",
        "blademaster_warp.paper_impact",
        "blademaster_warp.franka_run",
        "blademaster_warp.render_franka",
        "blademaster_warp.render_free",
        "scripts.force_convergence",
        "scripts.live_latency",
        "scripts.indentation",
        "scripts.topocut_compare",
        "scripts.fig3_freefall",
        "scripts.fig4_compare",
    ],
)
def test_cli_help(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()
