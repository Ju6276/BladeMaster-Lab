"""Code revision recorded in run manifests."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git_revision():
    def git(*args):
        return subprocess.run(['git', '-C', str(ROOT), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    try:
        return {'commit': git('rev-parse', 'HEAD'),
                'dirty': bool(git('status', '--porcelain', '--untracked-files=no'))}
    except (OSError, subprocess.CalledProcessError):
        return {'commit': None, 'dirty': None}
