import subprocess

from blademaster_warp import provenance


def test_revision_includes_new_source_but_ignores_run_output(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        )

    git("init")
    (tmp_path / ".gitignore").write_text("runs/\n")
    git("add", ".gitignore")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "Initialize")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    revision = provenance.git_revision()
    assert revision["commit"] and revision["dirty"] is False
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "report.json").write_text("{}")
    assert provenance.git_revision() == revision
    (tmp_path / "new_solver.py").write_text("value = 1\n")
    assert provenance.git_revision() == {**revision, "dirty": True}
