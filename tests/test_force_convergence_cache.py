import json
import sys
from dataclasses import asdict

import pytest

from scripts import force_convergence as study


@pytest.fixture(autouse=True)
def clean_git_revision(monkeypatch):
    monkeypatch.setattr(study, "git_revision", lambda: {"commit": "fixture-commit", "dirty": False})


def fixture_run(identity):
    cfg = identity["config"]
    return dict(
        cache_identity=identity,
        config=cfg,
        spacing=cfg["grid_spacing"],
        seed=cfg["sampling_seed"],
        dt=cfg["dt"],
        blade_contact_radius=cfg["blade_contact_radius"],
        revision=identity["provenance"],
        x_offset_m=identity["protocol"]["x_offset_m"],
        particles=1,
        force_n=[[0, 0, 1]] * 2000,
        pieces=[1],
        wall_s=0,
        gpu_errors=[0],
    )


@pytest.mark.parametrize("changed", ["config", "provenance", "protocol", "legacy", "dt"])
def test_cache_rejects_mismatch(changed):
    identity = study.cache_identity(0.006, 1, None, study.git_revision())
    run = fixture_run(json.loads(json.dumps(identity)))
    if changed == "legacy":
        del run["cache_identity"]
    elif changed == "dt":
        run["dt"] *= 2
    else:
        run["cache_identity"][changed] = {}
    with pytest.raises(ValueError, match="Cache mismatch"):
        study.validate_cache(run, identity, "fixture.json")


def test_fresh_config_and_json_checkpoint_have_the_same_identity():
    identity = study.cache_identity(0.006, 1, None, study.git_revision())
    run = fixture_run(identity)
    run["config"] = asdict(study.effective_config(0.006, 1))
    study.validate_cache(run, identity, "fresh.json")
    study.validate_cache(json.loads(json.dumps(run)), identity, "loaded.json")


def test_regular_sampling_offset_study_keeps_distinct_traces(tmp_path, monkeypatch):
    computed = []

    def compute(spacing, seed, radius, fraction):
        computed.append((spacing, seed, fraction))
        identity = study.cache_identity(spacing, seed, radius, study.git_revision(), fraction)
        return fixture_run(identity)

    monkeypatch.setattr(study, "cut_trace", compute)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "force_convergence",
            "--output",
            str(tmp_path),
            "--spacings",
            "0.006",
            "0.003",
            "--sampling-seeds",
            "0",
            "--x-offset-fractions",
            "0",
            "0.5",
        ],
    )
    study.main()
    assert computed == [(0.006, 0, 0), (0.006, 0, 0.5), (0.003, 0, 0), (0.003, 0, 0.5)]
    for spacing in (0.006, 0.003):
        path = tmp_path / f"h{spacing * 1000:g}mm_seed0_phase0.5.json"
        run = json.loads(path.read_text())
        assert run["x_offset_m"] == spacing / 4
        identity = study.cache_identity(spacing, 0, None, study.git_revision(), 0.25)
        with pytest.raises(ValueError, match="protocol"):
            study.validate_cache(run, identity, path)
    computed.clear()
    study.main()
    assert not computed
    row = json.loads((tmp_path / "summary.json").read_text())["by_spacing"]["6 mm"]
    assert row["seeds"] == 1 and row["samples"] == 2
    assert row["x_offsets_m"] == [0, 0.0015]


def test_all_caches_validate_before_compute(tmp_path, monkeypatch):
    path = tmp_path / "h5mm_seed1.json"
    path.write_text(json.dumps({"config": {"blade_contact_radius": 0.0005}}))
    original = path.read_bytes()
    monkeypatch.setattr(
        sys,
        "argv",
        ["force_convergence", "--output", str(tmp_path), "--spacings", "0.006", "0.005", "--seeds", "1"],
    )

    def unexpected(*args):
        pytest.fail("Simulation started before all caches were validated")

    monkeypatch.setattr(study, "cut_trace", unexpected)
    with pytest.raises(ValueError, match="h5mm_seed1.json"):
        study.main()
    assert path.read_bytes() == original
    assert not (tmp_path / "summary.json").exists()


def test_matching_cache_is_reused(tmp_path, monkeypatch):
    identity = study.cache_identity(0.006, 1, None, study.git_revision())
    path = tmp_path / "h6mm_seed1.json"
    path.write_text(json.dumps(fixture_run(identity)))
    original = path.read_bytes()
    monkeypatch.setattr(
        sys, "argv", ["force_convergence", "--output", str(tmp_path), "--spacings", "0.006", "--seeds", "1"]
    )

    def unexpected(*args):
        pytest.fail("Matching cache was recomputed")

    monkeypatch.setattr(study, "cut_trace", unexpected)
    study.main()
    assert path.read_bytes() == original
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["revision"] == identity["provenance"]


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    "revision",
    [
        {"commit": "fixture-commit", "dirty": True},
        {"commit": None, "dirty": None},
    ],
)
def test_numeric_study_requires_clean_commit(tmp_path, monkeypatch, cached, revision):
    path = tmp_path / "h6mm_seed1.json"
    if cached:
        identity = study.cache_identity(0.006, 1, None, study.git_revision())
        path.write_text(json.dumps(fixture_run(identity)))
    original = path.read_bytes() if cached else None
    monkeypatch.setattr(study, "git_revision", lambda: revision)
    monkeypatch.setattr(
        sys, "argv", ["force_convergence", "--output", str(tmp_path), "--spacings", "0.006", "--seeds", "1"]
    )

    def unexpected(*args):
        pytest.fail("Simulation started without a clean commit")

    monkeypatch.setattr(study, "cut_trace", unexpected)
    with pytest.raises(ValueError, match="require a clean Git commit"):
        study.main()
    assert (path.read_bytes() if path.exists() else None) == original
    assert not (tmp_path / "summary.json").exists()


@pytest.mark.parametrize(
    "revision",
    [
        {"commit": "fixture-commit", "dirty": True},
        {"commit": "another-commit", "dirty": False},
    ],
)
def test_code_change_during_compute_is_not_cached(tmp_path, monkeypatch, revision):
    identity = study.cache_identity(0.006, 1, None, study.git_revision())

    def changed_code(*args):
        monkeypatch.setattr(study, "git_revision", lambda: revision)
        return fixture_run(identity)

    monkeypatch.setattr(study, "cut_trace", changed_code)
    monkeypatch.setattr(
        sys, "argv", ["force_convergence", "--output", str(tmp_path), "--spacings", "0.006", "--seeds", "1"]
    )
    with pytest.raises(ValueError, match="Code changed during computation"):
        study.main()
    assert not list(tmp_path.glob("*.json"))


def test_legacy_plot_does_not_require_clean_revision(tmp_path, monkeypatch):
    legacy = {"by_spacing": {}, "window_s": [0.4, 1.4], "bin_s": 0.02}
    (tmp_path / "summary.json").write_text(json.dumps(legacy))
    monkeypatch.setattr(study, "git_revision", lambda: {"commit": "fixture-commit", "dirty": True})
    plotted = []
    monkeypatch.setattr(study, "plot", lambda studies, path: plotted.append((studies, path)))
    output = str(tmp_path / "legacy.png")
    monkeypatch.setattr(
        sys, "argv", ["force_convergence", "--plot", f"legacy={tmp_path}", "--output", output]
    )
    study.main()
    assert plotted == [([("legacy", legacy)], output)]
