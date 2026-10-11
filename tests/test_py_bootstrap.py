"""scripts/_py.sh dependency refresh with a stub pip (T-205): no network, no real installer.

CI (runs 38001026605, 38001411863): verify_fast's reinstall failed after 3.5 min, the stamp stayed
stale, and scripts/demo.sh re-ran pip when it sourced _py.sh inside a 30 s port-ownership test.
"""
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def make_checkout(tmp_path, pip_rc):
    """A throwaway main checkout: git repo, pyproject.toml, scripts/_py.sh and a counting stub pip."""
    repo = tmp_path / "main"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(ROOT / "scripts" / "_py.sh", repo / "scripts" / "_py.sh")
    (repo / "pyproject.toml").write_text('[project]\nname = "stub"\n')
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    bin_dir = repo / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    pip = bin_dir / "pip"
    pip.write_text(f'#!/usr/bin/env bash\necho "$*" >> "{repo}/pip-calls"\nexit {pip_rc}\n')
    pip.chmod(0o755)
    return repo


def source(repo, script="source scripts/_py.sh", **env):
    clean = {k: v for k, v in os.environ.items()
             if k not in {"PYTHON", "HEARING_REINSTALL", "HEARING_MODELS_DIR", "HEARING_RESULTS_DIR",
                          "HEARING_FIXTURES_DIR", "HEARING_PLAYWRIGHT"}}
    return subprocess.run(["bash", "-c", script], cwd=repo, env={**clean, **env},
                          capture_output=True, text=True, timeout=20, check=True)


def pip_calls(repo):
    calls = repo / "pip-calls"
    return calls.read_text().splitlines() if calls.exists() else []


def test_failed_install_is_not_retried_by_later_launchers(tmp_path):
    repo = make_checkout(tmp_path, pip_rc=1)
    first = source(repo)
    assert "dependency reinstall failed" in first.stdout
    # demo.sh and every other launcher source _py.sh again, also as children of the first shell
    later = [source(repo), source(repo, "source scripts/_py.sh; bash -c 'source scripts/_py.sh'")]
    assert len(pip_calls(repo)) == 1
    for run in later:
        assert "HEARING_REINSTALL=1" in run.stdout  # still loud, and names the explicit retry
    assert not (repo / ".venv" / ".pyproject.md5").exists()


def test_explicit_retry_and_pyproject_change_still_reinstall(tmp_path):
    repo = make_checkout(tmp_path, pip_rc=1)
    source(repo)
    # the explicit retry runs once; a launcher started from that shell does not inherit it
    source(repo, "source scripts/_py.sh; bash -c 'source scripts/_py.sh'", HEARING_REINSTALL="1")
    assert len(pip_calls(repo)) == 2
    with (repo / "pyproject.toml").open("a") as f:
        f.write('version = "2"\n')
    source(repo)
    assert len(pip_calls(repo)) == 3


def test_successful_install_writes_stamp_once(tmp_path):
    repo = make_checkout(tmp_path, pip_rc=0)
    source(repo)
    source(repo)
    assert pip_calls(repo) == ["install -q -e .[dev,eval,ml]"]
    assert (repo / ".venv" / ".pyproject.md5").exists()
    assert not (repo / ".venv" / ".pyproject.failed").exists()
