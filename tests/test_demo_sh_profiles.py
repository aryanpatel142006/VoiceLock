"""T-328: ``demo.sh --judge --team`` defaults HEARING_PROFILES_DIR and names it on the JUDGE line.

Runs the real launcher down its ``--check`` path with a fake interpreter (``$PYTHON``) that records what
judge_check would inherit, so no server, mic or model starts."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAKE_PY = """#!/bin/bash
[ "$1" = -m ] && exit 0  # hearing.tse.headroom
echo "INHERITED=${HEARING_PROFILES_DIR:-unset}"
"""


def run_team_check(tmp_path: Path, profiles: dict[str, list[str]] | None, explicit: str | None = None):
    fixtures = tmp_path / "generated"
    fixtures.mkdir()
    for name, files in (profiles or {}).items():
        (fixtures / "profiles" / name).mkdir(parents=True)
        for f in files:
            (fixtures / "profiles" / name / f).write_bytes(b"")
    fake = tmp_path / "python"
    fake.write_text(FAKE_PY)
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "HEARING_PROFILES_DIR"}
    env.update(PYTHON=str(fake), HEARING_FIXTURES_DIR=str(fixtures))
    if explicit is not None:
        env["HEARING_PROFILES_DIR"] = explicit
    r = subprocess.run(["bash", "scripts/demo.sh", "--judge", "--team", "--check", "--no-hw", "--no-open"],
                       cwd=ROOT, env=env, text=True, capture_output=True, timeout=30, check=False)
    assert r.returncode == 0, r.stderr
    return r.stdout, fixtures / "profiles"


def test_team_defaults_profiles_dir_and_names_it(tmp_path):
    out, profiles = run_team_check(tmp_path, {"aryan": ["profile-01.wav"], "joseph": ["profile-01.wav"]})
    assert f"INHERITED={profiles}" in out
    assert f"TEAM · profiles {profiles}" in out


def test_explicit_profiles_dir_wins(tmp_path):
    out, _ = run_team_check(tmp_path, {"aryan": ["profile-01.wav"], "joseph": ["profile-01.wav"]},
                            explicit="/somewhere/sat-gate")
    assert "INHERITED=/somewhere/sat-gate" in out
    assert "TEAM · profiles /somewhere/sat-gate" in out


def test_missing_profiles_warn_and_do_not_fail(tmp_path):
    out, _ = run_team_check(tmp_path, None)
    assert "INHERITED=unset" in out
    assert "TEAM · WARN no voice profiles (aryan joseph missing" in out


def test_one_missing_profile_warns(tmp_path):
    out, _ = run_team_check(tmp_path, {"aryan": ["profile-01.wav"], "joseph": []})
    assert "INHERITED=unset" in out
    assert "WARN no voice profiles (joseph missing" in out


def test_judge_line_carries_the_profiles_note():
    judge = [ln for ln in (ROOT / "scripts" / "demo.sh").read_text().splitlines()
             if re.match(r'\s*echo "JUDGE · ', ln)]
    assert len(judge) == 1 and "${PROFILES_NOTE:+ · $PROFILES_NOTE}" in judge[0]
