import json
import sys
from pathlib import Path

import pytest

from eval.prep_cpu_executor import (
    PlanError,
    execute_plan,
    load_plan,
)


def _plan(tmp_path: Path, script: str = "import pathlib; pathlib.Path('done').write_text('ok')"):
    output = tmp_path / "out"
    command = {
        "case_id": "case-1",
        "argv": [sys.executable, "-c", script],
        "output_dir": str(output / "case-1"),
        "reviewed": True,
        "available": True,
    }
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps({"output_root": str(output), "jobs": [command]}), encoding="utf-8"
    )
    return path, output


def test_dry_run_reports_jobs_without_running(tmp_path):
    plan_path, output = _plan(tmp_path)

    report = execute_plan(load_plan(plan_path), dry_run=True)

    assert report["status"] == "dry_run"
    assert report["planned_cases"] == 1
    assert not output.exists()


def test_execution_checkpoint_and_resume_skip(tmp_path):
    plan_path, output = _plan(tmp_path)
    plan = load_plan(plan_path)

    first = execute_plan(plan)
    second = execute_plan(plan)

    assert first["status"] == "completed"
    assert first["executed_cases"] == 1
    assert second["skipped_cases"] == 1
    assert (output / "case-1" / "done").read_text() == "ok"
    checkpoint = output / "case-1" / "checkpoint.json"
    assert json.loads(checkpoint.read_text())["status"] == "success"


def test_rejects_shell_and_unreviewed_commands(tmp_path):
    path, _ = _plan(tmp_path)
    data = json.loads(path.read_text())
    data["jobs"][0]["argv"] = ["sh", "-c", "echo unsafe"]
    data["jobs"][0]["reviewed"] = False
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PlanError, match="reviewed"):
        load_plan(path)


def test_rejects_case_limit_and_output_escape(tmp_path):
    path, _ = _plan(tmp_path)
    data = json.loads(path.read_text())
    data["jobs"][0]["output_dir"] = str(tmp_path / "outside")
    path.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(PlanError, match="output"):
        load_plan(path)


def test_stop_file_finishes_current_job(tmp_path):
    stop = tmp_path / "stop"
    script = (
        "import pathlib,time; pathlib.Path('done').write_text('ok'); "
        f"pathlib.Path({str(stop)!r}).write_text('stop'); time.sleep(.05)"
    )
    path, output = _plan(tmp_path, script)

    report = execute_plan(load_plan(path), stop_file=stop)

    assert report["status"] == "stopped"
    assert report["executed_cases"] == 1
    assert (output / "case-1" / "done").exists()
