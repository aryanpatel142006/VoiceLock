"""Bounded, resumable executor for reviewed offline CPU preparation jobs.

The plan is deliberately data-only: each job supplies an argv vector, not a
shell command.  This module never discovers work, retries completed work, or
contacts a service.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

MAX_CASES = 300
MAX_JOB_SECONDS = 30 * 60
MAX_TOTAL_SECONDS = 8 * 60 * 60
MIN_FREE_BYTES = 5 * 1024**3
_DENIED = {
    "ssh",
    "scp",
    "sftp",
    "curl",
    "wget",
    "pip",
    "git",
    "sbatch",
    "srun",
    "qsub",
    "python-gpu",
    "torchrun",
    "uvicorn",
    "server",
    "cuda",
    "mps",
    "gpu",
    "slurm",
}


class PlanError(ValueError):
    """A plan is not safe or does not satisfy the executor limits."""


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _executable(argv0: str) -> Path:
    candidate = Path(argv0)
    if not candidate.is_absolute():
        resolved = shutil.which(argv0)
        if resolved is None:
            raise PlanError(f"executable is unavailable: {argv0}")
        candidate = Path(resolved)
    candidate = candidate.resolve()
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise PlanError(f"executable is not runnable: {argv0}")
    if candidate.name.lower() in _DENIED:
        raise PlanError(f"forbidden executable: {candidate.name}")
    return candidate


def _validate_job(job: dict[str, Any], root: Path) -> dict[str, Any]:
    if job.get("reviewed") is not True or job.get("available") is not True:
        raise PlanError("each job must be explicitly reviewed and available")
    case_id = job.get("case_id")
    argv = job.get("argv")
    if not isinstance(case_id, str) or not case_id.strip():
        raise PlanError("job case_id must be a non-empty string")
    if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) for arg in argv):
        raise PlanError(f"{case_id}: argv must be a non-empty string list")
    executable = _executable(argv[0])
    forbidden = {
        Path(arg).name.lower()
        for arg in argv
        if Path(arg).name.lower() in _DENIED
        or arg.lower() in {"--gpu", "--device=cuda", "--device=mps"}
    }
    if forbidden:
        raise PlanError(f"{case_id}: forbidden command token: {min(forbidden)}")
    output_value = job.get("output_dir")
    if not isinstance(output_value, str) or not output_value:
        raise PlanError(f"{case_id}: output_dir is required")
    output = Path(output_value)
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise PlanError(f"{case_id}: output directory escapes output root") from exc
    timeout = job.get("timeout_s", MAX_JOB_SECONDS)
    if not isinstance(timeout, (int, float)) or timeout <= 0 or timeout > MAX_JOB_SECONDS:
        raise PlanError(f"{case_id}: timeout_s must be in (0, {MAX_JOB_SECONDS}]")
    return {
        **job,
        "case_id": case_id,
        "argv": [str(executable), *argv[1:]],
        "output_dir": str(output),
        "timeout_s": float(timeout),
    }


def load_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate a reviewed plan without creating or running anything."""
    source = Path(path).resolve()
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise PlanError("plan must be an object with a jobs list")
    root_value = data.get("output_root")
    if not isinstance(root_value, str) or not root_value:
        raise PlanError("plan requires output_root")
    root = Path(root_value).expanduser().resolve()
    total = data.get("total_timeout_s", MAX_TOTAL_SECONDS)
    if not isinstance(total, (int, float)) or total <= 0 or total > MAX_TOTAL_SECONDS:
        raise PlanError(f"total_timeout_s must be in (0, {MAX_TOTAL_SECONDS}]")
    if len(data["jobs"]) > MAX_CASES:
        raise PlanError(f"at most {MAX_CASES} jobs are permitted")
    jobs = [_validate_job(job, root) for job in data["jobs"] if isinstance(job, dict)]
    if len(jobs) != len(data["jobs"]):
        raise PlanError("every jobs entry must be an object")
    ids = [job["case_id"] for job in jobs]
    if len(set(ids)) != len(ids):
        raise PlanError("case_id values must be unique")
    return {"output_root": str(root), "total_timeout_s": float(total), "jobs": jobs}


def _job_hash(job: dict[str, Any]) -> str:
    executable = Path(job["argv"][0])
    return _json_hash(
        {
            "argv": job["argv"],
            "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
            "input_hashes": job.get("input_hashes", {}),
            "config_hash": job.get("config_hash"),
        }
    )


def _free_space(root: Path) -> int:
    return shutil.disk_usage(root).free


def execute_plan(
    plan: dict[str, Any],
    *,
    dry_run: bool = False,
    stop_file: str | Path | None = None,
    minimum_free_bytes: int = MIN_FREE_BYTES,
) -> dict[str, Any]:
    """Execute a validated plan, checkpointing each case atomically."""
    root = Path(plan["output_root"]).resolve()
    stop = Path(stop_file).resolve() if stop_file else None
    if dry_run:
        return {
            "status": "dry_run",
            "planned_cases": len(plan["jobs"]),
            "available_cases": len(plan["jobs"]),
            "executed_cases": 0,
            "skipped_cases": 0,
            "failed_cases": 0,
            "output_root": str(root),
        }
    root.mkdir(parents=True, exist_ok=True)
    if _free_space(root) < minimum_free_bytes:
        raise PlanError(f"less than {minimum_free_bytes} bytes free under output root")

    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "completed",
        "planned_cases": len(plan["jobs"]),
        "available_cases": len(plan["jobs"]),
        "executed_cases": 0,
        "skipped_cases": 0,
        "failed_cases": 0,
        "cases": [],
        "output_root": str(root),
    }
    interrupted = False
    previous_handler = signal.getsignal(signal.SIGTERM)

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGTERM, request_stop)
    try:
        for job in plan["jobs"]:
            if interrupted or (stop is not None and stop.exists()):
                result["status"] = "stopped"
                break
            if time.monotonic() - started >= plan["total_timeout_s"]:
                result["status"] = "time_limit"
                break
            if _free_space(root) < minimum_free_bytes:
                result["status"] = "disk_limit"
                break
            output = Path(job["output_dir"])
            output.mkdir(parents=True, exist_ok=True)
            checkpoint = output / "checkpoint.json"
            digest = _job_hash(job)
            if checkpoint.is_file():
                try:
                    prior = json.loads(checkpoint.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    prior = {}
                if prior.get("status") == "success" and prior.get("job_hash") == digest:
                    result["skipped_cases"] += 1
                    result["cases"].append({"case_id": job["case_id"], "status": "skipped"})
                    continue
            stdout_path = output / "stdout.log"
            stderr_path = output / "stderr.log"
            started_case = time.monotonic()
            status = "success"
            return_code: int | None = None
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                try:
                    process = subprocess.Popen(
                        job["argv"],
                        cwd=output,
                        stdout=stdout,
                        stderr=stderr,
                        shell=False,
                        text=True,
                    )
                    while process.poll() is None:
                        if time.monotonic() - started_case >= job["timeout_s"]:
                            process.terminate()
                            process.wait()
                            status = "timeout"
                            break
                        time.sleep(0.02)
                    return_code = process.returncode
                    if return_code != 0 and status == "success":
                        status = "failed"
                except (OSError, ValueError) as exc:
                    stderr.write(f"{type(exc).__name__}: {exc}\n")
                    status = "failed"
            _atomic_json(
                checkpoint,
                {
                    "case_id": job["case_id"],
                    "status": status,
                    "job_hash": digest,
                    "argv": job["argv"],
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "return_code": return_code,
                    "elapsed_s": round(time.monotonic() - started_case, 6),
                },
            )
            case_result = {"case_id": job["case_id"], "status": status}
            result["cases"].append(case_result)
            if status != "success":
                result["failed_cases"] += 1
                result["status"] = status
                break
            result["executed_cases"] += 1
            if interrupted or (stop is not None and stop.exists()):
                result["status"] = "stopped"
                break
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
    result["elapsed_s"] = round(time.monotonic() - started, 6)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    try:
        report = execute_plan(
            load_plan(args.plan), dry_run=args.dry_run, stop_file=args.stop_file
        )
    except (OSError, PlanError, json.JSONDecodeError) as exc:
        print(f"prep_cpu_executor: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] in {"completed", "dry_run", "stopped"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
