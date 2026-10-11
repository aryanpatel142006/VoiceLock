"""Build failure-safe, non-executing overnight experiment job manifests.

This module only validates and describes jobs.  It deliberately does not invoke
experiment scripts, copy data, acquire GPUs, or contact remote services.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

MAX_INPUTS = 500
MAX_OUTPUT_BYTES = 1024 ** 3
MAX_CASES = 300
MAX_SECONDS = 8 * 60 * 60
DEFAULT_LOCK = "/tmp/selective-hearing-overnight-gpu.lock"
METRIC_DEPENDENCIES = {
    "si_sdri": ["target_reference", "mix_reference"],
    "wer": ["transcript"],
    "leak_db": ["target_reference", "interferer_reference"],
    "stoi": ["target_reference"],
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args], capture_output=True, check=True,
            text=True, timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return "unknown"


def _revision(repo_root: Path) -> str:
    return _git(repo_root, "rev-parse", "HEAD")


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def verified_checkpoint(path: Path, recipe_sha256: str, case_id: str, root: Path) -> bool:
    """Read-only resume predicate; marker alone never authorizes a skip."""
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or data.get("status") != "complete":
            return False
        if data.get("recipe_sha256") != recipe_sha256 or data.get("case_id") != case_id:
            return False
        outputs = data.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            return False
        details, _, errors = _hash_inputs(outputs, root, reference_provenance=False)
        return not errors and len(details) == len(outputs)
    except (OSError, ValueError, TypeError):
        return False


def _path_value(value: Any) -> tuple[str, str | None]:
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict) and isinstance(value.get("path"), str):
        return value["path"], value.get("sha256")
    raise ValueError("path entries must be strings or {path, sha256} objects")


def _hash_inputs(
    values: list[Any], root: Path, *, reference_provenance: bool = True,
) -> tuple[list[dict[str, Any]], int, list[str]]:
    details: list[dict[str, Any]] = []
    available = 0
    errors: list[str] = []
    for value in values:
        try:
            raw_path, declared = _path_value(value)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        path = Path(raw_path)
        resolved = path if path.is_absolute() else root / path
        item: dict[str, Any] = {"path": raw_path, "exists": resolved.is_file()}
        if reference_provenance and resolved.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a", ".mp4"}:
            provenance = value.get("provenance") if isinstance(value, dict) else None
            item["provenance"] = provenance
            required = {"speaker_id", "utterance_id", "interval_s", "source_sha256",
                        "licence", "licence_evidence", "source_group", "partition"}
            if not isinstance(provenance, dict) or not required.issubset(provenance):
                errors.append(f"missing reference provenance: {raw_path}")
            else:
                interval = provenance["interval_s"]
                if (not isinstance(interval, list) or len(interval) != 2
                        or not all(_finite(t) for t in interval)
                        or not 0 <= interval[0] < interval[1]
                        or not all(isinstance(provenance[k], str) and provenance[k]
                                   for k in required - {"interval_s"})
                        or not re.fullmatch(r"[0-9a-f]{64}", provenance["source_sha256"])
                        or provenance["partition"] not in {"development", "final"}):
                    errors.append(f"invalid reference provenance: {raw_path}")
        if resolved.is_file():
            actual = _sha256(resolved)
            item["sha256"] = actual
            available += 1
            if not isinstance(declared, str) or declared != actual:
                errors.append(f"input hash mismatch: {raw_path}")
                item["hash_match"] = False
            else:
                item["hash_match"] = True
        else:
            errors.append(f"missing input: {raw_path}")
            item["sha256"] = None
            item["hash_match"] = False
        details.append(item)
    return details, available, errors


def _validate_hash_paths(job: dict[str, Any], root: Path) -> list[str]:
    errors: list[str] = []
    for field in ("model", "profile", "config"):
        value = job.get(f"{field}_hash")
        path_value = job.get(f"{field}_path")
        if path_value is None:
            if value is not None:
                errors.append(f"missing {field} path")
            if field == "config":
                errors.append("missing config path and hash")
            continue
        if not isinstance(path_value, str) or not path_value:
            errors.append(f"invalid {field} path")
            continue
        path = Path(path_value)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file():
            errors.append(f"missing {field}: {path_value}")
        elif not isinstance(value, str) or value != _sha256(resolved):
            errors.append(f"{field} hash mismatch: {path_value}")
    return errors


def build_manifest(
    plan_paths: list[Path],
    *,
    repo_root: Path,
    git_revision: str | None = None,
    capabilities: set[str] | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Validate plans and return a deterministic, replayable manifest."""
    current_revision = _revision(repo_root)
    jobs: list[dict[str, Any]] = []
    errors: list[str] = []
    plans: list[dict[str, Any]] = []
    planned_cases = 0
    total_seconds = 0
    available_inputs = 0
    planned_inputs = 0

    for plan_path in plan_paths:
        try:
            plan = json.loads(plan_path.read_text())
            if not isinstance(plan, dict) or not isinstance(plan.get("jobs"), list):
                raise TypeError("plan must be an object with jobs list")
        except (OSError, ValueError, TypeError) as exc:
            reason = f"unavailable plan {plan_path.name}: {type(exc).__name__}"
            plans.append({"name": plan_path.name, "status": "pending", "reason": reason})
            errors.append(reason)
            continue
        plans.append({"name": plan_path.name, "status": "available", "sha256": _sha256(plan_path)})
        intended = plan.get("source_revision")
        source = (_git(repo_root, "rev-parse", "--verify", f"{intended}^{{commit}}")
                  if isinstance(intended, str) and re.fullmatch(r"[0-9a-f]{7,40}", intended)
                  else "unknown")
        for raw in plan.get("jobs", []):
            if not isinstance(raw, dict):
                errors.append(f"invalid job in {plan_path.name}")
                continue
            job_id = raw.get("id")
            if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
                errors.append(f"invalid job id in {plan_path}")
                continue
            job_errors: list[str] = []
            script = raw.get("script")
            script_path = repo_root / script if isinstance(script, str) else None
            if script_path is None or not script_path.is_file():
                job_errors.append(f"missing script: {script}")
            elif raw.get("script_sha256") != _sha256(script_path):
                job_errors.append(f"script hash mismatch: {script}")
            if source == "unknown":
                job_errors.append("missing or unresolvable source revision")
            elif source != current_revision or (git_revision and git_revision != current_revision):
                job_errors.append("source revision mismatch with intended checkout")
            if script_path and script_path.is_file() and source != "unknown":
                try:
                    relative = script_path.resolve().relative_to(repo_root.resolve()).as_posix()
                except ValueError:
                    job_errors.append("script must be inside source checkout")
                else:
                    blob = _git(repo_root, "rev-parse", f"{source}:{relative}")
                    local = _git(repo_root, "hash-object", str(script_path))
                    if blob == "unknown" or blob != local:
                        job_errors.append("script absent or different on source revision")
            input_values = raw.get("inputs", [])
            if not isinstance(input_values, list):
                job_errors.append("inputs must be a list")
                input_values = []
            if planned_inputs + len(input_values) > MAX_INPUTS:
                input_details, count = [], 0
                input_errors = [f"input cap exceeded: {MAX_INPUTS}"]
            else:
                input_details, count, input_errors = _hash_inputs(input_values, repo_root)
            planned_inputs += len(input_values)
            available_inputs += count
            job_errors.extend(input_errors)
            job_errors.extend(_validate_hash_paths(raw, repo_root))
            cases = raw.get("cases", 0)
            timeout = raw.get("timeout_s", 0)
            if type(cases) is not int or cases <= 0:
                job_errors.append("cases must be a positive integer")
                cases = 0
            if not _finite(timeout) or timeout <= 0:
                job_errors.append("timeout_s must be positive")
                timeout = 0
            planned_cases += cases
            total_seconds += cases * timeout
            capability = raw.get("capability")
            if capability is not None and (not isinstance(capability, str) or not capability):
                job_errors.append("capability must be a nonempty name")
                capability = None
            if raw.get("requires_cuda") and (capabilities is None or "cuda" not in capabilities):
                job_errors.append("CUDA capability unknown or unavailable")
            args = raw.get("args", [])
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                job_errors.append("args must be a list of literal strings")
                args = []
            if type(raw.get("requires_cuda")) is not bool:
                job_errors.append("requires_cuda must be an explicit boolean")
            if raw.get("requires_cuda") and not capability:
                job_errors.append("CUDA capability unknown: explicit capability required")
            if capability and capabilities is None:
                job_errors.append("capability unknown")
            elif capability and capabilities is not None and capability not in capabilities:
                job_errors.append(f"capability unavailable: {capability}")
            job_outputs = raw.get("outputs", [])
            if not isinstance(job_outputs, list) or not job_outputs or not all(
                isinstance(item, str) and item for item in job_outputs
            ):
                job_errors.append("outputs must be a nonempty list of paths")
                job_outputs = []
            metrics = raw.get("metrics", [])
            if not isinstance(metrics, list) or not all(isinstance(m, str) for m in metrics):
                job_errors.append("metrics must be a list of names")
                metrics = []
            for metric in metrics:
                if metric not in METRIC_DEPENDENCIES:
                    job_errors.append(f"unknown metric: {metric}")
            dependencies = sorted({dep for metric in metrics
                                   for dep in METRIC_DEPENDENCIES.get(metric, [])})
            dependency_paths = raw.get("metric_dependencies", {})
            if not isinstance(dependency_paths, dict):
                job_errors.append("metric dependencies must map names to hashed input paths")
                dependency_paths = {}
            verified_inputs = {i["path"] for i in input_details if i["hash_match"]}
            for dep in dependencies:
                if not isinstance(dependency_paths.get(dep), str) or dependency_paths[dep] not in verified_inputs:
                    job_errors.append(f"missing verified metric dependency: {dep}")
            thresholds = raw.get("thresholds", {})
            if not isinstance(thresholds, dict):
                job_errors.append("thresholds must map known metrics to finite numbers")
                thresholds = {}
            for metric, threshold in thresholds.items():
                if metric not in metrics or not _finite(threshold):
                    job_errors.append(f"invalid threshold: {metric}")
            # Do not serialize NaN/Infinity even in a rejected recipe.
            thresholds = {m: v for m, v in thresholds.items() if _finite(v)}
            byte_cap = raw.get("max_output_bytes")
            if type(byte_cap) is not int or not 0 < byte_cap <= MAX_OUTPUT_BYTES:
                job_errors.append(f"max_output_bytes must be a positive integer <= {MAX_OUTPUT_BYTES}")
                byte_cap = 0
            if type(raw.get("seed")) is not int:
                job_errors.append("seed must be an explicit integer")
            if raw.get("lock_path", DEFAULT_LOCK) != DEFAULT_LOCK:
                job_errors.append("serial GPU lock must use the shared lock path")
            status = "ready" if not job_errors else "pending"
            jobs.append(
                {
                    "id": job_id,
                    "script": script,
                    "script_sha256": _sha256(script_path) if script_path and script_path.is_file() else None,
                    "source_revision": source,
                    "model": {"path": raw.get("model_path"), "sha256": raw.get("model_hash")},
                    "profile": {"path": raw.get("profile_path"), "sha256": raw.get("profile_hash")},
                    "config": {"path": raw.get("config_path"), "sha256": raw.get("config_hash")},
                    "args": args,
                    "inputs": input_details,
                    "outputs": job_outputs,
                    "cases": cases,
                    "max_output_bytes": byte_cap,
                    "timeout_s": timeout,
                    "timeout_total_s": cases * timeout,
                    "cuda_required": bool(raw.get("requires_cuda", False)),
                    "capability": capability,
                    "serial_gpu_lock": raw.get("lock_path", DEFAULT_LOCK),
                    "thresholds": thresholds,
                    "seed": raw.get("seed"),
                    "dependency_inputs": dependency_paths,
                    "metrics": metrics,
                    "metric_dependencies": dependencies,
                    "evidence": "planned",
                    "intended_evidence": raw.get("evidence", "planned"),
                    "status": status,
                    "reasons": job_errors,
                    "resume": resume,
                }
            )

    jobs = _json_safe(jobs)
    # Cross-job constraints invalidate every participant, including the first.
    id_owners: dict[str, list[int]] = {}
    output_owners: dict[str, list[int]] = {}
    for index, job in enumerate(jobs):
        id_owners.setdefault(job["id"], []).append(index)
        for output in job["outputs"]:
            normalized = str((repo_root / output).resolve())
            output_owners.setdefault(normalized, []).append(index)
    for owners in id_owners.values():
        if len(owners) > 1:
            for index in owners:
                jobs[index]["reasons"].append(f"duplicate job id: {jobs[index]['id']}")
    for owners in output_owners.values():
        if len(owners) > 1:
            for index in owners:
                jobs[index]["reasons"].append("output collision")
    cap_errors = []
    if planned_cases > MAX_CASES:
        cap_errors.append(f"planned cases {planned_cases} exceed cap {MAX_CASES}")
    if total_seconds > MAX_SECONDS:
        cap_errors.append(f"planned timeout {total_seconds}s exceed cap {MAX_SECONDS}s")
    if planned_inputs > MAX_INPUTS:
        cap_errors.append(f"planned inputs exceed cap {MAX_INPUTS}")
    for job in jobs:
        job["reasons"].extend(cap_errors)
        job["status"] = "pending" if job["reasons"] else "ready"
        errors.extend(f"{job['id']}: {reason}" for reason in job["reasons"])
        job["recipe_sha256"] = _digest({key: value for key, value in job.items()
                                        if key not in {"status", "reasons", "resume"}})
    jobs.sort(key=lambda item: item["id"])
    return {
        "schema": "overnight-prep-job-manifest/v2",
        "plans": plans,
        "mode": "resume" if resume else "dry-run",
        "source_revision": current_revision,
        "jobs": jobs,
        "errors": sorted(set(errors)),
        "summary": {
            "planned_jobs": len(jobs),
            "available_plans": sum(p["status"] == "available" for p in plans),
            "pending_plans": sum(p["status"] == "pending" for p in plans),
            "planned_cases": planned_cases,
            "planned_seconds": total_seconds,
            "available_inputs": available_inputs,
            "planned_inputs": planned_inputs,
            "missing_inputs": planned_inputs - available_inputs,
            "ready_jobs": sum(item["status"] == "ready" for item in jobs),
            "pending_jobs": sum(item["status"] == "pending" for item in jobs),
            "caps": {"cases": MAX_CASES, "seconds": MAX_SECONDS,
                     "inputs": MAX_INPUTS, "output_bytes_per_job": MAX_OUTPUT_BYTES},
        },
        "contract": {
            "never_executes": True,
            "smoke_first_cases": 12,
            "execution_gate": "all selected jobs ready; preserve full suite global ledger across shards",
            "timeout": "timeout_s per case; timeout_total_s hard job wall limit including setup",
            "gpu_lock": "executor holds shared lock for entire job; no parallel GPU jobs",
            "output_byte_cap": "executor stops before max_output_bytes of new job outputs",
            "heldout": "preserve source groups; never tune on final heldout",
            "never_syncs_or_submits": True,
            "resume_checkpoint": {
                "path_template": "checkpoints/{job_id}/{case_id}.json",
                "atomic": True,
                "completion_marker": "status=complete; matching recipe_sha256 and case_id; all output bytes rehashed",
                "verifier": "eval.prep_job_manifest.verified_checkpoint",
                "write_protocol": "write temporary file, fsync, atomic rename on same filesystem",
                "failure_status": "failed; stop job without retry",
                "budget_on_resume": "full original cases/time remain reserved; never reset global ledger",
                "skip_only_verified_complete": True,
                "failed_cases_stop_deterministically": True,
            },
            "evidence": {
                "planned": "recipe only; no measurement",
                "oracle": "reference-assisted result; not deployment evidence",
                "deployable": "result from the named script and immutable inputs",
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--git-revision")
    parser.add_argument("--capability", action="append", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate and write recipes without execution")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_manifest(
        args.plan,
        repo_root=args.repo_root,
        git_revision=args.git_revision,
        capabilities=set(args.capability) if args.capability is not None else None,
        resume=args.resume,
    )
    result["mode"] = "dry-run" if args.dry_run or not args.resume else "resume"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], sort_keys=True))
    return 0 if not result["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
