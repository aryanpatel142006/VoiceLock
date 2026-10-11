"""scripts/gpu_up.sh ilab: the srun command and the login-node launch script, built offline (--dry-run)."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from hearing.server.session import parse_pipelines

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "gpu_up.sh"


def stub_path(tmp_path) -> tuple[str, Path]:
    """A PATH whose ssh/rsync/curl/srun/pip/caffeinate record their invocation and exit 1."""
    bin_dir = tmp_path / "stubs"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "calls.log"
    for name in ("ssh", "rsync", "curl", "srun", "scp", "pip", "caffeinate"):
        stub = bin_dir / name
        stub.write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\nexit 1\n')
        stub.chmod(0o755)
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}", log


def dry_run(tmp_path, *, script=SCRIPT, **env) -> subprocess.CompletedProcess:
    path, _ = stub_path(tmp_path)
    clean = {k: v for k, v in os.environ.items()
             if not k.startswith(("ILAB_", "REMOTE_")) and k not in ("SSH_OPTS", "LOCAL_PORT")}
    return subprocess.run(["bash", str(script), "ilab", "--dry-run"], capture_output=True, text=True,
                          env={**clean, "XDG_STATE_HOME": str(tmp_path), "PATH": path, **env}, timeout=30, check=False)


def launch(tmp_path, **env) -> str:
    result = dry_run(tmp_path, **env)
    assert result.returncode == 0, result.stderr
    return result.stdout


def srun_line(script: str) -> str:
    lines = [line for line in script.splitlines() if " srun " in line]
    assert len(lines) == 1, script
    return lines[0]


def job_exec_line(script: str) -> str:
    """Decode the ``JOB=$'...'`` assignment the way the login node's bash does; return its exec line."""
    assignment = next(line for line in script.splitlines() if line.startswith("JOB="))
    job = subprocess.run(["bash", "-c", f'{assignment}\nprintf %s "$JOB"'], capture_output=True,
                         text=True, check=True).stdout
    return job.splitlines()[-1]


def remote_pipelines(script: str) -> str:
    """Run the job's exec line with ``printenv`` in place of the server: HEARING_PIPELINES as the job sees it."""
    line = job_exec_line(script)
    line = line.replace("exec env", "env", 1)
    line = line[: line.index(".venv/bin/python -m hearing serve")] + "printenv HEARING_PIPELINES"
    return subprocess.run(["bash", "-c", line], capture_output=True, text=True, check=True).stdout.strip()


def test_srun_defaults_pin_the_login_node_with_one_gpu_and_a_time_limit(tmp_path):
    line = srun_line(launch(tmp_path))
    assert line.startswith("nohup setsid srun ")
    assert "--gres=gpu:1" in line
    assert re.search(r"-w ilab3\b", line)
    assert "--time=02:00:00" in line  # the partition default is 7 days
    assert "-J hearing-hq" in line
    assert 'bash -c "$JOB"' in line
    assert line.endswith("> serve.log 2>&1 < /dev/null &")
    for rejected in ("--cpus-per-task", "--mem", " -c "):
        assert rejected not in line.replace('bash -c "$JOB"', "")


def test_dry_run_runs_no_ssh_network_or_installer(tmp_path):
    import shutil

    # _py.sh invokes pip by absolute path, so isolate its git main checkout too.
    root = tmp_path / "checkout"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    for name in ("gpu_up.sh", "_py.sh"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    shutil.copy2(ROOT / "pyproject.toml", root / "pyproject.toml")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    venv_bin = root / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    _, log = stub_path(tmp_path)
    pip = venv_bin / "pip"
    pip.write_text(f'#!/bin/sh\necho "absolute pip $*" >> "{log}"\nexit 1\n')
    pip.chmod(0o755)
    result = dry_run(tmp_path, script=scripts / "gpu_up.sh")
    assert result.returncode == 0, result.stderr
    assert not log.exists(), log.read_text()


def test_launch_script_is_valid_bash_and_serves_on_localhost(tmp_path):
    script = launch(tmp_path)
    subprocess.run(["bash", "-n"], input=script, text=True, check=True)
    assert "cd hearing-remote" in script
    assert "http://127.0.0.1:18765/health" in script  # REMOTE_PORT defaults to 18765 on the shared login node
    assert "--host 127.0.0.1 --port 18765" in job_exec_line(script)
    assert "squeue -h -u \"$USER\" -n hearing-hq" in script  # only our own job name is ever cancelled


def test_remote_focus_hq_is_the_gpu_ladder_with_av_mossformer2_two_second_windows(tmp_path):
    # T-075: the remote serves the tiers-remote ladder (enrolled with the same-gender guard -> AV 2 s / 1 s
    # -> gate), not the single AV stage, and not the CPU DFN tiers that put the remote behind real time.
    value = remote_pipelines(launch(tmp_path))
    spec = parse_pipelines(value)["focus_hq"]
    assert spec == 'tiers(config="config/tiers-remote.yaml")'
    ladder = yaml.safe_load((ROOT / "config" / "tiers-remote.yaml").read_text())["tiers"]
    pipelines = {t["name"]: t["pipeline"] for t in ladder}
    assert pipelines["focus_hq"] == 'tse(candidate="av_mossformer2", window_s=2.0, hop_s=1.0, require_gpu=true)>mixer'
    assert not any("dfn" in p for p in pipelines.values())
    # T-086: enrolled outranks AV once enrolled (AV = cold start / same-gender), at the A100's 0.5 s / 0.125 s.
    assert [t["name"] for t in ladder][:2] == ["enrolled", "focus_hq"]
    assert "window_s=0.5, hop_s=0.125, threaded=true" in pipelines["enrolled"]
    assert pipelines["enrolled"].endswith(">mixer(max_duck_db=20)")  # T-110: no holes on a missed window (T-095)


def test_reuse_starts_a_step_in_the_running_job_and_never_cancels_it(tmp_path):
    # T-110: restart the server on current code inside a live job (its GPU stays ours) instead of scancel + srun.
    script = launch(tmp_path, ILAB_REUSE="432518", REMOTE_PORT="18766", ILAB_GRES="gpu:a100:1")
    line = srun_line(script)
    assert "--jobid=432518 --overlap --gres=gpu:a100:1" in line
    assert "--time=" not in line
    assert "> serve-18766.log" in line  # the old server keeps writing serve.log
    assert "scancel" not in script.replace('elif [ -n "$id" ]; then scancel', "")
    assert "--port 18766" in job_exec_line(script)


def test_reuse_step_takes_no_cpus_of_its_own(tmp_path):
    # An --overlap step shares the job's CPUs (and ILAB_CPUS is a no-op since T-118).
    line = srun_line(launch(tmp_path, ILAB_REUSE="432518", ILAB_CPUS="8"))
    assert "--jobid=432518 --overlap" in line
    assert "--cpus-per-task" not in line


def test_overrides_reach_srun_and_the_server(tmp_path):
    script = launch(tmp_path, ILAB_TIME="00:02:00", ILAB_JOB="hq-test", ILAB_SRUN_ARGS="--partition=unlimited",
                    REMOTE_PORT="18999", REMOTE_PIPELINES="focus_hq=passthrough", REMOTE_DIR="/freespace/x")
    line = srun_line(script)
    assert "--time=00:02:00" in line
    assert "-J hq-test" in line
    assert "--partition=unlimited" in line
    assert "cd /freespace/x" in script
    assert "--port 18999" in job_exec_line(script)
    assert remote_pipelines(script) == "focus_hq=passthrough"


def test_job_on_another_node_listens_on_the_cluster_network(tmp_path):
    script = launch(tmp_path, ILAB_NODE="ilab1")
    assert re.search(r"-w ilab1\b", srun_line(script))
    assert "--host 0.0.0.0" in job_exec_line(script)
    assert "http://ilab1:18765/health" in script


@pytest.mark.parametrize("extra", ["--mem=8G", "--cpus-per-task=4", "-c 4"])
def test_rejected_srun_options_fail_before_anything_runs(tmp_path, extra):
    result = dry_run(tmp_path, ILAB_SRUN_ARGS=extra)
    assert result.returncode != 0
    assert "cluster rejects them" in result.stderr
    assert "srun" not in result.stdout


SINFO = "ilab1 gpu:1080ti:4\nrlab2 gpu:a100:2(S:0-1)\nrlab3 (null)"


def test_gres_with_a_gpu_type_picks_the_node_from_sinfo(tmp_path):
    script = launch(tmp_path, ILAB_GRES="gpu:a100:1", ILAB_SINFO=SINFO)
    line = srun_line(script)
    assert re.search(r"-w rlab2\b", line)
    assert "--gres=gpu:a100:1" in line
    assert "http://rlab2:18765/health" in script  # another node: health over the cluster network


def test_gres_type_no_node_has_fails(tmp_path):
    result = dry_run(tmp_path, ILAB_GRES="gpu:h100:1", ILAB_SINFO=SINFO)
    assert result.returncode != 0 and "no node has a h100" in result.stderr


def test_explicit_node_wins_over_sinfo(tmp_path):
    assert re.search(r"-w ilab1\b", srun_line(launch(tmp_path, ILAB_GRES="gpu:a100:1", ILAB_SINFO=SINFO, ILAB_NODE="ilab1")))


def test_ilab_cpus_is_a_no_op_that_warns(tmp_path):
    # T-118: iLab rejects every CPU flag ("Please do not specify the number of CPUs") and a step sees 96 cores.
    for value in ("6", "lots"):
        result = dry_run(tmp_path, ILAB_CPUS=value)
        assert result.returncode == 0, result.stderr
        assert "--cpus-per-task" not in srun_line(result.stdout)
        assert "ILAB_CPUS=" + value + " is ignored" in result.stderr


def test_side_job_keeps_its_own_state_name_and_ports(tmp_path):
    # T-118: an A/B job must not cancel the live job nor close its tunnel (state in hearing-gpu/side).
    result = subprocess.run(["bash", str(SCRIPT), "ilab", "--side", "--dry-run"], capture_output=True, text=True,
                            env={**{k: v for k, v in os.environ.items() if not k.startswith(("ILAB_", "REMOTE_"))
                                    and k != "LOCAL_PORT"}, "XDG_STATE_HOME": str(tmp_path),
                                 "PATH": stub_path(tmp_path)[0]}, timeout=30, check=False)
    assert result.returncode == 0, result.stderr
    assert "-J hearing-hq-side" in srun_line(result.stdout)
    assert "-n hearing-hq-side" in result.stdout and "-n hearing-hq " not in result.stdout
    assert "--port 18767" in job_exec_line(result.stdout)
    port = int(re.search(r"local port (\d+)", result.stderr).group(1))
    assert port != 18765
    assert (tmp_path / "hearing-gpu" / "side").is_dir()
    env_side = launch(tmp_path, ILAB_SIDE="1", LOCAL_PORT="19001", REMOTE_PORT="18770")
    assert "--port 18770" in job_exec_line(env_side)


def test_launch_installs_missing_core_deps_before_serving(tmp_path):
    script = launch(tmp_path)
    deps = next(i for i, line in enumerate(script.splitlines()) if "check_deps.py --install" in line)
    srun = next(i for i, line in enumerate(script.splitlines()) if " srun " in line)
    assert deps < srun


def test_check_deps_lists_missing_core_deps(tmp_path):
    from scripts import check_deps

    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\ndependencies = ["numpy>=1.26", "uvicorn[standard]>=0.30", '
                         '"surely-not-installed-pkg>=1"]\n')
    assert check_deps.missing(check_deps.core_deps(pyproject)) == ["surely-not-installed-pkg>=1"]
    assert check_deps.main(["--pyproject", str(pyproject)]) == 1
    assert check_deps.main([]) == 0  # this checkout's venv has every core dep


def run_launch(tmp_path, script: str, *, health: str, srun: str = "sleep 3") -> subprocess.CompletedProcess:
    """Run a generated launch script on a fake login node: REMOTE_DIR with REVISION ``new``, a .venv whose
    python only stubs the dep checks, squeue saying the job runs, and ``health`` as curl's body (``fail``
    exits 7; ``after-srun:<json>`` fails until the srun stub has run)."""
    import sys

    remote = tmp_path / "remote"
    (remote / ".venv" / "bin").mkdir(parents=True)
    (remote / "REVISION").write_text("new\n")
    py = remote / ".venv" / "bin" / "python"
    py.write_text('#!/bin/sh\ncase "$1$2" in scripts/check_deps.py*|*"import hearing"*) exit 0 ;; esac\n'
                  f'exec {sys.executable} "$@"\n')
    py.chmod(0o755)
    stubs = tmp_path / "remote-stubs"
    stubs.mkdir()
    started = tmp_path / "srun.started"
    body = health.removeprefix("after-srun:")
    gate = f'[ -f "{started}" ] || exit 7\n' if health.startswith("after-srun:") else ""
    curl = "exit 7\n" if health == "fail" else f"{gate}cat <<'EOF'\n{body}\nEOF\n"
    for name, text in {"curl": curl, "squeue": 'echo "435775 gpu hearing-hq"\n', "setsid": 'exec "$@"\n',
                       "srun": f'touch "{started}"\n{srun}\n', "scancel": "exit 0\n"}.items():
        (stubs / name).write_text(f"#!/bin/sh\n{text}")
        (stubs / name).chmod(0o755)
    script = script.replace("cd hearing-remote", f"cd {remote}", 1)
    env = {**os.environ, "PATH": f"{stubs}{os.pathsep}{os.environ['PATH']}"}
    return subprocess.run(["bash", "-s"], input=script, capture_output=True, text=True, env=env, timeout=20,
                          check=False)


def health_json(rev: str) -> str:
    return f'{{"status": "ok", "git_rev": "{rev}", "pid": 4242, "started_at": "2026-10-08T22:30:00+00:00"}}'


REUSE = {"ILAB_REUSE": "435775", "ILAB_NODE": "rlab2", "ILAB_LOGIN": "ilab2", "ILAB_WAIT": "5"}


def test_reuse_refuses_a_port_an_old_server_still_answers_on_the_job_node(tmp_path):
    # T-137: the bind check ran on the login node, srun failed to bind rlab2:18765 and /health answered from
    # the old server; the launch must probe the job node itself and fail before starting a step.
    script = launch(tmp_path, **REUSE)
    assert "http://rlab2:18765/health" in script
    assert 's.bind(("127.0.0.1", 18765))' not in script  # a login-node bind says nothing about rlab2
    result = run_launch(tmp_path, script, health=health_json("bc1d8dc"))
    assert result.returncode == 1
    assert "rlab2:18765 already serves git_rev bc1d8dc (pid 4242" in result.stderr
    assert "REMOTE_PORT" in result.stderr
    assert "JOBID=" not in result.stdout
    assert not (tmp_path / "srun.started").exists()


def test_reuse_fails_naming_the_mismatch_when_an_old_rev_answers_after_start(tmp_path):
    result = run_launch(tmp_path, launch(tmp_path, **REUSE), health=f"after-srun:{health_json('bc1d8dc')}")
    assert result.returncode == 1, result.stdout
    assert "answers git_rev bc1d8dc" in result.stderr
    assert "not the synced REVISION new" in result.stderr
    assert "JOBID=" not in result.stdout


def test_reuse_reports_the_rev_that_actually_serves(tmp_path):
    # Success returns on the first /health poll; the long wait only matters when the machine is loaded.
    # Keep the fake server step alive through polling, like the real long-running server.
    # An immediate exit races the marker-backed curl before the first successful health response.
    result = run_launch(tmp_path, launch(tmp_path, **{**REUSE, "ILAB_WAIT": "30"}),
                        health=f"after-srun:{health_json('new')}")
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-2:] == ["JOBID=435775", "REV=new"]


def test_reuse_step_that_cannot_bind_fails_without_success(tmp_path):
    result = run_launch(tmp_path, launch(tmp_path, **REUSE), health="fail", srun="exit 1")
    assert result.returncode == 1
    assert "step in job 435775 ended" in result.stderr
    assert "JOBID=" not in result.stdout


def test_cmd_ilab_checks_the_served_rev_and_prints_it():
    # The laptop side: the launch's REV= and the tunnel's /health git_rev must both equal the synced rev.
    text = SCRIPT.read_text()
    body = text[text.index("cmd_ilab() {"): text.index("ilab_close_tunnel() {")]
    assert '[ "$rev" = "$want" ] || die' in body
    assert '[ "$served" = "$want" ] || die' in body
    assert 'say "serving git_rev $served' in body


def job_env(script: str, name: str, cwd: Path) -> str:
    """Run the job's exec line with ``printenv NAME`` in place of the server, in ``cwd``."""
    line = job_exec_line(script).replace("exec env", "env", 1)
    line = line[: line.index(".venv/bin/python -m hearing serve")] + f"printenv {name} || true"
    return subprocess.run(["bash", "-c", line], capture_output=True, text=True, check=True, cwd=cwd).stdout.strip()


def test_ilab_profiles_sets_the_job_profiles_dir_to_the_synced_copy(tmp_path):
    # T-129: voice profiles are gitignored, so the code sync never carries them; ILAB_PROFILES rsyncs a
    # local dir to $REMOTE_DIR/profiles and the job serves with HEARING_PROFILES_DIR pointing there.
    profiles = tmp_path / "profiles-all"
    (profiles / "aryan").mkdir(parents=True)
    work = tmp_path / "remote"
    work.mkdir()
    script = launch(tmp_path, ILAB_PROFILES=str(profiles))
    assert job_env(script, "HEARING_PROFILES_DIR", work) == str(work.resolve() / "profiles")
    assert job_env(launch(tmp_path), "HEARING_PROFILES_DIR", work) == ""  # unset: profiles stay off


def test_ilab_profiles_must_be_a_directory(tmp_path):
    result = dry_run(tmp_path, ILAB_PROFILES=str(tmp_path / "missing"))
    assert result.returncode != 0
    assert "ILAB_PROFILES" in result.stderr


def test_cmd_ilab_rsyncs_the_profiles_through_the_control_master():
    text = SCRIPT.read_text()
    body = text[text.index("cmd_ilab() {"): text.index("ilab_close_tunnel() {")]
    sync = body[body.index('if [ -n "$ILAB_PROFILES" ]'):]
    assert "rsync -az --delete" in sync and '"$ILAB_PROFILES/"' in sync and "/profiles/" in sync
    assert "ControlPath=$ILAB_SOCKET" in sync
    assert body.index('if [ -n "$ILAB_PROFILES" ]') < body.index("ilab_launch \"$node\" \"$login\"))")


def test_the_t129_side_server_ladder_differs_only_in_the_enrolled_window():
    # T-129: the A100 on the Mac's 1.0 s window (0.25 s hop), served by a REMOTE_PIPELINES side server.
    base = yaml.safe_load((ROOT / "config" / "tiers-remote.yaml").read_text())
    w1 = yaml.safe_load((ROOT / "config" / "tiers-remote-w1.yaml").read_text())
    enrolled = w1["tiers"][0]["pipeline"]
    assert "window_s=1.0, hop_s=0.25, threaded=true" in enrolled
    w1["tiers"][0]["pipeline"] = base["tiers"][0]["pipeline"]
    assert w1 == base
