"""T-166: the literal judge wrapper prints caption WER and propagates failure."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(("caption", "percent", "verdict", "exit_code"), [
    ("one two red blue green gold black white purple orange", "80%", "PASS", 0),
    ("pink brown red blue green gold black white purple orange", "100%", "FAIL", 1),
])
def test_judge_team_check_no_hw_prints_text_wer_and_propagates_failure(
    tmp_path, caption, percent, verdict, exit_code,
):
    # Execute an unchanged copy of the wrapper in a sandbox: its interpreter
    # bootstrap must not install dependencies or write to the shared model cache.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(ROOT / "scripts/demo.sh", scripts / "demo.sh")
    (scripts / "_py.sh").write_text('PY="$TEST_JUDGE_PYTHON"\n')
    clip = tmp_path / "clip.mp4"
    clip.touch()  # Replay itself is replaced below; no media decoder is run.
    (tmp_path / "manifest.json").write_text(json.dumps({
        "duration_s": 10.0,
        "sentences": [{"who": "target", "text": "one two three four five six seven eight nine ten",
                       "start_s": 0.0, "end_s": 1.0}],
    }))
    session = tmp_path / "session"
    session.mkdir()
    (session / "events.jsonl").write_text("\n".join(json.dumps(row) for row in [
        {"dir": "in", "event": {"type": "binary", "kind": 1, "capture_ts": 100.0}},
        {"dir": "out", "event": {"type": "caption", "final": True, "text": caption,
                                 "t0": 100.0, "t1": 101.0}},
    ]) + "\n")
    runner = tmp_path / "fixture-python"
    runner.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
        import json
        import os
        import sys
        from pathlib import Path

        # The headroom benchmark is unrelated and may load model weights.
        if sys.argv[1:] == ["-m", "hearing.tse.headroom"]:
            raise SystemExit(0)
        assert sys.argv[1] == "scripts/judge_check.py", sys.argv
        assert sys.argv[2:] == ["--no-hw", "--team"], sys.argv
        assert os.environ["HEARING_DEMO_PRESET"] == "team"
        sys.path.insert(0, os.environ["TEST_JUDGE_ROOT"])
        from scripts import judge_check as jc
        from scripts import session_score as ss

        fixture = Path.cwd()
        ss.demo_dir = lambda: fixture
        jc.judge_clip = lambda demo, team=False: fixture / "clip.mp4"
        jc.check_team_preset = lambda: ("PASS", "Team preset", "fixture")
        jc.check_flow = lambda *args: ("PASS", "fixture flow", [])
        jc.check_regress = lambda *args, **kwargs: [("PASS", "fixture regress", "ok")]
        jc.check_devices = lambda *args, **kwargs: []

        def forbidden(*args, **kwargs):
            raise AssertionError("--no-hw accessed hardware or GPU")

        jc.check_latency = jc.check_camera = jc.probe_gpu = forbidden

        def recorded_text(clip):
            # Keep session parsing, WER calculation, bar and hygiene checks real;
            # replace the server/model replay with its recorded result fixture.
            manifest = json.loads((clip.parent / "manifest.json").read_text())
            return [jc.text_row(jc.session_rows(fixture / "session"), manifest, 30.0, "fixture")]

        jc.check_synthetic_text = recorded_text
        raise SystemExit(jc.main(sys.argv[2:]))
    """))
    runner.chmod(0o755)
    result = subprocess.run(
        ["bash", str(scripts / "demo.sh"), "--judge", "--team", "--check", "--no-hw"],
        env={**os.environ, "TEST_JUDGE_PYTHON": str(runner), "TEST_JUDGE_ROOT": str(ROOT)},
        capture_output=True, text=True, check=False, timeout=15,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    text_rows = [line for line in result.stdout.splitlines()
                 if line.startswith(("PASS", "FAIL")) and "Text caption WER" in line]
    assert len(text_rows) == 1, result.stdout
    assert text_rows[0].startswith(verdict), result.stdout
    assert percent in text_rows[0], result.stdout
    assert "0 empty, 0 repetition loop(s)" in text_rows[0], result.stdout
    # No unrelated failure should be responsible for the non-zero exit.
    failures = [line for line in result.stdout.splitlines() if line.startswith("FAIL")]
    assert failures == (text_rows if exit_code else []), result.stdout
