"""T-368: old (phone-memo) vs new (Oct 10 demo-mic) voice profiles on held-out new-session scenes.

    .venv/bin/python -m eval.oct10_reenroll cos                 # cos(aryan, joseph) for profiles/ and profiles-v2/
    .venv/bin/python -m eval.oct10_reenroll scenes              # held-out scenes -> $HEARING_FIXTURES_DIR/t368/<target>/<scene>
    .venv/bin/python -m eval.oct10_reenroll train-head          # T-336 head retrained on separated crops of profile-02..07
    .venv/bin/python -m eval.oct10_reenroll absent              # target-absent accepts (T-336 method), old LDA / new LDA / new head
    .venv/bin/python -m eval.oct10_reenroll batch > b.sh        # the demo_regress rows (run under ONE swarm-lock --exclusive);
                                                                # --labels mixed,new --sirs=-6 selects rows
    .venv/bin/python -m eval.oct10_reenroll score a.log b.log   # T-284 + T-326 columns per row (a later log's row wins) -> JSON + table
    .venv/bin/python -m eval.oct10_reenroll export ROWS.json    # level-matched A/B wavs -> ~/Desktop/selective-hearing-v2/

**Profiles.** New ones are enrolled by ``scripts/enroll_profiles.sh`` (T-172) from ``team/<name>/profile-02..07.wav`` into
``$HEARING_FIXTURES_DIR/profiles-v2`` (the old ``profiles/`` is never touched). **Scenes.** Built ONLY from
``team/<name>/holdout-01/02.wav`` (never used for enrollment or training) with ``eval.demo_scene.compose``, the T-141 shape
(target alone, turns, overlap, target alone; clip.mp4 with two grey cards, target face left). Overlap SIR 0 or -6 dB (the
other talker that much LOUDER, applied to the other stem everywhere, as ``eval.knob_sweep``), with no noise or DEMAND
PCAFETER cafeteria noise at SNR 10 dB under the target's speech. Both target roles: 2 x 2 x 2 = 8 scenes.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SR = 16000
NAMES = ("aryan", "joseph")
HOLDOUT = ("holdout-01.wav", "holdout-02.wav")
SIRS = (0.0, -6.0)
NOISES = ("none", "cafe")
CAFE_SNR_DB = 10.0
CAFE_ZIP, CAFE_MEMBER = "public_voice/PCAFETER_16k.zip", "PCAFETER/ch01.wav"
PEAK = 0.95
SCENE_SET = "t368"
MIXED_DIR = "profiles-v2/mixed"
HEAD_FILE = "T-368_team_head_v3.json"  # v2 (attempt 1): 3 takes, LDA features from profiles-v2/v1
# (label, profiles dir name, env) for the live rows; the head row's env is filled in by batch()
CONDITIONS = (("old", "profiles", {}), ("old+floor20", "profiles", {"HEARING_GATED_FLOOR_DB": "20"}),
              ("new", "profiles-v2", {}), ("new+floor20", "profiles-v2", {"HEARING_GATED_FLOOR_DB": "20"}),
              ("new+head+floor20", "profiles-v2", {"HEARING_GATED_FLOOR_DB": "20", "HEARING_TEAM_HEAD": HEAD_FILE}),
              # Q-1160: new joseph + old aryan (profiles-v2/mixed holds symlinks to profiles-v2/joseph and profiles/aryan)
              ("mixed", MIXED_DIR, {}), ("mixed+floor20", MIXED_DIR, {"HEARING_GATED_FLOOR_DB": "20"}))
# train-head: one 15 s segment of each of these takes from HEAD_SEG_OFFSET_S (skips the take's start), the T-336 recipe
# otherwise, LDA features from profiles-v2 as it is applied live. All six new takes: 42 training scenes.
HEAD_TAKES = tuple(f"profile-0{k}.wav" for k in range(2, 8))
HEAD_SEG_OFFSET_S = 20.0


def fixtures_dir() -> Path:
    return Path(os.environ.get("HEARING_FIXTURES_DIR") or Path(__file__).resolve().parents[1] / "eval/fixtures/generated")


def results_dir() -> Path:
    return Path(os.environ.get("HEARING_RESULTS_DIR") or Path(__file__).resolve().parents[1] / "eval/results")


def scene_name(sir_db: float, noise: str) -> str:
    return f"holdout-sir{int(sir_db):+d}-{noise}".replace("+0", "0")


def scene_list(root: Path | None = None) -> list[tuple[str, str, float, str, Path]]:
    """``[(target, other, sir dB, noise, scene dir)]`` for every held-out scene, target-major."""
    root = root or fixtures_dir() / SCENE_SET
    return [(t, o, s, n, root / t / scene_name(s, n)) for t in NAMES for o in NAMES if o != t for s in SIRS for n in NOISES]


def remix(stems: dict[str, np.ndarray], sir_db: float, peak: float = PEAK) -> dict[str, np.ndarray]:
    """Put the other stem ``-sir_db`` dB above its SIR-0 level, rebuild the mix, renormalise the peak."""
    other = stems["other"] * np.float32(10 ** (-sir_db / 20))
    mix = stems["target"] + other + stems["noise"]
    g = min(1.0, peak / max(float(np.max(np.abs(mix))), 1e-9))
    return {"mix": mix * g, "target": stems["target"] * g, "other": other * g, "noise": stems["noise"] * g}


def cafe_noise(n: int, root: Path | None = None) -> np.ndarray:
    """``n`` samples of DEMAND PCAFETER from minute 1 (the noise bank T-340 used)."""
    import io
    import zipfile

    import soundfile as sf

    from hearing.audio import io as aio

    with zipfile.ZipFile((root or fixtures_dir()) / CAFE_ZIP) as z:
        x, sr = sf.read(io.BytesIO(z.read(CAFE_MEMBER)), dtype="float32")
    if sr != SR:
        x = aio.resample(x, sr, SR)
    return np.asarray(x[60 * SR: 60 * SR + n], np.float32)


def build_scenes(out_root: Path, team: Path) -> list[Path]:
    """Every held-out scene under ``out_root/<target>/<scene>``; returns the manifest paths."""
    from eval import demo_scene as ds
    from eval import demo_team as dt
    from eval.fixtures import video as fxvideo
    from hearing.audio import io as aio

    sents = {}
    with tempfile.TemporaryDirectory(prefix="t368-team-") as tmp:
        for name in NAMES:
            d = Path(tmp) / name
            d.mkdir()
            for k, f in enumerate(HOLDOUT, 1):  # fixture-0k -> holdout-0k: demo_team cuts only "fixture*" takes
                os.symlink(team / name / f, d / f"fixture-0{k}.wav")
            mate = dt.Teammate(name, list(dt._takes(d, "fixture").values()), [])
            sents[name] = dt.teammate_sentences(mate)
            for s in sents[name]:
                s.utt = s.utt.replace("fixture-0", "holdout-0")
    n_noise = round(ds.DURATION_S * SR) * 2
    cafe = cafe_noise(n_noise)
    paths = []
    for target, other, sir, noise, d in scene_list(out_root):
        actors = {who: {"actor": f"team-{n}", "sex": "", "box": dt.FACE_BOX_DEFAULT, "bucket": None}
                  for who, n in (("target", target), ("other", other))}
        nz = cafe if noise == "cafe" else np.zeros(n_noise, np.float32)
        src = {"kind": "demand", "path": f"{CAFE_ZIP}:{CAFE_MEMBER}", "offset_s": 60} if noise == "cafe" else {"kind": "none"}
        note = f"team-{target} (target, left) vs team-{other}, held-out Oct 10 takes, SIR {sir:g} dB, noise {noise}"
        source = {"dataset": "team recordings (Oct 10, MacBook Pro Microphone)", "license": "own recordings (consented, not redistributed)",
                  "takes": {n: list(HOLDOUT) for n in (target, other)}, "task": "T-368"}
        mpath = ds.compose(d.name, {"target": sents[target], "other": sents[other]}, actors, CAFE_SNR_DB, nz, src,
                           d.parent, note=note, source=source, kind="team_scene",
                           extra={"standin": False, "heldout": True, "noise_kind": noise})
        st = {k: aio.load(d / "stems" / f"{k}.wav", SR)[0] for k in ("mix", "target", "other", "noise")}
        st = remix(st, sir)
        for k, x in st.items():
            aio.save(d / "stems" / f"{k}.wav", np.asarray(x, np.float32), SR)
        tmp_clip = d / "clip.sil.mp4"
        shutil.move(d / "clip.mp4", tmp_clip)
        fxvideo.mux_audio(tmp_clip, d / "stems" / "mix.wav", d / "clip.mp4")
        tmp_clip.unlink()
        m = json.loads(mpath.read_text())
        m.update({"sir_db_overlap": sir, "snr_db": CAFE_SNR_DB if noise == "cafe" else None})
        mpath.write_text(json.dumps(m, indent=2) + "\n")
        paths.append(mpath)
        print(f"{target}/{d.name}: {m['duration_s']} s, sections "
              f"{[(s['name'], s['start_s'], s['end_s']) for s in m['sections']]}", flush=True)
    (out_root / "scenes_manifest.json").write_text(json.dumps({
        "task": "T-368", "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "takes": list(HOLDOUT), "sirs_db": list(SIRS),
        "noises": list(NOISES), "cafe_snr_db": CAFE_SNR_DB,
        "scenes": [{"target": t, "other": o, "dir": str(d.relative_to(out_root))} for t, o, _, _, d in scene_list(out_root)],
    }, indent=1) + "\n")
    return paths


def cosines() -> dict:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.enroll_profiles import load_embedder, pair_cosines

    emb = load_embedder("ecapa")
    out = {}
    for d in ("profiles", "profiles-v2"):
        root = fixtures_dir() / d
        out[d] = {f"{a}:{b}": round(c, 4) for a, b, c in pair_cosines({n: root for n in NAMES}, emb)}
        print(d, out[d], flush=True)
    return out


def candidates(root: Path, head=None) -> dict:
    """Per target an ``EnrolledCandidate`` prepared as live from the profiles under ``root``."""
    from hearing.tse.enroll import list_profiles, profile_pairs
    from hearing.tse.enrolled import EnrolledCandidate

    pairs = dict(zip(list_profiles(root), profile_pairs(root), strict=True))
    out, shared = {}, None
    for name in NAMES:
        c = EnrolledCandidate() if shared is None else EnrolledCandidate(shared.separator, shared.embedder)
        c.team_head = head
        c.prepare(*pairs[name])
        shared = shared or c
        out[name] = (c, pairs[name])
    return out


def head_training_scenes(team: Path, seed: int = 368):
    """T-336's training recipe on one 15 s segment of each ``HEAD_TAKES`` take (new takes only)."""
    from eval import train_team_head as th

    rng = np.random.default_rng(seed)
    n = int(th.SEG_S * SR)
    off = int(HEAD_SEG_OFFSET_S * SR)
    scenes = []
    for k, take in enumerate(HEAD_TAKES):
        seg_a = th._norm(th._load(team / "aryan" / take)[off: off + n])
        seg_j = th._norm(th._load(team / "joseph" / take)[off: off + n])
        ea, ej = th._turns(n, rng), th._turns(n, rng)
        for r in th.RATIOS_DB:
            sa, sj = seg_a * ea * 10 ** (r / 20), seg_j * ej
            speech = sa + sj
            noise = th._babble(n, rng)
            noise *= np.sqrt(np.mean(speech ** 2) / max(np.mean(noise ** 2), 1e-12)) * 10 ** (-th.SNR_DB / 20)
            scenes.append((f"{take}_r{r:+.0f}", speech + noise, {"aryan": sa, "joseph": sj}))
        for who, x in (("aryan", seg_a * ea), ("joseph", seg_j * ej)):
            noise = th._babble(n, rng)
            noise *= np.sqrt(np.mean(x ** 2) / max(np.mean(noise ** 2), 1e-12)) * 10 ** (-th.SNR_DB / 20)
            z = np.zeros(n, np.float32)
            scenes.append((f"{take}_{who}_only", x + noise, {"aryan": x if who == "aryan" else z, "joseph": x if who == "joseph" else z}))
        print(f"segment {k}: {take} [{HEAD_SEG_OFFSET_S:g}, {HEAD_SEG_OFFSET_S + th.SEG_S:g}] s", flush=True)
    return scenes


def train_head(out: Path, threshold: float = 0.5, l2: float = 1.0) -> dict:
    from eval import train_team_head as th
    from hearing.tse.team_head import FEATURES, TeamHead, fit_logistic

    t0 = time.time()
    team, prof = fixtures_dir() / "team", fixtures_dir() / "profiles-v2"
    cands = candidates(prof)
    xs, ys = [], []
    for name, mix, stems in head_training_scenes(team):
        for r in th.window_rows(mix, stems, cands):
            xs.append(r["x"])
            ys.extend(r["y"])
        print(f"train {name}: {len(ys)} streams ({time.time() - t0:.0f} s)", flush=True)
    x, y = np.concatenate(xs), np.asarray(ys)
    mean, scale, coef, intercept = fit_logistic(x, y, l2=l2)
    head = TeamHead(mean, scale, coef, intercept, threshold, meta={
        "task": "T-368", "method": "T-336 (eval/train_team_head.py window_rows, ratios, babble SNR, labels)",
        "label": "tuned to the two teammates' voices, trained on separated crops of the Oct 10 enrollment takes",
        "train_audio": {n: [f"team/{n}/{t}" for t in HEAD_TAKES] for n in NAMES},
        "segment": [HEAD_SEG_OFFSET_S, HEAD_SEG_OFFSET_S + th.SEG_S], "lda_profiles": str(prof),
        "train_streams": len(y), "train_positive": int(y.sum()), "features": FEATURES,
    })
    p = head.prob(x)
    train = {"streams": len(y), "positive": int(y.sum()), "acc": float(np.mean((p >= head.threshold) == (y == 1))),
             "false_accept": float(np.mean(p[y == 0] >= head.threshold)),
             "lda_false_accept": float(np.mean(x[y == 0, 0] >= 0)), "lda_acc": float(np.mean((x[:, 0] >= 0) == (y == 1)))}
    head.meta["train"] = train
    out.parent.mkdir(parents=True, exist_ok=True)
    head.save(out)
    print(json.dumps({"train": train, "coef": dict(zip(FEATURES, np.round(coef, 3).tolist(), strict=True)),
                      "weights": str(out), "wall_s": round(time.time() - t0)}, indent=1), flush=True)
    return train


def absent_accepts(labels: tuple[str, ...] = ("old", "new", "mixed"), rep: dict | None = None) -> dict:
    """Seconds accepted as target on each scene's target-absent take (other + noise stems), T-336's method,
    per profiles set in ``labels`` (``new`` also scores the head); merged into ``rep``."""
    from eval import train_team_head as th
    from hearing.tse.team_head import TeamHead

    head = TeamHead.load(results_dir() / HEAD_FILE)
    rep = rep if rep is not None else {}
    dirs = {"old": "profiles", "new": "profiles-v2", "mixed": MIXED_DIR}
    for label in labels:
        cands = candidates(fixtures_dir() / dirs[label])
        for target, other, _, _, sd in scene_list():
            st = {k: th._load(sd / "stems" / f"{k}.wav") for k in ("other", "noise")}
            n = min(map(len, st.values()))
            rows = th.window_rows((st["other"][:n] + st["noise"][:n]).astype(np.float32),
                                  {target: np.zeros(n, np.float32), other: st["other"][:n]}, cands, (target,))
            a = th.summarise(rows, head)
            key = f"{target}/{sd.name}"
            rep.setdefault(key, {"windows": a["windows"]})[f"{label}_lda_s"] = (a["lda_accept_right"] + a["lda_accept_wrong"]) * th.HOP / SR
            if label == "new":
                rep[key]["new_head_s"] = (a["head_accept_right"] + a["head_accept_wrong"]) * th.HOP / SR
            print(key, label, rep[key], flush=True)
    return rep


def batch(py: str = ".venv/bin/python", seconds: int = 30, labels: list[str] | None = None,
          sirs: list[float] | None = None) -> str:
    """One shell script: every (scene, condition) demo_regress row (only ``labels`` / ``sirs`` when given), the T-284
    command, load logged before each."""
    fx = fixtures_dir()
    conds = [c for c in CONDITIONS if labels is None or c[0] in labels]
    # scene-major, SIR 0 + cafe (the demo-like scene) first: a cut-off batch keeps whole scenes
    order = sorted((s for s in scene_list() if sirs is None or s[2] in sirs),
                   key=lambda s: (s[2] != 0.0, s[3] != "cafe", s[0]))
    lines = ["#!/usr/bin/env bash", "set -u", f"# T-368 batch: {len(order) * len(conds)} rows"]
    for (target, other, _, _, d), (label, prof, env) in itertools.product(order, conds):
        e = {"HEARING_PROFILES_DIR": str(fx / prof), "HEARING_REGRESS_NO_BASELINE": "1", **env}
        if "HEARING_TEAM_HEAD" in e:
            e["HEARING_TEAM_HEAD"] = str(results_dir() / HEAD_FILE)
        pre = " ".join(f"{k}={v}" for k, v in e.items())
        lines += [f'echo "ROW {label} {target}:{other} {d.name} load $(uptime | sed \'s/.*averages*: //\')"',
                  (f"env {pre} {py} scripts/demo_regress.py --modes focus_hq --seconds {seconds} --no-free --no-baseline "
                   f"--profile team --isolation standard --voice-profile {target}:{other} --clip {d}"),
                  'echo "EXIT $?"']
    return "\n".join(lines) + "\n"


ROW_RE = re.compile(r"^ROW (\S+) (\w+):(\w+) (\S+) load (.*)$")
RUN_RE = re.compile(r"(demo_regress_\d{8}-\d{6})\.json")


def parse_log(text: str) -> list[dict]:
    """Rows of a batch log: label, target, scene, load, exit, demo_regress run id (the last one printed in the row)."""
    rows, cur = [], None
    for line in text.splitlines():
        m = ROW_RE.match(line.strip())
        if m:
            cur = {"label": m[1], "target": m[2], "other": m[3], "scene": m[4],
                   "load": " / ".join(p.strip() for p in m[5].replace(",", " ").split()[:3]), "run": None, "exit": None}
            rows.append(cur)
            continue
        if cur is None:
            continue
        r = RUN_RE.search(line)
        if r:
            cur["run"] = r[1]
        if line.startswith("EXIT "):
            cur["exit"] = int(line.split()[1])
    return rows


def telemetry_columns(events: Path) -> dict:
    """enrolled share, deadline misses, p95 compute, gated ticks from a session's events.jsonl (T-284 / T-342 columns)."""
    tele = []
    for line in events.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("dir") == "out" and r["event"].get("type") == "telemetry":
            tele.append(r["event"])
    tiers = [e.get("tier") for e in tele if e.get("tier")]
    enrolled = [e for e in tele if e.get("tier") == "enrolled"]
    gated = sum(1 for e in enrolled if (e.get("tse") or {}).get("confidence") == 0)
    health = None
    for e in reversed(tele):  # the last tick's enrolled health: tiers.tiers[name == enrolled].health[0]
        tier = next((t for t in (e.get("tiers") or {}).get("tiers") or [] if t.get("name") == "enrolled"), None)
        h = next((h for h in (tier or {}).get("health") or [] if "windows" in h), None)
        if h:
            health = h
            break
    out = {"enrolled_share": round(100 * tiers.count("enrolled") / len(tiers), 1) if tiers else None,
           "gated": f"{gated}/{len(enrolled)}"}
    if health:
        out["misses"] = f"{int(health.get('overruns', 0)) + int(health.get('late', 0))} / {health.get('windows')}"
        out["p95"] = health.get("compute_p95_ms")
    return out


def merge_logs(logs: list[Path]) -> list[dict]:
    """Rows of several batch logs; a later log's row replaces an earlier one with the same label, target and scene."""
    rows = {}
    for log in logs:
        for r in parse_log(log.read_text()):
            rows[(r["label"], r["target"], r["scene"])] = r
    return list(rows.values())


def score(logs: list[Path], sessions_root: Path) -> list[dict]:
    from eval.knob_sweep import live_atten

    rows = merge_logs(logs)
    res = results_dir()
    fx = fixtures_dir() / SCENE_SET
    for r in rows:
        if not r["run"]:
            continue
        j = json.loads((res / f"{r['run']}.json").read_text())
        s = j["scores"][0]
        d = sessions_root / s["session"]
        r.update({"session": str(d), "sdri": s.get("si_sdri_db"), "sdri_pre": s.get("si_sdri_pre_agc_db"),
                  "leak": s.get("leak_db"), "od_s": s.get("other_dominant_s"), "seconds": s.get("seconds"),
                  "od_pct": round(100 * s["other_dominant_s"] / s["seconds"], 1) if s.get("seconds") else None,
                  "drop": s.get("true_dropout_s"), "first_isolated_s": s.get("first_isolated_s"), "git": j.get("git", "")[:8]})
        if (d / "events.jsonl").is_file():
            r.update(telemetry_columns(d / "events.jsonl"))
            r.update(live_atten(d, fx / r["target"] / r["scene"] / "manifest.json"))
    return rows


def measured(r: dict) -> bool:
    """demo_regress's own rule: a row counts only when it ran and the enrolled tier played in >= 50 % of windows."""
    return r.get("od_s") is not None and (r.get("enrolled_share") or 0) >= 50.0


def nm(r: dict) -> str:
    return "" if measured(r) else (" **crash**" if r.get("od_s") is None else " **NOT MEASURED**")


def table(rows: list[dict], absent: dict | None = None) -> str:
    absent = absent or {}
    hdr = ("| target:other | scene | profiles + knob | load (1/5/15) | exit | enrolled share | deadline misses | p95 compute | "
           "SI-SDRi (pre-AGC) | leak | wrong-speaker | true dropouts | gated | target >= 10 dB down (median gain) | "
           "target-absent accepts | Run |")
    out = [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    for r in sorted(rows, key=lambda r: (r["target"], r["scene"], [c[0] for c in CONDITIONS].index(r["label"]))):
        a = absent.get(f"{r['target']}/{r['scene']}", {})
        acc = a.get("new_head_s") if "head" in r["label"] else a.get(f"{r['label'].split('+')[0]}_lda_s")
        f = lambda v, fmt="{:+.2f}": "-" if v is None else fmt.format(v)
        out.append(
            f"| {r['target']}:{r['other']} | {r['scene']} | {r['label']} | {r['load']} | {f(r.get('exit'), '{}')}{nm(r)} | "
            f"{f(r.get('enrolled_share'), '{:.1f} %')} | {r.get('misses', '-')} | {f(r.get('p95'), '{:.1f} ms')} | "
            f"{f(r.get('sdri'))} ({f(r.get('sdri_pre'))}) | {f(r.get('leak'), '{:.2f} dB')} | "
            f"{f(r.get('od_pct'), '{:.1f} %')} ({f(r.get('od_s'), '{:.1f} s')}) | {f(r.get('drop'), '{:.2f} s')} | "
            f"{r.get('gated', '-')} | {f(r.get('target_down_10dB_s'), '{:.1f} s')} ({f(r.get('median_target_gain_db'), '{:.2f} dB')}) | "
            f"{f(acc, '{:.1f} s')} | `{r.get('run') or '-'}` |")
    return "\n".join(out) + "\n"


def totals(rows: list[dict], target: str | None = None) -> dict:
    """Per condition over the scenes where EVERY condition measured: summed wrong-speaker s, target-down s, dropouts,
    mean leak / pre-AGC SI-SDRi, summed deadline misses (``target`` limits it to one target role)."""
    labels = [c[0] for c in CONDITIONS]
    rows = [r for r in rows if target is None or r["target"] == target]
    keys = {(r["target"], r["scene"]) for r in rows}
    full = {k for k in keys if all(any(measured(r) and (r["target"], r["scene"]) == k and r["label"] == lab for r in rows)
                                   for lab in labels)}
    out = {}
    for label in labels:
        rs = [r for r in rows if r["label"] == label and (r["target"], r["scene"]) in full]
        if not rs:
            continue
        out[label] = {"rows": len(rs), "scenes": sorted(f"{t}/{s}" for t, s in full), "wrong_s": round(sum(r["od_s"] for r in rs), 1),
                      "atten10_s": round(sum(r.get("target_down_10dB_s") or 0 for r in rs), 1),
                      "drop_s": round(sum(r["drop"] or 0 for r in rs), 2),
                      "leak_db": round(float(np.mean([r["leak"] for r in rs])), 2),
                      "sdri_pre": round(float(np.mean([r["sdri_pre"] for r in rs])), 2),
                      "misses": sum(int(r.get("misses", "0 /").split("/")[0]) for r in rs)}
    return out


def level_match(x: np.ndarray, ref_db: float) -> np.ndarray:
    """Scale ``x`` so its active-speech RMS (|x| > -60 dBFS samples) is ``ref_db`` dBFS, peak-limited to -1 dBFS."""
    a = x[np.abs(x) > 1e-3]
    rms = float(np.sqrt(np.mean(np.square(a, dtype=np.float64)))) if len(a) else 1e-9
    y = x * (10 ** (ref_db / 20) / max(rms, 1e-9))
    pk = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * min(1.0, 0.89 / pk) if pk > 0 else y).astype(np.float32)


def export(rows: list[dict], scene: str, labels: dict[str, str], dest: Path, ref_db: float = -26.0) -> list[Path]:
    """``dest/<target>-<n>-<tag>.wav`` for each label (``{label: tag}``) on ``scene`` + the scene's input mix."""
    from hearing.audio import io as aio

    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for target in NAMES:
        mix = aio.load(fixtures_dir() / SCENE_SET / target / scene / "stems" / "mix.wav", SR)[0]
        p = dest / f"{target}-0-input-mix.wav"
        aio.save(p, level_match(mix, ref_db), SR)
        written.append(p)
        for k, (label, tag) in enumerate(labels.items(), 1):
            r = next((r for r in rows if r["label"] == label and r["target"] == target and r["scene"] == scene and r.get("session")), None)
            if r is None:
                continue
            y = aio.load(Path(r["session"]) / "out.wav", SR)[0]
            p = dest / f"{target}-{k}-{tag}.wav"
            aio.save(p, level_match(y, ref_db), SR)
            written.append(p)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cos")
    sub.add_parser("scenes")
    sub.add_parser("train-head")
    b = sub.add_parser("batch")
    b.add_argument("--seconds", type=int, default=30)
    b.add_argument("--labels", default=None, help="comma-separated condition labels (default: all)")
    b.add_argument("--sirs", default=None, help="comma-separated overlap SIRs in dB (default: all)")
    a = sub.add_parser("absent")
    a.add_argument("--labels", default="old,new,mixed")
    a.add_argument("--merge", action="store_true", help="update the existing T-368_target_absent.json")
    s = sub.add_parser("score")
    s.add_argument("log", type=Path, nargs="+", help="batch logs; a later log's row replaces an earlier one")
    s.add_argument("--sessions", type=Path, default=Path("results/sessions"))
    s.add_argument("--absent", type=Path, default=None)
    e = sub.add_parser("export")
    e.add_argument("rows", type=Path)
    e.add_argument("--scene", default="holdout-sir0-cafe")
    e.add_argument("--best", default="new+floor20", help="the label of the best setting measured")
    e.add_argument("--also", action="append", default=[], help="more labels to export (e.g. the mixed pair)")
    e.add_argument("--dest", type=Path, default=Path.home() / "Desktop/selective-hearing-v2")
    args = ap.parse_args(argv)
    res = results_dir()
    if args.cmd == "cos":
        (res / "T-368_cos.json").write_text(json.dumps(cosines(), indent=1) + "\n")
    elif args.cmd == "scenes":
        build_scenes(fixtures_dir() / SCENE_SET, fixtures_dir() / "team")
    elif args.cmd == "train-head":
        train_head(res / HEAD_FILE)
    elif args.cmd == "absent":
        p = res / "T-368_target_absent.json"
        prev = json.loads(p.read_text()) if args.merge and p.is_file() else None
        p.write_text(json.dumps(absent_accepts(tuple(args.labels.split(",")), prev), indent=1) + "\n")
    elif args.cmd == "batch":
        sys.stdout.write(batch(seconds=args.seconds, labels=args.labels.split(",") if args.labels else None,
                               sirs=[float(x) for x in args.sirs.split(",")] if args.sirs else None))
    elif args.cmd == "score":
        rows = score(args.log, args.sessions)
        absent = json.loads(args.absent.read_text()) if args.absent else {}
        stamp = time.strftime("%Y%m%d-%H%M")
        tot = {"all": totals(rows), **{n: totals(rows, n) for n in NAMES}}
        (res / f"T-368_rows_{stamp}.json").write_text(json.dumps({"rows": rows, "totals": tot, "absent": absent}, indent=1) + "\n")
        print(table(rows, absent))
        print(json.dumps(tot, indent=1))
        print("rows:", res / f"T-368_rows_{stamp}.json")
    elif args.cmd == "export":
        rows = json.loads(args.rows.read_text())["rows"]
        labels = {"old": "old-profiles-today", "new": "new-profiles-today", args.best: f"best-{args.best.replace('+', '-')}",
                  **{lab: lab.replace("+", "-") for lab in args.also}}
        for p in export(rows, args.scene, labels, args.dest.expanduser()):
            print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
