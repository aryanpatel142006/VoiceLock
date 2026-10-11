"""judge_check Synthetic first-word latency and Text WER (T-117): onsets in out.wav, pairing, bars, hygiene."""
from __future__ import annotations

import json
import wave

import numpy as np

from scripts import judge_check as jc

SR = 16000


def _session(tmp_path, bursts, finals, buckets=("male_adult",), total_s=8.0, base=1000.0):
    """A tap directory: out.wav with 0.3 s tone bursts at ``bursts`` (stream seconds), 20 ms out frames
    stamped from ``base``, final captions ``(text, t1)`` and one telemetry per bucket."""
    n = int(total_s * SR)
    x = np.zeros(n, np.float32)
    for t in bursts:
        i = int(t * SR)
        x[i:i + int(0.3 * SR)] = 0.2 * np.sin(2 * np.pi * 220 * np.arange(int(0.3 * SR)) / SR)
    with wave.open(str(tmp_path / "out.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())
    rows = [{"t": 0, "dir": "out", "event": {"type": "binary", "kind": 3, "seq": k, "capture_ts": base + k * 0.02,
                                              "bytes": 640}} for k in range(n // 320)]
    rows += [{"t": 0, "dir": "out", "event": {"type": "caption", "text": t, "final": True, "t0": base, "t1": base + t1}}
             for t, t1 in finals]
    rows += [{"t": 0, "dir": "out", "event": {"type": "telemetry", "voice": {"bucket": b},
                                              "latency_ms": {"synth_phrase_p50": 900.0}}} for b in buckets]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return tmp_path


def test_first_word_latency_is_final_to_first_nonsilent_sample(tmp_path):
    d = _session(tmp_path, bursts=[1.9, 4.6], finals=[("Bin blue.", 1.0), ("Bin green.", 3.5)])
    s = jc.phrase_latencies(d)
    assert s["finals"] == 2 and s["engine_p50_ms"] == 900.0
    assert s["latencies_s"] == [0.9, 1.1] or all(abs(a - b) < 0.03 for a, b in zip(s["latencies_s"], [0.9, 1.1]))
    assert s["empty_captions"] == 0 and s["voice_flaps"] == 0 and s["late"] == 0


def test_a_queued_phrase_waits_for_the_next_onset_and_a_cut_off_one_is_dropped(tmp_path):
    d = _session(tmp_path, bursts=[2.0, 5.0], finals=[("One.", 1.0), ("Two.", 1.5), ("Three.", 7.9)])
    s = jc.phrase_latencies(d)
    assert len(s["latencies_s"]) == 2  # no burst left for the third
    assert abs(s["latencies_s"][1] - 3.5) < 0.03 and s["late"] == 1


def test_hygiene_counts_empty_captions_and_voice_flaps(tmp_path):
    d = _session(tmp_path, bursts=[2.0], finals=[(". . .", 1.0), ("Hi.", 1.5)],
                 buckets=("male_adult", "female_adult", "male_adult", "male_adult"))
    s = jc.phrase_latencies(d)
    assert s["empty_captions"] == 1 and s["voice_flaps"] == 2
    assert s["voice_buckets"] == ["female_adult", "male_adult"]


def test_synthetic_rows_pass_on_the_engines_bar_and_name_the_engine():
    stats = {"finals": 4, "latencies_s": [1.3, 1.6, 1.7, 2.2], "late": 0, "empty_captions": 0,
             "voice_buckets": ["male_adult"], "voice_flaps": 1, "engine_p50_ms": None}
    kokoro = jc.synthetic_rows(stats, "kokoro", 30.0, "demo-mf-easy")
    assert [r[0] for r in kokoro] == ["PASS", "PASS"] and "kokoro" in kokoro[0][1] and "<= 2.5" in kokoro[0][1]
    eleven = jc.synthetic_rows(stats, "elevenlabs", 30.0, "demo-mf-easy")
    assert eleven[0][0] == "FAIL" and "<= 1.5" in eleven[0][1] and "p50 1.65 s" in eleven[0][2]
    assert "p95" in kokoro[0][2] and "4 phrases of 4" in kokoro[0][2]


def test_synthetic_hygiene_fails_on_empty_captions_and_flapping():
    stats = {"finals": 2, "latencies_s": [1.0], "late": 0, "empty_captions": 1,
             "voice_buckets": ["a", "b"], "voice_flaps": 5, "engine_p50_ms": None}
    assert jc.synthetic_rows(stats, "kokoro", 30.0, "c")[1][0] == "FAIL"
    assert jc.synthetic_rows({**stats, "latencies_s": []}, "kokoro", 30.0, "c")[0][0] == "FAIL"


def test_text_wer_scores_final_captions_against_the_target_sentences():
    manifest = {"duration_s": 10.0, "sentences": [
        {"who": "target", "text": "bin blue by f six now", "start_s": 0.5, "end_s": 2.0},
        {"who": "other", "text": "place red at c four soon", "start_s": 2.0, "end_s": 3.5},
        {"who": "target", "text": "bin green by b two please", "start_s": 4.0, "end_s": 5.5}]}
    rows = [{"dir": "in", "event": {"type": "binary", "kind": 1, "capture_ts": 100.0}}] + [
        {"dir": "out", "event": {"type": "caption", "final": True, "text": t, "t0": 100.0 + a, "t1": 100.0 + b}}
        for t, a, b in (("Bin blue by F six now.", 0.5, 2.0), ("Bin green by B two please.", 4.0, 5.5))]
    verdict, crit, detail = jc.text_row(rows, manifest, 30.0, "demo-mf-easy")
    assert verdict == "PASS" and "0%" in detail and "2 final captions" in detail and "WER" in crit
    rows[1]["event"]["text"] = "Place red at C four soon."  # the other talker's words in a caption
    assert jc.text_row(rows, manifest, 30.0, "demo-mf-easy")[2].startswith("50%")  # 6 of 12 words wrong
    assert jc.text_row(rows, manifest, 30.0, "demo-mf-easy")[0] == "PASS"  # 50 % is inside the 90 % bar
    rows[1]["event"]["text"] = "Bin green by B two please " + "C-A-" * 20  # T-139: a repetition loop fails
    verdict, _, detail = jc.text_row(rows, manifest, 30.0, "demo-mf-easy")
    assert verdict == "FAIL" and "1 repetition loop(s)" in detail
    rows[1]["event"]["text"] = "Place red at C four soon now one two three four five six seven"
    assert jc.text_row(rows, manifest, 30.0, "demo-mf-easy")[0] == "FAIL"  # WER over the bar
