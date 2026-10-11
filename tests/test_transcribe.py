import json
import time
from pathlib import Path

import numpy as np
import pytest

from eval import transcribe
from eval.transcribe import (
    MockTranscriber,
    Transcriber,
    get_transcriber,
    wer_for_fixture,
)
from hearing.audio import io

TINY = Path(__file__).resolve().parents[1] / "eval" / "fixtures" / "tiny"


def test_mock_returns_configured_text():
    t = get_transcriber("mock", transcript="hello world")
    assert isinstance(t, Transcriber)
    assert t.transcribe(np.zeros(16000, np.float32), 16000) == "hello world"


def test_mock_placeholder_is_deterministic_and_length_based():
    t = MockTranscriber()
    a = t.transcribe(np.zeros(16000, np.float32), 16000)
    assert a == t.transcribe(np.ones(16000, np.float32), 16000)
    assert a != t.transcribe(np.zeros(32000, np.float32), 16000)


def test_auto_without_faster_whisper(monkeypatch):
    monkeypatch.setattr(transcribe, "_whisper_available", lambda: False)
    assert isinstance(get_transcriber("auto"), MockTranscriber)


def test_unknown_name():
    with pytest.raises(ValueError):
        get_transcriber("nope")


@pytest.mark.parametrize("manifest_key", ["transcript", "transcripts"])
def test_wer_for_fixture_mock(tmp_path, manifest_key):
    io.save(tmp_path / "mix.wav", np.zeros(16000, np.float32), 16000)
    ref = "the quick brown fox"
    value = ref if manifest_key == "transcript" else {"target": ref}
    (tmp_path / "manifest.json").write_text(json.dumps({manifest_key: value}))
    ok = wer_for_fixture(tmp_path, MockTranscriber(ref))
    assert ok["wer"] == 0.0 and ok["ref"] == ref and ok["hyp"] == ref
    assert (
        wer_for_fixture(tmp_path, MockTranscriber("the quick brown dog"))["wer"] == 0.25
    )


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("clip", ["speech_a", "speech_b"])
def test_whisper_wer_on_tiny_fixture(clip, capsys):
    pytest.importorskip("faster_whisper")
    wav = TINY / f"{clip}.wav"
    if not wav.exists():
        pytest.skip(f"{wav} not present")
    manifest = json.loads((TINY / "manifest.json").read_text())
    ref = manifest[clip]["transcript"] if clip in manifest else manifest["transcript"]
    t = transcribe.WhisperTranscriber()
    x, sr = io.load(wav)
    t0 = time.perf_counter()
    t.transcribe(x[:1600], sr)  # loads the model
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    hyp = t.transcribe(x, sr)
    rtf = (time.perf_counter() - t0) / (len(x) / sr)
    w = transcribe.wer(hyp, ref)
    with capsys.disabled():
        print(f"\n{clip}: wer={w:.3f} load={load_s:.1f}s rtf={rtf:.2f} hyp={hyp!r}")
    assert w <= 0.2
