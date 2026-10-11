import numpy as np
import pytest

from eval import sepformer_trial as st


def test_word_losses_counts_missing_and_wrong_voice():
    r = st.word_losses(
        "the quick brown over lazy", "the quick brown fox jumps", "over the lazy dog"
    )
    assert r["target_words"] == 5 and r["kept"] == 3 and r["missing"] == 2
    assert r["wrong_voice"] == 2  # "over", "lazy" belong to the other speaker


def test_silence_counts_as_missing_not_success():
    r = st.word_losses("", "the quick brown fox", "hello there")
    assert r["missing"] == 4 and r["wrong_voice"] == 0 and r["missing_frac"] == 1.0


def test_pinned_revision_and_separate_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    assert len(st.REVISION) == 40 and st.REVISION in str(st.cache_dir())
    assert "MossFormer2" not in str(st.cache_dir())
    with pytest.raises(st.Blocked, match="checkpoint files missing"):
        st.provenance()


def test_level_match_is_rms_of_reference():
    x, ref = np.ones(100, np.float32) * 0.1, np.ones(100, np.float32) * 0.4
    assert abs(float(np.sqrt(np.mean(st.level_match(x, ref) ** 2))) - 0.4) < 1e-5


def test_dry_run_rows_share_one_input_and_flag_offline_reference():
    args = st.parse(["--dry-run", "--contexts", "1.0", "--fake-seconds", "4"])
    data = st.dry_run(args)
    assert data["fake"] and {r["mix_sha"] for r in data["rows"]} == {
        data["rows"][0]["mix_sha"]
    }
    ref = [r for r in data["rows"] if r["context"] == "offline_reference"]
    assert ref and not any(r["streaming_valid"] for r in ref)
    assert "NO (offline)" in st.table(st.aggregate(data["rows"]))


def test_real_run_without_inputs_is_blocked(capsys):
    assert st.main([]) == 2
    assert "BLOCKED" in capsys.readouterr().err


def test_offline_reference_never_takes_the_deadline_fallback():
    class Slow:
        mixer_s = 1.0

        def run(self, m, st):
            from eval.isolation_diagnostic import WindowOut

            return WindowOut(streams=np.stack([m, m]), choice=0, compute_ms=1e9)

    assert st.NoDeadline(Slow()).run(np.zeros(160, np.float32), 0).compute_ms == 0.0


def test_warmup_call_is_not_timed():
    calls = []

    class Sep:
        device = "fake"

        def separate(self, mix, sr=16000):
            calls.append(len(mix))
            return np.stack([mix, mix])

    timed = st.Timed(Sep())
    st.warmup(Sep(), 1600)
    assert calls == [1600] and timed.times == []


def test_sisdri_uses_overlap_windows_only():
    """Target-only windows of a noiseless mix have input SI-SDR at the +100 dB cap; they must not enter SI-SDRi."""
    args = st.parse(["--dry-run", "--contexts", "1.0", "--fake-seconds", "4"])
    rows = st.dry_run(args)["rows"]
    for r in rows:
        assert r["n_overlap_windows"] <= r["n_windows"]
        assert "clip_sisdri_db" in r and r["deadline_applies"] == (r["context"] != "offline_reference")


def test_timing_stats_use_window_length_calls_only():
    class Sep:
        device = "fake"

        def separate(self, mix, sr=16000):
            return np.stack([mix, mix])

    timed = st.Timed(Sep())
    for n in (16000, 16000, 48000):
        timed.separate(np.zeros(n, np.float32))
    s = timed.stats(1.0)
    assert s["n_calls"] == 3 and s["n_timed_window_calls"] == 2


def test_long_inputs_are_memoised_window_calls_are_not():
    calls = []

    class Sep:
        device = "fake"

        def separate(self, mix, sr=16000):
            calls.append(len(mix))
            return np.stack([mix, mix])

    timed = st.Timed(Sep(), memo_above=1000)
    prof, win = np.ones(5000, np.float32), np.ones(500, np.float32)
    for _ in range(3):
        timed.separate(prof)
        timed.separate(win)
    assert calls == [5000, 500, 500, 500]
