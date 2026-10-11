import numpy as np

from eval import enhanced_replay as er

SR = er.SR


def _tone(f, s, amp=0.1):
    t = np.arange(int(s * SR)) / SR
    return (amp * np.sin(2 * np.pi * f * t)).astype(np.float32)


def test_leak_db_reads_the_other_talkers_gain():
    rng = np.random.default_rng(0)
    target = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
    other = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
    noise = rng.standard_normal(SR * 2).astype(np.float32) * 0.01
    y = target + 0.1 * other  # other 20 dB below the target
    assert abs(er.leak_db(y, target, other, noise) + 20.0) < 0.5


def test_level_match_sets_active_rms():
    x = _tone(220, 2, amp=0.01)
    y = er.level_match(x)
    assert abs(er.db(er.active_level(y)) - er.MATCH_DBFS) < 0.1


def test_dropouts_count_silent_runs_in_target_active_time():
    y = _tone(220, 4)
    y[SR : SR + SR // 2] = 0.0  # 0.5 s hole
    active = np.ones(len(er.frame_rms(y)), bool)
    n, secs = er.dropouts(y, active)
    assert n == 1 and abs(secs - 0.5) < 1e-6


def test_proxy_active_marks_target_windows():
    windows = [{"t": 0.0, "cos_t": 0.6, "cos_o": 0.2}, {"t": 0.75, "cos_t": 0.1, "cos_o": 0.5}]
    m = er.proxy_active(windows, 12)
    assert m[:4].any() and er.wrong_pct(windows) == 50.0
