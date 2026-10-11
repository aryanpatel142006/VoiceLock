import numpy as np

from experiments.ps4_tse_ab.run import extract, si_sdr


def test_si_sdr_scale_invariant_and_ordered():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000)
    assert si_sdr(0.3 * ref, ref) > 60
    assert si_sdr(ref + rng.standard_normal(16000), ref) < 5


def test_extract_chunked_stitching_identity():
    class Ident:
        def run(self, mix, scale):
            return mix.copy()

    mix = np.random.default_rng(1).standard_normal(16000 * 5).astype(np.float32)
    for win, hop in ((1.0, 0.5), (2.0, 1.0)):
        y, times = extract(Ident(), mix, win, hop)
        n = len(times) * round(hop * 16000) + round((win - hop) * 16000)
        assert np.allclose(y[:n], mix[:n])
    y, times = extract(Ident(), mix, None, None)
    assert np.allclose(y, mix) and len(times) == 1
