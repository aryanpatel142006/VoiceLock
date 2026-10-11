import numpy as np
import pytest

from hearing.tse import blind


def test_default_is_mossformer2(monkeypatch):
    monkeypatch.delenv("HEARING_SEPARATOR", raising=False)
    assert blind.separator_model() == blind.MODEL
    monkeypatch.setenv("HEARING_SEPARATOR", "mossformer2")
    assert blind.separator_model() == blind.MODEL


@pytest.mark.parametrize("kind,name", [("tiger", "TIGER-speech-small"), ("tiger-small", "TIGER-speech-small"), ("tiger-large", "TIGER-speech")])
def test_tiger_selection(monkeypatch, kind, name):
    monkeypatch.setenv("HEARING_SEPARATOR", kind)
    assert blind.separator_model() == name
    assert len(blind.TIGER_REPOS[kind][2]) == 40  # pinned revision


def test_unknown_separator_rejected(monkeypatch):
    monkeypatch.setenv("HEARING_SEPARATOR", "nope")
    with pytest.raises(ValueError):
        blind.separator_model()


def test_tiger_backend_output_shape_conformed(monkeypatch):
    monkeypatch.setenv("HEARING_SEPARATOR", "tiger")

    class Fake:
        device = "cpu"

        def separate(self, mix):
            return np.stack([mix, mix * 0.5])

    sep = blind.BlindSeparator(backend=Fake())
    out = sep.separate(np.random.default_rng(0).standard_normal(16000).astype(np.float32) * 0.1)
    assert out.shape == (2, 16000)
