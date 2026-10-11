import numpy as np
import pytest

from hearing.tse import TSECandidate, get_candidate
from hearing.tse.av_mossformer import AVMossFormer2, MockBackend, prepare_lips


def test_candidate_mock_contract():
    candidate = AVMossFormer2(backend=MockBackend())
    assert isinstance(candidate, TSECandidate)
    assert isinstance(get_candidate("av_mossformer2"), AVMossFormer2)
    mix = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
    out = candidate.extract(mix, 16000, lips=np.zeros((25, 88, 88), np.uint8))
    np.testing.assert_array_equal(out, mix)
    assert out.dtype == np.float32


def test_lips_resize_normalize_and_align():
    lips = np.stack([np.full((88, 88), v, np.uint8) for v in (0, 255)])
    prepared = prepare_lips(lips, 16000, 2)
    assert prepared.shape == (1, 25, 112, 112)
    assert prepared.dtype == np.float32
    assert prepared[0, 0, 0, 0] == pytest.approx(-2.4650474)
    assert prepared[0, -1, 0, 0] == pytest.approx(3.4591232)
    assert prepared[0, 6, 0, 0] < prepared[0, 12, 0, 0]


@pytest.mark.parametrize(
    "lips,fps",
    [
        (None, 25),
        (np.zeros((0, 88, 88), np.uint8), 25),
        (np.zeros((25, 88, 88), np.uint8), 0),
    ],
)
def test_invalid_lips(lips, fps):
    with pytest.raises(ValueError):
        prepare_lips(lips, 16000, fps)


@pytest.mark.slow
@pytest.mark.network
def test_real_one_second():
    from hearing.tse.av_mossformer import download_model

    download_model()
    candidate = AVMossFormer2()
    mix = (0.1 * np.sin(np.arange(16000) * 2 * np.pi * 220 / 16000)).astype(np.float32)
    out = candidate.extract(mix, 16000, lips=np.full((25, 88, 88), 106, np.uint8))
    assert out.shape == mix.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()


def test_decoder_overlap_add_on_cpu():
    # Two length-four frames overlap by two: [1,2,3,4] + [5,6,7,8].
    from types import SimpleNamespace

    import torch

    from hearing.tse.av_mossformer import _decoder_forward

    decoder = SimpleNamespace(L=4, basis_signals=torch.nn.Identity())
    encoded = torch.tensor([[[1.0, 5.0], [2.0, 6.0], [3.0, 7.0], [4.0, 8.0]]])
    actual = _decoder_forward(decoder, encoded, torch.ones_like(encoded))
    np.testing.assert_array_equal(actual.numpy(), [[1, 2, 8, 10, 7, 8]])


def test_loading_never_downloads_missing_weights(tmp_path, monkeypatch):
    from hearing.tse import av_mossformer as adapter

    monkeypatch.setenv("HEARING_MODELS_DIR", str(tmp_path))
    monkeypatch.setattr(
        adapter, "download_model", lambda: pytest.fail("implicit download")
    )
    with pytest.raises(RuntimeError, match="download_model"):
        adapter._NetworkBackend()
    assert list(tmp_path.iterdir()) == []


def test_mps_unsupported_forward_retries_cpu(caplog):
    import torch

    from hearing.tse.av_mossformer import _NetworkBackend

    class UnsupportedMPS(torch.nn.Module):
        def forward(self, audio, visual):
            return audio

    # The CPU tensor's .to('mps') fails before forward on CPU-only CI too.
    # Force the same unsupported-op failure deterministically on any hardware.
    backend = _NetworkBackend.__new__(_NetworkBackend)
    backend.device = "mps"
    backend.network = UnsupportedMPS()
    original_to = torch.Tensor.to

    def tensor_to(tensor, target, *args, **kwargs):
        if target == "mps":
            raise NotImplementedError("unsupported operation on MPS")
        return original_to(tensor, target, *args, **kwargs)

    from unittest.mock import patch

    with patch.object(torch.Tensor, "to", tensor_to):
        out = backend(
            np.ones((1, 160), np.float32), np.zeros((1, 1, 112, 112), np.float32)
        )
    np.testing.assert_array_equal(out, np.ones((1, 160), np.float32))
    assert backend.device == "cpu"
    assert "retrying CPU" in caplog.text
