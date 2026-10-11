import numpy as np
import pytest

from eval import overnight_noise_collect as m

LIC = """- [1-100032-A.ogg]: clip derived from rose_bark.wav (http://www.freesound.org/people/nfrae/sounds/100032/) by nfrae [CC0]
- [1-110389-A.ogg]: clip derived from dog.wav (http://www.freesound.org/people/x/sounds/110389/) by soundscalpel.com [CC-BY]
"""


def test_parse_esc50_license():
    d = m.parse_esc50_license(LIC)
    assert d["1-100032-A"]["licence_tag"] == "CC0"
    assert d["1-110389-A"]["author"] == "soundscalpel.com"


def test_select_requires_clip_licence():
    rows = m.parse_esc50_meta(
        "filename,fold,target,category\n1-100032-A-0.wav,1,0,rain\n9-999-A-0.wav,1,0,rain\n"
    )
    chosen = m.select_esc50(rows, m.parse_esc50_license(LIC))
    assert [r["filename"] for r in chosen] == ["1-100032-A-0.wav"]


def test_budget_enforced():
    b = m.Budget(10)
    b.add(10)
    with pytest.raises(RuntimeError):
        b.add(1)


def test_rir_room_labels():
    assert m.rir_room("air_type1_air_phone_kitchen_hhp.wav") == "AIR_kitchen"
    assert m.rir_room("air_type1_air_binaural_booth_0_1.wav") == "AIR_booth"
    assert m.rir_room("RVB2014_type1_rir_smallroom1_near_angla.wav") == "RVB2014_smallroom1"


def test_commons_licence_filter():
    assert m.commons_licence_ok("CC BY-SA 4.0")
    assert m.commons_licence_ok("CC0")
    assert not m.commons_licence_ok(None)
    assert not m.commons_licence_ok("All rights reserved")


def test_to_16k_mono_downmix_and_rate():
    x = np.zeros((44100, 2), dtype=np.float32)
    y = m.to_16k_mono(x, 44100)
    assert y.ndim == 1 and abs(len(y) - 16000) <= 2
