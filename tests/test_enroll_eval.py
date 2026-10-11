"""eval.enroll_eval helpers that need no vision or weights (T-092)."""

import json

import pytest

from eval.enroll_eval import solo_before


def test_solo_before_counts_target_alone_seconds(tmp_path):
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"faces": [
        {"role": "target", "speaking": [[0.0, 2.0], [3.0, 5.0]]},
        {"role": "other", "speaking": [[4.0, 6.0]]},
    ]}))
    assert solo_before(man, 1.0) == pytest.approx(1.0)
    assert solo_before(man, 10.0) == pytest.approx(3.0)  # 2 s + (3-4 s); 4-5 s overlaps the other
    assert solo_before(man, None) is None
    assert solo_before(tmp_path / "missing.json", 5.0) is None
