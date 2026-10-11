"""Tests for the T-255 blind A/B harness manifest contract.

The harness itself is static HTML/JS (web/experiments/T-255/). These tests pin
the trials.json contract that ab.js validates at load: every trial needs an id
and >=2 clips, every clip needs id/file/condition. Mirrors ab.js validateManifest.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "web" / "experiments" / "T-255"


def validate_manifest(m):
    """Python mirror of ab.js validateManifest. Returns error string or None."""
    if not isinstance(m, dict) or not m.get("trials"):
        return "manifest needs a non-empty trials array"
    for t in m["trials"]:
        if not t.get("trial_id") or len(t.get("clips", [])) < 2:
            return f"trial {t.get('trial_id', '?')} needs trial_id and >=2 clips"
        for c in t["clips"]:
            if not c.get("id") or not c.get("file") or not c.get("condition"):
                return f"trial {t['trial_id']}: every clip needs id, file, condition"
    return None


def test_example_manifest_valid():
    m = json.loads((PROTO / "trials.example.json").read_text())
    assert validate_manifest(m) is None


def test_example_manifest_blind_labels_not_leaked():
    """The manifest must not contain X/Y labels — blinding happens at render."""
    raw = (PROTO / "trials.example.json").read_text()
    m = json.loads(raw)
    for t in m["trials"]:
        for c in t["clips"]:
            assert c["id"] not in ("X", "Y"), "clip ids must not be presentation labels"


def test_rejects_empty():
    assert validate_manifest({}) is not None
    assert validate_manifest({"trials": []}) is not None


def test_rejects_single_clip():
    m = {"trials": [{"trial_id": "t1", "clips": [{"id": "c1", "file": "a.wav", "condition": "x"}]}]}
    assert validate_manifest(m) is not None


def test_rejects_clip_missing_condition():
    m = {"trials": [{"trial_id": "t1", "clips": [
        {"id": "c1", "file": "a.wav", "condition": "x"},
        {"id": "c2", "file": "b.wav"},
    ]}]}
    err = validate_manifest(m)
    assert err is not None and "condition" in err


def test_prototype_files_exist():
    for f in ["index.html", "ab.js", "ab.css", "trials.example.json", "README.md"]:
        assert (PROTO / f).exists(), f"missing {f}"
