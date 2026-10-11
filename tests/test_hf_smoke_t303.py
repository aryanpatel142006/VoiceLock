"""Offline checks for result integrity and scratch lifetime; no model imports."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    't303_smoke_common', Path(__file__).parents[1] / 'eval/hf_models/T-303/common.py'
)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_scratch_is_removed_on_failure():
    scratch = common.scratch

    with pytest.raises(RuntimeError), scratch() as location:
        (location / 'weights.bin').write_bytes(b'owned scratch fixture')
        raise RuntimeError('failed inference')
    assert not location.exists()


def test_measure_rejects_invalid_model_output():
    measure = common.measure

    with pytest.raises(ValueError, match='finite'):
        measure(lambda: np.array([np.nan]), repeats=1)


def test_measure_records_all_output_shapes():
    measure = common.measure

    result = measure(lambda: [np.zeros((1, 192)), np.ones((2, 1, 128))], repeats=2)
    assert result['output_shapes'] == [[1, 192], [2, 1, 128]]
    assert len(result['latencies_ms']) == 2
    assert result['first_inference_ms'] >= 0
