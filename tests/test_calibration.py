"""Tests for axiom.stats.calibration.ConformalCalibrator."""
import os
import tempfile

import numpy as np
import pytest

from axiom.stats.calibration import ConformalCalibrator


@pytest.fixture
def calibrator():
    """Fresh calibrator with temp cache path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_cal.pkl")
        c = ConformalCalibrator(cache_path=path)
        yield c


@pytest.fixture
def calibrated(calibrator):
    """Calibrator fitted on synthetic natural scores."""
    scores = np.random.default_rng(42).normal(loc=-2.0, scale=1.0, size=200)
    calibrator.fit(scores)
    return calibrator


class TestConformalCalibrator:
    def test_fit_sets_attributes(self, calibrator):
        scores = np.array([-3.0, -2.5, -2.0, -1.5, -1.0])
        calibrator.fit(scores)
        assert calibrator.calibration_scores is not None
        assert calibrator.natural_min == -3.0
        assert calibrator.natural_max == -1.0

    def test_fit_empty_raises(self, calibrator):
        with pytest.raises(ValueError, match="empty"):
            calibrator.fit(np.array([]))

    def test_compute_p_value_scalar(self, calibrated):
        # A very low score (anomalous) should have low p-value
        p_anomalous = calibrated.compute_p_value(-10.0)
        assert 0.0 < p_anomalous <= 1.0
        # A typical score should have higher p-value
        p_typical = calibrated.compute_p_value(-2.0)
        assert p_typical >= p_anomalous

    def test_compute_p_value_array(self, calibrated):
        scores = np.array([-10.0, -5.0, -2.0, 0.0])
        p_vals = calibrated.compute_p_value(scores)
        assert p_vals.shape == (4,)
        assert all(0.0 <= p <= 1.0 for p in p_vals)
        # P-values should be monotonically non-decreasing with score
        assert all(p_vals[i] <= p_vals[i + 1] for i in range(len(p_vals) - 1))

    def test_compute_p_value_uncalibrated_returns_ones(self, calibrator):
        """Before fit(), p-values should default to 1.0 (neutral)."""
        p = calibrator.compute_p_value(-5.0)
        assert p == 1.0

    def test_save_load_roundtrip(self, calibrator):
        scores = np.random.default_rng(42).normal(-2.0, 1.0, size=100)
        calibrator.fit(scores)
        calibrator.save()
        # Load into a new instance
        calibrator2 = ConformalCalibrator(cache_path=calibrator.cache_path)
        np.testing.assert_array_equal(calibrator2.calibration_scores, calibrator.calibration_scores)
        assert calibrator2.natural_min == calibrator.natural_min

    def test_load_corrupted_cache(self, calibrator):
        """Corrupted cache should be silently ignored."""
        with open(calibrator.cache_path, "wb") as f:
            f.write(b"not a valid pickle")
        # Should not raise — just ignore the corrupted cache
        calibrator.load()
        assert calibrator.calibration_scores is None

    def test_p_value_bounds(self, calibrated):
        """P-values must always be in (0, 1] for any input."""
        for val in [-100.0, -10.0, 0.0, 10.0, 100.0]:
            p = calibrated.compute_p_value(val)
            assert 0.0 < p <= 1.0, f"p-value {p} for input {val}"
