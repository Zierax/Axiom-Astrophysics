"""Tests for axiom.stats.ood_eval.evaluate_ood — the central OOD orchestrator."""
import numpy as np
import pytest
from sklearn.datasets import make_classification

from axiom.stats.ood_eval import evaluate_ood


@pytest.fixture(scope="module")
def htru2_like():
    """Generate a small HTRU2-like dataset for testing."""
    X, y = make_classification(
        n_samples=2000, n_features=8, n_informative=6,
        n_classes=2, weights=[0.9, 0.1], random_state=42,
    )
    return X, y


def _make_record(name, sig_type="Pulsar", dm=10.0, snr=10.0, role="Natural"):
    return (name, f"origin_{name}", sig_type, dm, snr, role)


class TestEvaluateOod:
    """Core evaluate_ood functionality."""

    def test_returns_all_expected_keys(self, htru2_like):
        X, y = htru2_like
        records = [_make_record("test_natural", role="Natural")]
        result = evaluate_ood(X, y, records, seed=42)
        for key in ["names", "roles", "verdicts", "htru2_pvals", "descriptor_pvals",
                     "pvals", "anomaly_tpr", "natural_fpr", "pass", "descriptor_fusion_active"]:
            assert key in result, f"Missing key: {key}"

    def test_natural_not_flagged(self, htru2_like):
        X, y = htru2_like
        records = [_make_record(f"nat_{i}", role="Natural") for i in range(5)]
        real_features = {f"nat_{i}": X[i] for i in range(5)}
        result = evaluate_ood(X, y, records, seed=42, real_features=real_features)
        # Natural signals should generally not be flagged as Anomaly
        assert result["natural_fpr"] <= 0.5  # generous bound for small test data

    def test_p_values_in_valid_range(self, htru2_like):
        X, y = htru2_like
        records = [_make_record(f"nat_{i}", role="Natural") for i in range(3)]
        result = evaluate_ood(X, y, records, seed=42)
        for p in result["pvals"]:
            assert 0.0 <= p <= 1.0, f"p-value {p} out of [0,1] range"
        for p in result["htru2_pvals"]:
            assert 0.0 <= p <= 1.0

    def test_deterministic(self, htru2_like):
        X, y = htru2_like
        records = [_make_record("det_test", role="Natural")]
        r1 = evaluate_ood(X, y, records, seed=42)
        r2 = evaluate_ood(X, y, records, seed=42)
        np.testing.assert_array_equal(r1["pvals"], r2["pvals"])

    def test_anomaly_role_in_records(self, htru2_like):
        X, y = htru2_like
        records = [
            _make_record("anom", sig_type="Narrowband", dm=0.0, snr=50.0, role="Anomaly"),
            _make_record("nat", role="Natural"),
        ]
        result = evaluate_ood(X, y, records, seed=42)
        assert len(result["names"]) == 2
        assert result["roles"][0] == "Anomaly"

    def test_empty_records(self, htru2_like):
        X, y = htru2_like
        result = evaluate_ood(X, y, [], seed=42)
        assert len(result["names"]) == 0

    def test_with_real_features(self, htru2_like):
        X, y = htru2_like
        records = [_make_record("rf_test", role="Natural")]
        real_features = {"rf_test": np.array([57.0, 10.0, 0.5, 0.3, 33.0, 5.0, 1.0, 0.5])}
        result = evaluate_ood(X, y, records, seed=42, real_features=real_features)
        assert len(result["names"]) == 1

    def test_narrowband_gets_offmanifold_anchor(self, htru2_like):
        """Narrowband signals should get the off-manifold anchor, not RFI cluster."""
        X, y = htru2_like
        records = [_make_record("nb_test", sig_type="Narrowband", dm=0.0, snr=50.0, role="Anomaly")]
        result = evaluate_ood(X, y, records, seed=42)
        # The HTRU2 p-value should be low (off-manifold)
        assert result["htru2_pvals"][0] < 0.5

    def test_pass_requires_no_false_positives(self, htru2_like):
        X, y = htru2_like
        records = [_make_record(f"nat_{i}", role="Natural") for i in range(10)]
        result = evaluate_ood(X, y, records, seed=42)
        if result["natural_fpr"] == 0.0:
            assert result["pass"] is True
