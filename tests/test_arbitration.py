"""Tests for the SignalArbitrator verdict logic.

The arbitrator must:
  * call a confidently-recognised natural/interference source by its class,
  * flag a genuinely out-of-manifold (OOD) source as Anomaly,
  * surface moderate-evidence cases as Candidate.
"""
import numpy as np

from axiom.stats.arbitration import SignalArbitrator


def _make_arbitrator():
    return SignalArbitrator(fdr_alpha=0.05, conformal_alpha=0.05)


def test_confident_natural_not_flagged():
    arb = _make_arbitrator()
    # pred_cls=0 (Natural), high natural confidence, in-distribution p-value.
    probs = np.array([[0.99, 0.01, 0.0]])
    verdicts, scores = arb.arbitrate(
        ["sig"], np.array([0]), probs, np.array([0.8]), np.array([3.0]),
        ["Pulsar"], ood_mask=np.array([False]),
    )
    assert verdicts[0] == "Natural"


def test_confident_interference_not_flagged():
    arb = _make_arbitrator()
    probs = np.array([[0.01, 0.99, 0.0]])
    verdicts, _ = arb.arbitrate(
        ["sig"], np.array([1]), probs, np.array([0.7]), np.array([3.0]),
        ["RFI"], ood_mask=np.array([False]),
    )
    assert verdicts[0] == "Interference"


def test_ood_is_anomaly():
    arb = _make_arbitrator()
    # OOD mask set: likelihood far below any natural candidate -> Anomaly,
    # regardless of classifier confidence.
    probs = np.array([[0.01, 0.99, 0.0]])  # classifier spuriously says RFI
    verdicts, _ = arb.arbitrate(
        ["tone"], np.array([1]), probs, np.array([0.024]), np.array([0.5]),
        ["Narrowband"], ood_mask=np.array([True]),
    )
    assert verdicts[0] == "Anomaly"


def test_low_confidence_is_candidate():
    arb = _make_arbitrator()
    # Not OOD, but classifier uncertain and not strongly significant.
    probs = np.array([[0.45, 0.45, 0.10]])
    verdicts, _ = arb.arbitrate(
        ["sig"], np.array([2]), probs, np.array([0.20]), np.array([5.0]),
        ["Unknown"], ood_mask=np.array([False]),
    )
    assert verdicts[0] == "Candidate — Requires Review"


def test_batch_fdr_control():
    """All-in-distribution batch -> no false anomalies."""
    arb = _make_arbitrator()
    n = 20
    probs = np.tile(np.array([[0.95, 0.05, 0.0]]), (n, 1))
    verdicts, _ = arb.arbitrate(
        [f"s{i}" for i in range(n)], np.zeros(n, dtype=int), probs,
        np.full(n, 0.9), np.full(n, 2.0), ["Pulsar"] * n,
        ood_mask=np.zeros(n, dtype=bool),
    )
    assert set(verdicts) == {"Natural"}
