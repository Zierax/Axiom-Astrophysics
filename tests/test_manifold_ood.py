"""Tests for axiom.stats.manifold_ood.

Synthetic Gaussian clusters exercise the conformal detector logic without
network/data; a real-manifold test runs the full evaluation when the pinned
filterbanks are cached.
"""
import os

import numpy as np
import pytest

from axiom.stats.manifold_ood import (
    ManifoldConformalDetector,
    ManifoldOODError,
    _auroc,
    _conformal_threshold,
    evaluate_manifold_ood,
)


def test_auroc_perfect_and_random():
    neg = np.array([0.0, 1.0, 2.0, 3.0])
    pos = np.array([10.0, 11.0, 12.0])
    assert _auroc(neg, pos) == 1.0
    # Identical distributions -> AUROC 0.5 (all ties).
    same = np.arange(10.0)
    assert abs(_auroc(same, same) - 0.5) < 1e-9


def test_conformal_threshold_monotone_in_alpha():
    scores = np.linspace(0, 1, 100)
    t_lo = _conformal_threshold(scores, alpha=0.01)
    t_hi = _conformal_threshold(scores, alpha=0.5)
    assert t_lo >= t_hi  # smaller alpha => higher (stricter) threshold


def test_detector_separates_gaussian_clusters():
    rng = np.random.default_rng(0)
    # Two normal classes (codes 0 and 2 are in NORMAL_CLASSES).
    Xa_cls = rng.normal(0.0, 1.0, size=(60, 6))
    Xb_cls = rng.normal(8.0, 1.0, size=(60, 6))
    X = np.vstack([Xa_cls, Xb_cls])
    y = np.array([0] * 60 + [2] * 60)
    det = ManifoldConformalDetector(alpha=0.1).fit(X, y)
    det.calibrate(X)
    # A far-away point must score higher (more anomalous) than an in-cluster one.
    inlier = det.score(np.array([[0.0] * 6]))
    outlier = det.score(np.array([[40.0] * 6]))
    assert outlier[0] > inlier[0]
    assert det.predict(np.array([[40.0] * 6]))[0]
    assert not det.predict(np.array([[0.0] * 6]))[0]


def test_detector_pvalues_in_unit_range():
    rng = np.random.default_rng(1)
    X = rng.normal(0.0, 1.0, size=(50, 5))
    y = np.zeros(50, dtype=int)
    det = ManifoldConformalDetector(alpha=0.1).fit(X, y).calibrate(X)
    p = det.p_values(rng.normal(0.0, 1.0, size=(10, 5)))
    assert np.all((p > 0.0) & (p <= 1.0))


def test_detector_validation():
    det = ManifoldConformalDetector(alpha=0.1)
    with pytest.raises(ManifoldOODError):
        det.score(np.zeros((2, 3)))  # not fitted
    with pytest.raises(ValueError):
        ManifoldConformalDetector(alpha=0.0)
    X = np.random.default_rng(2).normal(size=(20, 4))
    with pytest.raises(ManifoldOODError):
        ManifoldConformalDetector().fit(X, np.zeros(19))  # shape mismatch


# ---------------------------------------------------------------------------
# Real manifold (skipped unless the pinned filterbanks are cached).
# ---------------------------------------------------------------------------
def _all_cached():
    from axiom.data.populations import MANIFOLD_PLAN
    from axiom.data.provenance import get_spec, resolve_cache_dir
    cdir = resolve_cache_dir()
    return all(
        os.path.exists(os.path.join(cdir, get_spec(p.dataset).filename))
        for p in MANIFOLD_PLAN
    )


@pytest.mark.skipif(not _all_cached(), reason="pinned filterbanks not cached")
@pytest.mark.parametrize("seed", [42, 7, 123])
def test_real_manifold_ood_is_valid(seed):
    r = evaluate_manifold_ood(alpha=0.1, seed=seed)
    # Lane 1 is built from a small number of individual real observations
    # (one per class), windowed into a limited number of non-independent
    # segments. We therefore do NOT assert a strong AUROC / TPR here -- that
    # would encode a by-construction separation. Instead we assert the detector
    # runs and produces statistically valid, finite outputs:
    assert 0.0 <= r.auroc <= 1.0
    all_p = list(r.anomaly_pvalues) + list(r.normal_pvalues)
    assert len(all_p) > 0
    assert np.all(np.isfinite(all_p))
    assert np.all((np.asarray(all_p) > 0.0) & (np.asarray(all_p) <= 1.0))
    # Conformal false-positive control at the target level (finite-sample slack).
    assert r.normal_fpr <= r.alpha + 0.10
    assert r.conformal_coverage >= 1.0 - r.alpha - 0.10
