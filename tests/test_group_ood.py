"""Tests for axiom.stats.group_ood and the population manifold.

Pure statistics/detector logic runs without network. The full population
protocols run only when the catalog artifacts are cached (skipped otherwise so
offline CI stays green).
"""
import os

import numpy as np
import pytest

from axiom.stats.group_ood import (
    GroupOODError,
    _auroc,
    _bootstrap_ci,
    _conformal_threshold,
    _MahalanobisDetector,
)


# ---------------------------------------------------------------------------
# Pure logic (no network).
# ---------------------------------------------------------------------------
def test_auroc_perfect_and_tie():
    assert _auroc(np.array([0.0, 1.0, 2.0]), np.array([5.0, 6.0])) == 1.0
    same = np.arange(8.0)
    assert abs(_auroc(same, same) - 0.5) < 1e-9


def test_conformal_threshold_monotone_in_alpha():
    s = np.linspace(0, 1, 200)
    assert _conformal_threshold(s, 0.01) >= _conformal_threshold(s, 0.5)


def test_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    vals = rng.normal(5.0, 1.0, size=500)
    lo, hi = _bootstrap_ci(vals, lambda idx: float(vals[idx].mean()),
                           n_boot=500, seed=1)
    assert lo < 5.0 < hi


def test_bootstrap_ci_degenerate():
    lo, hi = _bootstrap_ci(np.array([1.0]), lambda idx: 1.0, n_boot=10, seed=0)
    assert np.isnan(lo) and np.isnan(hi)


def test_mahalanobis_detector_separates_clusters():
    rng = np.random.default_rng(3)
    Xa = rng.normal(0.0, 1.0, size=(80, 6))
    Xb = rng.normal(10.0, 1.0, size=(80, 6))
    X = np.vstack([Xa, Xb])
    y = np.array([0] * 80 + [1] * 80)
    det = _MahalanobisDetector().fit(X, y)
    inlier = det.score(np.zeros((1, 6)))
    outlier = det.score(np.full((1, 6), 50.0))
    assert outlier[0] > inlier[0]


def test_detector_rejects_non_finite():
    X = np.zeros((4, 3))
    X[0, 0] = np.nan
    with pytest.raises(GroupOODError):
        _MahalanobisDetector().fit(X, np.zeros(4))


def test_detector_not_fitted():
    with pytest.raises(GroupOODError):
        _MahalanobisDetector().score(np.zeros((2, 3)))


# ---------------------------------------------------------------------------
# Real population protocols (skipped unless catalogs are cached).
# ---------------------------------------------------------------------------
def _population_available():
    if os.environ.get("AXIOM_OFFLINE", "0").strip().lower() in ("1", "true", "yes"):
        # Offline is fine *if* everything is already cached; probe the cache.
        pass
    try:
        from axiom.data.catalogs import REGISTRY
        from axiom.data.downloader import resolve_cache_dir
        cdir = resolve_cache_dir()
        for key in ("atnf_pulsars", "chime_frb_cat1"):
            if not os.path.exists(os.path.join(cdir, REGISTRY[key].filename)):
                return False
        return os.path.exists(os.path.join("data", "HTRU_2.csv"))
    except Exception:
        return False


_REASON = "population catalogs not cached"


@pytest.mark.skipif(not _population_available(), reason=_REASON)
def test_population_assembles_independent_objects():
    from axiom.data.population import build_population
    pop = build_population(cache=True)
    assert pop.n_objects() > 5000
    # Every object is unique -> group CV is leakage-free by construction.
    assert np.unique(pop.group_ids).size == pop.n_objects()
    assert np.all(np.isfinite(pop.X[:, -4:]))  # indicator block finite
    counts = pop.class_counts()
    assert counts.get("PULSAR", 0) > 100
    assert counts.get("FRB", 0) > 100
    assert counts.get("RFI", 0) > 100


@pytest.mark.skipif(not _population_available(), reason=_REASON)
@pytest.mark.parametrize("seed", [42, 7])
def test_population_classification_is_strong(seed):
    from axiom.data.population import build_population
    from axiom.stats.group_ood import evaluate_population_classification
    pop = build_population(cache=True)
    r = evaluate_population_classification(pop, seed=seed, n_bootstrap=200)
    assert r.mcc >= 0.60
    assert r.weighted_f1 >= 0.90
    assert r.mcc_ci[0] <= r.mcc <= r.mcc_ci[1]
    assert r.passed


@pytest.mark.skipif(not _population_available(), reason=_REASON)
def test_population_ood_frb_is_separable_and_calibrated():
    from axiom.data.population import build_population
    from axiom.stats.group_ood import evaluate_population_ood
    pop = build_population(cache=True)
    r = evaluate_population_ood(pop, novel_class="FRB", seed=42, n_bootstrap=200)
    assert r.auroc >= 0.90
    assert r.novel_tpr >= 0.90
    assert r.normal_fpr <= r.alpha + 0.10
    assert r.conformal_coverage >= 1.0 - r.alpha - 0.10
    assert r.passed
