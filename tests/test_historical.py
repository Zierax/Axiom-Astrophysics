"""Tests for axiom.historical audit (curated lane is offline; real lane gated)."""
import os

import numpy as np
import pytest

from axiom.historical import run_curated_audit


def test_curated_runs_and_flags_off_manifold():
    res = run_curated_audit()
    assert len(res.rows) >= 20  # curated historical anomalies list
    # Narrowband/telemetry placeholder rows must be detected as placeholders.
    placeholders = [r for r in res.rows if r.is_placeholder]
    assert placeholders, "expected placeholder rows to be flagged"
    for r in placeholders:
        assert r.kind in {"Narrowband", "Telemetry", "Transmitted"}
    # Every placeholder row must be ranked OOD (off both HTRU2 clusters).
    verdicts = dict((n, v) for n, _, _, v, _ in res.ranked)
    for r in placeholders:
        assert verdicts.get(r.name) == "OOD", f"{r.name} should be OOD"
    # Scores are finite and the natural floor is finite.
    assert np.all(np.isfinite(res.density_score))
    assert np.isfinite(res.natural_floor)


def test_curated_real_signals_on_manifold():
    res = run_curated_audit()
    verdicts = dict((n, v) for n, _, _, v, _ in res.ranked)
    # Real FRBs / pulsars should sit on-manifold (not flagged as the OOD anchors).
    for name in ["FRB 010724 (Lorimer Burst)", "Crab Pulsar (PSR B0531+21)"]:
        assert verdicts.get(name) == "on-manifold", f"{name} should be on-manifold"


def _has_astropy() -> bool:
    try:
        import astropy
        return True
    except ImportError:
        return False


def _catalogs_cached() -> bool:
    return os.path.isfile(os.path.join("data", "historical_cache", "density_real.pkl"))


@pytest.mark.skipif(not (_catalogs_cached() and _has_astropy()),
                    reason="real catalogs not fetched/cached or astropy not installed")
def test_real_catalog_audit_runs():
    from axiom.historical import run_real_catalog_audit
    res = run_real_catalog_audit()
    assert res.n_objects > 1000
    assert np.all(np.isfinite(res.pvals))
    assert res.pvals.min() >= 0.0 and res.pvals.max() <= 1.0
    top = res.top(5)
    assert len(top) == 5
    assert "object_id" in top.columns


def test_unclassified_lane_c_protocol():
    from axiom.historical import run_unclassified_htru2_audit
    res = run_unclassified_htru2_audit()
    # The uncertainty band yields a non-empty unclassified pool.
    assert len(res.indices) > 0
    # p_max is a conformal p-value in [0, 1] and finite.
    assert np.all(np.isfinite(res.p_max))
    assert res.p_max.min() >= 0.0 and res.p_max.max() <= 1.0
    # top() ranks by p_max ascending and returns the requested columns.
    top = res.top(5)
    assert len(top) == 5
    assert "p_max" in top.columns
    assert "ensemble_p_pulsar" in top.columns
    # p_max is monotonically non-decreasing down the ranked list.
    assert (top["p_max"].to_numpy() == np.sort(top["p_max"].to_numpy())).all()
