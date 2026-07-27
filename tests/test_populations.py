"""Tests for axiom.data.populations (real-waterfall manifold assembly).

The full assembly requires the four pinned filterbanks to be cached (they are
downloaded once via axiom.data.provenance). Tests are skipped otherwise so CI
without network / data still runs. Config-hash and windowing logic that need no
data are tested unconditionally.
"""
import os

import numpy as np
import pytest

from axiom.data.populations import (
    ANOMALY_CLASSES,
    CLASS_ARTIFICIAL,
    CLASS_NAMES,
    CLASS_PULSAR,
    CLASS_RFI,
    MANIFOLD_PLAN,
    NORMAL_CLASSES,
    WindowPlan,
    _config_hash,
    _iter_windows,
    build_manifold,
)
from axiom.dsp.waterfall import N_WATERFALL_FEATURES


# ---------------------------------------------------------------------------
# Data-free logic.
# ---------------------------------------------------------------------------
def test_config_hash_is_stable_and_sensitive():
    h1 = _config_hash(MANIFOLD_PLAN)
    h2 = _config_hash(MANIFOLD_PLAN)
    assert h1 == h2 and len(h1) == 16
    changed = MANIFOLD_PLAN[:-1] + (
        WindowPlan(**{**MANIFOLD_PLAN[-1].__dict__, "max_windows": 99}),
    )
    assert _config_hash(changed) != h1


def test_iter_windows_time_axis():
    data = np.arange(100 * 8, dtype=float).reshape(100, 8)
    freqs = np.linspace(1400, 1300, 8)
    plan = WindowPlan("x", CLASS_PULSAR, "time", window=40, stride=20,
                      max_windows=10, dm_max=100.0, n_dm=16, n_subbands=8)
    wins = list(_iter_windows(data, freqs, plan))
    assert len(wins) == 4  # starts 0,20,40,60
    for wd, wf in wins:
        assert wd.shape == (40, 8)
        assert wf.shape == (8,)


def test_iter_windows_freq_axis_and_cap():
    data = np.arange(20 * 100, dtype=float).reshape(20, 100)
    freqs = np.linspace(1400, 1300, 100)
    plan = WindowPlan("x", CLASS_RFI, "freq", window=25, stride=25,
                      max_windows=2, dm_max=100.0, n_dm=16, n_subbands=8)
    wins = list(_iter_windows(data, freqs, plan))
    assert len(wins) == 2  # capped by max_windows (4 possible)
    for wd, wf in wins:
        assert wd.shape == (20, 25)
        assert wf.shape == (25,)


def test_iter_windows_rejects_oversized_window():
    data = np.zeros((10, 10))
    freqs = np.linspace(1400, 1300, 10)
    plan = WindowPlan("x", CLASS_PULSAR, "time", window=50, stride=10,
                      max_windows=5, dm_max=100.0, n_dm=16, n_subbands=8)
    with pytest.raises(ValueError):
        list(_iter_windows(data, freqs, plan))


def test_class_partition_is_disjoint_and_complete():
    assert set(NORMAL_CLASSES).isdisjoint(ANOMALY_CLASSES)
    assert set(NORMAL_CLASSES) | set(ANOMALY_CLASSES) == set(range(len(CLASS_NAMES)))


# ---------------------------------------------------------------------------
# Full manifold (requires cached filterbanks).
# ---------------------------------------------------------------------------
def _all_cached():
    from axiom.data.provenance import get_spec, resolve_cache_dir
    cdir = resolve_cache_dir()
    return all(
        os.path.exists(os.path.join(cdir, get_spec(p.dataset).filename))
        for p in MANIFOLD_PLAN
    )


pytestmark_real = pytest.mark.skipif(
    not _all_cached(), reason="pinned filterbanks not cached")


@pytestmark_real
def test_build_manifold_shapes_and_classes():
    m = build_manifold(cache=True)
    assert m.X.ndim == 2 and m.X.shape[1] == N_WATERFALL_FEATURES
    assert m.X.shape[0] == m.y.shape[0]
    assert np.all(np.isfinite(m.X))
    counts = m.counts()
    for name in CLASS_NAMES:
        assert counts[name] >= 1
    assert m.normal_mask().sum() + m.anomaly_mask().sum() == m.X.shape[0]


@pytestmark_real
def test_manifold_cache_roundtrip_is_identical():
    a = build_manifold(cache=True)
    b = build_manifold(cache=True)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.y, b.y)
    assert a.config_hash == b.config_hash


@pytestmark_real
def test_manifold_physics_separates_classes():
    m = build_manifold(cache=True)
    fi = {n: i for i, n in enumerate(m.feature_names)}

    def cls_mean(code, feat):
        return float(m.X[m.y == code][:, fi[feat]].mean())

    # Astrophysical pulsar is dispersed; terrestrial/artificial are at DM~0.
    assert cls_mean(CLASS_PULSAR, "best_dm") > 5.0
    assert cls_mean(CLASS_RFI, "best_dm") < 5.0
    # The Voyager 1 carrier is a genuinely narrowband signal: its measured
    # narrowband_fraction is high (a real spectral property of the carrier,
    # not a planted anchor). RFI here is broadband GBT observation, so its
    # narrowband fraction is lower.
    assert cls_mean(CLASS_ARTIFICIAL, "narrowband_fraction") > \
        cls_mean(CLASS_RFI, "narrowband_fraction")
