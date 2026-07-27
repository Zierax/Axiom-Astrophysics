"""Tests for axiom.dsp.physical_features (no network required).

The commensurate physical featurizer is exercised on synthetic catalog frames
covering every class regime and every missingness pattern.
"""
import numpy as np
import pandas as pd
import pytest

from axiom.dsp.physical_features import (
    INDICATOR_FEATURE_NAMES,
    N_CONTINUOUS,
    N_PHYSICAL_FEATURES,
    PHYSICAL_FEATURE_NAMES,
    PhysicalFeatureError,
    featurize_frame,
    featurize_imputed,
    impute_apply,
    impute_fit,
)


def _frame():
    # One pulsar, one FRB, one RFI row with class-appropriate missingness.
    return pd.DataFrame({
        "object_id": ["p1", "f1", "r1"],
        "source": ["atnf", "chime", "htru2"],
        "class_name": ["PULSAR", "FRB", "RFI"],
        "dm": [26.8, 716.6, 0.0],
        "dm_gal_max": [40.0, 40.0, 0.0],
        "snr": [np.nan, 18.0, 12.0],
        "glat": [-5.0, 21.3, np.nan],
        "glon": [120.0, 200.0, np.nan],
        "width_ms": [10.0, 0.3, np.nan],
        "period_s": [0.714, np.nan, np.nan],
    })


def test_shapes_and_names():
    X, names = featurize_frame(_frame())
    assert X.shape == (3, N_PHYSICAL_FEATURES)
    assert names == PHYSICAL_FEATURE_NAMES
    assert len(PHYSICAL_FEATURE_NAMES) == N_PHYSICAL_FEATURES
    assert N_CONTINUOUS < N_PHYSICAL_FEATURES


def test_indicator_block_is_finite_and_binary():
    X, _ = featurize_frame(_frame())
    ind = X[:, N_CONTINUOUS:]
    assert np.all(np.isfinite(ind))
    assert set(np.unique(ind)).issubset({0.0, 1.0})
    assert ind.shape[1] == len(INDICATOR_FEATURE_NAMES)


def test_missingness_indicators_track_data():
    X, names = featurize_frame(_frame())
    cols = {n: i for i, n in enumerate(names)}
    # RFI (row 2) has no sky position / width / snr(has) / period.
    assert X[2, cols["has_glat"]] == 0.0
    assert X[2, cols["has_width"]] == 0.0
    assert X[2, cols["has_period"]] == 0.0
    assert X[2, cols["has_snr"]] == 1.0
    # Pulsar (row 0) has a period; FRB (row 1) does not.
    assert X[0, cols["has_period"]] == 1.0
    assert X[1, cols["has_period"]] == 0.0


def test_dm_excess_is_extragalactic_for_frb():
    X, names = featurize_frame(_frame())
    cols = {n: i for i, n in enumerate(names)}
    # FRB DM (716.6) hugely exceeds the Galactic ceiling (40) -> large excess.
    assert X[1, cols["dm_excess"]] > 600.0
    # Pulsar DM (26.8) is below its ceiling -> negative excess.
    assert X[0, cols["dm_excess"]] < 0.0
    # RFI has DM == ceiling == 0 -> zero excess.
    assert X[2, cols["dm_excess"]] == 0.0


def test_continuous_missing_encoded_as_nan():
    X, names = featurize_frame(_frame())
    cols = {n: i for i, n in enumerate(names)}
    assert np.isnan(X[2, cols["abs_glat"]])
    assert np.isnan(X[1, cols["log_period"]])
    assert np.isfinite(X[0, cols["log_period"]])


def test_impute_fit_apply_removes_nan_without_leakage():
    X, _ = featurize_frame(_frame())
    med = impute_fit(X)
    assert med.shape == (N_CONTINUOUS,)
    Xf = impute_apply(X, med)
    assert np.all(np.isfinite(Xf))
    # Indicator block is untouched by imputation.
    assert np.array_equal(Xf[:, N_CONTINUOUS:], X[:, N_CONTINUOUS:])


def test_impute_apply_uses_reference_medians():
    X, _ = featurize_frame(_frame())
    med = np.zeros(N_CONTINUOUS)  # force fill value of 0
    Xf = impute_apply(X, med)
    # The RFI abs_glat NaN must be filled with the provided median (0).
    from axiom.dsp.physical_features import CONTINUOUS_FEATURE_NAMES
    j = CONTINUOUS_FEATURE_NAMES.index("abs_glat")
    assert Xf[2, j] == 0.0


def test_featurize_imputed_convenience():
    Xf, med, names = featurize_imputed(_frame())
    assert np.all(np.isfinite(Xf))
    assert med.shape == (N_CONTINUOUS,)


def test_validation_errors():
    with pytest.raises(PhysicalFeatureError):
        featurize_frame(pd.DataFrame())
    with pytest.raises(PhysicalFeatureError):
        featurize_frame(pd.DataFrame({"dm": [1.0]}))  # missing required cols
    bad = _frame()
    bad.loc[0, "dm"] = -1.0
    with pytest.raises(PhysicalFeatureError):
        featurize_frame(bad)
    with pytest.raises(PhysicalFeatureError):
        impute_apply(np.zeros((3, N_PHYSICAL_FEATURES)), np.zeros(N_CONTINUOUS + 1))


def test_determinism():
    X1, _ = featurize_frame(_frame())
    X2, _ = featurize_frame(_frame())
    np.testing.assert_array_equal(np.nan_to_num(X1), np.nan_to_num(X2))
