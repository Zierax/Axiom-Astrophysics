"""Commensurate physical featurizer for the population-scale catalog manifold.

Heterogeneous surveys (ATNF pulsars, CHIME/FRB, HTRU2 RFI) are mapped onto a
**single, physically-interpretable feature space** so that thousands of
independent real objects live on one commensurate manifold. Unlike the waterfall
featurizer (which operates on raw dynamic spectra), this map ingests the
normalized catalog schema produced by :mod:`axiom.data.catalogs` — one row per
independent astronomical object.

Every feature is a measured or catalog-derived physical quantity, applied
**identically to every class** (no per-class special-casing, no hand-tuned anchor
point). Quantities that are undefined for a class (e.g. rotation period for a
one-off FRB, sky position for an HTRU2 candidate) are represented as ``NaN`` in
the continuous block and flagged by an explicit binary *presence* indicator, so
missingness is modelled rather than silently imputed. Fold-local imputation
(:func:`impute_fit` / :func:`impute_apply`) fills the continuous block using
**training-fold** statistics only, preventing leakage.

Feature groups
--------------
Continuous (physical):
    log_dm            log10(1 + DM)                      dispersion magnitude
    dm_excess         DM - DM_gal_max                    extragalactic excess
    dm_excess_ratio   DM / (1 + DM_gal_max)              scale-free excess
    abs_glat          |Galactic latitude| [deg]          sky concentration
    log_width         log10(width_ms)                    pulse/burst duration
    log_snr           log10(1 + S/N)                     detection significance
    log_period        log10(period_s)                    rotation period
    log_duty          log10(duty cycle)                  width / period

Presence indicators (always finite, in {0, 1}):
    has_glat, has_width, has_snr, has_period

The DM-excess axes encode the single most physically decisive discriminator
between Galactic (pulsar/RRAT/magnetar) and extragalactic (FRB) dispersion, while
period/duty separate rotation-powered sources from bursts and DM≈0 separates
terrestrial RFI from all astrophysical dispersion.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Continuous physical feature names, in stable order.
CONTINUOUS_FEATURE_NAMES: Tuple[str, ...] = (
    "log_dm",
    "dm_excess",
    "dm_excess_ratio",
    "abs_glat",
    "log_width",
    "log_snr",
    "log_period",
    "log_duty",
)

#: Binary presence-indicator names, in stable order.
INDICATOR_FEATURE_NAMES: Tuple[str, ...] = (
    "has_glat",
    "has_width",
    "has_snr",
    "has_period",
)

#: Full feature name vector (continuous block followed by indicator block).
PHYSICAL_FEATURE_NAMES: Tuple[str, ...] = (
    CONTINUOUS_FEATURE_NAMES + INDICATOR_FEATURE_NAMES
)

#: Number of features emitted per object.
N_PHYSICAL_FEATURES: int = len(PHYSICAL_FEATURE_NAMES)

#: Index (in the full vector) at which the continuous block ends.
N_CONTINUOUS: int = len(CONTINUOUS_FEATURE_NAMES)

#: Required schema columns consumed by the featurizer.
_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "dm", "dm_gal_max", "snr", "glat", "width_ms", "period_s",
)

_EPS = 1e-12


class PhysicalFeatureError(RuntimeError):
    """Raised on malformed catalog input to the physical featurizer."""


def _column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame.columns:
        raise PhysicalFeatureError(f"input frame missing required column {name!r}.")
    return pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)


def featurize_frame(frame: pd.DataFrame) -> Tuple[np.ndarray, Tuple[str, ...]]:
    """Map a normalized catalog frame to the commensurate physical feature matrix.

    Parameters
    ----------
    frame : pandas.DataFrame
        Normalized catalog rows following :data:`axiom.data.catalogs.SCHEMA`
        (must contain at least :data:`_REQUIRED_COLUMNS`). One row per object.

    Returns
    -------
    X : ndarray of shape (n_objects, N_PHYSICAL_FEATURES)
        Continuous block (columns ``0:N_CONTINUOUS``) may contain ``NaN`` where a
        quantity is physically undefined for the object; the indicator block
        (columns ``N_CONTINUOUS:``) is always finite and in ``{0, 1}``.
    names : tuple of str
        :data:`PHYSICAL_FEATURE_NAMES`.

    Notes
    -----
    The continuous block is deliberately *not* imputed here; use
    :func:`impute_fit`/:func:`impute_apply` with training-fold data to fill it
    without leakage. This keeps the featurizer stateless and deterministic.
    """
    if not isinstance(frame, pd.DataFrame):
        raise PhysicalFeatureError("frame must be a pandas DataFrame.")
    if frame.empty:
        raise PhysicalFeatureError("frame is empty.")

    dm = _column(frame, "dm")
    dm_gal = _column(frame, "dm_gal_max")
    snr = _column(frame, "snr")
    glat = _column(frame, "glat")
    width = _column(frame, "width_ms")
    period = _column(frame, "period_s")

    if np.any(~np.isfinite(dm)):
        raise PhysicalFeatureError(
            "DM column contains non-finite values; normalize the catalog first."
        )
    if np.any(dm < 0.0):
        raise PhysicalFeatureError("DM column contains negative values.")

    n = dm.shape[0]

    # Presence indicators (finite by construction).
    has_glat = np.isfinite(glat).astype(np.float64)
    has_width = (np.isfinite(width) & (width > 0.0)).astype(np.float64)
    has_snr = np.isfinite(snr).astype(np.float64)
    has_period = (np.isfinite(period) & (period > 0.0)).astype(np.float64)

    # Galactic DM ceiling: where absent, fall back to DM itself so the excess is
    # zero (a conservative, non-informative default for that object).
    dm_gal_safe = np.where(np.isfinite(dm_gal), dm_gal, dm)

    log_dm = np.log10(1.0 + dm)
    dm_excess = dm - dm_gal_safe
    dm_excess_ratio = dm / (1.0 + np.clip(dm_gal_safe, 0.0, None))

    abs_glat = np.where(np.isfinite(glat), np.abs(glat), np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        log_width = np.where(has_width > 0, np.log10(np.clip(width, _EPS, None)), np.nan)
        log_snr = np.where(has_snr > 0, np.log10(1.0 + np.clip(snr, 0.0, None)), np.nan)
        log_period = np.where(has_period > 0, np.log10(np.clip(period, _EPS, None)), np.nan)
        both = (has_width > 0) & (has_period > 0)
        duty = np.where(both, (width / 1000.0) / np.clip(period, _EPS, None), np.nan)
        log_duty = np.where(np.isfinite(duty) & (duty > 0), np.log10(np.clip(duty, _EPS, None)), np.nan)

    continuous = np.column_stack([
        log_dm, dm_excess, dm_excess_ratio, abs_glat,
        log_width, log_snr, log_period, log_duty,
    ])
    indicators = np.column_stack([has_glat, has_width, has_snr, has_period])
    X = np.hstack([continuous, indicators]).astype(np.float64)

    if X.shape != (n, N_PHYSICAL_FEATURES):  # pragma: no cover - invariant
        raise PhysicalFeatureError(
            f"internal feature shape error: got {X.shape}, "
            f"expected {(n, N_PHYSICAL_FEATURES)}."
        )
    if not np.all(np.isfinite(X[:, N_CONTINUOUS:])):  # pragma: no cover
        raise PhysicalFeatureError("indicator block produced non-finite values.")
    return X, PHYSICAL_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Leakage-free imputation of the continuous block.
# ---------------------------------------------------------------------------
def impute_fit(X_train: np.ndarray) -> np.ndarray:
    """Compute per-feature medians of the continuous block from training data.

    Parameters
    ----------
    X_train : ndarray of shape (n, N_PHYSICAL_FEATURES)
        Training-fold feature matrix (as returned by :func:`featurize_frame`).

    Returns
    -------
    ndarray of shape (N_CONTINUOUS,)
        Median of each continuous feature over finite training entries. Features
        with no finite training entry default to 0.0.
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    if X_train.ndim != 2 or X_train.shape[1] != N_PHYSICAL_FEATURES:
        raise PhysicalFeatureError(
            f"X_train must have {N_PHYSICAL_FEATURES} columns; got {X_train.shape}."
        )
    cont = X_train[:, :N_CONTINUOUS]
    medians = np.zeros(N_CONTINUOUS, dtype=np.float64)
    for j in range(N_CONTINUOUS):
        col = cont[:, j]
        finite = col[np.isfinite(col)]
        medians[j] = float(np.median(finite)) if finite.size else 0.0
    return medians


def impute_apply(X: np.ndarray, medians: np.ndarray) -> np.ndarray:
    """Fill NaNs in the continuous block using precomputed training medians.

    Parameters
    ----------
    X : ndarray of shape (n, N_PHYSICAL_FEATURES)
        Feature matrix to impute (train, calibration or test).
    medians : ndarray of shape (N_CONTINUOUS,)
        Output of :func:`impute_fit` on the corresponding training fold.

    Returns
    -------
    ndarray
        A finite copy of ``X`` with continuous NaNs replaced by ``medians``.
    """
    X = np.array(X, dtype=np.float64, copy=True)
    medians = np.asarray(medians, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != N_PHYSICAL_FEATURES:
        raise PhysicalFeatureError(
            f"X must have {N_PHYSICAL_FEATURES} columns; got {X.shape}."
        )
    if medians.shape != (N_CONTINUOUS,):
        raise PhysicalFeatureError(
            f"medians must have shape ({N_CONTINUOUS},); got {medians.shape}."
        )
    cont = X[:, :N_CONTINUOUS]
    mask = ~np.isfinite(cont)
    if mask.any():
        fill = np.broadcast_to(medians, cont.shape)
        cont[mask] = fill[mask]
    X[:, :N_CONTINUOUS] = cont
    if not np.all(np.isfinite(X)):  # pragma: no cover - invariant
        raise PhysicalFeatureError("imputation left non-finite values.")
    return X


def featurize_imputed(
    frame: pd.DataFrame, medians: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...]]:
    """Convenience: featurize and self-impute (for exploratory, single-set use).

    Returns ``(X_imputed, medians, names)``. If ``medians`` is None it is fit on
    ``frame`` itself — appropriate only for exploratory/whole-set use, **never**
    for cross-validated evaluation (use fold-local :func:`impute_fit`).
    """
    X, names = featurize_frame(frame)
    if medians is None:
        medians = impute_fit(X)
    return impute_apply(X, medians), medians, names
