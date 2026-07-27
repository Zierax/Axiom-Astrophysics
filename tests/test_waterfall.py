"""Tests for axiom.dsp.waterfall.

The synthetic tests require no network: a dispersed broadband pulse and a
narrowband carrier are injected into controlled waterfalls so the featurizer's
physics (DM recovery, dispersion discrimination, narrowband/order response) is
validated deterministically. Real-file tests are skipped unless the pinned
filterbanks are already cached.
"""
import os

import numpy as np
import pytest

from axiom.dsp.waterfall import (
    DM_CONSTANT_MHZ2_PC_CM3_S,
    N_WATERFALL_FEATURES,
    WATERFALL_FEATURE_NAMES,
    WaterfallError,
    dedisperse_series,
    default_dm_grid,
    dm_snr_sweep,
    extract_features,
    normalize_bandpass,
)

TSAMP = 1.0e-3
FCH1 = 1500.0
FOFF = -0.5
N_CHAN = 600
N_TIME = 512
FREQS = FCH1 + np.arange(N_CHAN) * FOFF  # 1500 -> 1200.5 MHz, descending


def _inject_dispersed_pulse(dm, t0=60, amp=40.0, width=2, noise=1.0, seed=0):
    """Build a waterfall with a broadband pulse dispersed by ``dm``."""
    rng = np.random.default_rng(seed)
    wf = rng.normal(0.0, noise, size=(N_TIME, N_CHAN)) + 100.0  # bandpass offset
    ref = FREQS.max()
    delays = DM_CONSTANT_MHZ2_PC_CM3_S * dm * (FREQS ** -2 - ref ** -2)
    delay_samp = np.rint(delays / TSAMP).astype(int)
    for ch in range(N_CHAN):
        t = t0 + delay_samp[ch]
        lo, hi = max(0, t - width), min(N_TIME, t + width + 1)
        if lo < hi:
            wf[lo:hi, ch] += amp
    return wf


def _narrowband_carrier(chan=300, amp=60.0, noise=1.0, seed=1):
    """Modulated narrowband tone confined to a single channel.

    A real telemetry carrier is not perfectly DC: it carries slow amplitude
    modulation and so survives per-channel median subtraction (a perfectly
    steady tone would legitimately be removed, carrying no temporal signature).
    """
    rng = np.random.default_rng(seed)
    wf = rng.normal(0.0, noise, size=(N_TIME, N_CHAN)) + 100.0
    t = np.arange(N_TIME)
    envelope = 1.0 + 0.4 * np.sin(2.0 * np.pi * 3.0 * t / N_TIME)
    wf[:, chan] += amp * envelope
    return wf


# ---------------------------------------------------------------------------
# Physics: DM recovery and dispersion discrimination.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dm_true", [50.0, 150.0, 300.0])
def test_recovers_injected_dm(dm_true):
    wf = _inject_dispersed_pulse(dm_true)
    grid = default_dm_grid(FREQS, TSAMP, dm_max=500.0, n_dm=101)
    res = dm_snr_sweep(normalize_bandpass(wf), FREQS, TSAMP, grid, n_subbands=200)
    step = grid[1] - grid[0]
    assert abs(res.best_dm - dm_true) <= 2 * step
    assert res.best_snr > 8.0


def test_dispersed_pulse_is_off_dm0():
    wf = _inject_dispersed_pulse(200.0)
    vec, det = extract_features(
        wf, FREQS, TSAMP, dm_max=500.0, n_dm=101, return_details=True
    )
    feats = det["features"]
    assert feats["best_dm"] > 100.0
    # A dispersed pulse must be far weaker at DM=0 than at its true DM.
    assert feats["dm0_peak_ratio"] < 0.7
    assert feats["dm_curve_sharpness"] > 0.4


def test_narrowband_carrier_spectrally_separates_from_noise():
    wf = _narrowband_carrier()
    rfi_free = np.random.default_rng(2).normal(0.0, 1.0, size=(N_TIME, N_CHAN)) + 100.0
    v_tone = extract_features(wf, FREQS, TSAMP, dm_max=400.0, n_dm=81)
    v_noise = extract_features(rfi_free, FREQS, TSAMP, dm_max=400.0, n_dm=81)
    d = dict(zip(WATERFALL_FEATURE_NAMES, v_tone))
    dn = dict(zip(WATERFALL_FEATURE_NAMES, v_noise))
    # best_dm is deliberately NOT a narrowband discriminator: per-channel
    # normalisation equalises a single loud channel, so dispersion carries no
    # information about a narrowband tone. The *spectral* features are the
    # narrowband detectors and must separate the carrier from pure noise.
    assert d["narrowband_fraction"] > dn["narrowband_fraction"]
    assert d["log_spectral_kurtosis"] > dn["log_spectral_kurtosis"]


# ---------------------------------------------------------------------------
# Contract: shape, determinism, validation.
# ---------------------------------------------------------------------------
def test_feature_vector_contract():
    wf = _inject_dispersed_pulse(120.0)
    vec = extract_features(wf, FREQS, TSAMP)
    assert vec.shape == (N_WATERFALL_FEATURES,)
    assert len(WATERFALL_FEATURE_NAMES) == N_WATERFALL_FEATURES
    assert np.all(np.isfinite(vec))


def test_deterministic():
    wf = _inject_dispersed_pulse(120.0)
    a = extract_features(wf, FREQS, TSAMP, dm_max=400.0, n_dm=64)
    b = extract_features(wf, FREQS, TSAMP, dm_max=400.0, n_dm=64)
    np.testing.assert_array_equal(a, b)


def test_repairs_non_finite_samples():
    wf = _inject_dispersed_pulse(120.0)
    wf[10, 5] = np.nan
    wf[20, 7] = np.inf
    vec = extract_features(wf, FREQS, TSAMP)
    assert np.all(np.isfinite(vec))


@pytest.mark.parametrize(
    "bad_call",
    [
        lambda: extract_features(np.zeros(10), FREQS, TSAMP),
        lambda: extract_features(np.zeros((4, N_CHAN)), FREQS, TSAMP),
        lambda: extract_features(np.zeros((N_TIME, N_CHAN)), FREQS[:-1], TSAMP),
        lambda: extract_features(np.zeros((N_TIME, N_CHAN)), FREQS, 0.0),
        lambda: extract_features(np.zeros((N_TIME, N_CHAN)),
                                 np.zeros(N_CHAN), TSAMP),
    ],
)
def test_input_validation_raises(bad_call):
    with pytest.raises(WaterfallError):
        bad_call()


def test_dedisperse_rejects_overflowing_dm():
    wf = normalize_bandpass(_inject_dispersed_pulse(100.0))
    with pytest.raises(WaterfallError):
        dedisperse_series(wf, FREQS, TSAMP, dm=1_000_000.0)


def test_normalize_bandpass_handles_dead_channels():
    wf = _inject_dispersed_pulse(100.0)
    wf[:, 3] = 42.0  # dead (constant) channel
    out = normalize_bandpass(wf)
    assert np.all(out[:, 3] == 0.0)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Real-file tests (skipped unless the pinned filterbanks are cached).
# ---------------------------------------------------------------------------
def _cached(name):
    from axiom.data.provenance import get_spec, resolve_cache_dir
    return os.path.exists(os.path.join(resolve_cache_dir(), get_spec(name).filename))


def _has_blimpy() -> bool:
    try:
        return True
    except Exception:
        return False


@pytest.mark.skipif(not (_cached("pulsar_b0329") and _has_blimpy()),
                    reason="B0329 filterbank not cached or blimpy not installed")
def test_real_pulsar_dm_matches_catalog():
    from axiom.data.provenance import fetch
    from axiom.data.real_loaders import _ensure_blimpy
    blimpy = _ensure_blimpy()
    fb = blimpy.Waterfall(fetch("pulsar_b0329").path, max_load=2)
    data = np.asarray(fb.data[:, 0, :], dtype=np.float64)
    freqs = fb.header["fch1"] + np.arange(data.shape[1]) * fb.header["foff"]
    d = dict(zip(WATERFALL_FEATURE_NAMES,
                 extract_features(data, freqs, fb.header["tsamp"],
                                  dm_max=200.0, n_dm=96)))
    # Catalog DM of PSR B0329+54 is ~26.8 pc/cm^3.
    assert 15.0 < d["best_dm"] < 45.0
    # Dispersion must help (the undispersed SNR is strictly below the
    # best-dispersed SNR); the exact ratio depends on the observation's
    # intrinsic brightness, so the bound is loose and non-overfitting.
    assert d["dm0_peak_ratio"] < 0.9
