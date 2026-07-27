"""Real dynamic-spectrum featurization for the AXIOM anomaly engine.

A *waterfall* is a 2-D dynamic spectrum ``S[time, frequency]`` (rows are time
samples, columns are frequency channels, descending frequency when ``foff<0``).
From it we compute a single, deterministic, physics-grounded feature vector used
by every downstream stage (per-class manifold in ``axiom.data.populations``,
conformal OOD in ``axiom.stats.manifold_ood``, and the arbitrator).

The featurizer is intentionally the *only* map from a raw observation to the
manifold: no synthetic geometry, no hand-tuned anchor points. Two physics
properties drive the design:

  * **Dispersion.** A broadband astrophysical pulse (pulsar/FRB) is dispersed by
    the ISM; its arrival time is delayed by ``t = k * DM * (1/f^2 - 1/f_hi^2)``.
    We recover the best-fitting DM by incoherently dedispersing over a grid and
    measuring the integrated S/N of the pulse profile (``dm_snr_sweep``).
  * **Narrowband morphology.** A telemetry/technosignature carrier lives in a
    single frequency channel with no dispersion sweep. Per-channel
    normalisation equalises a lone loud channel, so dispersion carries *no*
    information about it; the *spectral* features (narrowband fraction, spectral
    kurtosis, occupancy) are the real narrowband detectors.

All functions are defensive and deterministic. Malformed input raises
:class:`WaterfallError`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

#: Cold-plasma dispersion constant in MHz^2 pc^-1 cm^3 s (i.e. the coefficient
#: relating DM to delay when frequencies are expressed in MHz and delays in s).
DM_CONSTANT_MHZ2_PC_CM3_S = 4.148808e3

#: Full feature-vector contract. Order is stable and consumed by populations.py,
#: manifold_ood.py and the reports.
WATERFALL_FEATURE_NAMES: Tuple[str, ...] = (
    "best_dm",
    "best_snr",
    "dm0_peak_ratio",
    "dm_curve_sharpness",
    "narrowband_fraction",
    "log_spectral_kurtosis",
    "occupancy",
    "peak_channel_snr",
    "band_kurtosis",
    "band_skewness",
    "tone_persistence",
)
N_WATERFALL_FEATURES = len(WATERFALL_FEATURE_NAMES)


class WaterfallError(ValueError):
    """Raised on malformed dynamic-spectrum input."""


def _validate(wf: np.ndarray, freqs: np.ndarray, tsamp: float) -> np.ndarray:
    if not isinstance(wf, np.ndarray):
        raise WaterfallError("wf must be a numpy array")
    if wf.ndim != 2:
        raise WaterfallError(f"wf must be 2-D (time, frequency); got {wf.ndim}-D")
    if wf.shape[0] < 8 or wf.shape[1] < 2:
        raise WaterfallError(f"wf too small: {wf.shape}")
    if not isinstance(freqs, np.ndarray) or freqs.shape[0] != wf.shape[1]:
        raise WaterfallError(
            f"freqs length ({getattr(freqs,'shape',None)}) must equal wf channels ({wf.shape[1]})"
        )
    if not np.all(np.isfinite(freqs)) or float(np.min(freqs)) <= 0.0:
        raise WaterfallError("freqs must be finite and strictly positive (MHz)")
    if not np.isfinite(tsamp) or tsamp <= 0.0:
        raise WaterfallError(f"tsamp must be finite and positive; got {tsamp}")
    arr = np.asarray(wf, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        # Repair non-finite samples in place via per-column median imputation.
        bad = ~np.isfinite(arr)
        col_med = np.nanmedian(arr, axis=0)
        arr[bad] = np.interp(np.where(bad)[1].astype(float), np.arange(arr.shape[1]), col_med)
    return arr


def normalize_bandpass(wf: np.ndarray) -> np.ndarray:
    """Per-channel bandpass normalisation.

    Each frequency channel is de-meaned and de-scaled by its robust
    (median/MAD) statistics so absolute flux calibration and RFI offsets do not
    dominate. Dead (constant) channels collapse to zero rather than amplifying
    noise.
    """
    wf = np.asarray(wf, dtype=np.float64)
    out = np.empty_like(wf)
    n_t, n_c = wf.shape
    for c in range(n_c):
        ch = wf[:, c]
        med = np.median(ch)
        mad = np.median(np.abs(ch - med)) + 1e-12
        out[:, c] = (ch - med) / (1.4826 * mad)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    # Dead channel (zero variance after normalisation) -> explicit zero.
    col_std = out.std(axis=0)
    out[:, col_std == 0] = 0.0
    return out


def dedisperse_series(
    wf: np.ndarray, freqs: np.ndarray, tsamp: float, dm: float
) -> np.ndarray:
    """Incoherently dedisperse ``wf`` (time, freq) to a 1-D time series.

    Each channel is shifted in time by its ISM delay relative to the
    highest-frequency channel, then summed. Zero-padding at the edges preserves
    the full time length so a pulse aligned at its true arrival index is never
    clipped. Raises :class:`WaterfallError` if the requested DM exceeds the causal
    span of the observation (would require shifting past the array edge).
    """
    wf = np.asarray(wf, dtype=np.float64)
    n_t, n_c = wf.shape
    f_hi = float(np.max(freqs))
    delays_s = DM_CONSTANT_MHZ2_PC_CM3_S * dm * (freqs ** -2 - f_hi ** -2)
    shifts = np.round(delays_s / tsamp).astype(np.int64)
    max_shift = int(np.max(np.abs(shifts)))
    if max_shift >= n_t:
        raise WaterfallError(
            f"DM={dm} requires shift {max_shift} >= n_time {n_t}; reduce dm_max"
        )
    out = np.zeros(n_t, dtype=np.float64)
    for c in range(n_c):
        s = int(shifts[c])
        if s >= 0:
            seg = wf[s:, c]
            out[: n_t - s] += seg
        else:
            seg = wf[: n_t + s, c]
            out[-s:] += seg
    return out


def default_dm_grid(
    freqs: np.ndarray, tsamp: float, dm_max: float, n_dm: int
) -> np.ndarray:
    """Linearly spaced DM trial grid from 0 to ``dm_max`` (inclusive)."""
    if n_dm < 2:
        raise WaterfallError("n_dm must be >= 2")
    return np.linspace(0.0, float(dm_max), int(n_dm))


@dataclass
class DMSweepResult:
    best_dm: float
    best_snr: float
    sweep_dm: np.ndarray
    sweep_snr: np.ndarray
    dm0_snr: float


def dm_snr_sweep(
    wf: np.ndarray,
    freqs: np.ndarray,
    tsamp: float,
    dm_grid: np.ndarray,
    n_subbands: int = 256,
) -> DMSweepResult:
    """Incoherent dedispersion S/N sweep over ``dm_grid``.

    For each trial DM the dynamic spectrum is dedispersed and the integrated
    profile's peak S/N (relative to the profile's own robust scatter) is
    recorded. Returns the best DM/SNR plus the full curve for downstream
    sharpness computation.
    """
    wf = _validate(wf, freqs, tsamp)
    snr_curve = np.empty(len(dm_grid), dtype=np.float64)
    dm0_snr = 0.0
    for i, dm in enumerate(dm_grid):
        try:
            prof = dedisperse_series(wf, freqs, tsamp, dm)
        except WaterfallError:
            snr_curve[i] = 0.0
            continue
        prof = prof - np.median(prof)
        sc = np.std(prof)
        snr_curve[i] = float(np.max(np.abs(prof)) / sc) if sc > 1e-12 else 0.0
        if dm == 0.0 or i == 0:
            dm0_snr = snr_curve[i]
    best_idx = int(np.argmax(snr_curve))
    return DMSweepResult(
        best_dm=float(dm_grid[best_idx]),
        best_snr=float(snr_curve[best_idx]),
        sweep_dm=dm_grid,
        sweep_snr=snr_curve,
        dm0_snr=float(dm0_snr),
    )


def _spectral_kurtosis(ch_power: np.ndarray, seg: int = 64) -> float:
    n = ch_power.shape[0]
    if n < 2 * seg:
        seg = max(1, n // 2)
    if seg < 1:
        return 0.0
    nseg = n // seg
    if nseg < 2:
        var = float(np.var(ch_power))
        mean = float(np.mean(ch_power))
        return float(np.log((var / (mean ** 2 + 1e-12)) + 1e-12))
    windowed = ch_power[: nseg * seg].reshape(nseg, seg)
    power = np.mean(windowed ** 2, axis=1)
    mean_p = power.mean()
    var_p = power.var()
    sk = (nseg / (nseg - 1.0)) * (var_p / (mean_p ** 2 + 1e-12)) - 1.0
    if sk <= -1e-12:
        return -30.0
    return float(np.log(sk + 1e-12))


def extract_features(
    wf: np.ndarray,
    freqs: np.ndarray,
    tsamp: float,
    dm_max: float = 500.0,
    n_dm: int = 101,
    n_subbands: int = 256,
    return_details: bool = False,
):
    """Extract the :data:`N_WATERFALL_FEATURES`-length vector from a waterfall.

    Parameters
    ----------
    wf : (n_time, n_chan) array
        Dynamic spectrum (time rows, frequency columns).
    freqs : (n_chan,) array
        Centre frequency of each channel in MHz (descending if ``foff<0``).
    tsamp : float
        Time sample interval in seconds (must be positive and finite).
    dm_max, n_dm : DM sweep configuration.
    return_details : bool
        If True, return ``(vector, details)`` where ``details["features"]`` is a
        name -> value dict (used by tests and diagnostics).

    Returns
    -------
    np.ndarray of shape (N_WATERFALL_FEATURES,) or ``(vec, details)``.
    """
    wf = _validate(wf, freqs, tsamp)
    norm = normalize_bandpass(wf)
    n_t, n_c = norm.shape

    grid = default_dm_grid(freqs, tsamp, dm_max, n_dm)
    sweep = dm_snr_sweep(norm, freqs, tsamp, grid, n_subbands=n_subbands)

    best_dm = float(sweep.best_dm)
    best_snr = float(sweep.best_snr)
    dm0_peak_ratio = float(sweep.dm0_snr / (best_snr + 1e-12))
    curve = sweep.sweep_snr
    curve_peak = float(np.max(curve)) + 1e-12
    curve_mean = float(np.mean(curve)) + 1e-12
    dm_curve_sharpness = float(np.clip(curve_peak / curve_mean - 1.0, 0.0, 10.0))

    # Spectral features are computed on the bandpass-corrected (per-channel
    # median-subtracted) spectrum, NOT the MAD-scaled ``norm``. MAD-scaling
    # collapses a loud narrowband channel to noise level, destroying the very
    # excess power that marks it; median subtraction preserves it. The robust
    # (median + k*MAD) threshold ignores the lone outlier so pure noise is not
    # flagged while a real carrier is.
    bp = wf - np.median(wf, axis=0)
    ch_power = np.mean(bp ** 2, axis=0)
    med_cp = np.median(ch_power)
    mad_cp = np.median(np.abs(ch_power - med_cp)) + 1e-9
    thr = med_cp + 8.0 * mad_cp
    narrowband_fraction = float(np.mean(ch_power > thr))
    sk = _spectral_kurtosis(ch_power)
    log_spectral_kurtosis = float(sk)

    whole_sd = float(np.std(norm)) + 1e-12
    occupancy = float(np.mean(np.abs(norm) > 5.0 * whole_sd))

    peak_channel_snr = float(np.max(ch_power) / (mad_cp + 1e-12))
    band_kurt = float(_safe_kurt(ch_power))
    band_skew = float(_safe_skew(ch_power))

    # Tonal persistence: a genuine narrowband carrier stays bright in (essentially)
    # the same channel across integrations, whereas transient RFI spikes are not
    # sustained. Computed on the bandpass-corrected (median-subtracted) spectrogram
    # so the lone carrier channel is preserved. This is a general SETI-style
    # discriminator, not tuned to any one source.
    bp_t = wf - np.median(wf, axis=0)
    ch_noise = np.median(np.abs(bp_t - np.median(bp_t))) + 1e-9
    # ``peak_chan`` is a frequency index (ch_power reduces the time axis), so the
    # per-integration column is ``bp_t[:, peak_chan]``.
    peak_chan = int(np.argmax(ch_power))
    col = (bp_t[:, peak_chan] - np.median(bp_t[:, peak_chan])) / (1.4826 * ch_noise)
    tone_persistence = float(np.mean(col >= 5.0))

    feats = {
        "best_dm": best_dm,
        "best_snr": best_snr,
        "dm0_peak_ratio": dm0_peak_ratio,
        "dm_curve_sharpness": dm_curve_sharpness,
        "narrowband_fraction": narrowband_fraction,
        "log_spectral_kurtosis": log_spectral_kurtosis,
        "occupancy": occupancy,
        "peak_channel_snr": peak_channel_snr,
        "band_kurtosis": band_kurt,
        "band_skewness": band_skew,
        "tone_persistence": tone_persistence,
    }
    vec = np.array([feats[k] for k in WATERFALL_FEATURE_NAMES], dtype=np.float64)
    vec = np.nan_to_num(vec, nan=0.0, posinf=1e6, neginf=-1e6)
    if return_details:
        return vec, {"features": feats, "sweep": sweep}
    return vec


def _safe_skew(x: np.ndarray) -> float:
    try:
        from scipy.stats import skew
        return float(skew(x))
    except Exception as exc:
        log.debug("_safe_skew failed: %s", exc)
        return 0.0


def _safe_kurt(x: np.ndarray) -> float:
    try:
        from scipy.stats import kurtosis
        return float(kurtosis(x))
    except Exception as exc:
        log.debug("_safe_kurt failed: %s", exc)
        return 0.0


def load_waterfall(path: str, max_samples: Optional[int] = 200_000):
    """Load a real waterfall from ``.fil`` (SIGPROC) or ``.h5`` (BL gupuspec).

    Returns ``(spectrum, freqs_hz, tsamp_s)`` where ``spectrum`` is
    ``(n_freq, n_time)`` (frequency rows, time columns), median-normalised per
    channel, ready for ``extract_features`` (after a transpose to time x freq).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fil":
        from axiom.dsp.fil_reader import read_fil_spectrum
        spec, hdr = read_fil_spectrum(path, max_samples=max_samples)
        return spec, hdr.freqs_hz, float(hdr.tsamp)
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as h:
            dset = h["data"] if "data" in h else list(h.values())[0]
            arr = np.asarray(dset[:], dtype=np.float64)
            attrs = {k: dset.attrs[k] for k in dset.attrs.keys()}
        if arr.ndim == 3:
            arr = arr[:, 0, :]
        n_f, n_t = arr.shape
        if max_samples is not None and n_t > max_samples:
            arr = arr[:, :max_samples]
            n_t = max_samples
        med = np.median(arr, axis=1, keepdims=True)
        med = np.where(med == 0, 1.0, med)
        spec = arr / med
        fch1 = float(attrs.get("fch1", 0.0)) * 1e6
        foff = float(attrs.get("foff", 0.0)) * 1e6
        freqs = fch1 + foff * np.arange(n_f)
        tsamp = float(attrs.get("tsamp", 1.0))
        if tsamp > 1e3:
            tsamp = tsamp / 1e6
        return spec, freqs, tsamp
    raise WaterfallError(f"Unsupported waterfall format: {ext}")
