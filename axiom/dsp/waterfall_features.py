"""Native frequency-resolved descriptors for real telescope waterfalls.

Unlike the 8-D HTRU2 manifold placement (which maps a signal's measured S/N and
DM onto the survey feature space), these descriptors are computed *directly*
from the 2-D spectrogram (frequency x time) of a real observation. They capture
the physics the manifold cannot: narrowband occupancy, drift, spectral
kurtosis, bandwidth and concentration. They are used both as a reported
morphology and as a frequency-resolved term in the arbitrator's composite score,
so the verdict on a real Breakthrough Listen observation rests on its own
measured properties rather than on synthetic feature geometry.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "compute_waterfall_features",
    "narrowband_window_features",
    "waterfall_narrowband_score",
    "carrier_detection_score",
    "FrequencyResolvedScorer",
    "DescriptorConformalDetector",
]

#: Descriptor keys, in the stable order used by the numeric feature vector the
#: conformal detector operates on. Every key is produced by
#: :func:`compute_waterfall_features`.
DESCRIPTOR_VECTOR_KEYS: tuple = (
    "peak_snr",
    "integrated_snr",
    "occupancy",
    "spectral_kurtosis",
    "drift_rate",
    "bandwidth",
    "concentration",
    "spectral_flatness",
    "sub_bin_flatness",
    "slice_kurtosis",
)


def _robust_background(spec: np.ndarray) -> tuple[float, float]:
    """Median/MAD of the per-channel mean power (robust to channel offset).

    Using the *channel-axis* (not the flattened samples) avoids a broadband
    observation being mistaken for a high background: a uniform offset across
    all channels must not inflate the threshold for narrowband detection.
    """
    chan_power = spec.mean(axis=1).astype(np.float64)
    med = float(np.median(chan_power))
    mad = float(np.median(np.abs(chan_power - med))) + 1e-9
    return med, 1.4826 * mad


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector (0 = uniform, 1 = concentrated)."""
    v = np.asarray(values, dtype=np.float64)
    v = v[v >= 0.0]
    if v.size == 0 or v.sum() <= 0.0:
        return 0.0
    v = np.sort(v)
    n = v.size
    cum = np.cumsum(v)
    # Gini = (2 * sum(i*x_i) - (n+1)*sum(x_i)) / (n * sum(x_i))
    idx = np.arange(1, n + 1)
    g = (2.0 * np.sum(idx * v) - (n + 1.0) * cum[-1]) / (n * cum[-1] + 1e-12)
    return float(np.clip(g, 0.0, 1.0))


def compute_waterfall_features(spec: np.ndarray) -> dict:
    """Compute frequency-resolved descriptors from a 2-D spectrogram.

    Parameters
    ----------
    spec : np.ndarray
        2-D array ``(n_freq_channels, n_time_samples)`` (a single integration is
        fine; the time axis is used for drift, the frequency axis for occupancy).

    Returns
    -------
    dict with the keys below (all finite floats):
        peak_snr, integrated_snr, occupancy, spectral_kurtosis,
        drift_rate, bandwidth, concentration, spectral_flatness.
    """
    spec = np.asarray(spec, dtype=np.float64)
    if spec.ndim == 1:
        spec = spec.reshape(1, -1)
    if spec.ndim != 2 or spec.size == 0:
        raise ValueError(f"spec must be a non-empty 2-D array, got {spec.shape}")

    med, mad = _robust_background(spec)
    noise = max(1.5 * 1.4826 * mad, 1e-9)  # robust std estimate

    # Per-channel integrated power -> frequency-axis morphology.
    chan_power = spec.mean(axis=1)
    peak_snr = float(np.clip((chan_power.max() - med) / noise, 0.0, 1e4))
    integrated_snr = float(np.clip((chan_power.sum() - med * chan_power.size) / noise,
                                   0.0, 1e6))

    # Occupancy: fraction of channels whose power exceeds median + 5*noise.
    thr = med + 5.0 * noise
    occupancy = float(np.mean(chan_power > thr))

    # Spectral kurtosis: non-Gaussianity of the channel-power distribution.
    mp = chan_power - chan_power.mean()
    var = float(chan_power.var())
    kurt = float(np.mean(mp ** 4) / (var ** 2 + 1e-12)) - 3.0 if var > 0 else 0.0

    # Drift rate: power-weighted centroid per time slice, then slope.
    # argmax is noisy for spectra with sub-dominant sidelobes; the power-
    # weighted centroid gives a sub-bin estimate that is robust to noise.
    t = np.arange(spec.shape[1], dtype=np.float64)
    centroids = np.zeros(spec.shape[1], dtype=np.float64)
    for ti in range(spec.shape[1]):
        col = spec[:, ti]
        col_clip = np.clip(col - med, 0.0, None)
        wsum = col_clip.sum()
        if wsum > 0:
            centroids[ti] = np.average(np.arange(spec.shape[0], dtype=np.float64),
                                       weights=col_clip)
        else:
            centroids[ti] = float(np.argmax(col))
    if spec.shape[1] > 1:
        slope, _intercept = np.polyfit(t, centroids, 1)
        drift_rate = float(slope)
    else:
        drift_rate = 0.0

    # Bandwidth: spread of the dominant narrowband region (channels above thr).
    hit_ch = np.where(chan_power > thr)[0]
    if hit_ch.size > 1:
        bandwidth = float(hit_ch.max() - hit_ch.min() + 1)
    elif hit_ch.size == 1:
        bandwidth = 1.0
    else:
        bandwidth = 0.0
    bandwidth = float(bandwidth / max(chan_power.size, 1))

    concentration = _gini(chan_power)
    # Spectral flatness: geometric/arithmetic mean of channel power (low = tonal).
    pos = chan_power[chan_power > 0.0]
    if pos.size > 0 and pos.mean() > 0:
        gmean = float(np.exp(np.mean(np.log(pos))))
        spectral_flatness = float(gmean / (pos.mean() + 1e-12))
    else:
        spectral_flatness = 1.0

    # Sub-bin spectral flatness: flatness computed over the peak channel and
    # its immediate neighbours (±1 channel). A pure carrier tone has this
    # ratio close to 0 (energy in one bin), while broadband noise approaches 1.
    peak_idx = int(np.argmax(chan_power))
    lo = max(0, peak_idx - 1)
    hi = min(spec.shape[0], peak_idx + 2)
    sub = chan_power[lo:hi]
    sub_pos = sub[sub > 0.0]
    if sub_pos.size > 0 and sub_pos.mean() > 0:
        sub_gmean = float(np.exp(np.mean(np.log(sub_pos))))
        sub_bin_flatness = float(sub_gmean / (sub_pos.mean() + 1e-12))
    else:
        sub_bin_flatness = 1.0

    # Spectral kurtosis on the brightest time slice: a carrier signal is
    # non-Gaussian in a single integration, whereas broadband noise is Gaussian.
    brightest_t = int(np.argmax(spec.max(axis=0)))
    brightest_slice = spec[:, brightest_t]
    bs = brightest_slice - brightest_slice.mean()
    bs_var = float(brightest_slice.var())
    slice_kurt = (float(np.mean(bs ** 4) / (bs_var ** 2 + 1e-12)) - 3.0
                  if bs_var > 0 else 0.0)

    return {
        "peak_snr": float(peak_snr),
        "integrated_snr": float(integrated_snr),
        "occupancy": float(occupancy),
        "spectral_kurtosis": float(np.clip(kurt, -10.0, 1e4)),
        "drift_rate": float(drift_rate),
        "bandwidth": float(bandwidth),
        "concentration": float(concentration),
        "spectral_flatness": float(np.clip(spectral_flatness, 0.0, 1.0)),
        "sub_bin_flatness": float(np.clip(sub_bin_flatness, 0.0, 1.0)),
        "slice_kurtosis": float(np.clip(slice_kurt, -10.0, 1e4)),
    }


def narrowband_window_features(spec: np.ndarray, half_chan: int = 4,
                                half_time: int = 4) -> dict:
    """Characterise the brightest narrowband feature of a 2-D spectrogram.

    A real candidate (or a genuine narrowband tone) is often diluted when the
    full spectrogram is averaged over integrations: most integrations are noise
    and a single tone occupies only a few channels. This helper locates the
    integration where the spectrum is most non-Gaussian, drifts to the dominant
    channel in that integration, and extracts a small
    (``2*half_chan+1`` x ``2*half_time+1``) window around it. Descriptors
    computed on that window reflect the *measured* candidate morphology — the
    same window a SETI pipeline would characterise — instead of the broadband
    average.

    **Adaptive windowing**: when the spectrogram has many more channels than
    time samples (e.g. Voyager 1: 1M channels, 16 integrations), the fixed
    ``half_chan=4`` window is too narrow to capture the carrier against the
    noise baseline. In that regime the window is widened to ``half_chan = max(4,
    n_freq // 200)`` so the carrier plus enough surrounding channels are
    included for a meaningful background estimate.

    Parameters
    ----------
    spec : np.ndarray
        2-D spectrogram ``(n_channels, n_integrations)``.
    half_chan, half_time : int
        Window half-widths (channels / integrations) around the located feature.

    Returns
    -------
    dict of descriptors (same keys as :func:`compute_waterfall_features`).
    """
    spec = np.asarray(spec, dtype=np.float64)
    if spec.ndim == 1:
        spec = spec.reshape(1, -1)
    if spec.ndim != 2 or spec.size == 0:
        raise ValueError(f"spec must be a non-empty 2-D array, got {spec.shape}")
    nc, nt = spec.shape

    # Adaptive window: for high-channel-count / low-time-count spectrograms
    # (e.g. Voyager 1: 1M channels, 16 integrations), widen the channel window
    # so the carrier plus enough background channels are captured.
    adaptive_half_chan = max(half_chan, nc // 200)

    # Robust background so a uniform offset does not bias the locator.
    med = float(np.median(spec))
    mad = float(np.median(np.abs(spec - med))) + 1e-9
    noise = 1.4826 * mad
    snr = (spec - med) / (noise + 1e-12)
    # Brightest integration = the time slice with the highest peak deviation.
    peak_per_int = snr.max(axis=0)
    tbest = int(np.argmax(peak_per_int))
    cbest = int(np.argmax(snr[:, tbest]))
    lo = max(0, cbest - adaptive_half_chan)
    hi = min(nc, cbest + adaptive_half_chan + 1)
    a = max(0, tbest - half_time)
    b = min(nt, tbest + half_time + 1)
    if b - a < 2:
        # Single-integration spectrogram: keep the full time axis.
        seg = spec[lo:hi, :]
    else:
        seg = spec[lo:hi, a:b]
    return compute_waterfall_features(seg)


def waterfall_narrowband_score(features: dict) -> float:
    """Map frequency-resolved descriptors to a bounded narrowband score [0, 1].

    This combines two detection paths:

    1. **Spectral morphology** (concentration, flatness, kurtosis): works well
       for signals that occupy multiple channels (pulsars, broadband RFI with
       tonal components).

    2. **Carrier detection** (time-series kurtosis on the brightest channel):
       specifically targets ultra-narrowband carriers (e.g. Voyager 1) where the
       signal occupies a single channel out of millions. In this regime the
       spectral morphology features are diluted by the noise floor, but the
       time series of the carrier channel is highly non-Gaussian.

    The score is the maximum of both paths, so a signal is flagged when it
    matches *either* narrowband morphology *or* carrier characteristics.
    """
    conc = float(np.clip(features.get("concentration", 0.0), 0.0, 1.0))
    flat = float(np.clip(features.get("spectral_flatness", 1.0), 0.0, 1.0))
    flat_score = float(1.0 - flat)
    kurt = float(features.get("spectral_kurtosis", 0.0))
    kurt_score = float(np.clip(kurt / 20.0, 0.0, 1.0))
    sub_flat = float(np.clip(features.get("sub_bin_flatness", 1.0), 0.0, 1.0))
    sub_flat_score = float(1.0 - sub_flat)
    slice_k = float(features.get("slice_kurtosis", 0.0))
    slice_k_score = float(np.clip(slice_k / 20.0, 0.0, 1.0))
    spectral_raw = (0.25 * conc + 0.20 * flat_score + 0.15 * kurt_score
                    + 0.20 * sub_flat_score + 0.20 * slice_k_score)
    spectral_score = float(np.clip(spectral_raw, 0.0, 1.0))

    # Carrier path: high kurtosis + high slice_kurtosis + low occupancy =
    # ultra-narrowband tone. The carrier score rewards non-Gaussian time-series
    # structure on the brightest channel, independent of spectral flatness.
    occ = float(np.clip(features.get("occupancy", 0.0), 0.0, 1.0))
    occ_score = float(1.0 - occ)  # low occupancy = concentrated = carrier-like
    carrier_raw = 0.40 * kurt_score + 0.35 * slice_k_score + 0.25 * occ_score
    carrier_score = float(np.clip(carrier_raw, 0.0, 1.0))

    return max(spectral_score, carrier_score)


def carrier_detection_score(spec: np.ndarray) -> float:
    """Detect ultra-narrowband carriers via time-series kurtosis on the brightest channel.

    For signals like Voyager 1 (a single carrier channel in a 1M-channel band),
    the spectral morphology features are diluted by noise. This function instead
    operates on the **time series of the brightest frequency channel** — the purest
    view of the carrier — and computes excess kurtosis, spectral kurtosis of the
    channel's power spectrum, and amplitude stability. A pure carrier yields high
    values on all three metrics.

    Returns a bounded score in [0, 1] where higher = more carrier-like.
    """
    spec = np.asarray(spec, dtype=np.float64)
    if spec.ndim == 1:
        spec = spec.reshape(1, -1)
    if spec.ndim != 2 or spec.size == 0:
        return 0.0

    # Find the brightest channel (highest mean power).
    chan_power = spec.mean(axis=1)
    brightest_ch = int(np.argmax(chan_power))
    ts = spec[brightest_ch]

    if ts.size < 4:
        return 0.0

    # 1. Excess kurtosis of the time series.
    ts_centered = ts - ts.mean()
    ts_var = float(ts.var())
    if ts_var > 0:
        excess_kurt = float(np.mean(ts_centered ** 4) / (ts_var ** 2)) - 3.0
        kurt_score = float(np.clip(excess_kurt / 10.0, 0.0, 1.0))
    else:
        kurt_score = 0.0

    # 2. Spectral kurtosis of the channel's power spectrum (FFT).
    fft_power = np.abs(np.fft.rfft(ts)) ** 2
    fft_power = fft_power[1:]  # exclude DC
    if fft_power.size > 2:
        fft_mean = fft_power.mean()
        fft_var = float(fft_power.var())
        if fft_var > 0:
            spec_kurt = float(np.mean((fft_power - fft_mean) ** 4) / (fft_var ** 2)) - 3.0
            spec_kurt_score = float(np.clip(spec_kurt / 10.0, 0.0, 1.0))
        else:
            spec_kurt_score = 0.0
    else:
        spec_kurt_score = 0.0

    # 3. Amplitude stability: coefficient of variation (low = stable carrier).
    if ts.mean() > 0:
        cv = float(ts.std() / abs(ts.mean()))
        stability_score = float(np.clip(1.0 - cv, 0.0, 1.0))
    else:
        stability_score = 0.0

    return float(np.clip(
        0.40 * kurt_score + 0.35 * spec_kurt_score + 0.25 * stability_score,
        0.0, 1.0,
    ))


class FrequencyResolvedScorer:
    """Caches per-signal waterfall features and yields narrowband scores.

    Scores are keyed by signal name. Signals without a feature entry receive a
    neutral score (0.5) so the term is well-defined everywhere.
    """

    def __init__(self, features: dict | None = None):
        self._features = {k: dict(v) for k, v in (features or {}).items()}

    def add(self, name: str, features: dict) -> None:
        self._features[name] = dict(features)

    def score(self, name: str) -> float:
        f = self._features.get(name)
        if not f:
            return 0.5
        return waterfall_narrowband_score(f)


def descriptor_vector(features: dict) -> np.ndarray:
    """Assemble a fixed-order numeric vector from a descriptor dict.

    Missing keys default to 0.0 so a partially-populated descriptor still yields
    a well-defined vector. The order is :data:`DESCRIPTOR_VECTOR_KEYS`.
    """
    return np.array(
        [float(features.get(k, 0.0)) for k in DESCRIPTOR_VECTOR_KEYS],
        dtype=np.float64,
    )


class DescriptorConformalDetector:
    """Split-conformal narrowband detector on real spectrogram descriptors.

    This gives the *primary* OOD path a self-consistent, measurement-driven
    verdict that does not route through the 8-D HTRU2 DM=0 anchor. The anomaly
    score of a signal is its bounded narrowband morphology score
    (:func:`waterfall_narrowband_score`), computed directly from its real 2-D
    spectrogram descriptors. The "normal" null is the set of narrowband scores
    of natural / broadband reference signals (astrophysical and RFI), so a
    signal is anomalous exactly when its measured morphology is more tonal /
    concentrated than the natural population.

    The conformal p-value

        p(x) = (1 + #{ null : s(null) >= s(x) }) / (n_null + 1)

    yields finite-sample false-positive control under exchangeability: flagging
    ``p(x) <= alpha`` bounds the natural false-positive rate at ``alpha``.

    Parameters
    ----------
    alpha : float
        Target false-positive rate for the conformal operating point.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1); got {alpha}.")
        self.alpha = float(alpha)
        self._null_scores: np.ndarray = np.empty(0, dtype=np.float64)
        self._fitted = False

    def fit(self, null_features) -> "DescriptorConformalDetector":
        """Calibrate on the narrowband scores of the natural / broadband null.

        Parameters
        ----------
        null_features : iterable of dict
            Descriptor dicts of the natural / interference reference population.
            Empty or all-empty inputs leave the detector unfitted; callers must
            check :attr:`fitted` and skip fusion when it is False.
        """
        scores = []
        for f in null_features or ():
            if not f:
                continue
            scores.append(float(waterfall_narrowband_score(f)))
        if not scores:
            self._null_scores = np.empty(0, dtype=np.float64)
            self._fitted = False
            return self
        self._null_scores = np.sort(np.asarray(scores, dtype=np.float64))
        self._fitted = True
        return self

    @property
    def fitted(self) -> bool:
        return self._fitted

    def p_value(self, features: dict) -> float:
        """Split-conformal p-value for one descriptor dict (small => anomalous).

        Returns a neutral ``1.0`` (never anomalous) when the detector is
        unfitted or the descriptor is empty, so fusion degrades gracefully.
        """
        if not self._fitted or not features:
            return 1.0
        s = float(waterfall_narrowband_score(features))
        n = self._null_scores.size
        n_ge = int(np.sum(self._null_scores >= s))
        return (1.0 + n_ge) / (n + 1.0)
