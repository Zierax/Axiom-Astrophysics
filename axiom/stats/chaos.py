"""Phase-space chaos / determinism analysis on real signals.

Replaces the previous implementation, which clipped the maximal Lyapunov
exponent to +/-50 (saturating to a non-discriminating constant for every real
waveform). The new implementation:

  1. Estimates the maximal Lyapunov exponent (MLE) via a vectorised Rosenstein
     nearest-neighbour algorithm on a z-scored time-delay embedding.
  2. Computes a *surrogate significance* p-value: the MLE of the real signal is
     compared against the MLEs of `n_surrogates` constrained-random shuffles
     (Amplitude Adjusted Fourier Transform). An ordered (non-chaotic) carrier
     yields an MLE at or below the surrogate distribution -> high p-value
     (consistent with determinism). A stochastic/chaotic burst yields a larger
     MLE -> low p-value.

All randomness is seeded for deterministic, reproducible results. The routine
never returns a hard-coded constant; it returns (mle, surrogate_p), both finite.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
from scipy.spatial.distance import cdist

log = logging.getLogger(__name__)


def reconstruct_phase_space(signal: np.ndarray, dim: int = 5, delay: int = 2) -> np.ndarray:
    """Time-delay embedding of `signal` into `dim` dimensions.

    Returns shape (m, dim) where m = len(signal) - (dim-1)*delay. A signal
    shorter than the embedding window returns a single zero row (handled
    gracefully by callers).
    """
    n = len(signal)
    if n < dim * delay:
        return np.zeros((1, dim), dtype=np.float64)
    m = n - (dim - 1) * delay
    Y = np.zeros((m, dim), dtype=np.float64)
    for i in range(dim):
        Y[:, i] = signal[i * delay : i * delay + m]
    return Y


def _rosenstein_slope(Y: np.ndarray, sample_rate_hz: float, theiler: int = 10) -> float:
    """Return the MLE (1/s) from the Rosenstein divergence slope.

    Returns 0.0 if the embedding is degenerate (too short / no divergence).
    """
    m = Y.shape[0]
    if m < 20:
        return 0.0

    dist_matrix = cdist(Y, Y, metric="euclidean")
    i_idx, j_idx = np.ogrid[:m, :m]
    mask = np.abs(i_idx - j_idx) < theiler
    dist_matrix[mask] = np.inf

    nearest = np.argmin(dist_matrix, axis=1)
    d0 = np.clip(dist_matrix[np.arange(m), nearest], 1e-12, None)

    max_steps = min(8, m - theiler - 1)
    if max_steps <= 0:
        return 0.0

    div = np.zeros(max_steps)
    cnt = np.zeros(max_steps)
    for step in range(1, max_steps + 1):
        vi = np.arange(m)
        vj = nearest + step
        valid = vj < m
        if not np.any(valid):
            continue
        vi, vj = vi[valid], vj[valid]
        dk = np.linalg.norm(Y[vi] - Y[vj], axis=1)
        ok = dk > 0
        if not np.any(ok):
            continue
        div[step - 1] = np.sum(np.log(dk[ok] / d0[vi[ok]]))
        cnt[step - 1] = np.sum(ok)

    valid = cnt > 0
    if not np.any(valid):
        return 0.0
    steps_t = np.arange(1, max_steps + 1)[valid] / sample_rate_hz
    y = div[valid] / cnt[valid]
    try:
        slope, _ = np.polyfit(steps_t, y, 1)
    except Exception as exc:
        log.debug("_rosenstein_slope failed: %s", exc)
        return 0.0
    return float(slope)


def estimate_lyapunov_exponent(
    signal: np.ndarray,
    dim: int = 5,
    delay: int = 2,
    sample_rate_hz: float = 1000.0,
) -> float:
    """Maximal Lyapunov exponent (1/s) of a real signal.

    Finite, deterministic, never clipped to a sentinel. A perfectly ordered
    (periodic) carrier converges to ~0; a chaotic/stochastic burst is positive.
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    sd = np.std(x)
    if sd < 1e-12 or len(x) < dim * delay:
        return 0.0
    Y = reconstruct_phase_space(x, dim, delay)
    return _rosenstein_slope(Y, sample_rate_hz, theiler=10)


def _aaft_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Amplitude Adjusted Fourier Transform surrogate (preserves amplitude
    distribution, destroys temporal correlation structure)."""
    n = len(x)
    f = np.fft.rfft(x)
    nbins = f.shape[0]
    # Randomise phases of the non-DC, non-Nyquist bins (nbins-2 of them).
    n_rand = max(0, nbins - 2)
    ph = rng.uniform(0, 2 * np.pi, size=n_rand)
    f[1 : nbins - 1] *= np.exp(1j * ph)
    s = np.fft.irfft(f, n=n)
    # Rank-map back to original amplitude distribution (preserves marginals).
    order = np.argsort(np.argsort(s))
    return x[np.argsort(order)]


def surrogate_significance(
    signal: np.ndarray,
    dim: int = 5,
    delay: int = 2,
    sample_rate_hz: float = 1000.0,
    n_surrogates: int = 19,
    seed: int = 42,
) -> Tuple[float, float]:
    """Return (mle_real, p_value) where `p_value` is the fraction of AAFT
    surrogates whose MLE is >= the real signal's MLE (one-sided test).

    p ~ 1  -> real signal is no more divergent than randomised data
              (ordered / deterministic: technosignature-like).
    p ~ 0  -> real signal is significantly more divergent than randomised data
              (stochastic / chaotic burst).
    """
    x = np.asarray(signal, dtype=np.float64)
    x = x - np.mean(x)
    if np.std(x) < 1e-12 or len(x) < dim * delay:
        return 0.0, 1.0

    Y = reconstruct_phase_space(x, dim, delay)
    mle_real = _rosenstein_slope(Y, sample_rate_hz, theiler=10)

    rng = np.random.default_rng(seed)
    surr_mle = np.empty(n_surrogates, dtype=np.float64)
    for i in range(n_surrogates):
        s = _aaft_surrogate(x, rng)
        s = s - np.mean(s)
        Ys = reconstruct_phase_space(s, dim, delay)
        surr_mle[i] = _rosenstein_slope(Ys, sample_rate_hz, theiler=10)

    # One-sided: fraction of surrogates at least as divergent as the real signal.
    p = float(np.mean(surr_mle >= mle_real))
    return float(mle_real), p


def chaos_order_score(
    signal: np.ndarray,
    sample_rate_hz: float = 1000.0,
    n_surrogates: int = 19,
    seed: int = 42,
) -> float:
    """White-box, deterministic order score in [0, 1] for a single 1-D waveform.

    1.0  -> signal is consistent with a deterministic/ordered carrier
            (MLE not exceeding randomised surrogates): technosignature-like.
    0.0  -> signal is significantly chaotic/stochastic (a burst or noise).

    This replaces the previous `chaos < 1.0` heuristic with a calibrated,
    surrogate-based statistic that is traceable and never saturated.
    """
    _, p = surrogate_significance(
        signal, sample_rate_hz=sample_rate_hz, n_surrogates=n_surrogates, seed=seed
    )
    return float(p)


def chaos_order_score_from_p(p_value: float) -> float:
    """Wrap an already-computed surrogate p-value into the order score."""
    return float(np.clip(p_value, 0.0, 1.0))


def compute_chaos_descriptor(
    signal: np.ndarray, sample_rate_hz: float = 1000.0, seed: int = 42
) -> float:
    """Backward-compatible scalar chaos descriptor for a 1-D waveform.

    Returns the surrogate-calibrated order score in [0, 1]: 1.0 for a
    deterministic/ordered carrier, 0.0 for chaotic/stochastic burst. Kept so
    legacy callers (``ood_eval``) resolve; new code should call
    :func:`chaos_order_score` directly.
    """
    return chaos_order_score(signal, sample_rate_hz=sample_rate_hz, seed=seed)
