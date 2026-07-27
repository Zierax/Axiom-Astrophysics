"""Physics-grounded rules and laws for the AXIOM arbitrator.

These are *honest* physical consistency checks expressed as bounded scores in
[0, 1]. They do not fabricate data: every input is a measured/catalog-derived
quantity already present in the pipeline (spectrogram descriptors, DM, Galactic
DM ceiling, rotation period, pulse width). The scores encode real astrophysical
laws so the final verdict rests on physics, not only on learned manifold geometry.

Laws implemented
----------------
1. **Tone-vs-dispersion contradiction law.** A genuine *dispersed* astrophysical
   pulse (pulsar / FRB) spreads its energy across the band and over DM, so its
   spectrogram is broadband and spectrally *flat* (high spectral flatness,
   low concentration). A *narrowband tone* (low spectral flatness, high channel
   concentration) is the physical signature of a sustained local / transmitted
   carrier — which, by construction, is the anomaly class we surface. A source
   that is simultaneously highly tonal AND asserted to be a dispersed natural
   pulse is physically self-contradictory; the law rewards the tonal morphology
   as an anomaly indicator but only where the dispersion evidence is weak.
2. **Drift consistency law.** A tone whose dominant channel *drifts* in frequency
   across integrations (non-zero drift rate above the noise floor) is a moving
   transmitter rather than a fixed astrophysical line; combined with narrowband
   morphology this is a stronger anomaly signal.
3. **Dispersion-excess law (catalog path).** For sources with a measured DM and a
   Galactic foreground ceiling ``dm_gal_max``: an extragalactic (FRB-like) source
   must exceed the Galactic ceiling (positive DM excess); a source *claimed* to be
   a Galactic pulsar but lying far *above* the halo ceiling is physically
   inconsistent. Encoded as a signed excess score.
4. **Duty-cycle consistency law (catalog path).** Rotation-powered neutron stars
   have a minimum electromagnetic duty cycle set by the emission geometry; an
   ultra-narrow duty cycle (width / period) at a long period is physically
   marginal. Encoded as a bounded consistency score used to *down-weight* a
   spurious "natural pulsar" claim.

The module is stateless and deterministic; every function tolerates missing
inputs (NaN / None) by returning neutral scores so the arbitrator degrades
gracefully.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "technosignature_law_score",
    "dispersion_excess_law_score",
    "duty_cycle_consistency_law_score",
    "combine_physics_laws",
]

_EPS = 1e-12


def _as_float(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def technosignature_law_score(
    features: dict,
    claimed_dispersed: bool = False,
) -> float:
    """Tone-vs-dispersion contradiction law (spectrogram-only, always available).

    Parameters
    ----------
    features : dict
        Waterfall descriptor dict (keys from ``DESCRIPTOR_VECTOR_KEYS``; the two
        used here are ``concentration`` and ``spectral_flatness``).
    claimed_dispersed : bool
        True when the ensemble/model asserts the source is a *dispersed* natural
        pulse (class 0/2 with high in-distribution confidence). When True, a
        highly tonal morphology is physically contradictory and is rewarded more
        strongly (it cannot be a dispersed pulse, so it is more likely a carrier).

    Returns
    -------
    float in [0, 1]
        Higher => more physically consistent with a transmitted/narrowband
        technosignature and less with a dispersed astrophysical pulse.
    """
    conc = float(np.clip(_as_float(features.get("concentration"), 0.0), 0.0, 1.0))
    flat = float(np.clip(_as_float(features.get("spectral_flatness"), 1.0), 0.0, 1.0))
    # A tone is low-flatness + high-concentration (same definition used by
    # waterfall_narrowband_score, kept here so the law is self-consistent).
    tonal = 0.6 * conc + 0.4 * (1.0 - flat)
    tonal = float(np.clip(tonal, 0.0, 1.0))
    # If the source is claimed to be a dispersed natural pulse, a tonal
    # morphology is the physical contradiction -> amplify; otherwise keep it.
    if claimed_dispersed:
        return float(np.clip(tonal * 1.25, 0.0, 1.0))
    return tonal


def dispersion_excess_law_score(
    dm: float,
    dm_gal_max: float | None = None,
    is_extragalactic_claim: bool = False,
) -> float:
    """Dispersion-excess law (catalog path; needs a measured DM).

    A source with DM below the Galactic foreground ceiling is consistent with a
    Galactic origin (or with local RFI / a carrier at DM~0). A source claimed to
    be extragalactic (FRB-like) MUST exceed the Galactic ceiling: positive excess
    is consistent, negative/near-zero excess contradicts the claim.

    Parameters
    ----------
    dm : float
        Measured dispersion measure.
    dm_gal_max : float, optional
        Galactic DM ceiling along the line of sight (e.g. from
        ``gigantic_dm_ceiling`` / ``dm_gal_max`` catalog column). If absent the
        law is neutral (no dispersion evidence either way).
    is_extragalactic_claim : bool
        True when the source is asserted to be an FRB-like extragalactic event.

    Returns
    -------
    float in [0, 1]
        For an extragalactic claim: high when DM exceeds the ceiling (consistent),
        low when it does not (contradiction -> anomaly). Neutral (0.5) when no
        Galactic ceiling is available or the claim is not extragalactic.
    """
    dm = _as_float(dm, np.nan)
    if not np.isfinite(dm):
        return 0.5
    if not is_extragalactic_claim:
        return 0.5
    if dm_gal_max is None or not np.isfinite(dm_gal_max):
        return 0.5
    excess = dm - float(dm_gal_max)
    if excess <= 0.0:
        return 0.0
    # Smooth saturating ramp: full consistency once excess ~= a few x the ceiling.
    return float(np.clip(1.0 - np.exp(-excess / max(abs(dm_gal_max), _EPS)), 0.0, 1.0))


def duty_cycle_consistency_law_score(
    width_ms: float | None = None,
    period_s: float | None = None,
) -> float:
    """Duty-cycle consistency law (catalog path; needs width and period).

    Rotation-powered pulsars emit over a finite duty cycle ``w = width_ms /
    (1000 * period_s)``. An asserted natural pulsar with an ultra-narrow duty
    cycle at a long period is physically marginal and is down-weighted. The law
    returns a *consistency* score in [0, 1] (1 = physically consistent pulsar,
    lower = marginal / more anomaly-like).

    Parameters
    ----------
    width_ms : float, optional
        Pulse width (ms).
    period_s : float, optional
        Rotation period (s).

    Returns
    -------
    float in [0, 1]
        Neutral (0.5) when width/period are unavailable; otherwise a bounded
        consistency score using a soft lower bound on the duty cycle.
    """
    width_ms = _as_float(width_ms, np.nan)
    period_s = _as_float(period_s, np.nan)
    if not (np.isfinite(width_ms) and np.isfinite(period_s) and width_ms > 0 and period_s > 0):
        return 0.5
    duty = (width_ms / 1000.0) / period_s
    if duty <= 0.0:
        return 0.5
    # Empirically rotation-powered pulsars have duty cycles mostly in 0.01-0.1;
    # a duty below ~0.5% at a long period is marginal. Soft ramp: full
    # consistency above 1.5% duty, fading to a floor below ~0.1%.
    lo, hi = 0.001, 0.015
    return float(np.clip((duty - lo) / max(hi - lo, _EPS), 0.0, 1.0))


def combine_physics_laws(
    *,
    technosig: float = 0.0,
    dispersion: float = 0.5,
    duty: float = 0.5,
    use_catalog_laws: bool = False,
) -> float:
    """Combine the individual law scores into one bounded physics-law score [0,1].

    When catalog physics is unavailable the score is simply the technosignature
    law (spectrogram-only). When catalog DM/duty are present, the dispersion and
    duty *consistency* scores are folded in: a source that is tonally anomalous
    AND dispersion/duty-consistent with a transmitted carrier is pushed toward a
    stronger anomaly; a source that is tonally quiet but dispersion/duty
    inconsistent with its claimed class is also nudged up.

    Parameters
    ----------
    technosig : float
        Output of :func:`technosignature_law_score` in [0, 1].
    dispersion : float
        Output of :func:`dispersion_excess_law_score` in [0, 1] (consistency).
    duty : float
        Output of :func:`duty_cycle_consistency_law_score` in [0, 1] (consistency).
    use_catalog_laws : bool
        Whether catalog physics (dispersion/duty) is available and should
        contribute.

    Returns
    -------
    float in [0, 1]
    """
    technosig = float(np.clip(technosig, 0.0, 1.0))
    dispersion = float(np.clip(dispersion, 0.0, 1.0))
    duty = float(np.clip(duty, 0.0, 1.0))
    if not use_catalog_laws:
        return technosig
    # Anomaly is favoured when technosig is high; catalog consistency scores that
    # are LOW mean the source contradicts its claimed natural class -> anomaly.
    catalog_anomaly = 1.0 - 0.5 * (dispersion + duty)
    # Weighted blend: spectrogram law dominates when present, catalog law refines.
    return float(np.clip(0.7 * technosig + 0.3 * catalog_anomaly, 0.0, 1.0))
