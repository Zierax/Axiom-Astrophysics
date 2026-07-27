"""Historical anomaly audit — reproducible real-catalog triage.

This module provides two complementary lanes for the question "what are the
weirdest signals in the historical/observational record?":

Lane A -- Curated historical signals (``run_curated_audit``, *illustrative*)
    Scores 25 famous signals (Wow!, BLC1, FRBs, perytons, Voyager/Pioneer
    telemetry, ...) through the HTRU2 manifold to show where documented signal
    *classes* fall. Several rows carry **placeholder** HTRU2-moment features and
    are NOT measured from raw data; this lane is a teaching/sanity-check
    illustration only and must never be cited as a measurement.

Lane B -- Real catalogs (``run_real_catalog_audit``, *measurement*)
    Fetches real catalogs (ATNF pulsars, CHIME/FRB Catalog 1) via
    :mod:`axiom.data.catalogs`, featurizes them on the commensurate physical
    manifold (:mod:`axiom.dsp.physical_features`), fits a per-class GMM density,
    and ranks objects by the **minimum per-class split-conformal off-manifold
    p-value**. This is a statistically defensible "most anomalous real signals"
    list.

Lane C -- Unclassified survey candidates (``run_unclassified_htru2_audit``,
*discovery protocol*)
    HTRU2 candidates on which the ensemble is uncertain (predicted pulsar
    probability near 0.5) form a genuine *unclassified* pool -- the survey itself
    could not decide pulsar vs RFI. Scoring them on the HTRU2 manifold by the
    **max-over-class** conformal p-value (rare under *both* natural classes)
    isolates the candidates that escape every known population. This is the exact
    protocol to apply to a truly unlabeled candidate set (MeerKAT/ASKAP/TNS) where
    a real unknown would hide. HTRU2 is labeled, so this lane *validates the
    protocol*, it does not claim a discovery.

Honesty boundary
----------------
An off-manifold object is a triage candidate for human follow-up -- **not**
evidence of artificial origin or a new astrophysical discovery. The curated lane
is illustrative; the real lane ranks rarity *within the known natural
populations*; the unclassified lane validates the discovery protocol on a labeled
survey.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from axiom.data.catalogs import load_catalog
from axiom.data.loader import load_htru2
from axiom.dsp.physical_features import featurize_frame, impute_apply, impute_fit
from axiom.ml.density import AnomalyDensityEstimator

#: Default cache directory for fetched catalogs and fitted density models.
CACHE_DIR = os.path.join("data", "historical_cache")

# ---------------------------------------------------------------------------
# Lane A -- curated historical signals (illustrative)
# ---------------------------------------------------------------------------
_CURATED_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "historical_anomalies.csv",
)
# Placeholder feature vector used for narrowband/telemetry rows in the CSV.
_PLACEHOLDER = np.array([30.0, 10.0, 0.0, 0.0, 0.5, 5.0, 0.0, 0.0])


@dataclass
class CuratedRow:
    name: str
    kind: str
    features: np.ndarray
    is_placeholder: bool = False
    true_dm: float = 0.0
    peak_snr: float = 0.0
    source: str = ""
    year: str = ""
    telescope: str = ""
    frequency_mhz: float = 0.0
    reference: str = ""


@dataclass
class CuratedResult:
    rows: List[CuratedRow]
    density_score: np.ndarray
    natural_floor: float
    verdict: np.ndarray
    ood_margin: float = 5.0

    @property
    def ranked(self) -> List[Tuple[str, str, float, str, bool]]:
        order = np.argsort(self.density_score)
        out = []
        for i in order:
            r = self.rows[i]
            out.append((r.name, r.kind, float(self.density_score[i]),
                        self.verdict[i], r.is_placeholder))
        return out

    def ranked_with_meta(self) -> List[dict]:
        """Return ranked signals with full metadata for reporting."""
        order = np.argsort(self.density_score)
        out = []
        for i in order:
            r = self.rows[i]
            out.append({
                "name": r.name,
                "type": r.kind,
                "loglik": float(self.density_score[i]),
                "verdict": self.verdict[i],
                "is_placeholder": r.is_placeholder,
                "true_dm": r.true_dm,
                "peak_snr": r.peak_snr,
                "source": r.source,
                "year": r.year,
                "telescope": r.telescope,
                "frequency_mhz": r.frequency_mhz,
                "reference": r.reference,
            })
        return out


def _safe_float(val: str, default: float = 0.0) -> float:
    """Convert a string to float, returning default for non-numeric values."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _read_curated_csv(path: Optional[str] = None) -> List[CuratedRow]:
    path = path or _CURATED_CSV
    rows: List[CuratedRow] = []
    with open(path, newline="") as fh:
        for rec in csv.DictReader(fh):
            feats = np.array([
                float(rec["profile_mean"]), float(rec["profile_std"]),
                float(rec["profile_kurtosis"]), float(rec["profile_skewness"]),
                float(rec["dmsnr_mean"]), float(rec["dmsnr_std"]),
                float(rec["dmsnr_kurtosis"]), float(rec["dmsnr_skewness"]),
            ], dtype=np.float64)
            placeholder = np.allclose(feats, _PLACEHOLDER, atol=1e-6)
            # Telemetry rows (spacecraft carriers) are illustrative and lack real feature measurements.
            # Treat them as placeholders regardless of exact numeric values.
            is_ph_placeholder = placeholder or (rec.get("type", "").lower() == "telemetry")
            rows.append(CuratedRow(
                name=rec["name"],
                kind=rec["type"],
                features=feats,
                is_placeholder=is_ph_placeholder,
                true_dm=_safe_float(rec.get("true_dm", "0")),
                peak_snr=_safe_float(rec.get("peak_snr", "0")),
                source=rec.get("source", ""),
                year=rec.get("year", ""),
                telescope=rec.get("telescope", ""),
                frequency_mhz=_safe_float(rec.get("frequency_mhz", "0")),
                reference=rec.get("reference", ""),
            ))
    return rows


def run_curated_audit(path: Optional[str] = None, ood_margin: float = 5.0,
                      seed: int = 42) -> CuratedResult:
    rows = _read_curated_csv(path)
    X_hist = np.stack([r.features for r in rows])

    X, y, _ = load_htru2()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    fit_idx = idx[:int(0.7 * len(X))]
    density = AnomalyDensityEstimator(
        n_components=5,
        cache_path=os.path.join(CACHE_DIR, "density_curated.pkl"),
    )
    density.fit(X[fit_idx], y[fit_idx])

    scores = density.log_prob(X_hist)
    natural_floor = float(np.min(density.log_prob(X[fit_idx])))
    verdict = np.where(scores < natural_floor - ood_margin, "OOD", "on-manifold")
    return CuratedResult(rows, scores, natural_floor, verdict, ood_margin)


# ---------------------------------------------------------------------------
# Lane B -- real catalogs (measurement)
# ---------------------------------------------------------------------------
@dataclass
class RealCatalogResult:
    frame: pd.DataFrame
    scores: np.ndarray
    pvals: np.ndarray
    labels: np.ndarray
    n_objects: int
    catalog_keys: Tuple[str, ...]
    max_pvals: Optional[np.ndarray] = None

    def top(self, k: int = 20, mode: str = "min") -> pd.DataFrame:
        """Return the top-k objects ranked by off-manifold p-value.

        mode="min" (default): an object is flagged if rare under *any* natural
          class (sensitive triage) -- the standard engine score.
        mode="max": an object is flagged only if rare under *every* natural class
          (it fits no known population) -- a stricter, discovery-relevant filter
          that surfaces genuinely unexplained objects rather than extreme-known ones.
        """
        ref = self.max_pvals if mode == "max" else self.pvals
        if ref is None:
            ref = self.pvals
        order = np.argsort(ref)[:k]
        out = self.frame.iloc[order].copy()
        out["manifold_logprob"] = self.scores[order]
        out["off_manifold_pval"] = self.pvals[order]
        out["off_manifold_pval_max"] = (self.max_pvals[order]
                                        if self.max_pvals is not None else np.nan)
        out["assigned_class"] = [int(self.labels[i]) for i in order]
        cols = [c for c in ["object_id", "class_name", "dm", "period_s",
                            "width_ms", "glat", "snr", "manifold_logprob",
                            "off_manifold_pval", "off_manifold_pval_max"]
                if c in out.columns]
        return out[cols].reset_index(drop=True)


def _load_and_featurize(keys: Tuple[str, ...], seed: int,
                        repin: bool, cache_dir: str
                        ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frames, all_feats, all_labels = [], [], []
    class_codes: Dict[str, int] = {}
    code = 0
    for key in keys:
        df = load_catalog(key, cache_dir=cache_dir, repin=repin)
        X, _ = featurize_frame(df)
        medians = impute_fit(X)
        X_imp = impute_apply(X, medians)
        if not np.all(np.isfinite(X_imp)):
            raise RuntimeError(f"non-finite features after impute for {key}")
        for cn in df["class_name"].unique():
            if cn not in class_codes:
                class_codes[cn] = code
                code += 1
        lab = df["class_name"].map(class_codes).to_numpy()
        frames.append(df)
        all_feats.append(X_imp)
        all_labels.append(lab)
    frame = pd.concat(frames, ignore_index=True)
    return frame, np.vstack(all_feats), np.concatenate(all_labels)


def run_real_catalog_audit(
    keys: Tuple[str, ...] = ("atnf_pulsars", "chime_frb_cat1"),
    seed: int = 42,
    repin: bool = False,
    cache_dir: str = CACHE_DIR,
) -> RealCatalogResult:
    """Score real catalogs on the physical manifold and rank off-manifold objects.

    Conformal protocol
    ------------------
    * Fit a per-class GMM density on a 70% in-fold (stratified by class label).
    * Score the held-out 30% calibration split; build an empirical CDF of
      negative log-likelihood per class.
    * Map every object to a per-class conformal p-value =
      ( #{cal objects in same class with loglik <= this} + 1 ) / (n_cal + 1).
    * Off-manifold p-value is the MINIMUM over classes (an object is flagged if
      it is rare under *any* natural class) -- a sensitive triage choice.
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    frame, feats, labels = _load_and_featurize(keys, seed, repin, cache_dir)
    n = feats.shape[0]

    rng = np.random.default_rng(seed)
    fit_mask = np.zeros(n, dtype=bool)
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        rng.shuffle(ci)
        fit_mask[ci[:max(1, int(0.7 * len(ci)))]] = True

    density = AnomalyDensityEstimator(
        n_components=5,
        cache_path=os.path.join(cache_dir, "density_real.pkl"),
    )
    density.fit(feats[fit_mask], labels[fit_mask])

    cal_mask = ~fit_mask
    cal_scores = density.log_prob(feats[cal_mask])
    cal_labels = labels[cal_mask]
    cal_by_class: Dict[int, np.ndarray] = {}
    for c in np.unique(cal_labels):
        cal_by_class[int(c)] = cal_scores[cal_labels == c]

    scores = density.log_prob(feats)
    pvals = np.ones(n, dtype=np.float64)
    max_pvals = np.ones(n, dtype=np.float64)
    assigned = np.zeros(n, dtype=np.int64)
    for i in range(n):
        s = scores[i]
        # Minimum over all classes => rarest under any natural population (triage).
        p_min = 1.0
        # Maximum over all classes => rare under EVERY natural population
        # (fits no known class) -- stricter, discovery-relevant.
        p_max = 0.0
        best_c = int(labels[i])
        for cc, refc in cal_by_class.items():
            pc = (int(np.sum(refc <= s)) + 1) / (len(refc) + 1)
            if pc < p_min:
                p_min, best_c = pc, cc
            if pc > p_max:
                p_max = pc
        pvals[i] = p_min
        max_pvals[i] = p_max
        assigned[i] = best_c

    return RealCatalogResult(frame, scores, pvals, assigned, n, keys, max_pvals)


# ---------------------------------------------------------------------------
# Lane C -- Unclassified survey candidates (discovery protocol)
# ---------------------------------------------------------------------------
@dataclass
class UnclassifiedResult:
    indices: np.ndarray          # HTRU2 row indices of the unclassified pool
    proba_pulsar: np.ndarray     # ensemble pulsar probability (near 0.5 = uncertain)
    p_min: np.ndarray            # rare under any natural class
    p_max: np.ndarray            # rare under BOTH natural classes (discovery filter)
    true_label: Optional[np.ndarray] = None  # HTRU2 label (validation only; absent in real use)

    def top(self, k: int = 20) -> pd.DataFrame:
        """Top-k unclassified candidates by p_max (rarest under BOTH classes)."""
        order = np.argsort(self.p_max)[:k]
        out = pd.DataFrame({
            "htru2_index": self.indices[order],
            "ensemble_p_pulsar": self.proba_pulsar[order],
            "p_min": self.p_min[order],
            "p_max": self.p_max[order],
        })
        if self.true_label is not None:
            out["true_label"] = self.true_label[order]
        return out.reset_index(drop=True)


def run_unclassified_htru2_audit(
    uncertainty_lo: float = 0.4,
    uncertainty_hi: float = 0.6,
    seed: int = 42,
    cache_dir: str = CACHE_DIR,
) -> UnclassifiedResult:
    """Surface HTRU2 candidates the ensemble cannot classify (unclassified pool).

    The pool is defined by ensemble pulsar-probability in ``[lo, hi]`` -- the
    survey's own uncertainty. Each candidate is scored on the HTRU2 manifold by
    per-class (pulsar / RFI) conformal p-values; ``p_max`` (rare under *both*
    classes) is the discovery filter: a candidate that escapes every known
    population. HTRU2 is labeled, so ``true_label`` is returned for *validation*;
    on a genuinely unlabeled survey this column is simply absent.

    Honesty note: this lane validates the discovery protocol. Because HTRU2 is
    labeled, it cannot produce a real discovery -- it shows the engine isolates
    the most manifold-escaping unclassified candidates, ready for an unlabeled set.
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    from axiom.ml.ensemble import AxiomEnsemble

    X, y, _ = load_htru2()
    ens = AxiomEnsemble(random_state=seed)
    ens.fit(X, y)
    proba = ens.predict_proba(X)[:, 1]

    pool = np.where((proba >= uncertainty_lo) & (proba <= uncertainty_hi))[0]
    if len(pool) == 0:
        raise RuntimeError("no unclassified candidates in the uncertainty band")

    rng = np.random.default_rng(seed)
    fit = rng.permutation(len(X))[:int(0.7 * len(X))]
    dens = AnomalyDensityEstimator(
        n_components=5,
        cache_path=os.path.join(cache_dir, "density_htru2_ens.pkl"),
    )
    dens.fit(X[fit], y[fit])

    cal_mask = ~np.isin(np.arange(len(X)), fit)
    sp = dens.log_prob_per_class(X, 1)
    sr = dens.log_prob_per_class(X, 0)

    def _conf(scores_all: np.ndarray) -> np.ndarray:
        cal = scores_all[cal_mask]
        out = np.ones(len(scores_all), dtype=np.float64)
        for i in range(len(scores_all)):
            out[i] = (int(np.sum(cal <= scores_all[i])) + 1) / (len(cal) + 1)
        return out

    pp = _conf(sp)
    pr = _conf(sr)
    p_min_pool = np.minimum(pp[pool], pr[pool])
    p_max_pool = np.maximum(pp[pool], pr[pool])

    return UnclassifiedResult(
        indices=pool,
        proba_pulsar=proba[pool],
        p_min=p_min_pool,
        p_max=p_max_pool,
        true_label=y[pool],
    )


# ---------------------------------------------------------------------------
# Lane D -- Historical signal verification (FULL AUDIT)
# ---------------------------------------------------------------------------
@dataclass
class HistoricalAuditResult:
    """Result of scoring all historical signals through the HTRU2 manifold."""
    rows: List[CuratedRow]
    density_score: np.ndarray
    natural_floor: float
    verdict: np.ndarray
    ood_margin: float = 5.0
    pvals: Optional[np.ndarray] = None

    def top(self, k: int = 20) -> List[dict]:
        """Top-k most off-manifold signals by density score."""
        order = np.argsort(self.density_score)[:k]
        out = []
        for i in order:
            r = self.rows[i]
            out.append({
                "name": r.name,
                "type": r.kind,
                "loglik": float(self.density_score[i]),
                "verdict": self.verdict[i],
                "true_dm": r.true_dm,
                "peak_snr": r.peak_snr,
                "source": r.source,
                "year": r.year,
                "telescope": r.telescope,
                "frequency_mhz": r.frequency_mhz,
                "reference": r.reference,
            })
        return out

    def by_type(self) -> Dict[str, dict]:
        """Aggregate statistics by signal type."""
        type_stats: Dict[str, dict] = {}
        for i, r in enumerate(self.rows):
            t = r.kind
            if t not in type_stats:
                type_stats[t] = {"count": 0, "ood_count": 0, "scores": []}
            type_stats[t]["count"] += 1
            type_stats[t]["scores"].append(float(self.density_score[i]))
            if self.verdict[i] == "OOD":
                type_stats[t]["ood_count"] += 1
        for t in type_stats:
            scores = type_stats[t]["scores"]
            type_stats[t]["mean_loglik"] = float(np.mean(scores))
            type_stats[t]["min_loglik"] = float(np.min(scores))
            type_stats[t]["fraction_ood"] = (
                type_stats[t]["ood_count"] / type_stats[t]["count"]
                if type_stats[t]["count"] > 0 else 0.0
            )
            del type_stats[t]["scores"]
        return type_stats


def run_historical_verification(
    path: Optional[str] = None,
    ood_margin: float = 5.0,
    seed: int = 42,
    cache_dir: str = CACHE_DIR,
) -> HistoricalAuditResult:
    """Score ALL historical signals through the HTRU2 manifold.

    This is the comprehensive audit: every famous signal, every FRB, every
    pulsar, every transient, every RFI event -- scored on the HTRU2 manifold
    to show where documented signal classes fall.

    The density model is fit on 70% of HTRU2 data (pulsars + RFI noise
    candidates), and all historical signals are scored against it. An OOD
    verdict means the signal's HTRU2-moment features are statistically
    unusual *even among known pulsars and RFI* -- a triage signal, NOT
    a claim of artificial origin.

    Returns HistoricalAuditResult with ranked signals, per-type stats,
    and metadata for reporting.
    """
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    rows = _read_curated_csv(path)
    X_hist = np.stack([r.features for r in rows])

    X, y, _ = load_htru2()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    fit_idx = idx[:int(0.7 * len(X))]
    cal_idx = idx[int(0.7 * len(X)):]

    density = AnomalyDensityEstimator(
        n_components=5,
        cache_path=os.path.join(cache_dir, "density_historical.pkl"),
    )
    density.fit(X[fit_idx], y[fit_idx])

    scores = density.log_prob(X_hist)
    natural_floor = float(np.min(density.log_prob(X[fit_idx])))
    verdict = np.where(scores < natural_floor - ood_margin, "OOD", "on-manifold")

    # Per-signal conformal p-value against calibration set
    cal_scores = density.log_prob(X[cal_idx])
    pvals = np.ones(len(X_hist), dtype=np.float64)
    for i in range(len(X_hist)):
        pvals[i] = (int(np.sum(cal_scores <= scores[i])) + 1) / (len(cal_scores) + 1)

    return HistoricalAuditResult(
        rows=rows,
        density_score=scores,
        natural_floor=natural_floor,
        verdict=verdict,
        ood_margin=ood_margin,
        pvals=pvals,
    )
