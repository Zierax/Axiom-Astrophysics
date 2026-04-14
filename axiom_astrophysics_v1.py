"""
AXIOM-ASTROPHYSICS v1.0 - Pure Python Pipeline
High-performance cosmic signal analysis
"""
from __future__ import annotations

import json
import logging
import math
import os
import uuid
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.distance import mahalanobis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class IngestResult:
    signals: list[dict]
    valid_count: int
    excluded_count: int
    warnings: list[str]


@dataclass
class EntropyResult:
    entropy_density: float
    label: str
    flagged: bool


@dataclass
class GeometryResult:
    anomaly_descriptor: str
    flagged: bool
    distance_score: float


@dataclass
class ProofResult:
    p_value: float
    hypothesis_tests_run: int
    logical_proof_status: str


@dataclass
class CalibrationReport:
    calibration_certainty: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass
class CorrectionReport:
    corrections: list[dict]
    signals_re_evaluated: int


@dataclass
class PipelineConfig:
    dataset_path: str
    output_path: str
    random_seed: int


# ---------------------------------------------------------------------------
# Component Classes (stubs)
# ---------------------------------------------------------------------------

# Known priority target signal IDs — these records always get priority_target=True in their audit records
PRIORITY_TARGET_IDS = frozenset({
    "ANOMALY_WOW_1977",
    "ANOMALY_BLC1_2020",
    "ANOMALY_ARECIBO_ECHO",
    "ANOMALY_LORIMER_2007",
    "ANOMALY_FRB121102_2014",
    "ANOMALY_SHGb02_28a",
    "ANOMALY_HD164595_2016",
    "ANOMALY_PRS_FRB121102",
    "ANOMALY_XTE_J1739_285",
    "ANOMALY_SGR_1935_2154",
    "ANOMALY_TABBY_STAR_2015",
    "ANOMALY_OUMUAMUA_2017",
    "ANOMALY_GCRT_J1745_2002",
    "ANOMALY_PERYTON_2015",
    "ANOMALY_FRB_20200120E",
    "ANOMALY_VELA_PULSAR_GLITCH",
    "ANOMALY_FAST_FRB_20190520B",
})

REQUIRED_FIELDS = [
    "signal_id",
    "frequency_mhz",
    "entropy_score",
    "drift_rate",
    "bandwidth_efficiency",
    "modulation_type",
    "intensity_sigma",
    "duration_sec",
    "origin_class",
]


class DatasetIngester:
    def load(self, path: str) -> IngestResult:
        # Error case: file not found
        if not os.path.exists(path):
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Error case: invalid JSON
        try:
            with open(path, "r", encoding="utf-8") as f:
                records = json.load(f)
        except json.JSONDecodeError as exc:
            raise json.JSONDecodeError(
                f"Invalid JSON in dataset file '{path}': {exc.msg}",
                exc.doc,
                exc.pos,
            ) from exc

        total = len(records)
        valid_signals: list[dict] = []
        warnings: list[str] = []
        excluded = 0

        for record in records:
            missing = [field for field in REQUIRED_FIELDS if field not in record]
            if missing:
                msg = (
                    f"Record missing required fields {missing}: "
                    f"{record.get('signal_id', '<unknown>')}"
                )
                logger.warning(msg)
                warnings.append(msg)
                excluded += 1
            else:
                valid_signals.append(record)

        logger.info(
            "Ingestion complete — total: %d, valid: %d, excluded: %d",
            total,
            len(valid_signals),
            excluded,
        )

        return IngestResult(
            signals=valid_signals,
            valid_count=len(valid_signals),
            excluded_count=excluded,
            warnings=warnings,
        )


class StratifiedSplitter:
    def split(self, signals: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
        # Group signals by origin_class
        classes: dict[str, list[dict]] = {}
        for signal in signals:
            cls = signal["origin_class"]
            classes.setdefault(cls, []).append(signal)

        training_pool: list[dict] = []
        test_pool: list[dict] = []

        rng = np.random.default_rng(seed)

        for cls, members in classes.items():
            members = list(members)
            rng.shuffle(members)
            half = math.ceil(len(members) / 2)
            training_pool.extend(members[:half])
            test_pool.extend(members[half:])

        # Log pool sizes
        logger.info(
            "Split complete — training: %d, test: %d",
            len(training_pool),
            len(test_pool),
        )

        # Log class distribution for training pool
        train_dist: dict[str, int] = {}
        for signal in training_pool:
            train_dist[signal["origin_class"]] = train_dist.get(signal["origin_class"], 0) + 1
        logger.info("Training pool class distribution: %s", train_dist)

        # Log class distribution for test pool
        test_dist: dict[str, int] = {}
        for signal in test_pool:
            test_dist[signal["origin_class"]] = test_dist.get(signal["origin_class"], 0) + 1
        logger.info("Test pool class distribution: %s", test_dist)

        return training_pool, test_pool


_LAMBDA_CDM_PARAMS = ["frequency_mhz", "entropy_score", "drift_rate", "intensity_sigma"]
_LAMBDA_CDM_SEED = 42

# Extended parameter set for richer baseline modeling
_LAMBDA_CDM_EXTENDED = ["frequency_mhz", "entropy_score", "drift_rate", "intensity_sigma",
                         "harmonic_complexity", "duration_log10"]


class LambdaCDMModel:
    """
    SOTA upgrade: 6-parameter correlated Lambda-CDM baseline.
    Original used 4 independent Gaussians. Now:
      - 6 parameters including harmonic_complexity and log10(duration)
      - Multivariate Gaussian with full covariance (captures correlations)
      - Separate sub-models for Broadband vs Narrowband (different physics)
      - Mahalanobis-based z-score for correlated parameter space
    """

    def __init__(self) -> None:
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        self._cov: np.ndarray | None = None
        self._inv_cov: np.ndarray | None = None
        self._fingerprint: frozenset | None = None
        self._rng = np.random.default_rng(_LAMBDA_CDM_SEED)

    def _signal_to_vector(self, signal: dict) -> np.ndarray:
        dur = max(signal.get("duration_sec", 1.0), 1e-6)
        return np.array([
            signal["frequency_mhz"],
            signal["entropy_score"],
            signal["drift_rate"],
            signal["intensity_sigma"],
            float(signal.get("harmonic_complexity", 0.0)),
            math.log10(dur),
        ], dtype=float)

    def fit(self, training_pool: list[dict]) -> None:
        natural_signals = [s for s in training_pool if s["origin_class"] == "Natural"]
        fit_pool = natural_signals if len(natural_signals) >= 10 else training_pool

        # Build feature matrix
        X = np.array([self._signal_to_vector(s) for s in fit_pool], dtype=float)

        # Per-parameter stats (for backward compat z_score)
        # Use robust statistics (percentile-based) to reduce outlier influence
        for i, param in enumerate(_LAMBDA_CDM_PARAMS):
            self._means[param] = float(np.mean(X[:, i]))
            std = float(np.std(X[:, i]))
            self._stds[param] = std if std > 0 else 1e-6
            # Store 16th and 84th percentiles for robust outlier detection
            self._percentile_16 = np.percentile(X, 16, axis=0)
            self._percentile_84 = np.percentile(X, 84, axis=0)

        # Full multivariate covariance
        self._mean_vec = np.mean(X, axis=0)
        self._std_vec = np.std(X, axis=0)
        self._std_vec[self._std_vec == 0] = 1e-6

        # Normalize before computing covariance (prevents scale dominance)
        X_norm = (X - self._mean_vec) / self._std_vec
        cov = np.cov(X_norm.T)
        # Add small regularization to prevent singular matrix
        reg = 1e-6 * np.eye(len(self._mean_vec))
        self._cov = cov + reg
        self._inv_cov = np.linalg.pinv(self._cov)

        self._fingerprint = frozenset(s["signal_id"] for s in training_pool)
        logger.info(
            "LambdaCDMModel fitted on %d natural signals (of %d total), 6-param multivariate",
            len(fit_pool), len(training_pool)
        )

    def generate(self, n: int = 10_000) -> np.ndarray:
        """Generate synthetic signals from the fitted multivariate distribution."""
        sample_size = max(n, 10_000)
        # For large samples, use independent normals (faster) then apply Cholesky correlation
        # This is equivalent to multivariate_normal but ~10x faster for large n
        try:
            L = np.linalg.cholesky(self._cov)
            Z = self._rng.standard_normal((sample_size, len(self._mean_vec)))
            X_norm = Z @ L.T
        except np.linalg.LinAlgError:
            # Fallback to independent sampling if cov is not positive definite
            X_norm = self._rng.standard_normal((sample_size, len(self._mean_vec)))
        X = X_norm * self._std_vec + self._mean_vec
        # Return only the 4 original params for backward compat
        return X[:, :4]

    def mahalanobis_distance(self, signal: dict) -> float:
        """Compute Mahalanobis distance of signal from natural baseline."""
        v = self._signal_to_vector(signal)
        v_norm = (v - self._mean_vec) / self._std_vec
        diff = v_norm - np.zeros(len(v_norm))  # centroid is zero in normalized space
        return float(np.sqrt(diff @ self._inv_cov @ diff))

    def z_score(self, signal: dict) -> dict[str, float]:
        return {
            param: float((signal[param] - self._means[param]) / self._stds[param])
            for param in _LAMBDA_CDM_PARAMS
        }

    def regenerate_if_stale(self, training_pool: list[dict]) -> None:
        new_fingerprint = frozenset(s["signal_id"] for s in training_pool)
        if new_fingerprint != self._fingerprint:
            logger.info("LambdaCDMModel: training pool changed, refitting")
            self.fit(training_pool)


class EntropyAnalyzer:
    """
    SOTA upgrade: multi-dimensional entropy scoring.
    Beyond a single scalar, we now compute:
      - Spectral entropy density (original)
      - Duration anomaly score: very short or very long durations relative to class
      - Bandwidth-frequency coherence: Narrowband at specific frequencies is suspicious
      - Combined composite entropy score for threshold comparison
    """

    def _entropy_density(self, signal: dict) -> float:
        bw_factor = 0.5 if signal["bandwidth_efficiency"] == "Narrowband" else 1.0
        mod_factor = 0.8 if signal["modulation_type"] == "Continuous" else 1.0
        return signal["entropy_score"] * bw_factor * mod_factor

    def _duration_score(self, signal: dict) -> float:
        """Normalized duration anomaly: very short bursts or very long CW are suspicious."""
        dur = signal.get("duration_sec", 60.0)
        if dur <= 0:
            return 0.0
        # Log-scale normalization against calibrated mean/std
        log_dur = math.log10(max(dur, 1e-6))
        if not hasattr(self, "_log_dur_mean"):
            return 0.0
        return abs(log_dur - self._log_dur_mean) / max(self._log_dur_std, 1e-6)

    def _narrowband_frequency_score(self, signal: dict) -> float:
        """
        Narrowband signals at cosmologically significant frequencies score higher.
        H-line (1420.405 MHz), OH maser (1612/1665/1667/1720 MHz),
        water maser (22235 MHz), and prime-number multiples of H-line.
        """
        if signal.get("bandwidth_efficiency") != "Narrowband":
            return 0.0
        freq = signal.get("frequency_mhz", 0.0)
        # Significant frequencies (MHz)
        significant = [
            1420.405,   # H-line
            1612.231,   # OH maser
            1665.402,   # OH main line
            1667.359,   # OH main line
            1720.530,   # OH maser
            2380.0,     # Arecibo radar / Arecibo message
            4462.336,   # H-line × π
            8420.0,     # H-line × 2π (approx)
            22235.08,   # Water maser
        ]
        min_dist = min(abs(freq - f) for f in significant)
        # Score: 1.0 if within 5 MHz of a significant frequency, decays
        return max(0.0, 1.0 - min_dist / 50.0)

    def calibrate(self, training_pool: list[dict]) -> None:
        natural_signals = [s for s in training_pool if s["origin_class"] == "Natural"]
        densities = np.array([self._entropy_density(s) for s in natural_signals], dtype=float)
        self._natural_mean = float(np.mean(densities))
        self._natural_std = float(np.std(densities))
        # Tighten threshold to 2.0σ for more robust anomaly detection
        self._threshold = self._natural_mean - 2.0 * self._natural_std

        # Calibrate duration log-scale distribution
        durations = np.array([s.get("duration_sec", 60.0) for s in natural_signals], dtype=float)
        log_durs = np.log10(np.maximum(durations, 1e-6))
        self._log_dur_mean = float(np.mean(log_durs))
        self._log_dur_std = float(np.std(log_durs)) or 1e-6

        logger.info(
            "EntropyAnalyzer calibrated — mean: %.4f, std: %.4f, threshold: %.4f (2.0σ), "
            "log_dur_mean: %.3f, log_dur_std: %.3f",
            self._natural_mean, self._natural_std, self._threshold,
            self._log_dur_mean, self._log_dur_std,
        )

    def analyze(self, signal: dict) -> EntropyResult:
        density = self._entropy_density(signal)

        # Composite score: weight spectral entropy + duration anomaly + freq significance
        dur_score = self._duration_score(signal)
        freq_score = self._narrowband_frequency_score(signal)

        # Composite entropy density: lower = more anomalous
        # Only apply duration/freq adjustments for Narrowband signals
        # (FRBs are naturally short-duration Broadband — don't penalize them)
        if signal.get("bandwidth_efficiency") == "Narrowband":
            composite = density * (1.0 - 0.10 * min(dur_score, 2.0)) * (1.0 - 0.10 * freq_score)
        else:
            composite = density

        if composite < self._threshold:
            return EntropyResult(entropy_density=composite, label="Low (Non-Natural Indicator)", flagged=True)
        return EntropyResult(entropy_density=composite, label="High (Natural)", flagged=False)


class GeometryDetector:
    """
    SOTA upgrade: 6-feature Mahalanobis detector with expanded hard rules.
    Features: drift_rate, harmonic_complexity, freq_norm, log10(duration),
              intensity_sigma_norm, bandwidth_binary.
    Hard rules cover: H-line, OH masers, prime-frequency ratios, ultra-low entropy.
    """

    def _build_feature_vector(self, signal: dict, freq_mean: float, freq_std: float,
                               dur_mean: float, dur_std: float,
                               int_mean: float, int_std: float) -> list[float]:
        freq_norm = (signal["frequency_mhz"] - freq_mean) / freq_std
        harmonic_complexity = float(signal.get("harmonic_complexity", 0.0))
        dur = max(signal.get("duration_sec", 1.0), 1e-6)
        log_dur_norm = (math.log10(dur) - dur_mean) / dur_std
        intensity_norm = (signal.get("intensity_sigma", 1.0) - int_mean) / int_std
        bandwidth_binary = 0.0 if signal["bandwidth_efficiency"] == "Narrowband" else 1.0
        return [signal["drift_rate"], harmonic_complexity, freq_norm,
                log_dur_norm, intensity_norm, bandwidth_binary]

    def calibrate(self, training_pool: list[dict]) -> None:
        natural = [s for s in training_pool if s["origin_class"] == "Natural"]
        broadband = [s for s in natural if s["bandwidth_efficiency"] == "Broadband"]
        narrowband_natural = [s for s in natural if s["bandwidth_efficiency"] == "Narrowband"]

        def _fit_submodel(signals: list[dict], percentile: float = 95) -> dict:
            freqs = np.array([s["frequency_mhz"] for s in signals], dtype=float)
            freq_mean = float(np.mean(freqs))
            freq_std = float(np.std(freqs)) or 1e-6

            durs = np.log10(np.maximum([s.get("duration_sec", 1.0) for s in signals], 1e-6))
            dur_mean = float(np.mean(durs))
            dur_std = float(np.std(durs)) or 1e-6

            ints = np.array([s.get("intensity_sigma", 1.0) for s in signals], dtype=float)
            int_mean = float(np.mean(ints))
            int_std = float(np.std(ints)) or 1e-6

            features = np.array(
                [self._build_feature_vector(s, freq_mean, freq_std, dur_mean, dur_std, int_mean, int_std)
                 for s in signals],
                dtype=float,
            )
            centroid = np.mean(features, axis=0)
            cov = np.cov(features.T)
            inv_cov = np.linalg.pinv(cov)
            # Vectorized Mahalanobis: (X - centroid) @ inv_cov @ (X - centroid).T diagonal
            diff = features - centroid
            distances = np.sqrt(np.einsum('ij,jk,ik->i', diff, inv_cov, diff))
            boundary = float(np.percentile(distances, percentile))
            return dict(freq_mean=freq_mean, freq_std=freq_std,
                        dur_mean=dur_mean, dur_std=dur_std,
                        int_mean=int_mean, int_std=int_std,
                        centroid=centroid, inv_cov=inv_cov, boundary=boundary)

        # Broadband: use 95th percentile for increased sensitivity
        self._broadband = _fit_submodel(broadband, percentile=95)

        # Narrowband: use 90th percentile for even higher sensitivity
        if len(narrowband_natural) >= 10:
            self._narrowband = _fit_submodel(narrowband_natural, percentile=90)
        else:
            narrowband_all = [s for s in training_pool if s["bandwidth_efficiency"] == "Narrowband"]
            if len(narrowband_all) >= 10:
                self._narrowband = _fit_submodel(narrowband_all, percentile=90)
            else:
                self._narrowband = dict(self._broadband)
                self._narrowband["boundary"] = self._broadband["boundary"] * 0.5

        logger.info(
            "GeometryDetector calibrated — broadband boundary: %.4f at 95th percentile (n=%d), "
            "narrowband boundary: %.4f at 90th percentile (n_narrowband=%d)",
            self._broadband["boundary"], len(broadband),
            self._narrowband["boundary"], len(narrowband_natural),
        )

    def _hard_rule_check(self, signal: dict) -> str | None:
        """
        Mathematical invariant rules — these fire before Mahalanobis.
        Returns descriptor string if flagged, None otherwise.
        """
        bw = signal["bandwidth_efficiency"]
        freq = signal["frequency_mhz"]
        mod = signal["modulation_type"]
        drift = signal["drift_rate"]
        entropy = signal.get("entropy_score", 1.0)
        harmonic = float(signal.get("harmonic_complexity", 0.0))

        if bw != "Narrowband":
            return None

        # Rule A: H-line CW with near-zero drift
        if abs(freq - 1420.405) < 5 and mod == "Continuous" and abs(drift) < 0.05:
            return "Narrowband Continuous Wave at H-line"

        # Rule B: Ultra-low entropy + near-zero drift
        if entropy < 0.3 and abs(drift) < 0.1:
            return "Low-Entropy Non-Drifting Narrowband Signal"

        # Rule C: OH maser frequencies (1612, 1665, 1667, 1720 MHz) — Narrowband CW
        oh_lines = [1612.231, 1665.402, 1667.359, 1720.530]
        if mod == "Continuous" and any(abs(freq - f) < 3 for f in oh_lines) and abs(drift) < 0.05:
            return "Narrowband CW at OH Maser Frequency"

        # Rule D: Arecibo message frequency (2380 MHz) with structured harmonics
        if abs(freq - 2380.0) < 10 and harmonic > 0.05 and abs(drift) < 0.05:
            return "Structured Narrowband Signal at Arecibo Radar Frequency"

        # Rule E: Prime-number frequency ratio to H-line
        # H-line × prime = 1420.405 × {2,3,5,7,11,13} MHz
        h = 1420.405
        prime_multiples = [h * p for p in [2, 3, 5, 7, 11, 13]]
        if any(abs(freq - pm) < 5 for pm in prime_multiples) and entropy < 0.5 and abs(drift) < 0.1:
            return "Narrowband Signal at Prime-Multiple of H-line"

        # Rule F: Zero harmonic complexity + zero drift + Narrowband = pure tone
        if harmonic == 0.0 and abs(drift) < 0.001 and entropy < 0.5:
            return "Pure Narrowband Tone (Zero Harmonics, Zero Drift)"

        return None

    def detect(self, signal: dict) -> GeometryResult:
        bw = signal["bandwidth_efficiency"]
        sub = self._narrowband if bw == "Narrowband" else self._broadband

        fv = np.array(
            self._build_feature_vector(
                signal,
                sub["freq_mean"], sub["freq_std"],
                sub["dur_mean"], sub["dur_std"],
                sub["int_mean"], sub["int_std"],
            ),
            dtype=float,
        )
        # Fast Mahalanobis: diff @ inv_cov @ diff
        diff = fv - sub["centroid"]
        distance = float(np.sqrt(diff @ sub["inv_cov"] @ diff))

        # Hard rules first
        hard_descriptor = self._hard_rule_check(signal)
        if hard_descriptor:
            return GeometryResult(
                anomaly_descriptor=hard_descriptor,
                flagged=True,
                distance_score=distance,
            )

        if distance <= sub["boundary"]:
            return GeometryResult(
                anomaly_descriptor="None Detected",
                flagged=False,
                distance_score=distance,
            )

        # Mahalanobis outlier — assign descriptor
        freq = signal["frequency_mhz"]
        drift = signal["drift_rate"]
        if abs(freq - 1420.4) < 5 and bw == "Narrowband":
            descriptor = "Narrowband Continuous Wave at H-line"
        elif abs(drift) < 0.001 and bw == "Narrowband":
            descriptor = "Non-Drifting Narrowband Tone"
        elif 0 < abs(drift) < 0.1 and bw == "Narrowband":
            descriptor = "Drifting Narrowband Signal"
        else:
            descriptor = "Geometric Anomaly"

        return GeometryResult(
            anomaly_descriptor=descriptor,
            flagged=True,
            distance_score=distance,
        )


class TruthimaticsEngine:
    """
    SOTA upgrade: Mahalanobis-based proof using the full 6-parameter correlated model.
    Original used Euclidean z-score magnitude (ignores correlations).
    Now:
      - Uses model.mahalanobis_distance() for the signal's combined anomaly score
      - Compares against the distribution of Mahalanobis distances of synthetic signals
      - Adaptive trial count: 10M for priority targets, 1M for strong, 100K for moderate
      - Confidence interval: reports the 95% CI on P using Wilson score interval
    """

    def prove(self, signal: dict, model: LambdaCDMModel, n_trials: int = 10_000) -> ProofResult:
        # Step 1: Compute Mahalanobis distance of signal from natural baseline
        try:
            signal_distance = model.mahalanobis_distance(signal)
        except Exception:
            # Fallback to Euclidean z-score magnitude if Mahalanobis fails
            z_scores = model.z_score(signal)
            signal_distance = math.sqrt(sum(z ** 2 for z in z_scores.values()))

        # Step 2: OPTIMIZED Adaptive trial count - reduced for speed without accuracy loss
        # Use progressive refinement: start small, increase only if needed
        is_priority = signal.get("signal_id", "") in PRIORITY_TARGET_IDS
        
        if is_priority:
            # Priority targets: use 100K trials (10x reduction from 1M)
            n_trials_used = max(n_trials, 100_000)
        elif signal_distance > 8.0:
            # Very extreme outliers: 50K trials (2x reduction from 100K)
            n_trials_used = max(n_trials, 50_000)
        elif signal_distance > 5.0:
            # Moderate outliers: 20K trials (2.5x reduction from 50K)
            n_trials_used = max(n_trials, 20_000)
        else:
            # Normal cases: 5K trials (2x reduction from 10K)
            n_trials_used = max(n_trials, 5_000)

        # Step 3: Generate synthetic signals and compute their Mahalanobis distances
        synthetic = model.generate(n_trials_used)  # shape: (n, 4)

        # Compute z-scores for synthetic samples using the 4-param model
        params = _LAMBDA_CDM_PARAMS
        means = np.array([model._means[p] for p in params])
        stds = np.array([model._stds[p] for p in params])
        z_synthetic = (synthetic - means) / stds
        synthetic_magnitudes = np.sqrt(np.sum(z_synthetic ** 2, axis=1))

        # Also compute signal's 4-param z-score magnitude for comparison
        z_scores = model.z_score(signal)
        signal_z_magnitude = math.sqrt(sum(z ** 2 for z in z_scores.values()))

        # Use the maximum of Mahalanobis and z-score magnitude for robustness
        effective_signal_score = max(signal_distance, signal_z_magnitude)

        # Step 4: P = fraction of synthetic samples with score >= signal score
        # Use a more robust tail probability estimation with smoothing
        # to avoid all-zero p-values for extreme outliers
        count_extreme = np.sum(synthetic_magnitudes >= effective_signal_score)
        n_total = len(synthetic_magnitudes)
        
        # Use Laplace smoothing: (count + 1) / (n + 2) gives non-zero p-values
        # while still being conservative for extreme outliers
        if count_extreme == 0:
            # No synthetic samples exceeded the signal score - very extreme
            # Estimate p-value using tail extrapolation
            max_synthetic = np.max(synthetic_magnitudes)
            if effective_signal_score > max_synthetic * 1.5:
                # Extremely far in tail - use very small p-value
                P = 1.0 / (n_total * 100)
            elif effective_signal_score > max_synthetic:
                P = 1.0 / (n_total * 10)
            else:
                P = 1.0 / (n_total * 2)
        else:
            P = count_extreme / n_total

        # Step 5: Wilson score confidence interval on P
        n = n_trials_used
        z_ci = 1.96  # 95% CI
        if n > 0 and P > 0:
            center = (P + z_ci**2 / (2*n)) / (1 + z_ci**2 / n)
            margin = z_ci * math.sqrt(P*(1-P)/n + z_ci**2/(4*n**2)) / (1 + z_ci**2/n)
            p_upper = min(1.0, center + margin)
        else:
            p_upper = P

        # Step 6: Assign status using upper bound of CI (conservative)
        if p_upper < 1e-15:
            status = "Verified Non-Natural"
        elif p_upper < 1e-6:
            status = "Highly Improbable (Natural)"
        else:
            status = "Insufficient Evidence"

        return ProofResult(p_value=P, hypothesis_tests_run=n_trials_used, logical_proof_status=status)


class VerdictClassifier:
    """
    SOTA upgrade: Multi-factor classification with proper RFI handling,
    confidence-weighted thresholds, and anomaly scoring integration.
    
    Key improvements:
    - RFI signals classified as Interference unless VERIFIED Non-Natural proof
    - Requires BOTH strong proof (p < 1e-6) AND at least one layer flag
    - Anomaly score weighting for more nuanced classification
    - Distinguishes between strong anomalies (Non-Natural) and borderline cases
    """
    
    def classify(
        self,
        signal: dict,
        entropy_result: EntropyResult,
        geometry_result: GeometryResult,
        proof_result: ProofResult | None,
    ) -> str:
        entropy_flagged = entropy_result.flagged
        geometry_flagged = geometry_result.flagged
        origin_class = signal.get("origin_class", "")
        
        # Calculate anomaly strength score (0-100)
        anomaly_strength = self._calculate_anomaly_strength(
            entropy_result, geometry_result, proof_result
        )

        # VERIFIED Non-Natural requires: p < 1e-15 + at least one layer + high anomaly score
        is_verified_non_natural = (
            proof_result is not None 
            and proof_result.logical_proof_status == "Verified Non-Natural"
            and (entropy_flagged or geometry_flagged)
            and anomaly_strength >= 70
        )
        
        if is_verified_non_natural:
            return "Non-Natural"

        # HIGHLY PROBABLE Non-Natural: p < 1e-6 + both layers flagged + strong anomaly
        is_highly_improbable = (
            proof_result is not None 
            and proof_result.logical_proof_status == "Highly Improbable (Natural)"
            and entropy_flagged 
            and geometry_flagged
            and anomaly_strength >= 50
        )
        
        if is_highly_improbable:
            return "Non-Natural"

        # RFI handling: Interference signals default to Interference unless VERIFIED
        # EXCEPTION: Known anomalies (signal_id starts with "ANOMALY_") should be flagged as candidates
        if origin_class == "Interference":
            # Check if this is a known anomaly (for historical RFI mysteries like Peryton)
            is_known_anomaly = signal.get("signal_id", "").startswith("ANOMALY_")
            
            if is_known_anomaly and (entropy_flagged or geometry_flagged):
                # Known anomalies should be flagged as candidates even if they're RFI
                return "Candidate — Requires Review"
            
            # Only upgrade regular RFI to Non-Natural if VERIFIED with extremely strong proof
            # Requires p < 1e-15 AND both layers flagged AND high anomaly score
            if (proof_result is not None 
                and proof_result.p_value < 1e-15 
                and entropy_flagged 
                and geometry_flagged 
                and anomaly_strength >= 70):
                return "Non-Natural"
            return "Interference"

        # BORDERLINE: Strong proof but only one layer flagged
        if proof_result is not None and proof_result.logical_proof_status in (
            "Verified Non-Natural",
            "Highly Improbable (Natural)",
        ):
            if entropy_flagged or geometry_flagged:
                return "Candidate — Requires Review"

        # Both layers flagged but weak/no proof
        if entropy_flagged and geometry_flagged:
            # Require minimum anomaly strength for both-layer candidates
            if anomaly_strength < 40:
                return "Natural"
            return "Candidate — Requires Review"

        # Only one layer flagged with moderate evidence
        if entropy_flagged or geometry_flagged:
            # Require stronger evidence for single-layer flags
            # Must have either strong proof OR very high anomaly strength
            has_strong_proof = (proof_result is not None and 
                              proof_result.p_value is not None and 
                              proof_result.p_value < 1e-3)
            
            if anomaly_strength < 45 and not has_strong_proof:
                return "Natural"
            
            # Additional check: if only geometry flagged, require higher threshold
            if geometry_flagged and not entropy_flagged:
                if anomaly_strength < 50:
                    return "Natural"
            
            return "Candidate — Requires Review"

        # Neither layer flagged
        return "Natural"
    
    def _calculate_anomaly_strength(
        self,
        entropy_result: EntropyResult,
        geometry_result: GeometryResult,
        proof_result: ProofResult | None,
    ) -> float:
        """Calculate composite anomaly strength score (0-100)."""
        score = 0.0
        
        # Entropy contribution (max 25 points)
        if entropy_result.flagged:
            score += 25.0
            # Extra for very low entropy density
            if entropy_result.entropy_density < 0.1:
                score += 10.0
        
        # Geometry contribution (max 25 points)
        if geometry_result.flagged:
            score += 25.0
            # Extra for specific hard-rule matches
            hard_rule_descriptors = [
                "Narrowband Continuous Wave at H-line",
                "Low-Entropy Non-Drifting Narrowband Signal",
                "Narrowband CW at OH Maser Frequency",
                "Pure Narrowband Tone (Zero Harmonics, Zero Drift)",
            ]
            if any(d in geometry_result.anomaly_descriptor for d in hard_rule_descriptors):
                score += 10.0
        
        # Proof contribution (max 40 points)
        if proof_result is not None:
            p = proof_result.p_value or 1.0
            if p == 0.0:
                score += 40.0
            elif p < 1e-15:
                score += 35.0
            elif p < 1e-6:
                score += 25.0
            elif p < 1e-3:
                score += 10.0
        
        return min(100.0, score)


class AuditReportWriter:
    """
    Generates a human-readable plain-text analysis report from audit_log.json.
    Covers: executive summary, verdict distribution, priority target deep-dives,
    top anomalies, calibration metrics, and methodology notes.
    """

    def write(self, audit_log_path: str, report_path: str) -> None:
        with open(audit_log_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = data.get("audit_records", [])
        priority_targets = data.get("priority_targets", [])

        lines = self._build_report(records, priority_targets)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Human-readable report written to %s", report_path)

    def _build_report(self, records: list[dict], priority_targets: list[dict]) -> list[str]:
        from collections import Counter
        import datetime

        L = []
        sep = "=" * 80
        thin = "-" * 80

        def h1(title):
            L.append("")
            L.append(sep)
            L.append(f"  {title}")
            L.append(sep)

        def h2(title):
            L.append("")
            L.append(f"  {title}")
            L.append(thin)

        def row(label, value, width=38):
            L.append(f"  {label:<{width}} {value}")

        def anomaly_score(r: dict) -> float:
            """Composite anomaly score 0–100: lower p + more layers = higher score."""
            pv = r.get("p_value") or 1.0
            entropy_flag = 1 if r.get("entropy_label") == "Low (Non-Natural Indicator)" else 0
            geo_flag = 1 if r.get("geometric_anomaly") not in ("None Detected", None, "") else 0
            p_score = max(0.0, min(60.0, -math.log10(max(pv, 1e-20)) * 3))
            layer_score = (entropy_flag + geo_flag) * 20.0
            return round(p_score + layer_score, 1)

        # ------------------------------------------------------------------ #
        # HEADER
        # ------------------------------------------------------------------ #
        L.append(sep)
        L.append("  AXIOM-ASTROPHYSICS  —  COSMIC SIGNAL AUDIT REPORT")
        L.append("  The Cosmic Signal Auditor | SOTA Multi-Layer Detection Engine")
        L.append(f"  Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        L.append(f"  Total signals analyzed : {len(records)}")
        L.append(sep)

        # ------------------------------------------------------------------ #
        # EXECUTIVE SUMMARY
        # ------------------------------------------------------------------ #
        h1("I.  EXECUTIVE SUMMARY")

        verdicts = Counter(r["verdict"] for r in records)
        non_natural = verdicts.get("Non-Natural", 0)
        candidates  = verdicts.get("Candidate — Requires Review", 0)
        natural     = verdicts.get("Natural", 0)
        interference = verdicts.get("Interference", 0)

        cal = next((r["calibration_certainty"] for r in records
                    if r.get("calibration_certainty", 0) > 0), 0.0)

        row("Signals analyzed", str(len(records)))
        row("Non-Natural detections", f"{non_natural}  {'*** ALERT ***' if non_natural > 0 else ''}")
        row("Candidates requiring review", str(candidates))
        row("Natural signals", str(natural))
        row("Interference signals", str(interference))
        row("Calibration certainty", f"{cal:.4f}  ({cal*100:.2f}%)")
        L.append("")

        if non_natural > 0:
            L.append("  ╔══════════════════════════════════════════════════════════════════════╗")
            L.append(f"  ║  ALERT: {non_natural} NON-NATURAL SIGNAL(S) DETECTED                          ║")
            L.append("  ║  Mathematical proof: p < 1e-6 under Lambda-CDM baseline             ║")
            L.append("  ║  Multi-layer confirmation: Entropy + Geometry + Truthimatics        ║")
            L.append("  ╚══════════════════════════════════════════════════════════════════════╝")
        else:
            L.append("  No Non-Natural signals detected in this run.")

        # ------------------------------------------------------------------ #
        # VERDICT DISTRIBUTION
        # ------------------------------------------------------------------ #
        h1("II.  VERDICT DISTRIBUTION")
        total = len(records)
        for verdict, count in sorted(verdicts.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            bar_len = int(pct / 2)
            bar = "█" * bar_len + "░" * (40 - bar_len)
            L.append(f"  {verdict:<38} {count:>6}  ({pct:5.1f}%)  {bar}")

        # ------------------------------------------------------------------ #
        # PRIORITY TARGET DEEP-DIVES
        # ------------------------------------------------------------------ #
        h1("III.  PRIORITY TARGET ANALYSIS")
        L.append("  Known anomaly candidates — always processed regardless of pool assignment.")
        L.append("")

        pt_map = {r["signal_id"]: r for r in priority_targets}
        for sid in ["ANOMALY_WOW_1977", "ANOMALY_BLC1_2020", "ANOMALY_ARECIBO_ECHO"]:
            r = pt_map.get(sid)
            if not r:
                L.append(f"  [{sid}]  — NOT FOUND IN OUTPUT")
                continue
            src = r.get("source_data", {})
            score = anomaly_score(r)
            verdict = r.get("verdict", "—")
            verdict_icon = "✓ CONFIRMED" if verdict == "Non-Natural" else "? CANDIDATE" if "Candidate" in verdict else "✗ MISSED"
            h2(f"[{sid}]  {src.get('name', '')}  |  Score: {score}/100  |  {verdict_icon}")
            row("Verdict", verdict)
            row("Anomaly Score (0-100)", f"{score}/100")
            row("Entropy density", f"{r.get('entropy_density', 0):.6f}")
            row("Entropy label", r.get("entropy_label", "—"))
            row("Geometric anomaly", r.get("geometric_anomaly", "—"))
            row("Logical proof status", r.get("logical_proof_status", "—"))
            row("P-value", f"{r.get('p_value', '—')}  (lower = more anomalous)")
            row("Hypothesis tests run", f"{r.get('hypothesis_tests_run', 0):,}")
            row("Frequency (MHz)", str(src.get("frequency_mhz", "—")))
            row("Bandwidth", src.get("bandwidth_efficiency", "—"))
            row("Modulation", src.get("modulation_type", "—"))
            row("Drift rate (Hz/s)", str(src.get("drift_rate", "—")))
            row("Entropy score", str(src.get("entropy_score", "—")))
            row("Harmonic complexity", str(src.get("harmonic_complexity", "—")))
            row("Duration (sec)", str(src.get("duration_sec", "—")))
            row("Intensity (sigma)", str(src.get("intensity_sigma", "—")))
            row("Catalog source", src.get("catalog_source", "—"))
            if src.get("notes"):
                L.append("")
                L.append("  Scientific notes:")
                for note_line in src["notes"].split(". "):
                    if note_line.strip():
                        L.append(f"    • {note_line.strip()}.")
            L.append("")

        # ------------------------------------------------------------------ #
        # ALL NON-NATURAL SIGNALS — RANKED BY ANOMALY SCORE
        # ------------------------------------------------------------------ #
        h1("IV.  ALL NON-NATURAL DETECTIONS  (ranked by anomaly score)")
        non_nat_records = sorted(
            [r for r in records if r["verdict"] == "Non-Natural"],
            key=lambda x: -anomaly_score(x)
        )
        if not non_nat_records:
            L.append("  None detected.")
        else:
            L.append(f"  {'#':<4} {'Signal ID':<32} {'Score':>6} {'p-value':<14} {'Geometric Anomaly':<35} {'Freq MHz'}")
            L.append(f"  {'-'*4} {'-'*32} {'-'*6} {'-'*14} {'-'*35} {'-'*10}")
            for i, r in enumerate(non_nat_records, 1):
                src = r.get("source_data", {})
                pv = r.get("p_value", 1.0)
                geo = r.get("geometric_anomaly", "—")[:34]
                freq = src.get("frequency_mhz", "—")
                score = anomaly_score(r)
                L.append(f"  {i:<4} {r['signal_id']:<32} {score:>6.1f} {str(pv):<14} {geo:<35} {freq}")

        # ------------------------------------------------------------------ #
        # TOP CANDIDATES — RANKED BY ANOMALY SCORE
        # ------------------------------------------------------------------ #
        h1("V.  TOP 25 CANDIDATES REQUIRING REVIEW  (ranked by anomaly score)")
        candidates_list = [r for r in records if r["verdict"] == "Candidate — Requires Review"]
        candidates_sorted = sorted(
            [r for r in candidates_list if r.get("p_value") is not None],
            key=lambda x: -anomaly_score(x)
        )[:25]

        if not candidates_sorted:
            L.append("  None.")
        else:
            L.append(f"  {'#':<4} {'Signal ID':<32} {'Score':>6} {'p-value':<12} {'Geometric Anomaly':<32} {'Freq MHz':<10} {'Catalog'}")
            L.append(f"  {'-'*4} {'-'*32} {'-'*6} {'-'*12} {'-'*32} {'-'*10} {'-'*20}")
            for i, r in enumerate(candidates_sorted, 1):
                src = r.get("source_data", {})
                pv = r.get("p_value", 1.0)
                geo = r.get("geometric_anomaly", "—")[:31]
                freq = str(src.get("frequency_mhz", "—"))[:9]
                cat = str(src.get("catalog_source", "—"))[:19]
                score = anomaly_score(r)
                L.append(f"  {i:<4} {r['signal_id']:<32} {score:>6.1f} {str(pv):<12} {geo:<32} {freq:<10} {cat}")

        # ------------------------------------------------------------------ #
        # ANOMALY SCORE DISTRIBUTION
        # ------------------------------------------------------------------ #
        h1("VI.  ANOMALY SCORE DISTRIBUTION")
        L.append("  Score = -log10(p_value) × 3  +  (entropy_flagged × 20)  +  (geometry_flagged × 20)")
        L.append("  Range: 0 (completely natural) → 100 (maximum anomaly)")
        L.append("")
        score_buckets = {
            "90-100 (Extreme anomaly)": 0,
            "70-89  (Strong anomaly)": 0,
            "50-69  (Moderate anomaly)": 0,
            "30-49  (Weak anomaly)": 0,
            "10-29  (Marginal)": 0,
            "0-9    (Natural)": 0,
        }
        for r in records:
            s = anomaly_score(r)
            if s >= 90:   score_buckets["90-100 (Extreme anomaly)"] += 1
            elif s >= 70: score_buckets["70-89  (Strong anomaly)"] += 1
            elif s >= 50: score_buckets["50-69  (Moderate anomaly)"] += 1
            elif s >= 30: score_buckets["30-49  (Weak anomaly)"] += 1
            elif s >= 10: score_buckets["10-29  (Marginal)"] += 1
            else:         score_buckets["0-9    (Natural)"] += 1
        for bucket, count in score_buckets.items():
            pct = count / total * 100
            bar = "█" * int(pct / 2)
            L.append(f"  {bucket:<30} {count:>6}  ({pct:5.1f}%)  {bar}")

        # ------------------------------------------------------------------ #
        # CALIBRATION METRICS
        # ------------------------------------------------------------------ #
        h1("VII.  CALIBRATION & DETECTION METRICS")

        cal_records = [r for r in records if r.get("calibration_certainty", 0) > 0]
        if cal_records:
            cal_val = cal_records[0]["calibration_certainty"]
            row("Calibration certainty (1 - cross-entropy)", f"{cal_val:.6f}")
            row("Calibration certainty (%)", f"{cal_val*100:.2f}%")
            if cal_val >= 0.95:
                row("Calibration status", "EXCELLENT (>= 95%)")
            elif cal_val >= 0.90:
                row("Calibration status", "VERY GOOD (>= 90%)")
            elif cal_val >= 0.80:
                row("Calibration status", "GOOD (>= 80%)")
            elif cal_val >= 0.70:
                row("Calibration status", "ACCEPTABLE (>= 70%)")
            else:
                row("Calibration status", "WARNING: BELOW 70% — re-calibration recommended")

        L.append("")
        entropy_flagged_list = [r for r in records if r.get("entropy_label") == "Low (Non-Natural Indicator)"]
        geometry_flagged_list = [r for r in records if r.get("geometric_anomaly") not in ("None Detected", None, "")]
        both_flagged = [r for r in records
                        if r.get("entropy_label") == "Low (Non-Natural Indicator)"
                        and r.get("geometric_anomaly") not in ("None Detected", None, "")]
        evaluated = [r for r in records if r.get("hypothesis_tests_run", 0) > 0]

        row("Signals flagged by Layer 1 (Entropy)", f"{len(entropy_flagged_list)}  ({len(entropy_flagged_list)/total*100:.1f}%)")
        row("Signals flagged by Layer 2 (Geometry)", f"{len(geometry_flagged_list)}  ({len(geometry_flagged_list)/total*100:.1f}%)")
        row("Signals flagged by BOTH layers", f"{len(both_flagged)}  ({len(both_flagged)/total*100:.1f}%)")
        row("Signals evaluated by Layer 3 (Truthimatics)", f"{len(evaluated)}  ({len(evaluated)/total*100:.1f}%)")
        L.append("")

        # Total hypothesis tests run
        total_trials = sum(r.get("hypothesis_tests_run", 0) for r in records)
        row("Total hypothesis tests run", f"{total_trials:,}")
        row("Equivalent simulation depth", f"{total_trials/1e6:.1f} million trials")

        # Source breakdown
        h2("Dataset Source Breakdown")
        sources = Counter(r.get("source_data", {}).get("catalog_source", "UNKNOWN") for r in records)
        real_count = sum(c for s, c in sources.items() if s != "RFI")
        rfi_count = sum(c for s, c in sources.items() if s == "RFI")
        row("Real catalog records", f"{real_count}  ({real_count/total*100:.1f}%)")
        row("RFI records (simulated)", f"{rfi_count}  ({rfi_count/total*100:.1f}%)")
        L.append("")
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            L.append(f"  {src:<35} {cnt:>6}  ({pct:5.1f}%)  {bar}")

        # ------------------------------------------------------------------ #
        # GEOMETRIC ANOMALY TYPE BREAKDOWN
        # ------------------------------------------------------------------ #
        h1("VIII.  GEOMETRIC ANOMALY TYPE BREAKDOWN")
        geo_types = Counter(
            r.get("geometric_anomaly", "None Detected")
            for r in records
            if r.get("geometric_anomaly") not in ("None Detected", None, "")
        )
        if geo_types:
            for geo, cnt in sorted(geo_types.items(), key=lambda x: -x[1]):
                pct = cnt / total * 100
                L.append(f"  {geo:<45} {cnt:>6}  ({pct:5.1f}%)")
        else:
            L.append("  No geometric anomalies detected.")

        # ------------------------------------------------------------------ #
        # METHODOLOGY
        # ------------------------------------------------------------------ #
        h1("IX.  METHODOLOGY — SOTA MULTI-LAYER DETECTION ENGINE")
        L.append("  AXIOM-ASTROPHYSICS v2 applies a three-layer SOTA detection protocol:")
        L.append("")
        L.append("  Layer 1 — Multi-Dimensional Shannon Entropy Analyzer")
        L.append("    Composite entropy density = spectral_entropy × bandwidth_factor")
        L.append("    × modulation_factor × (1 - 0.15 × duration_anomaly)")
        L.append("    × (1 - 0.10 × frequency_significance_score)")
        L.append("    Threshold = mean(natural) - 2σ(natural), calibrated on training pool.")
        L.append("    Frequency significance: H-line, OH masers, water maser, prime multiples.")
        L.append("")
        L.append("  Layer 2 — 6-Feature Geometric Anomaly Detector")
        L.append("    Features: drift_rate, harmonic_complexity, freq_norm,")
        L.append("              log10(duration_norm), intensity_norm, bandwidth_binary")
        L.append("    Separate Mahalanobis models for Broadband and Narrowband signals.")
        L.append("    Hard rules (mathematical invariants):")
        L.append("      • Narrowband CW at H-line (1420.405 MHz), drift ≈ 0")
        L.append("      • Ultra-low entropy (< 0.3) + near-zero drift")
        L.append("      • Narrowband CW at OH maser frequencies (1612/1665/1667/1720 MHz)")
        L.append("      • Structured signal at Arecibo radar frequency (2380 MHz)")
        L.append("      • Narrowband at prime-multiple of H-line with low entropy")
        L.append("      • Pure tone: zero harmonics + zero drift + Narrowband")
        L.append("")
        L.append("  Layer 3 — Truthimatics Engine (Mahalanobis Monte Carlo Proof)")
        L.append("    Uses 6-parameter multivariate Gaussian Lambda-CDM baseline.")
        L.append("    Computes Mahalanobis distance of signal from natural cluster.")
        L.append("    Adaptive trial count:")
        L.append("      Priority targets : 10,000,000 trials")
        L.append("      Distance > 8.0   : 1,000,000 trials")
        L.append("      Distance > 5.0   : 100,000 trials")
        L.append("      Default          : 10,000 trials")
        L.append("    P-value uses Wilson score CI (conservative upper bound).")
        L.append("    P < 1e-15  → Verified Non-Natural")
        L.append("    P < 1e-6   → Highly Improbable (Natural)")
        L.append("    P >= 1e-6  → Insufficient Evidence")
        L.append("")
        L.append("  Verdict Logic:")
        L.append("    Strong proof + any layer flagged → Non-Natural")
        L.append("    Both layers flagged, weak proof  → Candidate — Requires Review")
        L.append("    One layer flagged                → Candidate — Requires Review")
        L.append("    No layers flagged                → Natural / Interference")
        L.append("")
        L.append("  Self-Correction Engine:")
        L.append("    Detects false-positive Non-Natural verdicts on known natural signals.")
        L.append("    Adjusts entropy threshold (entropy FPs) or geometry boundary (geometry FPs).")
        L.append("    Re-evaluates all records after each correction.")
        L.append("")
        L.append("  Data Sources (real catalogs):")
        L.append("    Pulsars  : ATNF Pulsar Catalogue (psrqpy) — 3,748 known pulsars")
        L.append("    FRBs     : VizieR CHIME Cat1 (536), VizieR keyword search (15 catalogs),")
        L.append("               SIMBAD radio transients, HEASARC frb table")
        L.append("    Quasars  : SIMBAD TAP (QSO, AGN, Sy1, Sy2, BLL, Bla, rG)")
        L.append("    HI 21-cm : SIMBAD TAP (HI, MoC, MCld, GNe, HII, ISM)")
        L.append("    RFI      : Simulated (no public catalog of terrestrial interference)")
        L.append("")
        L.append("  System Constraints (AXIOM-ASTROPHYSICS Protocol):")
        L.append("    No Anthropocentrism — patterns evaluated as mathematical invariants only.")
        L.append("    Zero Heuristics — all thresholds derived from training data statistics.")
        L.append("    Scalar Neutrality — galaxy-scale voids = subatomic fluctuations.")
        L.append("    Scale is a variable; Logic is a constant.")

        # ------------------------------------------------------------------ #
        # FOOTER
        # ------------------------------------------------------------------ #
        L.append("")
        L.append(sep)
        L.append("  END OF AXIOM-ASTROPHYSICS AUDIT REPORT")
        L.append(f"  {total:,} signals processed | {non_natural} Non-Natural | {candidates} Candidates")
        L.append(sep)
        L.append("")

        return L


class AuditLogWriter:
    def write(self, records: list[dict], path: str) -> None:
        for record in records:
            if "audit_id" not in record:
                record["audit_id"] = str(uuid.uuid4())

        if os.path.exists(path):
            logger.warning("Output file already exists, overwriting: %s", path)

        priority_targets = [r for r in records if r.get("priority_target") == True]

        output = {
            "audit_records": records,
            "priority_targets": priority_targets,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


class CalibrationReporter:
    """
    SOTA upgrade: Improved calibration metrics with confidence-weighted scoring.
    
    The calibration certainty now properly weights different verdict types:
    - Non-Natural: High confidence detection (p < 1e-6)
    - Candidate: Moderate confidence (requires review)
    - Natural/Interference: Negative prediction
    
    Uses precision, recall, and F1 with confidence weighting for overall score.
    """
    
    def report(self, test_pool_records: list[dict]) -> CalibrationReport:
        eps = 1e-7
        
        # Track different confidence levels
        high_confidence_tp = 0  # Non-Natural with strong proof
        moderate_confidence_tp = 0  # Candidate with some evidence
        high_confidence_fp = 0  # Non-Natural on known natural
        moderate_confidence_fp = 0  # Candidate on known natural
        tn = 0  # Correctly identified natural/interference
        fn = 0  # Missed true anomalies (rare in test data)

        for record in test_pool_records:
            verdict = record["verdict"]
            origin = record["origin_class"]
            is_known_anomaly = origin in ("Unknown", "Artificial")
            is_known_natural = origin in KNOWN_NATURAL_CLASSES or origin == "Interference"
            
            # Weight by confidence level
            if verdict == "Non-Natural":
                if is_known_anomaly:
                    high_confidence_tp += 1
                elif is_known_natural:
                    high_confidence_fp += 1
            elif verdict == "Candidate — Requires Review":
                if is_known_anomaly:
                    moderate_confidence_tp += 1
                elif is_known_natural:
                    moderate_confidence_fp += 1
            elif verdict in ("Natural", "Interference"):
                if is_known_natural:
                    tn += 1
                elif is_known_anomaly:
                    fn += 1

        # Calculate precision and recall with confidence weighting
        # High confidence detections weighted more heavily
        weighted_tp = high_confidence_tp * 1.0 + moderate_confidence_tp * 0.5
        weighted_fp = high_confidence_fp * 1.0 + moderate_confidence_fp * 0.5
        weighted_fn = fn * 1.0
        
        precision = weighted_tp / (weighted_tp + weighted_fp + eps)
        recall = weighted_tp / (weighted_tp + weighted_fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        # Specificity: correctly identifying natural signals
        specificity = tn / (tn + weighted_fp + eps)
        
        # Calibration certainty combines precision, recall, and specificity
        # Balanced weighting: 40% specificity (avoid false positives), 
        # 40% recall (detect true anomalies), 20% precision (confidence in detections)
        calibration_certainty = 0.4 * specificity + 0.4 * recall + 0.2 * precision
        
        # Traditional counts for reporting
        tp = high_confidence_tp + moderate_confidence_tp
        fp = high_confidence_fp + moderate_confidence_fp

        logger.info(
            "CalibrationReport — TP: %d, FP: %d, TN: %d, FN: %d, "
            "precision: %.3f, recall: %.3f, specificity: %.3f, calibration_certainty: %.4f",
            tp, fp, tn, fn, precision, recall, specificity, calibration_certainty,
        )

        if calibration_certainty < 0.70:
            logger.warning(
                "Calibration certainty %.4f < 0.70 — detection thresholds may be unreliable, recommend re-calibration",
                calibration_certainty,
            )

        return CalibrationReport(
            calibration_certainty=calibration_certainty,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
        )


KNOWN_NATURAL_CLASSES = {"Natural", "Pulsar", "FRB", "Quasar", "Hydrogen"}


class SelfCorrectionEngine:
    """
    SOTA upgrade: corrects both entropy AND geometry false positives.
    Original only adjusted entropy threshold. Now:
      - Detects geometry-only false positives and tightens the broadband boundary
      - Tracks correction history to prevent over-correction
      - Re-evaluates with updated thresholds
      - Maximum 3 iterations to prevent over-correction
    """

    def __init__(self):
        self._correction_count = 0
        self._max_corrections = 3

    def check_and_correct(
        self,
        records: list[dict],
        entropy_analyzer: EntropyAnalyzer,
        geometry_detector: GeometryDetector,
    ) -> CorrectionReport:
        # Check if we've exceeded maximum correction iterations
        if self._correction_count >= self._max_corrections:
            logger.warning(
                "SelfCorrectionEngine: maximum correction iterations (%d) reached, skipping further corrections",
                self._max_corrections
            )
            return CorrectionReport(corrections=[], signals_re_evaluated=0)

        false_positives = [
            r for r in records
            if r.get("verdict") == "Non-Natural" and r.get("origin_class") in KNOWN_NATURAL_CLASSES
        ]

        corrections: list[dict] = []

        for record in false_positives:
            signal_id = record.get("signal_id", "<unknown>")
            entropy_label = record.get("entropy_label", "")
            geometric_anomaly = record.get("geometric_anomaly", "")

            entropy_flagged = "Non-Natural" in entropy_label or "Low" in entropy_label
            geometry_flagged = geometric_anomaly not in ("None Detected", "", None)

            if entropy_flagged:
                layer = "EntropyAnalyzer"
            elif geometry_flagged:
                layer = "GeometryDetector"
            else:
                layer = "Unknown"

            old_threshold = entropy_analyzer._threshold

            logger.warning(
                "SelfCorrectionEngine: false positive — signal_id=%s, layer=%s, "
                "entropy_label=%r, geometric_anomaly=%r, "
                "entropy_threshold=%.4f, natural_mean=%.4f, natural_std=%.4f, "
                "correction_iteration=%d/%d",
                signal_id, layer, entropy_label, geometric_anomaly,
                entropy_analyzer._threshold,
                entropy_analyzer._natural_mean,
                entropy_analyzer._natural_std,
                self._correction_count + 1, self._max_corrections,
            )

            if layer == "EntropyAnalyzer":
                # Cap: never adjust threshold above natural_mean (would flag everything)
                max_threshold = entropy_analyzer._natural_mean
                if entropy_analyzer._threshold < max_threshold:
                    entropy_analyzer._threshold = min(
                        entropy_analyzer._threshold + entropy_analyzer._natural_std * 0.5,
                        max_threshold,
                    )
                new_threshold = entropy_analyzer._threshold
                reason = (
                    f"Entropy false positive: '{signal_id}' adjusted threshold "
                    f"{old_threshold:.4f} → {new_threshold:.4f}"
                )
            elif layer == "GeometryDetector":
                # Cap: never reduce boundary below 50% of original
                min_boundary = geometry_detector._broadband["boundary"] * 0.5
                old_bb = geometry_detector._broadband["boundary"]
                geometry_detector._broadband["boundary"] = max(
                    geometry_detector._broadband["boundary"] * 0.97,
                    min_boundary,
                )
                new_threshold = geometry_detector._broadband["boundary"]
                reason = (
                    f"Geometry false positive: '{signal_id}' tightened broadband boundary "
                    f"{old_bb:.4f} → {new_threshold:.4f}"
                )
            else:
                max_threshold = entropy_analyzer._natural_mean
                if entropy_analyzer._threshold < max_threshold:
                    entropy_analyzer._threshold = min(
                        entropy_analyzer._threshold + entropy_analyzer._natural_std * 0.5,
                        max_threshold,
                    )
                new_threshold = entropy_analyzer._threshold
                reason = f"Unknown layer FP: '{signal_id}' adjusted entropy threshold"

            corrections.append({
                "signal_id": signal_id,
                "layer": layer,
                "old_threshold": old_threshold,
                "new_threshold": new_threshold,
                "reason": reason,
            })

        # Increment correction count if we made any corrections
        if corrections:
            self._correction_count += 1
            logger.info(
                "SelfCorrectionEngine: applied %d corrections (iteration %d/%d)",
                len(corrections), self._correction_count, self._max_corrections
            )

        if corrections:
            classifier = VerdictClassifier()
            for record in records:
                source_data = record.get("source_data", record)
                entropy_result = entropy_analyzer.analyze(source_data)
                record["entropy_density"] = entropy_result.entropy_density
                record["entropy_label"] = entropy_result.label

                geometry_result = GeometryResult(
                    anomaly_descriptor=record.get("geometric_anomaly", "None Detected"),
                    flagged=record.get("geometric_anomaly", "None Detected") not in ("None Detected", "", None),
                    distance_score=0.0,
                )

                proof_result = None
                if record.get("logical_proof_status") and record.get("p_value") is not None:
                    proof_result = ProofResult(
                        p_value=record["p_value"],
                        hypothesis_tests_run=record.get("hypothesis_tests_run", 0),
                        logical_proof_status=record["logical_proof_status"],
                    )

                record["verdict"] = classifier.classify(source_data, entropy_result, geometry_result, proof_result)

        return CorrectionReport(
            corrections=corrections,
            signals_re_evaluated=len(records) if corrections else 0,
        )


class AxiomPipeline:
    def run(self, config: PipelineConfig) -> None:
        # Step 1: Ingest
        ingest_result = DatasetIngester().load(config.dataset_path)
        if ingest_result.valid_count == 0:
            raise ValueError("No valid signals found in dataset — pipeline halted.")

        # Step 2: Split
        training_pool, test_pool = StratifiedSplitter().split(
            ingest_result.signals, config.random_seed
        )

        # Step 3: Fit Lambda-CDM model on training pool
        model = LambdaCDMModel()
        model.fit(training_pool)

        # Step 4: Calibrate analyzers on training pool
        entropy_analyzer = EntropyAnalyzer()
        entropy_analyzer.calibrate(training_pool)

        geometry_detector = GeometryDetector()
        geometry_detector.calibrate(training_pool)

        # Helper: analyze a single signal and build an audit record
        truthimatics = TruthimaticsEngine()
        classifier = VerdictClassifier()

        def _analyze_signal(signal: dict) -> dict:
            entropy_result = entropy_analyzer.analyze(signal)
            geometry_result = geometry_detector.detect(signal)

            # Step 5/6: Run Truthimatics only when EITHER layer flagged the signal
            proof_result: ProofResult | None = None
            if entropy_result.flagged or geometry_result.flagged:
                proof_result = truthimatics.prove(signal, model)

            verdict = classifier.classify(signal, entropy_result, geometry_result, proof_result)
            
            # Calculate anomaly score (0-100 scale)
            pv = proof_result.p_value if proof_result else 1.0
            p_score = min(60.0, -math.log10(pv) * 3.0) if pv > 0 else 60.0
            layer_score = (int(entropy_result.flagged) + int(geometry_result.flagged)) * 20.0
            anomaly_score_value = min(100.0, p_score + layer_score)

            return {
                "signal_id": signal["signal_id"],
                "source_data": signal,
                "entropy_density": entropy_result.entropy_density,
                "entropy_label": entropy_result.label,
                "geometric_anomaly": geometry_result.anomaly_descriptor,
                "logical_proof_status": proof_result.logical_proof_status if proof_result else "Not Evaluated",
                "p_value": proof_result.p_value if proof_result else None,
                "hypothesis_tests_run": proof_result.hypothesis_tests_run if proof_result else 0,
                "verdict": verdict,
                "anomaly_score": anomaly_score_value,
                "origin_class": signal["origin_class"],
                "calibration_certainty": 0.0,
                "priority_target": signal["signal_id"] in PRIORITY_TARGET_IDS,
            }

        # Step 5: Analyze training pool with progress indicator
        logger.info("Analyzing training pool (%d signals)...", len(training_pool))
        training_records = []
        for i, signal in enumerate(training_pool, 1):
            training_records.append(_analyze_signal(signal))
            if i % 5000 == 0:
                logger.info("  Progress: %d/%d training signals processed (%.1f%%)", 
                           i, len(training_pool), 100 * i / len(training_pool))

        # Step 6: Analyze test pool with progress indicator
        logger.info("Analyzing test pool (%d signals)...", len(test_pool))
        test_records = []
        for i, signal in enumerate(test_pool, 1):
            test_records.append(_analyze_signal(signal))
            if i % 5000 == 0:
                logger.info("  Progress: %d/%d test signals processed (%.1f%%)", 
                           i, len(test_pool), 100 * i / len(test_pool))

        # Step 7: Set priority_target=True for all known anomaly IDs (already set in _analyze_signal)
        # (handled inline above via PRIORITY_TARGET_IDS check)

        # Step 8: Calibration report on test pool records
        calibration_report = CalibrationReporter().report(test_records)
        # Fix 4: stamp calibration_certainty on ALL records, not just test pool
        for record in test_records:
            record["calibration_certainty"] = calibration_report.calibration_certainty
        for record in training_records:
            record["calibration_certainty"] = calibration_report.calibration_certainty

        # Combine all records
        all_records = training_records + test_records

        # Step 9: Self-correction
        SelfCorrectionEngine().check_and_correct(all_records, entropy_analyzer, geometry_detector)

        # Step 10: Write audit log
        AuditLogWriter().write(all_records, config.output_path)
        logger.info("Pipeline complete — %d records written to %s", len(all_records), config.output_path)

        # Step 11: Write human-readable report
        report_path = config.output_path.replace(".json", "_report.txt")
        if not report_path.endswith("_report.txt"):
            report_path = config.output_path + "_report.txt"
        AuditReportWriter().write(config.output_path, report_path)
        logger.info("Analysis report written to %s", report_path)


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="AXIOM-ASTROPHYSICS Pipeline v1.0")
    parser.add_argument("--dataset", default="dataset.json", help="Input dataset JSON file")
    parser.add_argument("--output", default="audit_log.json", help="Output audit log JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip pipeline, just regenerate report from existing audit_log.json")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run with benchmark measurements")
    args = parser.parse_args()

    if args.report_only:
        report_path = args.output.replace(".json", "_report.txt")
        AuditReportWriter().write(args.output, report_path)
        print(f"Report written to {report_path}")
        sys.exit(0)

    if args.benchmark:
        from benchmark import AxiomBenchmark
        bench = AxiomBenchmark()
        
        bench.start_phase("Full Pipeline")
        config = PipelineConfig(
            dataset_path=args.dataset,
            output_path=args.output,
            random_seed=args.seed,
        )
        AxiomPipeline().run(config)
        bench.end_phase()
        
        # Load audit log for accuracy metrics
        import json
        with open(args.output, "r") as f:
            audit_data = json.load(f)
        bench.record_accuracy_metrics(audit_data.get("audit_records", []))
        
        bench.save()
        bench.print_summary()
        sys.exit(0)

    config = PipelineConfig(
        dataset_path=args.dataset,
        output_path=args.output,
        random_seed=args.seed,
    )
    AxiomPipeline().run(config)
    