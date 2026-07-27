import logging
import os
from pathlib import Path

import joblib

log = logging.getLogger(__name__)
import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


class ConformalCalibrator:
    """
    Computes distribution-free, calibrated p-values via split conformal prediction.
    Takes outlier density scores (log-likelihoods) of signals and compares them 
    against a held-out calibration set of natural signals.
    
    Guarantee: False Alarm Rate is controlled below the specified significance level alpha.
    """
    def __init__(self, cache_path=None):
        if cache_path is None:
            cache_path = str(_PKG_ROOT / "data" / "models" / "conformal_calibration.pkl")
        self.cache_path = cache_path
        self.calibration_scores = None
        self.natural_min = None
        self.natural_max = None
        self.load()

    def fit(self, natural_log_probs):
        """
        Calibrate on a STRICTLY HELD-OUT set of natural signals.

        The calibration scores must NEVER overlap the training set used to fit
        the density estimator, otherwise the conformal false-alarm guarantee is
        invalid. `natural_log_probs` are per-class max log-likelihoods of
        real natural sources reserved for calibration only.
        """
        scores = np.asarray(natural_log_probs, dtype=np.float64)
        if scores.size == 0:
            raise ValueError("ConformalCalibrator.fit received an empty calibration set.")
        self.calibration_scores = np.sort(scores)
        self.natural_min = float(scores.min())
        self.natural_max = float(scores.max())
        self.save()
        return True

    def compute_p_value(self, log_prob):
        """
        Compute conformal p-value for a single log-probability score or an array of scores.
        Smaller p-value = more anomalous (less likely to be natural).
        """
        if self.calibration_scores is None:
            # Fallback if not calibrated
            return 1.0 if np.isscalar(log_prob) else np.ones(len(log_prob))
            
        N = len(self.calibration_scores)
        
        if np.isscalar(log_prob):
            # Number of calibration scores smaller than or equal to the test score
            count = np.sum(self.calibration_scores <= log_prob)
            p_val = (count + 1.0) / (N + 1.0)
            return float(p_val)
        else:
            log_prob = np.asarray(log_prob)
            p_vals = np.zeros(len(log_prob))
            for i, val in enumerate(log_prob):
                count = np.sum(self.calibration_scores <= val)
                p_vals[i] = (count + 1.0) / (N + 1.0)
            return p_vals

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            joblib.dump({
                "calibration_scores": self.calibration_scores,
                "natural_min": self.natural_min,
                "natural_max": self.natural_max,
            }, self.cache_path)
        except Exception as e:
            log.error("ConformalCalibrator save failed: %s", e)

    def load(self):
        if os.path.exists(self.cache_path):
            try:
                data = joblib.load(self.cache_path)
                self.calibration_scores = data["calibration_scores"]
                self.natural_min = data.get("natural_min", None)
                self.natural_max = data.get("natural_max", None)
            except Exception as exc:
                log.warning("Corrupted calibration cache, ignoring: %s", exc)
