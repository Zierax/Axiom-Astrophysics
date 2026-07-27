import logging
import os
from pathlib import Path

import joblib

log = logging.getLogger(__name__)
import numpy as np
from sklearn.mixture import GaussianMixture

_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


class AnomalyDensityEstimator:
    """
    Fits a non-parametric density estimator over natural signal features.

    Two modes:
      * Joint mode  (fit(X)): one GMM over all in-distribution features.
      * Per-class mode (fit(X, y)): one GMM per natural class (e.g. pulsar,
        RFI). A test point is scored by the MAXIMUM class log-likelihood,
        i.e. "is this point typical of ANY natural population?" This prevents
        the minority class from being penalised by a joint, imbalanced density
        and is what makes conformal OOD detection meaningful.

    If GMM fails or features have low variance, falls back to robust scaling
    + covariance.
    """
    def __init__(self, n_components=5, cache_path=None):
        if cache_path is None:
            cache_path = str(_PKG_ROOT / "data" / "models" / "density_model.pkl")
        self.n_components = n_components
        self.cache_path = cache_path
        self.model = None
        self.feature_means = None
        self.feature_stds = None
        self.per_class = False
        self.class_models = {}
        self.class_feature_means = {}
        self.class_feature_stds = {}

        # Load cached model if exists
        self.load()

    def fit(self, X_train, y_train=None):
        """
        Fit density estimator on natural signals.

        X_train : shape (N, D) numpy array of natural signal features.
        y_train : optional shape (N,) labels. If given (and >1 class present),
                  fit one GMM per class and score by max class likelihood.
        """
        if y_train is not None and len(np.unique(y_train)) > 1:
            return self._fit_per_class(X_train, y_train)

        if len(X_train) < self.n_components:
            # Fallback for small datasets
            self.n_components = max(1, len(X_train))

        # Reset any state inherited from a cached model so a fit on a different
        # feature dimension fully overwrites the previous estimator.
        self.per_class = False
        self.class_models = {}
        self.class_feature_means = {}
        self.class_feature_stds = {}
        self.cov = None
        self.inv_cov = None

        # Normalize features
        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0)
        self.feature_stds[self.feature_stds == 0.0] = 1e-6

        X_norm = (X_train - self.feature_means) / self.feature_stds

        # Fit GMM
        try:
            self.model = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                random_state=42,
                max_iter=100
            )
            self.model.fit(X_norm)
            self.save()
            return True
        except Exception as e:
            log.warning("GMM fit failed: %s — falling back to single-component", e)
            # Fallback to single Gaussian via sample covariance
            self.model = None
            self.cov = np.cov(X_norm.T)
            self.inv_cov = np.linalg.pinv(self.cov)
            return False

    def _fit_per_class(self, X_train, y_train):
        self.per_class = True
        # Reset all per-class state so a fresh fit never inherits stale classes
        # (e.g. a 12-D model loaded from disk) that would otherwise yield a
        # feature-dimension mismatch in log_prob. Only classes present in the
        # current fit survive.
        self.class_models = {}
        self.class_feature_means = {}
        self.class_feature_stds = {}
        self.feature_means = np.mean(X_train, axis=0)
        self.feature_stds = np.std(X_train, axis=0)
        self.feature_stds[self.feature_stds == 0.0] = 1e-6

        for c in np.unique(y_train):
            Xc = X_train[y_train == c]
            means = Xc.mean(axis=0)
            stds = Xc.std(axis=0)
            stds[stds == 0.0] = 1e-6
            Xn = (Xc - means) / stds
            n_comp = min(self.n_components, max(1, len(Xc) // 20))
            try:
                gmm = GaussianMixture(
                    n_components=max(1, n_comp),
                    covariance_type="full",
                    random_state=42,
                    max_iter=100,
                )
                gmm.fit(Xn)
            except Exception as exc:
                log.debug("GMM init failed, using single component: %s", exc)
                gmm = None
            self.class_models[int(c)] = gmm
            self.class_feature_means[int(c)] = means
            self.class_feature_stds[int(c)] = stds

        self.save()
        return True

    def log_prob(self, X):
        """
        Compute log-likelihood log p(x) for input features.
        X: shape (N, D) numpy array of signal features.
        Returns the max over per-class densities (if per-class) or the single
        joint density otherwise.
        """
        if self.feature_means is None:
            return np.zeros(len(X))

        if self.per_class and self.class_models:
            class_scores = []
            for c, gmm in self.class_models.items():
                means = self.class_feature_means[c]
                stds = self.class_feature_stds[c]
                Xn = (X - means) / stds
                if gmm is not None:
                    try:
                        class_scores.append(gmm.score_samples(Xn))
                        continue
                    except Exception as exc:
                        log.debug("Per-class log_prob failed: %s", exc)
                # Fallback: single Gaussian for this class
                diff = Xn
                cov = np.cov((X - means) / stds, rowvar=False)
                inv = np.linalg.pinv(cov)
                d2 = np.sum(diff @ inv * diff, axis=1)
                class_scores.append(-0.5 * d2)
            return np.max(np.stack(class_scores, axis=1), axis=1)

        X_norm = (X - self.feature_means) / self.feature_stds

        if self.model is not None:
            try:
                return self.model.score_samples(X_norm)
            except Exception as exc:
                log.debug("GMM log_prob failed: %s", exc)

        # Fallback to single Gaussian Mahalanobis log likelihood
        if not hasattr(self, "inv_cov") or self.inv_cov is None:
            return np.zeros(len(X))
        diff = X_norm
        d2 = np.sum(diff @ self.inv_cov * diff, axis=1)
        return -0.5 * d2

    def score_samples(self, X):
        """Alias for log_prob"""
        return self.log_prob(X)

    def log_prob_per_class(self, X, class_idx):
        """Log-density of X under a single natural class's GMM.

        Used for class-conditional conformal p-values so that a minority class
        (e.g. pulsars) is compared against its own population rather than the
        RFI-dominated joint pool.
        """
        if not self.per_class or int(class_idx) not in self.class_models:
            # Fall back to the joint score if no per-class model.
            return self.log_prob(X)
        c = int(class_idx)
        means = self.class_feature_means[c]
        stds = self.class_feature_stds[c]
        Xn = (np.asarray(X) - means) / stds
        gmm = self.class_models[c]
        if gmm is not None:
            try:
                return gmm.score_samples(Xn)
            except Exception as exc:
                log.debug("log_prob_per_class failed for class %d: %s", c, exc)
        diff = Xn
        cov = np.cov(Xn, rowvar=False)
        inv = np.linalg.pinv(cov)
        d2 = np.sum(diff @ inv * diff, axis=1)
        return -0.5 * d2

    def save(self):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            joblib.dump({
                "model": self.model,
                "per_class": self.per_class,
                "class_models": self.class_models,
                "class_feature_means": self.class_feature_means,
                "class_feature_stds": self.class_feature_stds,
                "n_components": self.n_components,
                "feature_means": self.feature_means,
                "feature_stds": self.feature_stds,
                "cov": getattr(self, "cov", None),
                "inv_cov": getattr(self, "inv_cov", None)
            }, self.cache_path)
        except Exception as e:
            log.error("Save failed: %s", e)

    def load(self):
        if os.path.exists(self.cache_path):
            try:
                data = joblib.load(self.cache_path)
                self.model = data["model"]
                self.per_class = data.get("per_class", False)
                self.class_models = data.get("class_models", {})
                self.class_feature_means = data.get("class_feature_means", {})
                self.class_feature_stds = data.get("class_feature_stds", {})
                self.n_components = data["n_components"]
                self.feature_means = data["feature_means"]
                self.feature_stds = data["feature_stds"]
                if "cov" in data:
                    self.cov = data["cov"]
                if "inv_cov" in data:
                    self.inv_cov = data["inv_cov"]
            except Exception as exc:
                log.warning("Corrupted cache, ignoring: %s", exc)
