"""Group-aware evaluation of the population-scale catalog manifold.

Two rigorous, leakage-free protocols operate on the per-object population manifold
(:mod:`axiom.data.population`), where every row is an *independent* astronomical
object identified by a unique ``group_id``:

1. :func:`evaluate_population_classification` — stratified **group** K-fold
   multiclass classification of real object types (pulsar / FRB / RRAT / magnetar
   / RFI). Because each object appears once and folds are keyed on ``group_id``,
   there is no within-observation leakage. Reports macro-F1, Matthews correlation
   and balanced accuracy with nonparametric bootstrap confidence intervals over
   the out-of-fold predictions.

2. :func:`evaluate_population_ood` — **leave-class-out** conformal novelty
   detection. One physical population (default: extragalactic FRBs) is withheld
   entirely; a per-class Mahalanobis density is fit on the remaining "normal"
   populations and cross-conformally calibrated. This is an honest OOD test — the
   detector never sees the novel class during fit *or* calibration — and, unlike
   the synthetic waterfall audit, the anomaly is a *real distinct population*.

All randomness is seeded; imputation is fit per training fold only.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import StratifiedGroupKFold

from axiom.data.population import CODE_TO_CLASS, PopulationData, build_population
from axiom.dsp.physical_features import (
    impute_apply,
    impute_fit,
)

log = logging.getLogger(__name__)

_EPS = 1e-12


class GroupOODError(RuntimeError):
    """Raised on malformed population input to the group-aware evaluators."""


# ---------------------------------------------------------------------------
# Shared statistics.
# ---------------------------------------------------------------------------
def _auroc(neg_scores: np.ndarray, pos_scores: np.ndarray) -> float:
    """AUROC via the Mann-Whitney U statistic (higher pos score = more anomalous)."""
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = np.asarray(pos_scores, dtype=np.float64)
    if neg.size == 0 or pos.size == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    order = np.argsort(allv, kind="stable")
    ranks = np.empty(allv.size, dtype=np.float64)
    ranks[order] = np.arange(1, allv.size + 1, dtype=np.float64)
    sorted_v = allv[order]
    i = 0
    while i < sorted_v.size:
        j = i
        while j + 1 < sorted_v.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    r_pos = ranks[neg.size:].sum()
    n_pos, n_neg = pos.size, neg.size
    u = r_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _conformal_threshold(cal_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal upper threshold: the ceil((1-alpha)(n+1))-th order stat."""
    s = np.sort(np.asarray(cal_scores, dtype=np.float64))
    n = s.size
    if n == 0:
        raise GroupOODError("empty calibration scores.")
    k = int(np.ceil((1.0 - alpha) * (n + 1))) - 1
    k = int(np.clip(k, 0, n - 1))
    return float(s[k])


def _bootstrap_ci(
    values: np.ndarray,
    statistic,
    *,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Percentile bootstrap CI of a statistic over resampled row indices.

    ``statistic`` maps an index array to a scalar. Returns ``(low, high)`` at the
    ``1 - alpha`` level; degenerate inputs yield ``(nan, nan)``.
    """
    n = int(values.shape[0])
    if n < 2 or n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[b] = statistic(idx)
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(stats, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(stats, 100.0 * (1.0 - alpha / 2.0)))
    return (lo, hi)


# ---------------------------------------------------------------------------
# Protocol 1: group K-fold multiclass classification.
# ---------------------------------------------------------------------------
@dataclass
class PopulationClassificationReport:
    """Out-of-fold multiclass classification metrics with bootstrap CIs.

    The primary, imbalance-robust headline metric is the Matthews correlation
    coefficient (``mcc``); ``macro_f1`` weights every class equally and is
    therefore dominated by rare pulsar subtypes (RRAT, magnetar) that overlap the
    pulsar population physically — it is reported for full transparency but is not
    the pass gate. ``weighted_f1`` weights classes by support.
    """

    mcc: float
    mcc_ci: Tuple[float, float]
    weighted_f1: float
    weighted_f1_ci: Tuple[float, float]
    macro_f1: float
    balanced_accuracy: float
    accuracy: float
    per_class_f1: Dict[str, float]
    class_counts: Dict[str, int]
    confusion: List[List[int]]
    labels: List[str]
    n_objects: int
    n_splits: int
    n_features: int
    passed: bool = False


def evaluate_population_classification(
    population: Optional[PopulationData] = None,
    *,
    seed: int = 42,
    n_splits: int = 5,
    n_bootstrap: int = 1000,
    mcc_min: float = 0.60,
    weighted_f1_min: float = 0.90,
) -> PopulationClassificationReport:
    """Stratified group K-fold multiclass classification of real object types.

    A :class:`~sklearn.ensemble.HistGradientBoostingClassifier` (deterministic
    under a fixed seed, native NaN handling) is trained out-of-fold with
    fold-local median imputation of the continuous physical block. Folds are keyed
    on ``group_id`` so no object leaks across the train/test boundary.

    The pass gate uses the imbalance-robust Matthews correlation coefficient and
    support-weighted F1; macro-F1 and per-class F1 are reported for transparency.
    """
    if population is None:
        population = build_population(cache=True)
    X, y = np.asarray(population.X, dtype=np.float64), np.asarray(population.y)
    groups = np.asarray(population.group_ids)
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise GroupOODError("population X/y shape mismatch.")

    classes, class_sizes = np.unique(y, return_counts=True)
    if classes.size < 2:
        raise GroupOODError("need >=2 classes for classification.")
    k = int(np.clip(n_splits, 2, int(class_sizes.min())))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

    oof_pred = np.full(y.shape[0], -1, dtype=np.int64)
    for train_idx, test_idx in splitter.split(X, y, groups):
        medians = impute_fit(X[train_idx])
        X_tr = impute_apply(X[train_idx], medians)
        X_te = impute_apply(X[test_idx], medians)
        clf = HistGradientBoostingClassifier(random_state=seed, max_iter=300)
        clf.fit(X_tr, y[train_idx])
        oof_pred[test_idx] = clf.predict(X_te)
    if np.any(oof_pred < 0):  # pragma: no cover - fold coverage
        raise GroupOODError("cross-validation left unpredicted objects.")

    labels_sorted = [int(c) for c in classes]
    label_names = [CODE_TO_CLASS.get(c, str(c)) for c in labels_sorted]
    macro_f1 = float(f1_score(y, oof_pred, average="macro", labels=labels_sorted))
    weighted_f1 = float(f1_score(y, oof_pred, average="weighted", labels=labels_sorted))
    mcc = float(matthews_corrcoef(y, oof_pred))
    bal_acc = float(balanced_accuracy_score(y, oof_pred))
    acc = float(np.mean(oof_pred == y))
    per_class = f1_score(y, oof_pred, average=None, labels=labels_sorted)
    per_class_f1 = {label_names[i]: float(per_class[i]) for i in range(len(labels_sorted))}

    n_cls = len(labels_sorted)
    code_to_pos = {c: i for i, c in enumerate(labels_sorted)}
    confusion = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y, oof_pred):
        confusion[code_to_pos[int(t)], code_to_pos[int(p)]] += 1

    def _mcc_stat(idx: np.ndarray) -> float:
        yt, yp = y[idx], oof_pred[idx]
        if np.unique(yt).size < 2:
            return float("nan")
        return float(matthews_corrcoef(yt, yp))

    def _wf1_stat(idx: np.ndarray) -> float:
        return float(f1_score(y[idx], oof_pred[idx], average="weighted",
                              labels=labels_sorted, zero_division=0))

    mcc_ci = _bootstrap_ci(y, _mcc_stat, n_boot=n_bootstrap, seed=seed)
    wf1_ci = _bootstrap_ci(y, _wf1_stat, n_boot=n_bootstrap, seed=seed + 1)

    report = PopulationClassificationReport(
        mcc=mcc,
        mcc_ci=mcc_ci,
        weighted_f1=weighted_f1,
        weighted_f1_ci=wf1_ci,
        macro_f1=macro_f1,
        balanced_accuracy=bal_acc,
        accuracy=acc,
        per_class_f1=per_class_f1,
        class_counts=population.class_counts(),
        confusion=confusion.tolist(),
        labels=label_names,
        n_objects=int(y.shape[0]),
        n_splits=k,
        n_features=int(X.shape[1]),
        passed=bool(mcc >= mcc_min and weighted_f1 >= weighted_f1_min),
    )
    log.info("[group_ood] classification MCC=%.3f weightedF1=%.3f macroF1=%.3f "
             "balAcc=%.3f (%d objects, k=%d)", mcc, weighted_f1, macro_f1,
             bal_acc, y.shape[0], k)
    return report


# ---------------------------------------------------------------------------
# Protocol 2: leave-class-out conformal novelty detection.
# ---------------------------------------------------------------------------
@dataclass
class _DensityModel:
    """Per-class robust location + Ledoit-Wolf precision in standardised space."""

    label: int
    mean: np.ndarray
    precision: np.ndarray
    n_fit: int


class _MahalanobisDetector:
    """Min-over-normal-class standardised Mahalanobis novelty scorer."""

    def __init__(self) -> None:
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._models: List[_DensityModel] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_MahalanobisDetector":
        X = np.asarray(X, dtype=np.float64)
        if not np.all(np.isfinite(X)):
            raise GroupOODError("detector fit received non-finite features.")
        self._mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std <= _EPS] = 1.0
        self._std = std
        Xs = (X - self._mean) / self._std
        self._models = []
        for c in np.unique(y):
            Xc = Xs[y == c]
            if Xc.shape[0] < 2:
                precision = np.eye(Xc.shape[1])
                mean = Xc.mean(axis=0) if Xc.shape[0] else np.zeros(Xc.shape[1])
            else:
                mean = np.median(Xc, axis=0)
                Xc_centered = Xc - mean
                precision = LedoitWolf(assume_centered=True).fit(Xc_centered).precision_
            self._models.append(_DensityModel(int(c), mean, precision, int(Xc.shape[0])))
        if not self._models:
            raise GroupOODError("no normal classes to fit.")
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._mean is None:
            raise GroupOODError("detector not fitted.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[None, :]
        Xs = (X - self._mean) / self._std
        dists = np.empty((Xs.shape[0], len(self._models)), dtype=np.float64)
        for j, m in enumerate(self._models):
            diff = Xs - m.mean
            d2 = np.einsum("ij,jk,ik->i", diff, m.precision, diff)
            dists[:, j] = np.sqrt(np.clip(d2, 0.0, None))
        return dists.min(axis=1)


@dataclass
class PopulationOODReport:
    """Leave-class-out conformal novelty-detection metrics."""

    novel_class: str
    normal_classes: List[str]
    auroc: float
    auroc_ci: Tuple[float, float]
    novel_tpr: float
    normal_fpr: float
    conformal_coverage: float
    threshold: float
    alpha: float
    n_normal: int
    n_novel: int
    n_splits: int
    n_features: int
    passed: bool = False
    normal_scores: List[float] = field(default_factory=list)
    novel_scores: List[float] = field(default_factory=list)


def evaluate_population_ood(
    population: Optional[PopulationData] = None,
    *,
    novel_class: str = "FRB",
    seed: int = 42,
    n_splits: int = 5,
    alpha: float = 0.1,
    n_bootstrap: int = 1000,
    auroc_min: float = 0.90,
    tpr_min: float = 0.90,
) -> PopulationOODReport:
    """Leave-class-out conformal OOD: withhold ``novel_class`` entirely.

    The normal manifold is every class except ``novel_class``. Normal objects are
    scored out-of-fold by a per-class Mahalanobis detector (group K-fold), giving
    an honest conformal null; the withheld novel population is scored by a
    detector fit on all normal objects. Reports threshold-free AUROC (primary,
    with bootstrap CI), conformal TPR at ``alpha`` and empirical normal coverage.
    """
    if population is None:
        population = build_population(cache=True)
    X = np.asarray(population.X, dtype=np.float64)
    y = np.asarray(population.y)
    groups = np.asarray(population.group_ids)

    novel_code = population.code_of(novel_class)
    novel_mask = y == novel_code
    if not novel_mask.any():
        raise GroupOODError(f"novel class {novel_class!r} absent from population.")
    normal_mask = ~novel_mask
    Xn, yn, gn = X[normal_mask], y[normal_mask], groups[normal_mask]
    Xa = X[novel_mask]
    normal_codes = np.unique(yn)
    if normal_codes.size < 1:
        raise GroupOODError("no normal classes remain after withholding novel.")

    _, class_sizes = np.unique(yn, return_counts=True)
    k = int(np.clip(n_splits, 2, int(class_sizes.min())))
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)

    oof_scores = np.full(Xn.shape[0], np.nan, dtype=np.float64)
    for train_idx, test_idx in splitter.split(Xn, yn, gn):
        medians = impute_fit(Xn[train_idx])
        X_tr = impute_apply(Xn[train_idx], medians)
        X_te = impute_apply(Xn[test_idx], medians)
        det = _MahalanobisDetector().fit(X_tr, yn[train_idx])
        oof_scores[test_idx] = det.score(X_te)
    if not np.all(np.isfinite(oof_scores)):  # pragma: no cover - fold coverage
        raise GroupOODError("cross-conformal left unscored normal objects.")

    medians_full = impute_fit(Xn)
    det_full = _MahalanobisDetector().fit(impute_apply(Xn, medians_full), yn)
    novel_scores = det_full.score(impute_apply(Xa, medians_full))

    auroc = _auroc(oof_scores, novel_scores)
    threshold = _conformal_threshold(oof_scores, alpha)
    novel_tpr = float(np.mean(novel_scores > threshold))

    normal_flags = np.empty(oof_scores.shape[0], dtype=bool)
    for i, s in enumerate(oof_scores):
        others = np.delete(oof_scores, i)
        normal_flags[i] = s > _conformal_threshold(others, alpha)
    normal_fpr = float(np.mean(normal_flags))

    n = oof_scores.size
    normal_p = np.asarray(
        [(1.0 + int(np.sum(np.delete(oof_scores, i) >= s))) / n
         for i, s in enumerate(oof_scores)]
    )
    coverage = float(np.mean(normal_p > alpha)) if normal_p.size else 1.0

    neg = oof_scores
    pos = novel_scores

    # Bootstrap over both score pools for the AUROC CI.
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        bn = neg[rng.integers(0, neg.size, size=neg.size)]
        bp = pos[rng.integers(0, pos.size, size=pos.size)]
        boot[b] = _auroc(bn, bp)
    boot = boot[np.isfinite(boot)]
    auroc_ci = (
        (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        if boot.size else (float("nan"), float("nan"))
    )

    report = PopulationOODReport(
        novel_class=novel_class,
        normal_classes=[CODE_TO_CLASS.get(int(c), str(int(c))) for c in normal_codes],
        auroc=float(auroc),
        auroc_ci=auroc_ci,
        novel_tpr=novel_tpr,
        normal_fpr=normal_fpr,
        conformal_coverage=coverage,
        threshold=float(threshold),
        alpha=float(alpha),
        n_normal=int(Xn.shape[0]),
        n_novel=int(Xa.shape[0]),
        n_splits=int(k),
        n_features=int(X.shape[1]),
        passed=bool(auroc >= auroc_min and novel_tpr >= tpr_min
                    and normal_fpr <= alpha + 0.10),
        normal_scores=[float(s) for s in oof_scores],
        novel_scores=[float(s) for s in novel_scores],
    )
    log.info("[group_ood] OOD novel=%s AUROC=%.3f TPR=%.3f FPR=%.3f cov=%.3f "
             "(normal=%d novel=%d k=%d)", novel_class, report.auroc,
             report.novel_tpr, report.normal_fpr, report.conformal_coverage,
             report.n_normal, report.n_novel, report.n_splits)
    return report
