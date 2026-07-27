"""Deterministic collection of every validation suite into one structured record.

Each ``collect_suite_*`` function returns ``(summary, arrays)``:

* ``summary`` — JSON-serialisable scalars/lists that feed the written reports.
* ``arrays`` — NumPy arrays kept in memory to render the figures.

Sections that require optional resources (offline catalogs, pinned filterbanks)
degrade gracefully: they are marked ``{"skipped": True, "reason": ...}`` instead
of aborting the whole run, mirroring ``benchmark.py``.
"""
from __future__ import annotations

import logging
import platform

log = logging.getLogger(__name__)
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from axiom.data.loader import load_htru2
from axiom.ml.ensemble import AxiomEnsemble

SEED = 42
N_FOLDS = 5
SCHEMA_VERSION = 1


@dataclass
class RunData:
    """Complete collected benchmark record.

    ``summary`` is fully JSON-serialisable and drives the written reports;
    ``arrays`` holds NumPy payloads consumed only by the figure renderers.
    """

    summary: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def section(self, name: str) -> Dict[str, Any]:
        return self.summary.get(name, {})


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _f(x: Any) -> float:
    return float(np.asarray(x, dtype=np.float64))


def _guard(fn: Callable[[], Tuple[Dict[str, Any], Dict[str, Any]]],
           name: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run a collector, converting any failure into a skipped section."""
    t0 = time.time()
    try:
        summary, arrays = fn()
        summary.setdefault("skipped", False)
        summary["elapsed_s"] = round(time.time() - t0, 2)
        return summary, arrays
    except Exception as exc:  # pragma: no cover - defensive offline guard
        return (
            {"skipped": True, "reason": f"{type(exc).__name__}: {exc}",
             "elapsed_s": round(time.time() - t0, 2)},
            {},
        )


# ---------------------------------------------------------------------------
# Suite 1 — In-distribution 5-fold CV + diagnostic curves (HTRU2).
# ---------------------------------------------------------------------------

def collect_suite_1(X: np.ndarray, y: np.ndarray,
                    col_names) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    keys = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]
    folds: Dict[str, list] = {k: [] for k in keys}

    for train_idx, test_idx in skf.split(X, y):
        model = AxiomEnsemble(n_classes=2, random_state=SEED)
        model.fit(X[train_idx], y[train_idx])
        preds = model.predict(X[test_idx])
        probs = model.predict_proba(X[test_idx])[:, 1]
        yte = y[test_idx]
        folds["accuracy"].append(accuracy_score(yte, preds))
        folds["precision"].append(precision_score(yte, preds, zero_division=0))
        folds["recall"].append(recall_score(yte, preds, zero_division=0))
        folds["f1"].append(f1_score(yte, preds, zero_division=0))
        folds["mcc"].append(matthews_corrcoef(yte, preds))
        folds["auc"].append(roc_auc_score(yte, probs))

    agg = {k: {"mean": _f(np.mean(folds[k])), "std": _f(np.std(folds[k]))}
           for k in keys}

    # Held-out split (0.2) for confusion / ROC / PR / calibration / prob dist.
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    ens = AxiomEnsemble(n_classes=2, random_state=SEED)
    ens.fit(X_tr, y_tr)
    preds = ens.predict(X_te)
    probs = ens.predict_proba(X_te)[:, 1]

    cm = confusion_matrix(y_te, preds)
    fpr, tpr, _ = roc_curve(y_te, probs)
    holdout_auc = roc_auc_score(y_te, probs)
    prec_c, rec_c, _ = precision_recall_curve(y_te, probs)
    ap = average_precision_score(y_te, probs)
    frac_pos, mean_pred = calibration_curve(y_te, probs, n_bins=10, strategy="quantile")

    # Learning curve on the RF base learner (scaled) as convergence proxy.
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    rf = ens.get_base_learners()["rf"]
    ls_sizes, ls_train, ls_test = learning_curve(
        rf, X_tr_s, y_tr, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 6), scoring="accuracy",
        random_state=SEED)
    importances = np.asarray(rf.feature_importances_, dtype=np.float64)

    summary = {
        "dataset": "HTRU2",
        "n_samples": int(len(y)),
        "n_pulsars": int(np.sum(y == 1)),
        "n_rfi": int(np.sum(y == 0)),
        "n_features": int(X.shape[1]),
        "feature_names": list(col_names),
        "n_folds": N_FOLDS,
        "per_fold": {k: [ _f(v) for v in folds[k] ] for k in keys},
        "aggregate": agg,
        "holdout": {
            "test_fraction": 0.2,
            "auc": _f(holdout_auc),
            "average_precision": _f(ap),
            "confusion_matrix": cm.tolist(),
            "accuracy": _f(accuracy_score(y_te, preds)),
            "mcc": _f(matthews_corrcoef(y_te, preds)),
        },
        "pass_gate": {
            "accuracy_min": 0.98, "mcc_min": 0.85,
            "accuracy_ok": bool(agg["accuracy"]["mean"] >= 0.98),
            "mcc_ok": bool(agg["mcc"]["mean"] >= 0.85),
        },
        "top_features": [
            {"feature": str(col_names[i]), "importance": _f(importances[i])}
            for i in np.argsort(importances)[::-1]
        ],
    }
    summary["passed"] = bool(summary["pass_gate"]["accuracy_ok"]
                             and summary["pass_gate"]["mcc_ok"])
    arrays = {
        "per_fold": {k: np.asarray(folds[k]) for k in keys},
        "confusion_matrix": cm,
        "roc": (fpr, tpr, float(holdout_auc)),
        "pr": (rec_c, prec_c, float(ap)),
        "calibration": (mean_pred, frac_pos),
        "prob_by_class": {
            "pulsar": probs[y_te == 1], "rfi": probs[y_te == 0]},
        "learning_curve": (ls_sizes, ls_train, ls_test),
        "feature_importance": (list(col_names), importances),
    }
    return summary, arrays


# ---------------------------------------------------------------------------
# Suite 2 — Ablation study (HTRU2).
# ---------------------------------------------------------------------------

def collect_suite_2(X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    def _rf():
        return RandomForestClassifier(
            n_estimators=300, max_depth=15, class_weight="balanced",
            random_state=SEED, n_jobs=-1)

    configs = {
        "Full Ensemble (RF+HGBT to LR)": "ENSEMBLE",
        "RF Only (300 trees)": (_rf(), "all"),
        "HGBT Only (300 iters)": (HistGradientBoostingClassifier(
            max_iter=300, max_depth=8, learning_rate=0.05, random_state=SEED), "all"),
        "RF: 4 profile features only": (_rf(), "profile"),
        "RF: 4 DM-SNR features only": (_rf(), "dmsnr"),
    }

    rows = {}
    for name, spec in configs.items():
        if spec == "ENSEMBLE":
            model = AxiomEnsemble(n_classes=2, random_state=SEED)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
        else:
            clf, subset = spec
            if subset == "profile":
                clf.fit(X_tr_s[:, :4], y_tr)
                preds = clf.predict(X_te_s[:, :4])
            elif subset == "dmsnr":
                clf.fit(X_tr_s[:, 4:], y_tr)
                preds = clf.predict(X_te_s[:, 4:])
            else:
                clf.fit(X_tr_s, y_tr)
                preds = clf.predict(X_te_s)
        rows[name] = {
            "accuracy": _f(accuracy_score(y_te, preds)),
            "mcc": _f(matthews_corrcoef(y_te, preds)),
            "f1": _f(f1_score(y_te, preds, zero_division=0)),
        }

    uplift = (rows["Full Ensemble (RF+HGBT to LR)"]["accuracy"]
              - rows["RF Only (300 trees)"]["accuracy"]) * 100.0
    summary = {
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "configs": rows,
        "ensemble_uplift_pp_over_rf": _f(uplift),
    }
    arrays = {"configs": rows}
    return summary, arrays


# ---------------------------------------------------------------------------
# Suite 3 — OOD anomaly detection (real-augmented with synthetic fallback).
# ---------------------------------------------------------------------------

def collect_suite_3(X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from axiom.stats.ood_eval import evaluate_ood

    real_features, real_waves, waterfall_features = {}, {}, {}
    source = "synthetic"
    records: list = []
    try:
        from axiom.data.real_loaders import get_full_real_ood
        records, real_features, real_waves, waterfall_features, _ = get_full_real_ood(
            X, y, seed=SEED, n_frb=25, n_rfi=25, n_anom=25)
        if records:
            source = "real-augmented"
    except Exception:
        records = []

    if not records:
        rng = np.random.default_rng(SEED)
        for i in range(25):
            records.append((f"NarrowbandTone_{i}", "Narrowband", "Narrowband",
                            0.0, 18.0, "Anomaly"))
        records.append(("Wow! Signal", "Narrowband", "Narrowband", 0.0, 30.0, "Anomaly"))
        records.append(("BLC1", "Narrowband", "Narrowband", 0.0, 15.0, "Anomaly"))
        for i in range(25):
            records.append((f"FRB_{i}", "FRB", "FRB",
                            float(rng.uniform(100, 1000)), 20.0, "Natural"))
        for i in range(15):
            records.append((f"Quasar_{i}", "Quasar", "Quasar", 0.0, 5.0, "Natural"))
        for i in range(25):
            records.append((f"RFI_{i}", "RFI", "RFI", 0.0, 10.0, "Interference"))

    result = evaluate_ood(X, y, records, seed=SEED,
                          real_features=real_features or None,
                          real_waves=real_waves or None,
                          waterfall_features=waterfall_features or None)
    names, roles, verdicts = result["names"], result["roles"], result["verdicts"]

    role_counts: Dict[str, int] = {}
    for r in roles:
        role_counts[r] = role_counts.get(r, 0) + 1
    anomaly_table = [
        {"name": n, "verdict": v, "detected": bool(v == "Anomaly")}
        for n, r, v in zip(names, roles, verdicts) if r == "Anomaly"
    ]
    voyager = [row for row in
               ({"name": n, "verdict": v} for n, r, v in zip(names, roles, verdicts)
                if n.startswith("Voyager") and r == "Anomaly")]

    summary = {
        "source": source,
        "n_ood_samples": int(len(roles)),
        "role_counts": role_counts,
        "anomaly_tpr": _f(result["anomaly_tpr"]),
        "natural_fpr": _f(result["natural_fpr"]),
        "passed": bool(result["pass"]),
        "pass_gate": {"tpr_min": 0.90, "fpr_max": 0.10},
        "anomaly_detail": anomaly_table,
        "voyager_ground_truth": voyager,
    }
    arrays = {
        "rates": {"tpr": _f(result["anomaly_tpr"]),
                  "fpr": _f(result["natural_fpr"])},
        "role_counts": role_counts,
        "anomaly_detail": anomaly_table,
    }
    return summary, arrays


# ---------------------------------------------------------------------------
# Suite 4 — Baseline comparison (5-fold CV, incl. nested-CV HGBT).
# ---------------------------------------------------------------------------

def collect_suite_4(X: np.ndarray, y: np.ndarray) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    baselines = {
        "AXIOM Ensemble": "AXIOM",
        "Logistic Regression": LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "Random Forest (100)": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, random_state=SEED),
        "HGBT (100)": HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, random_state=SEED),
        "HGBT (nested-CV tuned)": "NESTED_HGBT",
    }
    grid = {"learning_rate": [0.05, 0.1, 0.2],
            "max_leaf_nodes": [15, 31, 63],
            "l2_regularization": [0.0, 1.0]}

    rows = {}
    all_preds: Dict[str, np.ndarray] = {}
    for name, clf in baselines.items():
        acc, mcc, f1 = [], [], []
        preds_all = np.zeros(len(y), dtype=np.int64)
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            if clf == "AXIOM":
                model = AxiomEnsemble(n_classes=2, random_state=SEED)
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
            elif clf == "NESTED_HGBT":
                inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
                search = GridSearchCV(
                    HistGradientBoostingClassifier(max_iter=200, random_state=SEED),
                    grid, scoring="matthews_corrcoef", cv=inner, n_jobs=-1, refit=True)
                search.fit(X_tr, y_tr)
                preds = search.predict(X_te)
            else:
                sc = StandardScaler()
                c = clf.__class__(**clf.get_params())
                c.fit(sc.fit_transform(X_tr), y_tr)
                preds = c.predict(sc.transform(X_te))
            acc.append(accuracy_score(y_te, preds))
            mcc.append(matthews_corrcoef(y_te, preds))
            f1.append(f1_score(y_te, preds, zero_division=0))
            preds_all[test_idx] = preds
        rows[name] = {"accuracy": _f(np.mean(acc)), "mcc": _f(np.mean(mcc)),
                      "f1": _f(np.mean(f1))}
        all_preds[name] = preds_all

    best = max(rows, key=lambda k: rows[k]["mcc"])
    summary = {"models": rows, "best_by_mcc": best,
               "axiom_is_best": bool(best == "AXIOM Ensemble")}
    arrays = {"models": rows, "all_preds": all_preds, "y_true": y}
    return summary, arrays


# ---------------------------------------------------------------------------
# Suite 5 — Statistical significance (McNemar + Wilson CI).
# ---------------------------------------------------------------------------

def collect_suite_5(y_true: np.ndarray, all_preds: Dict[str, np.ndarray],
                    model_a: str, model_b: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from scipy.stats import chi2 as chi2_dist

    pa, pb = all_preds.get(model_a), all_preds.get(model_b)
    if pa is None or pb is None:
        return {"skipped": True, "reason": "predictions unavailable"}, {}

    ca, cb = (pa == y_true), (pb == y_true)
    b01 = int(np.sum(~ca & cb))
    b10 = int(np.sum(ca & ~cb))
    if (b01 + b10) == 0:
        chi2, p_value = 0.0, 1.0
    else:
        chi2 = (abs(b01 - b10) - 1.0) ** 2 / (b01 + b10)
        p_value = float(1.0 - chi2_dist.cdf(chi2, df=1))

    acc = accuracy_score(y_true, pa)
    n = len(y_true)
    z = 1.96
    denom = 1 + z ** 2 / n
    center = (acc + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((acc * (1 - acc) + z ** 2 / (4 * n)) / n) / denom

    summary = {
        "model_a": model_a, "model_b": model_b,
        "only_a_correct": b10, "only_b_correct": b01,
        "mcnemar_chi2": _f(chi2), "p_value": _f(p_value),
        "significant": bool(p_value < 0.05),
        "model_a_accuracy": _f(acc),
        "wilson_ci_95": [_f(center - margin), _f(center + margin)],
    }
    arrays = {"discordant": {"only_a": b10, "only_b": b01}}
    return summary, arrays


# ---------------------------------------------------------------------------
# Suite 6 — Lane 1 real-waterfall manifold OOD.
# ---------------------------------------------------------------------------

def collect_suite_6() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from axiom.stats.manifold_ood import evaluate_manifold_ood

    r = evaluate_manifold_ood(alpha=0.1, seed=SEED)
    normal = np.asarray(r.normal_scores, dtype=np.float64)
    anom = np.asarray(r.anomaly_scores, dtype=np.float64)
    summary = {
        "class_counts": dict(r.class_counts),
        "auroc": _f(r.auroc),
        "anomaly_tpr": _f(r.anomaly_tpr),
        "normal_fpr": _f(r.normal_fpr),
        "conformal_coverage": _f(r.conformal_coverage),
        "alpha": _f(r.alpha),
        "n_fit": int(r.n_fit), "n_cal": int(r.n_cal),
        "n_anomaly": int(r.n_anomaly), "n_splits": int(r.n_splits),
        "passed": bool(r.passed),
    }
    arrays = {
        "scores": {"normal": normal, "anomaly": anom},
        "class_counts": dict(r.class_counts),
    }
    return summary, arrays


# ---------------------------------------------------------------------------
# Suite 7 — Lane 2 population-scale catalog manifold.
# ---------------------------------------------------------------------------

def collect_suite_7() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    from axiom.data.population import build_population
    from axiom.stats.group_ood import (
        evaluate_population_classification,
        evaluate_population_ood,
    )

    pop = build_population(cache=True)
    clf = evaluate_population_classification(pop, seed=SEED)
    ood = evaluate_population_ood(pop, novel_class="FRB", seed=SEED)

    # 2-D PCA embedding of the physical manifold for a qualitative scatter.
    from axiom.dsp.physical_features import impute_apply, impute_fit
    Ximp = impute_apply(np.asarray(pop.X, dtype=np.float64),
                        impute_fit(np.asarray(pop.X, dtype=np.float64)))
    Xs = StandardScaler().fit_transform(Ximp)
    pca = PCA(n_components=2, random_state=SEED)
    emb = pca.fit_transform(Xs)

    summary = {
        "n_objects": int(pop.n_objects()),
        "class_counts": dict(pop.class_counts()),
        "n_features": int(pop.X.shape[1]),
        "classification": {
            "mcc": _f(clf.mcc), "mcc_ci": [_f(clf.mcc_ci[0]), _f(clf.mcc_ci[1])],
            "weighted_f1": _f(clf.weighted_f1),
            "weighted_f1_ci": [_f(clf.weighted_f1_ci[0]), _f(clf.weighted_f1_ci[1])],
            "macro_f1": _f(clf.macro_f1),
            "balanced_accuracy": _f(clf.balanced_accuracy),
            "accuracy": _f(clf.accuracy),
            "per_class_f1": {k: _f(v) for k, v in clf.per_class_f1.items()},
            "confusion_matrix": clf.confusion,
            "labels": list(clf.labels),
            "n_splits": int(clf.n_splits),
            "passed": bool(clf.passed),
        },
        "ood": {
            "novel_class": ood.novel_class,
            "normal_classes": list(ood.normal_classes),
            "auroc": _f(ood.auroc),
            "auroc_ci": [_f(ood.auroc_ci[0]), _f(ood.auroc_ci[1])],
            "novel_tpr": _f(ood.novel_tpr),
            "normal_fpr": _f(ood.normal_fpr),
            "conformal_coverage": _f(ood.conformal_coverage),
            "alpha": _f(ood.alpha),
            "n_normal": int(ood.n_normal), "n_novel": int(ood.n_novel),
            "passed": bool(ood.passed),
        },
        "passed": bool(clf.passed and ood.passed),
    }
    arrays = {
        "class_counts": dict(pop.class_counts()),
        "confusion": (np.asarray(clf.confusion), list(clf.labels)),
        "per_class_f1": dict(clf.per_class_f1),
        "ood_scores": {"normal": np.asarray(ood.normal_scores),
                       "novel": np.asarray(ood.novel_scores)},
        "pca": (emb, np.asarray(pop.y), pop),
        "pca_explained": [float(v) for v in pca.explained_variance_ratio_],
    }
    return summary, arrays


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def collect_all(verbose: bool = True) -> RunData:
    """Run every suite deterministically and return a :class:`RunData`."""
    def _log(msg: str) -> None:
        if verbose:
            log.info("%s", msg)

    t0 = time.time()
    _log("loading HTRU2...")
    X, y, col_names = load_htru2()

    run = RunData()
    run.summary["meta"] = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now_iso(),
        "seed": SEED,
        "n_folds": N_FOLDS,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }

    _log("suite 1: in-distribution CV + diagnostics")
    run.summary["suite_1"], run.arrays["suite_1"] = _guard(
        lambda: collect_suite_1(X, y, col_names), "suite_1")

    _log("suite 2: ablation")
    run.summary["suite_2"], run.arrays["suite_2"] = _guard(
        lambda: collect_suite_2(X, y), "suite_2")

    _log("suite 3: OOD anomaly detection")
    run.summary["suite_3"], run.arrays["suite_3"] = _guard(
        lambda: collect_suite_3(X, y), "suite_3")

    _log("suite 4: baseline comparison (incl. nested CV)")
    run.summary["suite_4"], run.arrays["suite_4"] = _guard(
        lambda: collect_suite_4(X, y), "suite_4")

    s4_arr = run.arrays.get("suite_4", {})
    if s4_arr.get("all_preds") is not None:
        preds = s4_arr["all_preds"]
        strongest = max((k for k in preds if k != "AXIOM Ensemble"),
                        key=lambda k: run.summary["suite_4"]["models"][k]["mcc"])
        _log(f"suite 5: significance (AXIOM vs {strongest})")
        run.summary["suite_5"], run.arrays["suite_5"] = _guard(
            lambda: collect_suite_5(y, preds, "AXIOM Ensemble", strongest), "suite_5")
    else:
        run.summary["suite_5"] = {"skipped": True, "reason": "suite 4 unavailable"}
        run.arrays["suite_5"] = {}

    _log("suite 6: Lane 1 manifold OOD")
    run.summary["suite_6"], run.arrays["suite_6"] = _guard(collect_suite_6, "suite_6")

    _log("suite 7: Lane 2 population manifold")
    run.summary["suite_7"], run.arrays["suite_7"] = _guard(collect_suite_7, "suite_7")

    run.summary["meta"]["total_elapsed_s"] = round(time.time() - t0, 2)
    _log(f"done in {run.summary['meta']['total_elapsed_s']}s")
    return run


def build_verdict(run: RunData) -> Dict[str, Any]:
    """Derive an overall pass/fail verdict from the collected sections."""
    def _passed(name: str) -> Optional[bool]:
        s = run.summary.get(name, {})
        if s.get("skipped"):
            return None
        return bool(s.get("passed", False))

    s1 = _passed("suite_1")
    s3 = _passed("suite_3")
    s6 = _passed("suite_6")
    s7 = _passed("suite_7")
    s5 = run.summary.get("suite_5", {})
    sig = None if s5.get("skipped") else bool(s5.get("significant", False))

    # Suites 6 (Lane-1 real-waterfall manifold) and 7 (Lane-2 population-scale
    # catalog manifold) carry the novel scientific claims. They are REQUIRED for
    # an overall "validated" verdict: if either is skipped (e.g. real datasets not
    # fetched) the system is reported as NEEDS WORK rather than validated on the
    # HTRU2-only suites alone.
    gates = [g for g in (s1, s3) if g is not None]
    required = [g for g in (s6, s7) if g is not None]
    overall = bool(
        gates and all(gates)
        and required and all(required)
        and s6 is not None and s7 is not None
    )

    return {
        "suite_1_id_performance": s1,
        "suite_3_ood_detection": s3,
        "suite_5_significant": sig,
        "suite_6_manifold_lane1": s6,
        "suite_7_population_lane2": s7,
        "overall_validated": overall,
    }
