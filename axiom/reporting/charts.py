"""Publication-quality figure rendering for the benchmark report.

Every ``chart_*`` function consumes the in-memory :class:`RunData` arrays and
writes one 300-dpi PNG, returning a :class:`ChartSpec` (or ``None`` when its
source suite was skipped). :func:`render_all` drives the whole gallery and never
lets a single failed figure abort the run.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

log = logging.getLogger(__name__)
from typing import Callable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .collect import RunData  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.5,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
})

_ACCENT = "#1f6feb"
_ACCENT2 = "#d9480f"
_GREEN = "#2f9e44"
_GREY = "#868e96"


@dataclass
class ChartSpec:
    """Metadata for one rendered figure."""

    filename: str
    title: str
    caption: str
    suite: str


def _save(fig, out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    fig.savefig(path)
    plt.close(fig)
    return name


def _has(run: RunData, suite: str, key: Optional[str] = None) -> bool:
    if run.summary.get(suite, {}).get("skipped"):
        return False
    arr = run.arrays.get(suite, {})
    return bool(arr) and (key is None or key in arr)


# ---------------------------------------------------------------------------
# Suite 1 diagnostics.
# ---------------------------------------------------------------------------

def chart_cv_per_fold(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "per_fold"):
        return None
    folds = run.arrays["suite_1"]["per_fold"]
    metrics = ["accuracy", "precision", "recall", "f1", "mcc", "auc"]
    n = len(folds["accuracy"])
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in metrics:
        ax.plot(x, folds[m], marker="o", label=m.upper())
    ax.set_xticks(x)
    ax.set_xlabel("Fold")
    ax.set_ylabel("Score")
    ax.set_ylim(min(0.8, float(np.min(folds["mcc"])) - 0.02), 1.005)
    ax.set_title("Suite 1 — Per-Fold Cross-Validation Metrics (HTRU2)")
    ax.legend(ncol=3, fontsize=8, loc="lower right")
    return ChartSpec(_save(fig, d, "01_cv_per_fold.png"),
                     "Per-fold cross-validation metrics",
                     "Stratified 5-fold CV metrics of the AXIOM ensemble on HTRU2.",
                     "suite_1")


def chart_confusion_htru2(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "confusion_matrix"):
        return None
    cm = run.arrays["suite_1"]["confusion_matrix"]
    return _confusion_fig(cm, ["RFI/Noise", "Pulsar"], d,
                          "02_confusion_htru2.png",
                          "Suite 1 — Confusion Matrix (HTRU2 hold-out)",
                          "Confusion matrix on a stratified 20% hold-out.",
                          "suite_1")


def chart_roc_htru2(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "roc"):
        return None
    fpr, tpr, auc_v = run.arrays["suite_1"]["roc"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color=_ACCENT, lw=2.5, label=f"AXIOM (AUC = {auc_v:.4f})")
    ax.plot([0, 1], [0, 1], color=_GREY, lw=1.2, ls="--", label="Chance")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Suite 1 — ROC Curve (HTRU2)")
    ax.legend(loc="lower right")
    return ChartSpec(_save(fig, d, "03_roc_htru2.png"),
                     "ROC curve (HTRU2)",
                     "Receiver operating characteristic of the ensemble on HTRU2.",
                     "suite_1")


def chart_pr_htru2(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "pr"):
        return None
    rec, prec, ap = run.arrays["suite_1"]["pr"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(rec, prec, color=_ACCENT2, lw=2.5, label=f"AXIOM (AP = {ap:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Suite 1 — Precision-Recall Curve (HTRU2)")
    ax.legend(loc="lower left")
    return ChartSpec(_save(fig, d, "04_pr_htru2.png"),
                     "Precision-recall curve (HTRU2)",
                     "PR curve; robust to the 10:1 HTRU2 class imbalance.",
                     "suite_1")


def chart_calibration(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "calibration"):
        return None
    mean_pred, frac_pos = run.arrays["suite_1"]["calibration"]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=1.2, label="Perfect")
    ax.plot(mean_pred, frac_pos, marker="o", color=_ACCENT,
            lw=2, label="AXIOM ensemble")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraction of pulsars")
    ax.set_title("Suite 1 — Reliability Diagram (HTRU2)")
    ax.legend(loc="upper left")
    return ChartSpec(_save(fig, d, "05_calibration_htru2.png"),
                     "Reliability diagram",
                     "Calibration of predicted probabilities (quantile bins).",
                     "suite_1")


def chart_prob_dist(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "prob_by_class"):
        return None
    p = run.arrays["suite_1"]["prob_by_class"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0, 1, 41)
    ax.hist(p["rfi"], bins=bins, alpha=0.6, color=_GREY, label="RFI/Noise", density=True)
    ax.hist(p["pulsar"], bins=bins, alpha=0.6, color=_ACCENT, label="Pulsar", density=True)
    ax.set_yscale("log")
    ax.set_xlabel("Predicted P(pulsar)")
    ax.set_ylabel("Density (log)")
    ax.set_title("Suite 1 — Predicted Probability Separation (HTRU2)")
    ax.legend()
    return ChartSpec(_save(fig, d, "06_prob_distribution.png"),
                     "Probability separation by class",
                     "Class-conditional predicted-probability histograms.",
                     "suite_1")


def chart_learning_curve(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "learning_curve"):
        return None
    sizes, tr, te = run.arrays["suite_1"]["learning_curve"]
    tm, ts = tr.mean(1), tr.std(1)
    vm, vs = te.mean(1), te.std(1)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(sizes, tm, "o-", color=_ACCENT2, label="Training")
    ax.plot(sizes, vm, "o-", color=_GREEN, label="Cross-validation")
    ax.fill_between(sizes, tm - ts, tm + ts, color=_ACCENT2, alpha=0.12)
    ax.fill_between(sizes, vm - vs, vm + vs, color=_GREEN, alpha=0.12)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Accuracy")
    ax.set_title("Suite 1 — Learning Curve (RF base learner)")
    ax.legend(loc="lower right")
    return ChartSpec(_save(fig, d, "07_learning_curve.png"),
                     "Learning curve",
                     "Convergence / overfitting diagnostic on the RF base learner.",
                     "suite_1")


def chart_feature_importance(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_1", "feature_importance"):
        return None
    names, imp = run.arrays["suite_1"]["feature_importance"]
    order = np.argsort(imp)[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(len(imp)), np.asarray(imp)[order], color="teal", width=0.65)
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([names[i] for i in order], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Relative importance")
    ax.set_title("Suite 1 — Feature Importance (RF base learner)")
    return ChartSpec(_save(fig, d, "08_feature_importance.png"),
                     "Feature importance (HTRU2)",
                     "Gini importance of the 8 HTRU2 survey features.",
                     "suite_1")


# ---------------------------------------------------------------------------
# Suite 2 ablation.
# ---------------------------------------------------------------------------

def chart_ablation(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_2", "configs"):
        return None
    rows = run.arrays["suite_2"]["configs"]
    names = list(rows)
    acc = [rows[n]["accuracy"] for n in names]
    mcc = [rows[n]["mcc"] for n in names]
    f1 = [rows[n]["f1"] for n in names]
    y = np.arange(len(names))
    h = 0.26
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + h, acc, height=h, color=_ACCENT, label="Accuracy")
    ax.barh(y, mcc, height=h, color=_ACCENT2, label="MCC")
    ax.barh(y - h, f1, height=h, color=_GREEN, label="F1")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Score")
    ax.set_title("Suite 2 — Ablation Study (HTRU2)")
    ax.legend(loc="lower right")
    return ChartSpec(_save(fig, d, "09_ablation.png"),
                     "Ablation study",
                     "Contribution of each ensemble component and feature block.",
                     "suite_2")


# ---------------------------------------------------------------------------
# Suite 4 baselines.
# ---------------------------------------------------------------------------

def chart_baselines(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_4", "models"):
        return None
    rows = run.arrays["suite_4"]["models"]
    names = sorted(rows, key=lambda k: rows[k]["mcc"])
    mcc = [rows[n]["mcc"] for n in names]
    colors = [_ACCENT if n == "AXIOM Ensemble" else _GREY for n in names]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(names, mcc, color=colors)
    for b, v in zip(bars, mcc):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=8)
    ax.set_xlabel("Matthews correlation coefficient (5-fold mean)")
    ax.set_xlim(min(mcc) - 0.02, 1.0)
    ax.set_title("Suite 4 — Baseline Comparison by MCC (HTRU2)")
    return ChartSpec(_save(fig, d, "10_baselines_mcc.png"),
                     "Baseline comparison (MCC)",
                     "AXIOM vs standard classifiers incl. nested-CV-tuned HGBT.",
                     "suite_4")


# ---------------------------------------------------------------------------
# Suite 3 OOD.
# ---------------------------------------------------------------------------

def chart_ood_rates(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_3", "rates"):
        return None
    rates = run.arrays["suite_3"]["rates"]
    gate = run.summary["suite_3"]["pass_gate"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    labels = ["Anomaly TPR", "Natural FPR"]
    vals = [rates["tpr"], rates["fpr"]]
    colors = [_GREEN, _ACCENT2]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    ax.axhline(gate["tpr_min"], color=_GREEN, ls="--", lw=1,
               label=f"TPR target ≥ {gate['tpr_min']:.2f}")
    ax.axhline(gate["fpr_max"], color=_ACCENT2, ls="--", lw=1,
               label=f"FPR target ≤ {gate['fpr_max']:.2f}")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Rate")
    ax.set_title("Suite 3 — OOD Anomaly Detection Rates")
    ax.legend(fontsize=8)
    return ChartSpec(_save(fig, d, "11_ood_rates.png"),
                     "OOD detection rates",
                     "Genuine-anomaly TPR vs natural/interference FPR against gates.",
                     "suite_3")


def chart_ood_roles(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_3", "role_counts"):
        return None
    counts = run.arrays["suite_3"]["role_counts"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(list(counts), list(counts.values()), color=_ACCENT, width=0.6)
    ax.set_ylabel("Count")
    ax.set_title("Suite 3 — OOD Evaluation Set Composition")
    for i, v in enumerate(counts.values()):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    return ChartSpec(_save(fig, d, "12_ood_composition.png"),
                     "OOD set composition",
                     "Ground-truth role composition of the OOD audit set.",
                     "suite_3")


# ---------------------------------------------------------------------------
# Suite 6 Lane 1.
# ---------------------------------------------------------------------------

def chart_lane1_scores(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_6", "scores"):
        return None
    s = run.arrays["suite_6"]["scores"]
    return _score_hist(s["normal"], s["anomaly"], d, "13_lane1_score_dist.png",
                       "Suite 6 — Lane 1 Manifold OOD Scores",
                       "Normal (natural)", "Artificial (Voyager)",
                       "Conformal score separation of the real-waterfall manifold.",
                       "suite_6")


def chart_lane1_roc(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_6", "scores"):
        return None
    s = run.arrays["suite_6"]["scores"]
    return _roc_from_scores(s["normal"], s["anomaly"], d, "14_lane1_roc.png",
                            "Suite 6 — Lane 1 OOD ROC (artificial vs natural)",
                            "ROC of Lane 1 conformal scores.", "suite_6")


def chart_lane1_counts(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_6", "class_counts"):
        return None
    counts = run.arrays["suite_6"]["class_counts"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.bar(list(counts), list(counts.values()), color=_ACCENT2, width=0.6)
    ax.set_ylabel("Real waterfalls")
    ax.set_title("Suite 6 — Lane 1 Manifold Class Counts")
    for i, v in enumerate(counts.values()):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=9)
    return ChartSpec(_save(fig, d, "15_lane1_counts.png"),
                     "Lane 1 class counts",
                     "Provenance-pinned real dynamic spectra per class.",
                     "suite_6")


# ---------------------------------------------------------------------------
# Suite 7 Lane 2.
# ---------------------------------------------------------------------------

def chart_population_dist(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_7", "class_counts"):
        return None
    counts = run.arrays["suite_7"]["class_counts"]
    names = list(counts)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.bar(names, [counts[n] for n in names], color=_ACCENT, width=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Independent objects (log)")
    ax.set_title("Suite 7 — Population Class Distribution (Lane 2)")
    for i, n in enumerate(names):
        ax.text(i, counts[n] * 1.05, str(counts[n]), ha="center", fontsize=9)
    return ChartSpec(_save(fig, d, "16_population_distribution.png"),
                     "Population class distribution",
                     "19k+ independent real objects from ATNF, CHIME/FRB, HTRU2.",
                     "suite_7")


def chart_population_confusion(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_7", "confusion"):
        return None
    cm, labels = run.arrays["suite_7"]["confusion"]
    return _confusion_fig(cm, labels, d, "17_population_confusion.png",
                          "Suite 7 — Population Confusion Matrix (group CV)",
                          "Row-normalised group-CV confusion of real object types.",
                          "suite_7", normalize=True)


def chart_population_f1(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_7", "per_class_f1"):
        return None
    f1 = run.arrays["suite_7"]["per_class_f1"]
    names = list(f1)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bars = ax.bar(names, [f1[n] for n in names], color=_GREEN, width=0.6)
    for b, n in zip(bars, names):
        ax.text(b.get_x() + b.get_width() / 2, f1[n] + 0.01, f"{f1[n]:.2f}",
                ha="center", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 score")
    ax.set_title("Suite 7 — Per-Class F1 (Lane 2 group CV)")
    return ChartSpec(_save(fig, d, "18_population_per_class_f1.png"),
                     "Population per-class F1",
                     "Per-class F1; rare pulsar subtypes overlap the pulsar class.",
                     "suite_7")


def chart_lane2_scores(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_7", "ood_scores"):
        return None
    s = run.arrays["suite_7"]["ood_scores"]
    return _score_hist(s["normal"], s["novel"], d, "19_lane2_score_dist.png",
                       "Suite 7 — Lane 2 OOD Scores (FRB withheld)",
                       "Normal (Galactic)", "Novel (FRB)",
                       "Mahalanobis score separation of extragalactic FRBs.",
                       "suite_7")


def chart_lane2_roc(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_7", "ood_scores"):
        return None
    s = run.arrays["suite_7"]["ood_scores"]
    return _roc_from_scores(s["normal"], s["novel"], d, "20_lane2_roc.png",
                            "Suite 7 — Lane 2 OOD ROC (FRB vs Galactic)",
                            "ROC of leave-class-out conformal FRB detection.",
                            "suite_7")


def chart_population_pca(run, d) -> Optional[ChartSpec]:
    if not _has(run, "suite_7", "pca"):
        return None
    from axiom.data.population import CODE_TO_CLASS
    emb, y, _pop = run.arrays["suite_7"]["pca"]
    var = run.arrays["suite_7"].get("pca_explained", [0.0, 0.0])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    codes = np.unique(y)
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(codes):
        m = y == c
        name = CODE_TO_CLASS.get(int(c), str(int(c)))
        ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.4,
                   color=cmap(i % 10), label=f"{name} (n={int(m.sum())})")
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
    ax.set_title("Suite 7 — Physical Manifold PCA Embedding (Lane 2)")
    ax.legend(markerscale=2, fontsize=8, loc="best")
    return ChartSpec(_save(fig, d, "21_population_pca.png"),
                     "Physical manifold PCA",
                     "2-D PCA of the 12-D physical feature space by object class.",
                     "suite_7")


# ---------------------------------------------------------------------------
# Cross-suite summary.
# ---------------------------------------------------------------------------

def chart_headline_summary(run, d) -> Optional[ChartSpec]:
    labels, vals, los, his = [], [], [], []
    s1 = run.summary.get("suite_1", {})
    if not s1.get("skipped"):
        m = s1["aggregate"]["mcc"]
        labels.append("HTRU2 CV\nMCC")
        vals.append(m["mean"]); los.append(m["std"]); his.append(m["std"])
    s7 = run.summary.get("suite_7", {})
    if not s7.get("skipped"):
        c = s7["classification"]
        labels.append("Population\nMCC")
        vals.append(c["mcc"])
        los.append(c["mcc"] - c["mcc_ci"][0]); his.append(c["mcc_ci"][1] - c["mcc"])
        o = s7["ood"]
        labels.append("FRB-OOD\nAUROC")
        vals.append(o["auroc"])
        los.append(o["auroc"] - o["auroc_ci"][0]); his.append(o["auroc_ci"][1] - o["auroc"])
    s6 = run.summary.get("suite_6", {})
    if not s6.get("skipped"):
        labels.append("Lane 1\nAUROC")
        vals.append(s6["auroc"]); los.append(0.0); his.append(0.0)
    if not labels:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    ax.bar(x, vals, yerr=[los, his], capsize=5, color=_ACCENT, width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Headline Metrics Across Suites (95% CI where available)")
    return ChartSpec(_save(fig, d, "00_headline_summary.png"),
                     "Headline metrics summary",
                     "Cross-suite headline metrics with confidence intervals.",
                     "summary")


# ---------------------------------------------------------------------------
# Shared figure primitives.
# ---------------------------------------------------------------------------

def _confusion_fig(cm, labels, out_dir, name, title, caption, suite,
                   normalize=False) -> ChartSpec:
    cm = np.asarray(cm, dtype=np.float64)
    disp = cm.copy()
    if normalize:
        rs = disp.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1.0
        disp = disp / rs
    fig, ax = plt.subplots(figsize=(1.4 * len(labels) + 3, 1.4 * len(labels) + 2.5))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() else 1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    thresh = disp.max() / 2 if disp.max() else 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{disp[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if disp[i, j] > thresh else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    return ChartSpec(_save(fig, out_dir, name), title.split("— ")[-1], caption, suite)


def _score_hist(normal, anomaly, out_dir, name, title, n_label, a_label,
                caption, suite) -> ChartSpec:
    normal = np.asarray(normal, dtype=np.float64)
    anomaly = np.asarray(anomaly, dtype=np.float64)
    allv = np.concatenate([normal, anomaly])
    lo, hi = np.percentile(allv, 0.5), np.percentile(allv, 99.5)
    bins = np.linspace(lo, hi, 40)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.hist(normal, bins=bins, alpha=0.6, color=_GREY, density=True, label=n_label)
    ax.hist(anomaly, bins=bins, alpha=0.6, color=_ACCENT2, density=True, label=a_label)
    ax.set_xlabel("Novelty score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    return ChartSpec(_save(fig, out_dir, name), title.split("— ")[-1], caption, suite)


def _roc_from_scores(normal, anomaly, out_dir, name, title, caption,
                     suite) -> ChartSpec:
    from sklearn.metrics import roc_auc_score, roc_curve
    normal = np.asarray(normal, dtype=np.float64)
    anomaly = np.asarray(anomaly, dtype=np.float64)
    y = np.concatenate([np.zeros(normal.size), np.ones(anomaly.size)])
    s = np.concatenate([normal, anomaly])
    fpr, tpr, _ = roc_curve(y, s)
    auc_v = roc_auc_score(y, s)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot(fpr, tpr, color=_ACCENT, lw=2.5, label=f"AUROC = {auc_v:.4f}")
    ax.plot([0, 1], [0, 1], color=_GREY, ls="--", lw=1.2, label="Chance")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    return ChartSpec(_save(fig, out_dir, name), title.split("— ")[-1], caption, suite)


_REGISTRY: List[Callable[[RunData, str], Optional[ChartSpec]]] = [
    chart_headline_summary,
    chart_cv_per_fold,
    chart_confusion_htru2,
    chart_roc_htru2,
    chart_pr_htru2,
    chart_calibration,
    chart_prob_dist,
    chart_learning_curve,
    chart_feature_importance,
    chart_ablation,
    chart_baselines,
    chart_ood_rates,
    chart_ood_roles,
    chart_lane1_scores,
    chart_lane1_roc,
    chart_lane1_counts,
    chart_population_dist,
    chart_population_confusion,
    chart_population_f1,
    chart_lane2_scores,
    chart_lane2_roc,
    chart_population_pca,
]


def render_all(run: RunData, out_dir: str,
               verbose: bool = True) -> List[ChartSpec]:
    """Render every registered chart; skips (not errors) on missing data."""
    os.makedirs(out_dir, exist_ok=True)
    specs: List[ChartSpec] = []
    for fn in _REGISTRY:
        try:
            spec = fn(run, out_dir)
        except Exception as exc:  # pragma: no cover - one bad chart never aborts
            if verbose:
                log.error("%s failed: %s", fn.__name__, exc)
            spec = None
        if spec is not None:
            specs.append(spec)
            if verbose:
                log.info("Wrote %s", spec.filename)
    return specs
