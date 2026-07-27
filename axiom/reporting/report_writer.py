"""Render the collected :class:`RunData` into markdown + JSON report files.

Layout written under ``benchmarks/reports``::

    README.md          executive summary + chart gallery + verdict
    summary.json       master machine-readable record (+ verdict)
    methodology.md     fixed methodology / caveats note
    suite_<n>_<slug>.md / .json   per-suite detailed reports
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .charts import ChartSpec

log = logging.getLogger(__name__)
from .collect import RunData

# suite key -> (file slug, human title)
SUITE_META = {
    "suite_1": ("suite_1_in_distribution", "In-Distribution Cross-Validation (HTRU2)"),
    "suite_2": ("suite_2_ablation", "Ablation Study (HTRU2)"),
    "suite_3": ("suite_3_ood_detection", "Out-of-Distribution Anomaly Detection"),
    "suite_4": ("suite_4_baselines", "Baseline Comparison"),
    "suite_5": ("suite_5_significance", "Statistical Significance"),
    "suite_6": ("suite_6_manifold_lane1", "Real-Waterfall Manifold OOD (Lane 1)"),
    "suite_7": ("suite_7_population_lane2", "Population-Scale Catalog Manifold (Lane 2)"),
}

_CAVEAT = (
    "> **Scientific caveat.** An out-of-distribution / anomaly verdict flags a "
    "signal that is statistically inconsistent with the learned natural manifold. "
    "It is **not**, by itself, proof of artificial origin. FRB separability "
    "reflects genuine extragalactic dispersion, not a technosignature. See "
    "`README.md` §07."
)


def _b(v: Optional[bool]) -> str:
    if v is None:
        return "SKIPPED"
    return "PASS" if v else "FAIL"


def _sig(v: Optional[bool]) -> str:
    if v is None:
        return "SKIPPED"
    return "SIGNIFICANT" if v else "NOT SIGNIFICANT (tied with strongest baseline)"


def _fmt(v: Any, nd: int = 4) -> str:
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _charts_for(specs: List[ChartSpec], suite: str) -> List[ChartSpec]:
    return [s for s in specs if s.suite == suite]


def _embed(spec: ChartSpec, rel_prefix: str) -> str:
    return (f"![{spec.title}]({rel_prefix}{spec.filename})\n\n"
            f"*{spec.caption}*\n")


# ---------------------------------------------------------------------------
# Per-suite detailed markdown.
# ---------------------------------------------------------------------------

def _detail_common_header(title: str, s: Dict[str, Any]) -> List[str]:
    lines = [f"# {title}", ""]
    if s.get("skipped"):
        lines += [f"> **Status: SKIPPED** — {s.get('reason', 'unavailable')}", ""]
    else:
        lines += [f"- **Runtime:** {s.get('elapsed_s', 'n/a')} s", ""]
    return lines


def _detail_suite_1(s, charts, rel) -> str:
    L = _detail_common_header(SUITE_META["suite_1"][1], s)
    if s.get("skipped"):
        return "\n".join(L)
    agg = s["aggregate"]
    L += [f"Dataset **{s['dataset']}** — {s['n_samples']} samples "
          f"({s['n_pulsars']} pulsars / {s['n_rfi']} RFI), "
          f"{s['n_features']} features, {s['n_folds']}-fold stratified CV.", "",
          "## Aggregate cross-validation metrics", "",
          "| Metric | Mean | Std |", "|---|---|---|"]
    for k in ["accuracy", "precision", "recall", "f1", "mcc", "auc"]:
        L.append(f"| {k.upper()} | {agg[k]['mean']:.4f} | {agg[k]['std']:.4f} |")
    L += ["", "## Per-fold detail", "",
          "| Fold | " + " | ".join(m.upper() for m in
          ["accuracy", "mcc", "auc", "f1"]) + " |",
          "|" + "---|" * 5]
    pf = s["per_fold"]
    for i in range(s["n_folds"]):
        L.append(f"| {i+1} | {pf['accuracy'][i]:.4f} | {pf['mcc'][i]:.4f} | "
                 f"{pf['auc'][i]:.4f} | {pf['f1'][i]:.4f} |")
    h = s["holdout"]
    L += ["", "## Hold-out diagnostics (20% stratified)", "",
          f"- AUC: **{h['auc']:.4f}**, Average Precision: **{h['average_precision']:.4f}**",
          f"- Accuracy: {h['accuracy']:.4f}, MCC: {h['mcc']:.4f}", "",
          "## Feature importance (RF base learner)", "",
          "| Rank | Feature | Importance |", "|---|---|---|"]
    for i, row in enumerate(s["top_features"], 1):
        L.append(f"| {i} | `{row['feature']}` | {row['importance']:.4f} |")
    g = s["pass_gate"]
    L += ["", "## Pass gate", "",
          f"- Accuracy ≥ {g['accuracy_min']}: **{_b(g['accuracy_ok'])}**",
          f"- MCC ≥ {g['mcc_min']}: **{_b(g['mcc_ok'])}**",
          f"- **Suite verdict: {_b(s['passed'])}**", ""]
    L += ["## Figures", ""] + [_embed(c, rel) for c in charts]
    return "\n".join(L)


def _detail_suite_2(s, charts, rel) -> str:
    L = _detail_common_header(SUITE_META["suite_2"][1], s)
    if s.get("skipped"):
        return "\n".join(L)
    L += [f"Train/test split: {s['n_train']} / {s['n_test']} (stratified 80/20).", "",
          "| Configuration | Accuracy | MCC | F1 |", "|---|---|---|---|"]
    for name, r in s["configs"].items():
        L.append(f"| {name} | {r['accuracy']:.4f} | {r['mcc']:.4f} | {r['f1']:.4f} |")
    L += ["", f"Ensemble accuracy uplift over RF alone: "
          f"**{s['ensemble_uplift_pp_over_rf']:+.2f}** percentage points.", ""]
    L += ["## Figures", ""] + [_embed(c, rel) for c in charts]
    return "\n".join(L)


def _detail_suite_3(s, charts, rel) -> str:
    L = _detail_common_header(SUITE_META["suite_3"][1], s)
    if s.get("skipped"):
        return "\n".join(L)
    L += [f"OOD source: **{s['source']}** | evaluation set: {s['n_ood_samples']} samples.", "",
          "| Role | Count |", "|---|---|"]
    for role, n in s["role_counts"].items():
        L.append(f"| {role} | {n} |")
    g = s["pass_gate"]
    L += ["", "## Detection rates", "",
          f"- Genuine-anomaly TPR: **{s['anomaly_tpr']*100:.1f}%** "
          f"(target ≥ {g['tpr_min']*100:.0f}%)",
          f"- Natural/interference FPR: **{s['natural_fpr']*100:.1f}%** "
          f"(target ≤ {g['fpr_max']*100:.0f}%)",
          f"- **Suite verdict: {_b(s['passed'])}**", ""]
    if s.get("voyager_ground_truth"):
        L += ["## Ground-truth artificial control (Voyager 1)", "",
              "| Signal | Verdict |", "|---|---|"]
        for row in s["voyager_ground_truth"]:
            L.append(f"| {row['name']} | {row['verdict']} |")
        L.append("")
    L += ["## Genuine-anomaly detail", "", "| Signal | Verdict | Detected |",
          "|---|---|---|"]
    for row in s["anomaly_detail"]:
        L.append(f"| {row['name']} | {row['verdict']} | "
                 f"{'yes' if row['detected'] else 'no'} |")
    L += ["", _CAVEAT, "", "## Figures", ""] + [_embed(c, rel) for c in charts]
    return "\n".join(L)


def _detail_suite_4(s, charts, rel) -> str:
    L = _detail_common_header(SUITE_META["suite_4"][1], s)
    if s.get("skipped"):
        return "\n".join(L)
    L += ["5-fold stratified CV. The nested-CV HGBT tunes hyper-parameters with an "
          "inner CV inside every outer fold (no test-fold leakage).", "",
          "| Model | Accuracy | MCC | F1 |", "|---|---|---|---|"]
    ordered = sorted(s["models"], key=lambda k: s["models"][k]["mcc"], reverse=True)
    for name in ordered:
        r = s["models"][name]
        star = " **(AXIOM)**" if name == "AXIOM Ensemble" else ""
        L.append(f"| {name}{star} | {r['accuracy']:.4f} | {r['mcc']:.4f} | {r['f1']:.4f} |")
    L += ["", f"Best by MCC: **{s['best_by_mcc']}** "
          f"(AXIOM is best: {s['axiom_is_best']}).", ""]
    L += ["## Figures", ""] + [_embed(c, rel) for c in charts]
    return "\n".join(L)


def _detail_suite_5(s, charts, rel) -> str:
    L = _detail_common_header(SUITE_META["suite_5"][1], s)
    if s.get("skipped"):
        return "\n".join(L)
    L += [f"McNemar test: **{s['model_a']}** vs **{s['model_b']}** "
          "(the strongest baseline).", "",
          f"- Only {s['model_a']} correct: {s['only_a_correct']}",
          f"- Only {s['model_b']} correct: {s['only_b_correct']}",
          f"- McNemar χ²: **{s['mcnemar_chi2']:.4f}**, p-value: **{s['p_value']:.6f}**",
          f"- Significant (p < 0.05): **{_b(s['significant'])}**",
          f"- {s['model_a']} accuracy: {s['model_a_accuracy']:.4f}, "
          f"95% Wilson CI: [{s['wilson_ci_95'][0]:.4f}, {s['wilson_ci_95'][1]:.4f}]", ""]
    L += ["## Figures", ""] + [_embed(c, rel) for c in charts]
    return "\n".join(L)


def _detail_suite_6(s, charts, rel) -> str:
    L = _detail_common_header(SUITE_META["suite_6"][1], s)
    if s.get("skipped"):
        return "\n".join(L)
    L += ["Every class is a real, provenance-pinned dynamic spectrum through one "
          "12-D featurizer; cross-conformal evaluation.", "",
          "| Class | Waterfalls |", "|---|---|"]
    for k, v in s["class_counts"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## Metrics", "",
          f"- AUROC (artificial vs natural): **{s['auroc']:.4f}**",
          f"- Artificial (Voyager) TPR: **{s['anomaly_tpr']*100:.1f}%**",
          f"- Normal FPR: **{s['normal_fpr']*100:.1f}%** "
          f"(target ≤ {s['alpha']*100:.0f}%)",
          f"- Conformal coverage: **{s['conformal_coverage']*100:.1f}%** "
          f"(target ≥ {(1-s['alpha'])*100:.0f}%)",
          f"- Cross-conformal folds: {s['n_splits']} | fit {s['n_fit']} / "
          f"cal {s['n_cal']} / anomaly {s['n_anomaly']}",
          f"- **Suite verdict: {_b(s['passed'])}**", "",
          _CAVEAT, "", "## Figures", ""] + [_embed(c, rel) for c in charts]
    return "\n".join(L)


def _detail_suite_7(s, charts, rel) -> str:
    L = _detail_common_header(SUITE_META["suite_7"][1], s)
    if s.get("skipped"):
        return "\n".join(L)
    c = s["classification"]
    o = s["ood"]
    L += [f"**{s['n_objects']} independent real objects** through one {s['n_features']}-D "
          "commensurate physical featurizer; cross-validation keyed on each object's "
          "unique group id (leakage-free by construction).", "",
          "## Population composition", "", "| Class | Objects |", "|---|---|"]
    for k, v in s["class_counts"].items():
        L.append(f"| {k} | {v} |")
    L += ["", "## 7a — Multiclass classification (StratifiedGroupKFold, HGBT)", "",
          f"- MCC (headline): **{c['mcc']:.4f}** "
          f"95% CI [{c['mcc_ci'][0]:.4f}, {c['mcc_ci'][1]:.4f}]",
          f"- Weighted F1: **{c['weighted_f1']:.4f}** "
          f"95% CI [{c['weighted_f1_ci'][0]:.4f}, {c['weighted_f1_ci'][1]:.4f}]",
          f"- Macro F1: {c['macro_f1']:.4f} | Balanced accuracy: {c['balanced_accuracy']:.4f} "
          f"| Accuracy: {c['accuracy']:.4f}", "",
          "| Class | F1 |", "|---|---|"]
    for k, v in c["per_class_f1"].items():
        L.append(f"| {k} | {v:.3f} |")
    L += ["", f"Classification verdict: **{_b(c['passed'])}** "
          f"({c['n_splits']} folds).", "",
          "## 7b — Leave-class-out conformal OOD (novel = FRB)", "",
          f"- Normal populations: {', '.join(o['normal_classes'])}",
          f"- AUROC (FRB vs normal): **{o['auroc']:.4f}** "
          f"95% CI [{o['auroc_ci'][0]:.4f}, {o['auroc_ci'][1]:.4f}]",
          f"- Novel (FRB) TPR: **{o['novel_tpr']*100:.1f}%**",
          f"- Normal FPR: **{o['normal_fpr']*100:.1f}%** "
          f"(target ≤ {o['alpha']*100:.0f}%)",
          f"- Conformal coverage: **{o['conformal_coverage']*100:.1f}%**",
          f"- Normal / novel objects: {o['n_normal']} / {o['n_novel']}",
          f"- OOD verdict: **{_b(o['passed'])}**", "",
          _CAVEAT, "", "## Figures", ""] + [_embed(c2, rel) for c2 in charts]
    return "\n".join(L)


_DETAIL = {
    "suite_1": _detail_suite_1, "suite_2": _detail_suite_2,
    "suite_3": _detail_suite_3, "suite_4": _detail_suite_4,
    "suite_5": _detail_suite_5, "suite_6": _detail_suite_6,
    "suite_7": _detail_suite_7,
}


# ---------------------------------------------------------------------------
# Executive summary README.
# ---------------------------------------------------------------------------

def _summary_md(run: RunData, verdict: Dict[str, Any],
                specs: List[ChartSpec]) -> str:
    m = run.summary["meta"]
    L = ["# axiom-astrophysics — Benchmark Report", "",
         f"_Generated {m['generated_utc']} · seed {m['seed']} · "
         f"Python {m['python']} · NumPy {m['numpy']} · scikit-learn {m['sklearn']}_", "",
         f"_Platform: {m['platform']} · total runtime "
         f"{m.get('total_elapsed_s', 'n/a')} s_", "",
         "This report is regenerated deterministically by "
         "`python3 scripts/generate_reports.py`. All numbers are computed from "
         "real data; none are hard-coded.", "",
         "## Overall verdict", "",
         f"**{'SYSTEM VALIDATED' if verdict['overall_validated'] else 'NEEDS WORK'}**", "",
         "| Suite | Result |", "|---|---|",
         f"| 1 — In-distribution performance | {_b(verdict['suite_1_id_performance'])} |",
         f"| 3 — OOD anomaly detection | {_b(verdict['suite_3_ood_detection'])} |",
         f"| 5 — Statistical significance | {_sig(verdict['suite_5_significant'])} |",
         f"| 6 — Manifold OOD (Lane 1) | {_b(verdict['suite_6_manifold_lane1'])} |",
         f"| 7 — Population manifold (Lane 2) | {_b(verdict['suite_7_population_lane2'])} |",
         ""]

    L += ["## Headline metrics", "", "| Metric | Value |", "|---|---|"]
    s1 = run.summary.get("suite_1", {})
    if not s1.get("skipped"):
        a = s1["aggregate"]
        L += [f"| HTRU2 5-fold accuracy | {a['accuracy']['mean']:.4f} ± "
              f"{a['accuracy']['std']:.4f} |",
              f"| HTRU2 5-fold MCC | {a['mcc']['mean']:.4f} ± {a['mcc']['std']:.4f} |",
              f"| HTRU2 5-fold AUC | {a['auc']['mean']:.4f} ± {a['auc']['std']:.4f} |"]
    s7 = run.summary.get("suite_7", {})
    if not s7.get("skipped"):
        c, o = s7["classification"], s7["ood"]
        L += [f"| Population objects (Lane 2) | {s7['n_objects']} |",
              f"| Population MCC | {c['mcc']:.4f} "
              f"[{c['mcc_ci'][0]:.4f}, {c['mcc_ci'][1]:.4f}] |",
              f"| Population weighted F1 | {c['weighted_f1']:.4f} |",
              f"| FRB-OOD AUROC | {o['auroc']:.4f} "
              f"[{o['auroc_ci'][0]:.4f}, {o['auroc_ci'][1]:.4f}] |"]
    s6 = run.summary.get("suite_6", {})
    if not s6.get("skipped"):
        L.append(f"| Lane 1 manifold AUROC | {s6['auroc']:.4f} |")
    L.append("")

    L += ["## Detailed reports", ""]
    for key, (slug, title) in SUITE_META.items():
        st = run.summary.get(key, {})
        status = "SKIPPED" if st.get("skipped") else "OK"
        L.append(f"- [{title}](./{slug}.md) — {status}")
    L += ["", "See also [methodology.md](./methodology.md) and the machine-readable "
          "[summary.json](./summary.json).", ""]

    L += ["## Chart gallery", ""]
    for spec in specs:
        L.append(_embed(spec, "../charts/"))
    L += [_CAVEAT, ""]
    return "\n".join(L)


_METHODOLOGY = """# Methodology & Caveats

## Reproducibility
- Fixed seed **42**; deterministic estimators throughout.
- Regenerate with `python3 scripts/generate_reports.py`.
- Environment (versions, platform) is recorded in `summary.json` under `meta`.

## Suites
1. **In-distribution (HTRU2).** Stratified 5-fold CV of the AXIOM stacking
   ensemble. A separate 20% hold-out drives the confusion matrix, ROC, PR,
   reliability and probability-separation figures, and an RF learning curve.
2. **Ablation.** Contribution of each ensemble component and feature block on a
   stratified 80/20 split.
 3. **OOD anomaly detection.** Real-augmented audit set with ground-truth roles
    (prefers genuine Breakthrough Listen observations — the real Voyager 1
    carrier/sidebands **and real GUPPI `.gpuspec` spectrograms of nearby stars
    from the Kaggle `tentotheminus9/breakthrough-listen-search-for-advanced-life`
    release** — over synthetic narrowband-tone controls,
    which remain only as a last-resort fallback). Reports honest genuine-anomaly
    TPR and natural/interference FPR.
4. **Baseline comparison.** 5-fold CV against standard classifiers, including an
   HGBT tuned by proper nested cross-validation (no test-fold leakage).
5. **Statistical significance.** McNemar's test (continuity-corrected) of AXIOM
   against the strongest baseline, plus a Wilson 95% accuracy interval.
6. **Lane 1 — real-waterfall manifold OOD.** Real, provenance-pinned dynamic
   spectra through one 12-D featurizer; cross-conformal AUROC and calibrated FPR.
7. **Lane 2 — population-scale catalog manifold.** ~19k independent real objects
   (ATNF, CHIME/FRB, HTRU2) through one commensurate physical featurizer.
   Multiclass typing uses StratifiedGroupKFold keyed on each object's unique
   group id (leakage-free); a leave-class-out conformal test withholds
   extragalactic FRBs entirely.

## Scientific caveat
An OOD/anomaly verdict flags a signal statistically inconsistent with the learned
natural manifold. It is **not** proof of artificial origin. FRB separability
reflects genuine extragalactic dispersion (DM far above the Galactic model), which
validates the physical feature space — nothing more.
"""


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def write_reports(run: RunData, verdict: Dict[str, Any],
                  specs: List[ChartSpec], reports_dir: str,
                  verbose: bool = True) -> Dict[str, Any]:
    os.makedirs(reports_dir, exist_ok=True)

    # Master JSON (summary + verdict + chart manifest).
    master = dict(run.summary)
    master["verdict"] = verdict
    master["charts"] = [
        {"filename": s.filename, "title": s.title, "caption": s.caption,
         "suite": s.suite} for s in specs
    ]
    with open(os.path.join(reports_dir, "summary.json"), "w") as f:
        json.dump(master, f, indent=2)

    # Per-suite JSON + markdown.
    for key, (slug, _title) in SUITE_META.items():
        section = run.summary.get(key, {})
        with open(os.path.join(reports_dir, f"{slug}.json"), "w") as f:
            json.dump(section, f, indent=2)
        detail_fn = _DETAIL[key]
        md = detail_fn(section, _charts_for(specs, key), "../charts/")
        with open(os.path.join(reports_dir, f"{slug}.md"), "w") as f:
            f.write(md + "\n")

    # Executive summary + methodology.
    with open(os.path.join(reports_dir, "README.md"), "w") as f:
        f.write(_summary_md(run, verdict, specs) + "\n")
    with open(os.path.join(reports_dir, "methodology.md"), "w") as f:
        f.write(_METHODOLOGY)

    if verbose:
        log.info("Wrote README.md")
    return master
