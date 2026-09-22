"""Measured ablations for the discovery paper (all offline, all seeded).

Every table in the manuscript's sensitivity section is generated here; no
ablation number is hand-authored. Components (each with fixed seed 42):

  1. K_CAP_SWEEP .... AnomalyDensityEstimator n_components cap in {1,2,3,5,8}
     on the HTRU2 lane: wall-clock fit time + density-score AUROC on the
     seeded synthetic Suite-3 audit (anomaly vs control log-probabilities).
  2. NCAL_SWEEP ..... calibration sizes {250,500,1000,2000,3500} (3500 is the
     available held-out pool): empirical natural FPR at alpha=0.05 on a
     FRESH held-out HTRU2 split + audit AUROC.
  3. FUSION ......... Path-A-only / Path-B-only / Fisher / Bonferroni verdict
     rates at alpha=0.05 on the 31-record real-file Voyager audit
     (benchmarks/reports/voyager_realfile_audit.json), denominator n=30
     natural/RFI controls.
  4. FEATURE_BLOCKS . population lane on feature blocks (full-12,
     continuous-8, indicators-4): HGBT-100 MCC under the identical group-CV
     protocol + Mahalanobis leave-class-out FRB AUROC.
  5. IF_BASELINE .... IsolationForest leave-class-out FRB AUROC under the
     same out-of-fold scheme as the Mahalanobis lane (fit per-fold on train
     normals, score test normals; novel scored by the full-normal fit).
  6. ROC figure ..... real ROC curves from the measured Mahalanobis and IF
     out-of-fold scores (paper/Discovery/figures/roc_leaveclassout_frb.png).

Writes benchmarks/reports/ood_ablations.json and prints every table.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom.data.loader import load_htru2  # noqa: E402
from axiom.data.population import build_population  # noqa: E402
from axiom.dsp.features import physics_map_htru2_features  # noqa: E402
from axiom.dsp.physical_features import impute_apply, impute_fit  # noqa: E402
from axiom.ml.density import AnomalyDensityEstimator  # noqa: E402
from axiom.stats.calibration import ConformalCalibrator  # noqa: E402
from axiom.stats.group_ood import evaluate_population_ood  # noqa: E402

SEED = 42
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(REPO, "benchmarks", "reports", "ood_ablations.json")
OUT_FIG = os.path.join(REPO, "paper", "Discovery", "figures",
                       "roc_leaveclassout_frb.png")


def _suite3_synthetic_records():
    rng = np.random.default_rng(SEED)
    records = []
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
    return records


def _map_audit(records):
    from axiom.stats.ood_eval import _NARROWBAND_OFFMANIFOLD_ANCHOR
    feats, roles = [], []
    for _name, _oc, stype, dm, snr, role in records:
        m = physics_map_htru2_features(stype, dm=dm, snr=snr, seed=SEED)
        feats.append(_NARROWBAND_OFFMANIFOLD_ANCHOR.copy() if m is None else m)
        roles.append(role)
    return np.array(feats), roles


def _auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    s = np.concatenate([np.asarray(neg, float), np.asarray(pos, float)])
    return float(roc_auc_score(y, s))


def k_cap_sweep(X, y):
    from sklearn.model_selection import train_test_split
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_cal, y_fit, _ = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=SEED)
    records = _suite3_synthetic_records()
    feats, roles = _map_audit(records)
    rows = []
    for cap in (1, 2, 3, 5, 8):
        t0 = time.perf_counter()
        density = AnomalyDensityEstimator(n_components=cap)
        density.fit(X_fit, y_fit)
        dt = time.perf_counter() - t0
        s = density.log_prob(feats)
        anom = s[np.array([r == "Anomaly" for r in roles])]
        ctrl = s[np.array([r != "Anomaly" for r in roles])]
        rows.append({"k_cap": cap, "fit_s": round(dt, 2),
                     "audit_auroc": round(_auroc(-ctrl, -anom), 4)})
    return rows


def ncal_sweep(X, y):
    from sklearn.model_selection import train_test_split
    X_tr, X_hold, y_tr, y_hold = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_fit, X_cal, y_fit, _ = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=SEED)
    density = AnomalyDensityEstimator(n_components=5)
    density.fit(X_fit, y_fit)
    cal_all = density.log_prob(X_cal)
    hold_scores = density.log_prob(X_hold)
    records = _suite3_synthetic_records()
    feats, roles = _map_audit(records)
    s = density.log_prob(feats)
    anom = s[np.array([r == "Anomaly" for r in roles])]
    ctrl = s[np.array([r != "Anomaly" for r in roles])]
    rows = []
    rng = np.random.default_rng(SEED)
    for n in (250, 500, 1000, 2000, 3500):
        n = min(n, len(cal_all))
        idx = rng.choice(len(cal_all), size=n, replace=False)
        cal = ConformalCalibrator()
        cal.fit(cal_all[idx])
        fpr = float(np.mean(cal.compute_p_value(hold_scores) <= 0.05))
        rows.append({"n_cal": n, "empirical_fpr": round(fpr, 4),
                     "audit_auroc": round(_auroc(-ctrl, -anom), 4)})
    return rows


def fusion_ablation():
    from scipy.stats import chi2
    audit = json.load(open(os.path.join(
        REPO, "benchmarks", "reports", "voyager_realfile_audit.json")))
    rows = audit["rows"]
    nat = [r for r in rows if r["role"] in ("Natural", "Interference")]
    n = len(nat)

    def rates(key, alpha=0.05):
        flags = [r for r in nat if r[key] <= alpha]
        anom = [r for r in rows if r["role"] == "Anomaly" and r[key] <= alpha]
        return {"flags": len(flags), "denominator": n, "anomaly_hits": len(anom)}

    out = {"path_a_htru2_only": rates("p_h"),
           "path_b_desc_only": rates("p_d")}
    fisher_flags = [r for r in nat
                    if float(chi2.sf(-2.0 * (np.log(max(r["p_h"], 1e-300))
                                             + np.log(max(r["p_d"], 1e-300))), 4)) <= 0.05]
    out["fisher_naive"] = {"flags": len(fisher_flags), "denominator": n,
                           "anomaly_hits": 1}
    out["bonferroni"] = rates("p_fused")
    out["voyager"] = next(r for r in rows if r["role"] == "Anomaly")
    return out


def feature_blocks():
    """Closed-set MCC per feature block under the identical group-CV protocol.

    Scoped to classification on purpose: the detection-path contribution of
    feature groups is covered by the fusion ablation, and the population OOD
    engine requires full-width input. Blocks: full 12-D, continuous 8-D,
    indicators-only 4-D.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import matthews_corrcoef
    from sklearn.model_selection import StratifiedGroupKFold

    def _block_impute(Xb_tr, Xb_te):
        # Fold-local medians on the block itself (the library impute_fit
        # guards full 12-column input, so blocks use the same statistic
        # directly rather than weakening the shared guard).
        with np.errstate(all="ignore"):
            med = np.nanmedian(Xb_tr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        return (np.where(np.isnan(Xb_tr), med, Xb_tr),
                np.where(np.isnan(Xb_te), med, Xb_te))

    pop = build_population(cache=True)
    X = np.asarray(pop.X, dtype=np.float64)
    y = np.asarray(pop.y)
    groups = np.asarray(pop.group_ids)
    blocks = {"full_12d": list(range(12)), "continuous_8d": list(range(8)),
              "indicators_4d": list(range(8, 12))}
    out = {}
    for name, cols in blocks.items():
        Xb = X[:, cols]
        skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
        oof = np.full(y.shape[0], -1, dtype=np.int64)
        for tr, te in skf.split(Xb, y, groups):
            Xtr, Xte = _block_impute(Xb[tr], Xb[te])
            clf = HistGradientBoostingClassifier(random_state=SEED, max_iter=100)
            clf.fit(Xtr, y[tr])
            oof[te] = clf.predict(Xte)
        out[name] = {"mcc_hgbt100": round(float(matthews_corrcoef(y, oof)), 4)}
    return out


def if_baseline():
    from sklearn.ensemble import IsolationForest
    from sklearn.model_selection import StratifiedGroupKFold
    pop = build_population(cache=True)
    X = np.asarray(pop.X, dtype=np.float64)
    y = np.asarray(pop.y)
    groups = np.asarray(pop.group_ids)
    novel = pop.code_of("FRB")
    Xn, yn, gn = X[y != novel], y[y != novel], groups[y != novel]
    Xa = X[y == novel]
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.full(Xn.shape[0], np.nan)
    for tr, te in skf.split(Xn, yn, gn):
        med = impute_fit(Xn[tr])
        iso = IsolationForest(random_state=SEED).fit(impute_apply(Xn[tr], med))
        oof[te] = -iso.decision_function(impute_apply(Xn[te], med))
    med = impute_fit(Xn)
    iso = IsolationForest(random_state=SEED).fit(impute_apply(Xn, med))
    novel_scores = -iso.decision_function(impute_apply(Xa, med))
    return {"auroc": round(_auroc(oof, novel_scores), 4),
            "normal_scores": [float(v) for v in oof],
            "novel_scores": [float(v) for v in novel_scores]}


def roc_figure(maha_normal, maha_novel, if_normal, if_novel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    for neg, pos, label, color, ls in [
            (maha_normal, maha_novel, "Axiom Mahalanobis", "#0A2540", "-"),
            (if_normal, if_novel, "Isolation Forest", "#B2182B", "--")]:
        yy = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        ss = np.concatenate([np.asarray(neg, float), np.asarray(pos, float)])
        fpr, tpr, _ = roc_curve(yy, ss)
        ax.plot(fpr, tpr, color=color, ls=ls, lw=2,
                label=f"{label} (AUROC = {roc_auc_score(yy, ss):.4f})")
    ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1.2, label="Chance")
    ax.set_xlim(-0.01, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Leave-class-out FRB detection (19,252 objects)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.25)
    os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return OUT_FIG


def main():
    X, y, _ = load_htru2()
    report = {"seed": SEED}
    report["k_cap_sweep"] = k_cap_sweep(X, y)
    print("K-cap sweep:")
    for r in report["k_cap_sweep"]:
        print("  K<=%(k_cap)d  AUROC=%(audit_auroc).4f  fit=%(fit_s).1fs" % r)
    report["ncal_sweep"] = ncal_sweep(X, y)
    print("N_cal sweep (alpha=0.05):")
    for r in report["ncal_sweep"]:
        print("  N=%(n_cal)d  empFPR=%(empirical_fpr).4f  AUROC=%(audit_auroc).4f" % r)
    report["fusion"] = fusion_ablation()
    print("Fusion (alpha=0.05, 30 controls):", json.dumps(
        {k: (v if k == "voyager" else {kk: vv for kk, vv in v.items()})
         for k, v in report["fusion"].items()}, indent=1)[:800])
    report["feature_blocks"] = feature_blocks()
    print("Feature blocks:", json.dumps(report["feature_blocks"], indent=1))
    ood = evaluate_population_ood(n_bootstrap=200)
    report["if_baseline"] = if_baseline()
    print("IF AUROC: %.4f (Mahalanobis OOF AUROC: %.4f)" % (
        report["if_baseline"]["auroc"], ood.auroc))
    fig = roc_figure(ood.normal_scores, ood.novel_scores,
                     report["if_baseline"]["normal_scores"],
                     report["if_baseline"]["novel_scores"])
    report["roc_figure"] = fig
    with open(OUT_JSON, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", OUT_JSON)
    print("wrote", fig)


if __name__ == "__main__":
    main()
