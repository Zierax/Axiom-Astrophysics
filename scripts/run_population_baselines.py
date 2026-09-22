"""Population-scale baseline comparison (12-D, 19,252 objects, group CV).

Runs standard classifiers through the EXACT protocol of
axiom.stats.group_ood.evaluate_population_classification so each cell is
directly comparable to the reported HGBT-300 numbers:

  * StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) on
    group_ids (one row per object; grouping is a no-op guard);
  * fold-local median imputation of the continuous block;
  * pooled out-of-fold predictions for accuracy / MCC / weighted-F1;
  * McNemar's test (continuity-corrected, paired on pooled OOF) of
    HGBT-300 against the best baseline, with the full 2x2 table saved.

Scaling: LogReg/RF/SVM use a per-fold StandardScaler fit on the training
fold only; HGBT variants are unscaled (tree splits are scale-invariant,
so scaling would be a no-op). All randomness is seeded (42).

Writes benchmarks/reports/population_baselines.json and prints the table.
"""
import json
import os
import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from axiom.data.population import build_population  # noqa: E402
from axiom.dsp.physical_features import impute_apply, impute_fit  # noqa: E402

SEED = 42
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "benchmarks", "reports", "population_baselines.json")


def _models():
    return {
        "LogReg(C=1)": LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "RF(100,d=10)": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),
        "SVM(RBF,C=1)": SVC(C=1.0, kernel="rbf", gamma="scale", random_state=SEED),
        "HGBT(100,d=6)": HistGradientBoostingClassifier(
            max_iter=100, max_depth=6, random_state=SEED),
        "HGBT(300)": HistGradientBoostingClassifier(random_state=SEED, max_iter=300),
    }


def _mcnemar(b10, b01):
    """Continuity-corrected McNemar chi2 (df=1) from discordant counts."""
    n = b10 + b01
    if n == 0:
        return 0.0
    return float((abs(b10 - b01) - 1.0) ** 2 / n)


def main():
    pop = build_population(cache=True)
    X = np.asarray(pop.X, dtype=np.float64)
    y = np.asarray(pop.y)
    groups = np.asarray(pop.group_ids)
    labels = sorted(int(c) for c in np.unique(y))

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = {}
    for name, proto in _models().items():
        pred = np.full(y.shape[0], -1, dtype=np.int64)
        for tr, te in splitter.split(X, y, groups):
            med = impute_fit(X[tr])
            Xtr, Xte = impute_apply(X[tr], med), impute_apply(X[te], med)
            if name.startswith("HGBT"):
                clf = proto.__class__(**proto.get_params())
                clf.fit(Xtr, y[tr])
                pred[te] = clf.predict(Xte)
            else:
                scaler = StandardScaler().fit(Xtr)
                clf = proto.__class__(**proto.get_params())
                clf.fit(scaler.transform(Xtr), y[tr])
                pred[te] = clf.predict(scaler.transform(Xte))
        assert np.all(pred >= 0)
        oof[name] = pred

    rows = {}
    for name, pred in oof.items():
        rows[name] = {
            "accuracy": float(accuracy_score(y, pred)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", labels=labels)),
        }

    # McNemar: HGBT(300) vs the best baseline by MCC (pre-specified primary
    # comparator is HGBT(300); baselines ranked post hoc for reporting only).
    base_name = max((n for n in oof if n != "HGBT(300)"),
                    key=lambda n: rows[n]["mcc"])
    a, b = oof["HGBT(300)"] == y, oof[base_name] == y
    b10, b01 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    both_ok, both_bad = int(np.sum(a & b)), int(np.sum(~a & ~b))
    mcnemar = {
        "model_a": "HGBT(300)", "model_b": base_name,
        "both_correct": both_ok, "only_a": b10, "only_b": b01,
        "both_wrong": both_bad, "n": int(y.shape[0]),
        "chi2_continuity_corrected": _mcnemar(b10, b01),
    }

    report = {"seed": SEED, "n": int(y.shape[0]), "protocol": {
        "splits": "StratifiedGroupKFold(5, shuffle, random_state=42) on object_id",
        "imputation": "fold-local medians", "scaling": "per-fold StandardScaler except HGBT (scale-invariant, unscaled)",
    }, "models": rows, "mcnemar_hgbt300_vs_best_baseline": mcnemar}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(report, f, indent=1)

    print(f"{'model':14s} {'acc':>7s} {'mcc':>7s} {'wF1':>7s}")
    for name, r in rows.items():
        print(f"{name:14s} {r['accuracy']:7.4f} {r['mcc']:7.4f} {r['weighted_f1']:7.4f}")
    print("McNemar HGBT(300) vs %s: b10=%d b01=%d both_ok=%d both_bad=%d chi2=%.3f" % (
        base_name, b10, b01, both_ok, both_bad, mcnemar["chi2_continuity_corrected"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
