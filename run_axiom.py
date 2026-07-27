"""
axiom-astrophysics v2 — Production Pipeline Orchestrator
===========================================================

Downloads and loads the real-world HTRU2 dataset, trains the Stacking
Classifier and GMM Density Estimator, and runs conformal anomaly detection.
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
)

from axiom.config import PipelineConfig
from axiom.data.loader import load_htru2, split_htru2
from axiom.ml.ensemble import AxiomEnsemble
from axiom.stats.ood_eval import evaluate_ood

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_axiom")


def run_production_pipeline(config_path=None):
    t_start = time.time()
    config = PipelineConfig(config_path)

    # 1. Load real dataset (HTRU2)
    log.info("Step 1: Loading real-world HTRU2 pulsar candidate dataset...")
    X, y, col_names = load_htru2()
    
    # Split dataset (single source of truth: pipeline_config.yaml →
    # train 0.8 / val 0.1 / test 0.1; overrides the 0.7/0.15 hardcode).
    train_ratio = config.get("data.train_ratio", 0.8)
    val_ratio = config.get("data.val_ratio", 0.1)
    seed = config.get("data.split_seed", 42)
    splits = split_htru2(X, y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    # 2. Train stacking ensemble classifier
    log.info("Step 2: Training AXIOM stacking ensemble classifier...")
    ensemble = AxiomEnsemble(n_classes=2, random_state=seed)
    ensemble.fit(X_train, y_train)

    # 3. The OOD density estimator is fit per-class inside the audit (Step 5)
    #    to stay consistent with the benchmark suite and the corrected logic.

    # 4. Evaluate on In-Distribution (ID) test set
    log.info("Step 4: Evaluating In-Distribution performance on HTRU2 test set...")
    preds_test = ensemble.predict(X_test)
    
    acc = accuracy_score(y_test, preds_test)
    mcc = matthews_corrcoef(y_test, preds_test)
    
    print("\n" + "=" * 60)
    print("  IN-DISTRIBUTION PERFORMANCE (Pulsars vs RFI)")
    print("=" * 60)
    print(f"  Test Samples : {len(y_test)}")
    print(f"  Accuracy     : {acc:.4f} (Target: >= 0.9800)")
    print(f"  MCC          : {mcc:.4f} (Target: >= 0.9500)")
    print("-" * 60)
    print(classification_report(y_test, preds_test, target_names=["RFI", "Pulsar"], zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds_test))
    print("=" * 60 + "\n")

    # 5. Out-of-Distribution (OOD) Anomaly Audit — corrected methodology
    log.info("Step 5: Physics-mapped OOD audit (corrected methodology)...")

    # Labelled audit: genuine narrowband carriers (anomalies) vs legitimate
    # astrophysical / RFI sources (controls). FRBs and quasars are NOT anomalies.
    # Real observational augmentation (AXIOM_REAL_OOD != 0, default on).
    # Replaces synthetic controls with measured data: Natural FRB -> CHIME/FRB
    # Catalog 2 DMs; Interference -> real HTRU2 RFI features; Anomaly -> real
    # Voyager 1 GBT carrier + sidebands (ground-truth artificial technosignature)
    # plus real Breakthrough Listen narrowband observation channels. See
    # axiom/data/real_loaders.py. Falls back to synthetic on any failure.
    ood_records, real_features, real_waves = [], {}, {}
    if os.environ.get("AXIOM_REAL_OOD", "1") != "0":
        try:
            from axiom.data.real_loaders import get_full_real_ood
            ood_records, real_features, real_waves, waterfall_features, _ = get_full_real_ood(
                X, y, seed=seed, n_frb=25, n_rfi=25, n_anom=25)
            log.info("Step 5: real-data OOD set assembled "
                     "(%d records; %d real waveforms, %d real feature vectors, "
                     "%d waterfall descriptors).",
                     len(ood_records), len(real_waves), len(real_features),
                     len(waterfall_features))
        except Exception as exc:  # pragma: no cover - offline guard
            log.warning("Step 5: real-data OOD unavailable (%s); "
                        "using synthetic OOD set.", exc)
            ood_records = []
            waterfall_features = {}

    if not ood_records:
        ood_records = []
        for i in range(25):
            ood_records.append((f"NarrowbandTone_{i}", "Narrowband", "Narrowband",
                                0.0, 18.0, "Anomaly"))
        ood_records.append(("Wow! Signal", "Narrowband", "Narrowband", 0.0, 30.0, "Anomaly"))
        ood_records.append(("BLC1", "Narrowband", "Narrowband", 0.0, 15.0, "Anomaly"))
        for i in range(25):
            ood_records.append((f"FRB_{i}", "FRB", "FRB",
                                float(np.random.uniform(100, 1000)), 20.0, "Natural"))
        for i in range(15):
            ood_records.append((f"Quasar_{i}", "Quasar", "Quasar", 0.0, 5.0, "Natural"))
        for i in range(25):
            ood_records.append((f"RFI_{i}", "RFI", "RFI", 0.0, 10.0, "Interference"))

    ood_result = evaluate_ood(X, y, ood_records, seed=seed,
                              real_features=real_features or None,
                              real_waves=real_waves or None,
                              waterfall_features=waterfall_features or None)

    verdicts = ood_result["verdicts"]
    roles = ood_result["roles"]
    names = ood_result["names"]

    n_anom = sum(r == "Anomaly" for r in roles)
    n_nat = sum(r in ("Natural", "Interference") for r in roles)

    print("=" * 60)
    print("  OUT-OF-DISTRIBUTION ANOMALY AUDIT (labelled controls)")
    print("=" * 60)
    print(f"  Genuine anomalies (narrowband carriers) : {n_anom}")
    print(f"  Natural / Interference controls         : {n_nat}")
    print(f"  Anomaly True-Positive Rate  : {ood_result['anomaly_tpr']*100:.1f}%")
    print(f"  Natural False-Positive Rate : {ood_result['natural_fpr']*100:.1f}%")
    print("-" * 60)

    # Flagship real-world validation: the Voyager 1 GBT carrier is a
    # ground-truth *artificial* technosignature. Report it when present.
    voy = [(n, v) for n, v, r in zip(names, verdicts, roles)
           if n.startswith("Voyager") and r == "Anomaly"]
    if voy:
        n_hit = sum(v == "Anomaly" for _, v in voy)
        print("  Real artificial technosignature (Voyager 1 GBT carrier):")
        for n, v in voy:
            print(f"    - {n:<24s} | Verdict: {v}")
        print(f"    Voyager 1 detection: {n_hit}/{len(voy)} (ground truth)")

    # Historical synthetic candidates (only present in the synthetic fallback).
    idx = {n: i for i, n in enumerate(names)}
    if "Wow! Signal" in idx or "BLC1" in idx:
        print("  Historical candidate verdicts:")
        if "Wow! Signal" in idx:
            print(f"    - Wow! Signal (1977) | Verdict: {verdicts[idx['Wow! Signal']]}")
        if "BLC1" in idx:
            print(f"    - BLC1 (2020)        | Verdict: {verdicts[idx['BLC1']]}")
    print("=" * 60 + "\n")
    
    # Save trained models
    ensemble.save("data/models/ensemble_model.pkl")
    
    log.info("Production pipeline execution finished in %.2f seconds.", time.time() - t_start)


def main():
    parser = argparse.ArgumentParser(description="axiom-astrophysics v2 Production Pipeline Runner")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    args = parser.parse_args()

    print("=" * 60)
    print("  axiom-astrophysics v2")
    print("  Production Pipeline Execution")
    print("=" * 60)

    try:
        run_production_pipeline(args.config)
    except Exception as exc:
        log.exception("Pipeline run encountered critical failure: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
