# axiom-astrophysics v2 — API Reference

Detailed specifications of core public packages and interfaces in the `axiom` module.

---

## 1. Machine Learning (`axiom.ml`)

### `AxiomEnsemble`
Stacked generalization model combining Random Forest and HistGradientBoosting classifiers.
```python
from axiom.ml.ensemble import AxiomEnsemble

ensemble = AxiomEnsemble(n_classes=2, random_state=42)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)
probabilities = ensemble.predict_proba(X_test)
```

### `AnomalyDensityEstimator`
Multivariate Gaussian Mixture Model fit **per class** (Pulsar / RFI) and scored
by the maximum class-likelihood.
```python
from axiom.ml.density import AnomalyDensityEstimator

density = AnomalyDensityEstimator(n_components=5)
density.fit(X_normal, y_normal)            # per-class fit
log_likelihood = density.log_prob(X_test)  # max_c log p(x | c)
class_scores = density.log_prob_per_class(X_test, class_idx)
```

---

## 2. Statistical Calibration (`axiom.stats`)

### `ConformalCalibrator`
Split conformal prediction calibrator producing valid p-values under covariate shift.
```python
from axiom.stats.calibration import ConformalCalibrator

calibrator = ConformalCalibrator()
calibrator.fit(validation_log_probs)
p_values = calibrator.compute_p_value(test_log_probs)
```

### `SignalArbitrator`
Benjamini-Hochberg FDR control and composite anomaly verdict arbitrator. The
`ood_mask` (pre-computed by the absolute-OOD rule) restricts anomaly candidacy
to signals lying outside the natural population manifold; `cnn_probs` is an
optional learned waveform branch fused with the ensemble via geometric mean.
```python
from axiom.stats.arbitration import SignalArbitrator

arbitrator = SignalArbitrator(fdr_alpha=0.05, conformal_alpha=0.05)
verdicts, anomaly_scores = arbitrator.arbitrate(
    signal_ids, meta_predictions, meta_probs, p_values,
    chaos_scores, origin_classes, ood_mask=ood_mask, cnn_probs=cnn_probs
)
```

- `meta_predictions` / `meta_probs`: 3-class mapping (Natural / Interference /
  Anomaly) where the Anomaly column is `1 - p_value`.
- `ood_mask`: boolean array; only `True` entries may receive an "Anomaly" verdict.
- `cnn_probs`: optional `(N, 3)` probabilities from `CosmicSignalCNN`; fused with
  `meta_probs` by geometric mean when provided and valid.
- `chaos_scores`: `(N,)` legacy Lyapunov scalars **or** `(N, D)` descriptors from
  `compute_chaos_descriptor`; reduced to a [0, 1] order score internally.

### `compute_chaos_descriptor` / `chaos_order_score`
Bounded, discriminating nonlinear-dynamics features (replaces the saturated
single Lyapunov exponent).
```python
from axiom.stats.chaos import compute_chaos_descriptor, chaos_order_score

descriptor = compute_chaos_descriptor(waveform)   # [perm_entropy, lyapunov_norm]
order = chaos_order_score(descriptor)             # 1.0 = highly ordered / tone
```
- `compute_chaos_descriptor` returns a 2-element float64 vector: normalized
  permutation entropy in [0, 1] and a tanh-squashed Lyapunov exponent in (-1, 1).
  Degenerate input returns a neutral `[0.5, 0.0]`.
- `chaos_order_score` accepts `(N,)` or `(N, D)` and returns `(N,)` in [0, 1]
  (1.0 == maximally ordered / deterministic, 0.0 == stochastic noise).

### `CosmicSignalCNN`
Trainable 1-D CNN waveform classifier (3 classes: Natural / Interference /
Anomaly). Uses PyTorch when available, otherwise a deterministic pure-NumPy
backend with manual backpropagation and Adam.
```python
from axiom.ml.cnn import CosmicSignalCNN

cnn = CosmicSignalCNN(seed=42)
cnn.train(waveforms, labels, epochs=20, batch_size=32)
probs = cnn.predict_proba(waveforms)   # (N, 3) probabilities
```

---

## 3. Data Processing (`axiom.data`)

### `load_htru2`
Utility to auto-download, unzip, verify, and load the UCI HTRU2 dataset.
```python
from axiom.data.loader import load_htru2

features, labels, column_names = load_htru2(dest_dir="data")
```

### `load_catalog` / `load_many` (Lane-2 catalogs)
Fetch and normalize verified real catalogs (ATNF `B/psr/psr`, CHIME/FRB
`J/ApJS/257/59/table2`, HTRU2) to one common per-object schema via the
reproducible, SHA-256-pinned downloader.
```python
from axiom.data.catalogs import load_catalog, load_many, REGISTRY

df = load_catalog("atnf_pulsars")          # one normalized catalog
pop = load_many(["atnf_pulsars", "chime_frb_cat1", "htru2_local"])
```

### `build_population`
Assemble the population-scale manifold (feature matrix, class codes, unique
`group_ids`, provenance) with a deterministic on-disk cache.
```python
from axiom.data.population import build_population

pop = build_population(cache=True)          # 19,252 independent objects
X, y, groups = pop.X, pop.y, pop.group_ids
```

## 4. Physical Featurizer (`axiom.dsp.physical_features`)

### `featurize_frame` / `impute_fit` / `impute_apply`
Map the normalized catalog schema to a commensurate 12-D physical feature space
(DM, extragalactic DM excess, Galactic latitude, width, S/N, period, duty cycle,
plus missingness indicators), with leakage-free fold-local imputation.
```python
from axiom.dsp.physical_features import featurize_frame, impute_fit, impute_apply

X, names = featurize_frame(df)              # NaN where a quantity is undefined
medians = impute_fit(X[train_idx])          # training-fold medians only
X_tr = impute_apply(X[train_idx], medians)
```

## 4b. Native Spectrogram DSP (`axiom.dsp.waterfall`, `axiom.dsp.waterfall_features`)

### `extract_features` (real-waterfall featurizer)
Maps any filterbank dynamic spectrum (frequency × time) to a **12-D** vector via
bandpass normalization, an incoherent-dedispersion DM–SNR sweep, and profile/spectral
descriptors. This is the single deterministic featurizer used by Lane 1.
```python
from axiom.dsp.waterfall import extract_features

vec = extract_features(waterfall, freqs_mhz, tsamp_s,
                       dm_max=500.0, n_dm=32, n_subbands=32)
# returns np.ndarray shape (12,) ; pass return_details=True for DM sweep + scores
```

### `compute_waterfall_features` / `waterfall_narrowband_score`
Direct frequency-resolved descriptors from a 2-D spectrogram (no synthetic
manifold placement). `waterfall_narrowband_score` returns a bounded [0, 1] tone
score (high channel concentration + low spectral flatness → tone-like).
```python
from axiom.dsp.waterfall_features import (
    compute_waterfall_features, waterfall_narrowband_score,
    descriptor_vector, DescriptorConformalDetector,
)

feat = compute_waterfall_features(spec)        # dict of 8 descriptors
score = waterfall_narrowband_score(feat)       # [0, 1]
vec = descriptor_vector(feat)                  # fixed-order (8,) numeric vector
```

### `DescriptorConformalDetector`
Split-conformal detector over the 8-D descriptor vector. The "normal" null is the
natural/broadband population; `p_value` is finite-sample-valid and returns `1.0`
(neutral) for unfitted detectors or empty descriptors.
```python
det = DescriptorConformalDetector(alpha=0.05).fit(natural_descriptor_dicts)
p = det.p_value(feat)        # small => off-manifold / tonal relative to natural
```

## 4c. Out-of-Distribution Evaluation (`axiom.stats.ood_eval`, `axiom.data.populations`, `axiom.stats.manifold_ood`)

### `evaluate_ood`
Full anomaly audit over a labelled OOD record set. Returns verdicts, roles, the
fused conformal p-values, and transparency fields for the descriptor fusion.
```python
from axiom.stats.ood_eval import evaluate_ood

res = evaluate_ood(X, y, records, seed=42,
                   real_features=real_features, real_waves=real_waves,
                   waterfall_features=waterfall_features)
# res keys: verdicts, roles, names, pvals (fused), htru2_pvals,
#           descriptor_pvals, descriptor_fusion_active, ood_mask,
#           anomaly_tpr, natural_fpr, pass
```
Records are `(name, origin_class, sig_type, dm, snr, true_role)` tuples with
`true_role ∈ {Anomaly, Natural, Interference}`. Real spectrograms supplied via
`waterfall_features` are scored on their measured morphology; the fused p-value is
`min(1, 2·min(p_htru2, p_descriptor))` (Bonferroni).

### `build_manifold` / `ManifoldConformalDetector` / `evaluate_manifold_ood` (Lane 1)
The self-consistent real-waterfall manifold. `build_manifold` featurises real
GUPPI/filterbank observations with `extract_features`; `ManifoldConformalDetector`
fits a robust per-normal-class Ledoit–Wolf precision and scores by the minimum
standardised Mahalanobis distance; `evaluate_manifold_ood` runs cross-conformal
OOD with exact finite-sample FDR control.
```python
from axiom.data.populations import build_manifold
from axiom.stats.manifold_ood import evaluate_manifold_ood, ManifoldConformalDetector

manifold = build_manifold(cache=True)          # real observations, 12-D features
report = evaluate_manifold_ood(manifold)        # AUROC, TPR, FPR, coverage
```

## 5. Group-Aware Evaluation (`axiom.stats.group_ood`)

### `evaluate_population_classification`
Stratified group K-fold multiclass typing (MCC/weighted-F1 headline, bootstrap
CIs, confusion matrix).

### `evaluate_population_ood`
Leave-class-out conformal novelty detection (default: extragalactic FRBs
withheld) reporting AUROC with bootstrap CI, conformal TPR/FPR and coverage.
```python
from axiom.stats.group_ood import (
    evaluate_population_classification, evaluate_population_ood,
)

clf = evaluate_population_classification()   # MCC 0.817
ood = evaluate_population_ood(novel_class="FRB")  # AUROC 0.9998
```

---

## 6. Reporting Pipeline (`axiom.reporting`)

Deterministic generator turning every validation suite into the
`benchmarks/` artifact tree. No figures are fabricated — every metric is
computed from real data under a fixed seed.

### `collect_all` / `RunData`
Runs all 7 suites and returns a `RunData` (`summary` JSON record + `arrays`
NumPy payload for charting).
```python
from axiom.reporting import collect_all

run = collect_all()          # RunData; run.summary["verdict"] via build_verdict
```

### `generate`
End-to-end orchestrator: collect → render charts → write reports.
```python
from axiom.reporting import generate

master = generate(root="benchmarks")   # writes benchmarks/reports + benchmarks/charts
```

### `render_all` / `write_reports`
Lower-level hooks if you want to render only the 22 figures (`render_all`) or
re-emit the markdown/JSON reports (`write_reports`) from a pre-collected run.

