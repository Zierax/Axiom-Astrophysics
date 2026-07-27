# axiom-astrophysics v2 — Core Logic & Scientific Philosophy

This document outlines the theoretical physics, mathematical logic, and scientific philosophy underlying the axiom-astrophysics signal audit engine.

---

## 1. The Axiom Protocol: Logic as a Universal Constant

The central premise of the AXIOM engine is that **Logic is a Constant**. When searching for anomalous out-of-distribution (OOD) signals or potential technosignatures, traditional SETI (Search for Extraterrestrial Intelligence) approaches often rely on heuristic "anthropocentric" templates—searching for patterns humans might use, such as prime number sequences or narrow-band FM modulation.

AXIOM discards anthropocentric assumptions. Instead, it audits signals based purely on their deviation from the natural stochastic and thermodynamic limits of the universe, modeled as the Lambda-CDM cosmological baseline.

### 1.1 Core Principles
1. **Zero-Heuristic Baseline**: The system learns the continuous probability density function of known natural signals (e.g., Pulsars, Quasars, RFI). It does not search for "aliens"; it searches for "mathematical impossibility" within the natural feature space.
2. **Thermodynamic Exclusion**: Natural macroscopic processes maximize entropy over time. Signals displaying sustained, low-entropy structural complexity that resists dispersion are flagged.
3. **Scale Invariance**: The mathematics of the engine apply equally to microsecond transient bursts (like FRBs) and continuous wave anomalies.

---

## 2. Feature Manifold Logic

AXIOM maps signals onto the 8-dimensional HTRU2 survey manifold (the 8 real
HTRU2 moments; the 9th dataset column is the class label) via physics-based
exemplars, augmented by auxiliary DSP complexity features (Shannon entropy,
permutation entropy, Higuchi dimension, LZ76, drift rate, harmonic complexity,
intensity). Each feature represents a fundamental physical or informational
property.

### 2.1 Information Theory Metrics
* **Shannon Entropy ($H$)**: Measures the information density of the signal. Random noise (RFI) possesses maximum entropy, while artificial communication possesses compressed, structured entropy.
* **Lempel-Ziv Complexity (LZ76)**: Quantifies the rate at which new patterns emerge in the signal. A highly periodic pulsar has low LZ76, noise has high LZ76, and an artificial signal rests in the critical regime between them.

### 2.2 Nonlinear Dynamics & Chaos Theory
* **Permutation Entropy**: Evaluates the topological complexity of the time series by analyzing the ordinal relationships between values, distinguishing deterministic chaos from uncorrelated noise.
* **Higuchi Fractal Dimension ($D_f$)**: Measures the self-similarity and "roughness" of the signal. Natural signals exhibit fractal properties characteristic of their physical emission mechanisms (e.g., synchrotron radiation).

### 2.3 Physical Astrometry
* **Drift Rate ($df/dt$)**: Measures the rate of change in frequency over time. High drift rates can indicate high planetary acceleration (Doppler shifting), a hallmark of signals originating from a localized rotating body.
* **Harmonic Complexity**: Analyzes the Fourier domain for unnaturally spaced harmonics, often indicating artificial modulation or terrestrial interference.

### 2.4 Native Frequency-Resolved Manifold (real spectrograms)
For real telescope observations a *second*, fully measurement-driven feature space
is derived directly from the 2-D spectrogram (frequency × time) by
`axiom.dsp.waterfall_features.compute_waterfall_features`, without reference to any
synthetic manifold placement. The descriptors — peak/integrated S/N, occupancy,
spectral kurtosis, drift rate, bandwidth, channel concentration (Gini), and
spectral flatness — encode the same physics the survey manifold cannot: *how*
concentrated and tonal the emission is across frequency. A narrowband carrier is
characterised by high channel concentration and low spectral flatness (a tone),
whereas a broadband astrophysical process spreads energy across the band. This
space is used for a conformal OOD test that is independent of the HTRU2 anchor, so
the verdict on a real observation rests on its own measured morphology.

---

## 3. The Arbitration Logic

Once a signal is mapped into the feature manifold, the AXIOM engine employs a multi-layered arbitration process to determine its origin.

### 3.1 The In-Distribution Boundary (Stacking Ensemble)
Pulsars and terrestrial Radio Frequency Interference (RFI) share significant feature overlap, particularly at low Signal-to-Noise Ratios (SNR). The AXIOM engine uses a Stacking Ensemble:
1. **Random Forest**: Builds uncorrelated decision trees to partition the feature space, reducing variance.
2. **Histogram Gradient Boosting (HGBT)**: Iteratively corrects the residual errors of the forest, establishing precise nonlinear boundaries.
3. **Logistic Meta-Learner**: Combines the predictions, acting as a final logical gate that prevents overfitting to specific training anomalies.

### 3.2 The Out-of-Distribution Boundary (GMM & Conformal Prediction)
How does the system react to a signal it has never seen before (e.g., an FRB or a technosignature)?
1. **Density Estimation**: A per-class Gaussian Mixture Model (GMM) learns the density $p(x \mid c)$ for each known natural class (Pulsar, RFI). A signal whose **maximum** class-likelihood is far below the entire natural population (the absolute OOD rule) lies outside the natural envelope and becomes an anomaly candidate.
2. **Conformal Calibration**: To avoid arbitrary thresholding, the engine uses Split Conformal Prediction. It computes a mathematically rigorous $p$-value representing the probability that a natural process could generate a signal with a density score less than or equal to the observed score.
3. **Dual conformal fusion (real observations)**: A real spectrogram is also scored on its native frequency-resolved descriptors (§2.4) against a natural/broadband null, yielding a second conformal p-value $p_{\text{descriptor}}$. The two p-values are combined by a Bonferroni min-rule,
   $$p_{\text{fused}} = \min\!\bigl(1,\; 2\cdot\min(p_{\text{htru2}},\,p_{\text{descriptor}})\bigr),$$
   which preserves finite-sample false-positive control while allowing a signal to be flagged when it is off-manifold in **either** space. Observations without a real spectrogram keep $p_{\text{descriptor}} = 1$ and are judged solely on the survey manifold.

### 3.3 False Discovery Rate (FDR) Control
In a dataset of billions of signals, relying on a static $p < 0.05$ threshold will yield millions of false positives. AXIOM applies the **Benjamini-Hochberg procedure** to control the expected proportion of false discoveries, dynamically scaling the significance threshold based on the total number of tests.

---

## 4. The Final Verdict

The `SignalArbitrator` executes the final logic loop:
1. If the Conformal FDR test fails (the signal is mathematically indistinguishable from the natural distribution), it is classed as **Natural** or **Interference**.
2. If the test passes (the signal is a severe outlier), the Lyapunov Exponent is calculated. If the signal demonstrates high deterministic order (low chaos), it is elevated to **Candidate — Requires Review**.
3. If it violates all known natural manifolds with high confidence, it is designated an **Anomaly**.

The AXIOM designation of "Anomaly" does not inherently mean "Artificial." It means a statistical invariant has been breached, requiring immediate human scientific review.
