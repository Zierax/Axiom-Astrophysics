# axiom-astrophysics v2 — Data Dictionary

This document defines the physical and statistical interpretation of the feature
parameters used by AXIOM. §1–§3 describe the **HTRU2 in-distribution manifold**
(the 8 survey moments used by the production classifier); §4 describes the
**Lane-2 population-scale physical manifold** (the 12-D commensurate vector used
for catalog-scale validation of thousands of independent real objects).

---

## 1. Integrated Profile Features (Time Domain)
The integrated profile is a 1D array of pulse intensities folded at the candidate's optimal period. It represents the "shape" of the pulse as a function of rotational phase.

| Feature Name | Column Index | Physical Meaning | Expected Behavior (Pulsar) | Expected Behavior (RFI) |
|---|---|---|---|---|
| `profile_mean` | 0 | The arithmetic mean of the integrated pulse profile. | Higher than noise floor due to coherent emission. | Highly variable, often low or extremely saturated. |
| `profile_std` | 1 | The standard deviation (dispersion) of the profile intensity. | Moderate; pulses are distinct but bounded. | Can be very high if interference is impulsive. |
| `profile_kurtosis` | 2 | The excess kurtosis (peakedness/sharpness) of the profile. | High (sharp, narrow peak compared to a Gaussian). | Low or negative (broadband or uniform noise). |
| `profile_skewness` | 3 | The skewness (asymmetry) of the profile. | Positive (rapid rise, slower exponential decay tail). | Near zero (symmetric noise) or highly erratic. |

---

## 2. DM-SNR Curve Features (Dispersion Domain)
The DM-SNR (Dispersion Measure - Signal-to-Noise Ratio) curve plots the SNR of the candidate as a function of trial Dispersion Measures. True astrophysical signals exhibit a peak SNR at a non-zero DM, dropping off symmetrically.

| Feature Name | Column Index | Physical Meaning | Expected Behavior (Pulsar) | Expected Behavior (RFI) |
|---|---|---|---|---|
| `dmsnr_mean` | 4 | The mean SNR across all trial dispersion measures. | Low (SNR peaks sharply at true DM, zero elsewhere). | High (RFI peaks at DM=0 and smears across all DMs). |
| `dmsnr_std` | 5 | Standard deviation of the DM-SNR curve. | Low to moderate. | High (large fluctuations across DM trials). |
| `dmsnr_kurtosis` | 6 | Excess kurtosis of the DM-SNR curve. | High (a very sharp spike exactly at the true DM). | Low (broad, flat distribution of SNR). |
| `dmsnr_skewness` | 7 | Skewness of the DM-SNR curve. | Positive (long tail on one side of the true DM). | Near zero (symmetric spread). |

---

## 3. The Target Variable

| Label Name | Column Index | Physical Meaning | Total Count |
|---|---|---|---|
| `class_label` | 8 | Ground-truth verification class. | 17,898 |
| `0` | - | Non-pulsar, Terrestrial RFI, or random stochastic noise. | 16,259 |
| `1` | - | Verified Pulsar (Astrophysical origin). | 1,639 |

---

---

## 4. Lane-2 Population-Scale Physical Manifold (12-D)

For population validation the engine bypasses the HTRU2 anchor and maps each
independent catalog object (pulsar, FRB, RRAT, magnetar, RFI) through one
commensurate featurizer, `axiom.dsp.physical_features.featurize_frame`. Because
every row is a distinct object with a unique `group_id`, cross-validation keyed on
that id is leakage-free by construction. The 12 features are:

| # | Feature | Transformation | Meaning | Defined for |
|---|---|---|---|---|
| 0 | `log_dm` | `log10(1 + DM)` | Dispersion measure (integrated electron column) | all |
| 1 | `dm_excess` | `DM − DM_gal` | Extragalactic DM above the Galactic model | all |
| 2 | `dm_excess_ratio` | `DM / (1 + DM_gal)` | Relative height above Galactic expectation | all |
| 3 | `abs_glat` | `|b|` | Absolute Galactic latitude (out-of-plane ⇒ Galactic) | all |
| 4 | `log_width` | `log10(W)` | Pulse/burst width (ms) | pulsars/FRBs |
| 5 | `log_snr` | `log10(1 + S/N)` | Signal-to-noise ratio | pulsars/FRBs |
| 6 | `log_period` | `log10(P)` | Spin period (s) | pulsars/RRAT/magnetars |
| 7 | `log_duty` | `log10(W / P)` | Duty cycle | pulsars/RRAT/magnetars |
| 8 | `has_glat` | `0/1` | Galactic latitude present | indicator |
| 9 | `has_width` | `0/1` | Width present | indicator |
| 10 | `has_snr` | `0/1` | S/N present | indicator |
| 11 | `has_period` | `0/1` | Period present | indicator |

Undefined quantities (e.g. width for an FRB catalog entry) are left as `NaN` and
imputed with the training-fold median via `impute_fit` / `impute_apply`, so the
same code path handles every class. `DM_gal` is taken from the catalog's own
Galactic model when available (CHIME `DMeYMW16`), otherwise from a deterministic
ceiling `DM_gal(panel, |b|)`.

---

## Note on Synthetic OOD Generation
When the AXIOM engine evaluates out-of-distribution signals, it does **not** treat
FRBs or quasars as anomalies — those are legitimate astrophysical sources used as
**controls**. Genuine anomalies are narrowband carriers (Wow!, BLC1-style, and the
real Voyager 1 GBT carrier) that are projected onto the HTRU2 manifold via
`physics_map_htru2_features` (the 8 real HTRU2 moments; the dataset's 9th column
is the class label) so the per-class density estimator can detect an
out-of-manifold gap. Synthetic waveforms are synthesized per signal type, and
their DSP complexity features feed the chaos / CNN / anchor logic. The Lane-2
manifold instead evaluates *real* population separation directly (see §4).
