# AGENTS.md — axiom-astrophysics

## Mission (overrides all other guidance)
This repo must reach **Q1-journal (Nature / ApJ) quality** and be **useful on real-world, normal inputs**, not demos. Every change must be production-hardened: no placeholders (`// TODO`, `# ... existing code ...`), no happy-path-only logic, deterministic and reproducible, with defensive input validation and graceful failure. Prefer fixing root causes over papering over symptoms.

## Repository map
- `axiom/` — package. Submodules: `data` (HTRU2 load/cache/catalogs), `dsp` (features, synthesis, C bindings), `ml` (ensemble, density, cnn), `stats` (calibration, arbitration, chaos).
- `run_axiom.py` — production pipeline (train + OOD anomaly test).
- `benchmark.py` — validation suite (5-fold CV, baselines, McNemar, OOD audit).
- `scripts/` — reports, plots, historical audit (`audit_historical.py`, `generate_historical_anomalies.py`, `generate_reports.py`).
- `axiom/reporting/` — the research-grade report generator: `collect.py` (runs every validation suite deterministically), `charts.py` (20+ scientific figures), `report_writer.py` (`benchmarks/reports/` markdown + JSON), `pipeline.py` (orchestrator).
- `tests/` — pytest. `configs/pipeline_config.yaml` — dot-path-overridable config (`axiom/config.py:82`).
- `data/`, `benchmarks/` (reports + charts), `*.pkl`, `*.so`, `*.csv` are **git-ignored** — they regenerate; do not commit them.

## Commands
- `make setup` → `pip install -r requirements.txt && pip install -e ".[test,waterfall]"` (installs pytest + blimpy for the full suite)
- `make test` → `python3 -m pytest tests/ -v`  (NOTE: first run downloads HTRU2 — needs network; see gotchas)
- `make benchmark` → `python3 benchmark.py`
- `make report` → `python3 scripts/generate_reports.py` (regenerates `benchmarks/reports/*.md|json` + `benchmarks/charts/*.png`)
- `make all` = test + benchmark + report
- `python3 run_axiom.py [--config configs/pipeline_config.yaml]`
- README's `wsl python3 ...` is for Windows+WSL hosts; in this Linux shell use plain `python3`.

## Critical gotchas (easy to get wrong)
- **Lint is configured via ruff** in `pyproject.toml` (rules E9/F/B/I). Run `ruff check axiom scripts benchmark.py run_axiom.py` before committing; CI should gate on it. There is no mypy/typecheck configured.
- **`pyyaml` is required and present** in both `requirements.txt` and `pyproject.toml`, so `configs/pipeline_config.yaml` overrides take effect. `axiom/config.py` still falls back to `DEFAULT_CONFIG` only if yaml import fails.
- **HTRU2 auto-downloads on first use** from UCI (`axiom/data/loader.py:11`) into `data/HTRU_2.csv`. Offline runs fail. Tests download it too (module-scoped fixtures). A fresh clone has no `data/` (git-ignored).
- **C extension is optional.** `axiom/lib/axiom_core.so` is git-ignored and is NOT prebuilt in a clean clone. `axiom/dsp/c_bindings.py` uses it if present, else falls back to pure-Python (slower). Rebuild with `bash axiom/lib/recompile.sh` (needs `gcc`). Never assume the `.so` exists in CI/clean clone.
- **CNN IS wired into the arbitrator.** `axiom/ml/cnn.py` (`CosmicSignalCNN`) trains deterministically (pure-NumPy backend implemented; PyTorch used when available) and is fused with the ensemble inside `SignalArbitrator` via `cnn_probs` (geometric-mean fusion). The previous uniform 1/3 fallback is gone.
- **Frequency-resolved features are wired in.** `axiom/dsp/waterfall_features.py` computes native spectrogram descriptors (occupancy, drift, spectral kurtosis, concentration, flatness) from real GUPPI waterfalls; they feed the arbitrator's composite score via `waterfall_features` threaded through `get_full_real_ood` → `evaluate_ood` → `SignalArbitrator`.
- **Seeds are load-bearing for reproducibility**: `split_seed=42`, `random_state=42` threaded through ensemble/density. Keep deterministic; never introduce unseeded randomness into reported metrics.

## Known limitations (may be flagged in review)
- **OOD audit now uses real telescope signals, not synthetic-by-construction ones.** Lane 1 ingests real, checksum-verified Voyager 1 carrier/sidebands + Kaggle Breakthrough Listen GUPPI spectrograms of nearby stars + real CHIME/FRB and HTRU2 RFI. The synthetic narrowband-tone controls remain only as a last-resort fallback. The engineered *technosignature* anomaly is represented by **unlabeled** real observations used as anomaly controls (we verify the engine surfaces off-manifold telescope signals), **not** as adjudicated artificial-origin claims — see README §06/§07.
- **Frequency-resolved descriptors are native AND drive the primary verdict.** `axiom/dsp/waterfall_features.py` computes occupancy, drift, spectral kurtosis, bandwidth, concentration and spectral flatness directly from real GUPPI spectrograms. As of v2 these feed a **descriptor-conformal p-value** (`DescriptorConformalDetector`) that is Bonferroni-fused with the HTRU2 conformal p-value inside `evaluate_ood` (`p_fused = min(1, 2·min(p_htru2, p_descriptor))`). A real signal is therefore flagged when its **measured** morphology is off-manifold in *either* space — the 8-D HTRU2 DM=0 anchor is no longer the sole primary driver. Records without a real spectrogram keep `p_descriptor = 1` (neutral) and degrade gracefully to the HTRU2 path. `evaluate_ood` returns `htru2_pvals`, `descriptor_pvals`, `descriptor_fusion_active` for transparency.
- **`_snr_to_dmsnr` is now physically motivated (v2).** It replaces the old linear percentile interpolation with a saturating **radiometer-equation** law: folded-profile integrated S/N grows as √(detection S/N) and saturates, anchored so the survey's working detection significance (~10σ) maps to the cluster DM-SNR band median. See `axiom/dsp/features.py:_snr_to_dmsnr` (`_SNR_REFERENCE`, sqrt compression). It remains a *placement* on the survey manifold, not an absolute flux calibration.
- Anomaly verdict ≠ proof of artificial origin (README §07). The reports keep this scientific caveat.

## Conventions
- Python 3.8+, numpy docstring style, sklearn-first (no hard torch dependency).
- Public complexity APIs (`shannon_entropy_norm`, etc. in `c_bindings.py`) must keep a pure-Python fallback with identical signature.
- Density scoring is **per-class**: `score(x) = max_c log p(x|c)` (`axiom/ml/density.py`). Do not regress to a single joint GMM for OOD.
- Decision rule invariant enforced by `tests/test_features.py`: natural/RFI sources land on-manifold (`density >= natural_min - 5`); narrowband sources land out-of-manifold (`< natural_min - 5`). Preserve this when touching the feature map.
