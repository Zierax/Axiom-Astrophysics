# axiom-astrophysics — Quick Start (v2)

Multi-layer cosmic signal audit engine. The two review-era limitations were
**fixed in code** (not just documented) for v2:

- **Native frequency-resolved verdicts.** Real telescope waterfalls are
  characterised directly by `axiom.dsp.waterfall_features` and feed a
  **descriptor-conformal p-value** that is Bonferroni-fused with the HTRU2
  conformal p-value inside `evaluate_ood`. A real signal is flagged when its
  *measured* morphology is off-manifold in either space.
- **Physical SNR→DM-SNR placement.** `_snr_to_dmsnr` now uses a saturating
  radiometer-equation law (√S/N compression) instead of a linear percentile map.

See `README.md` (§06) and `AGENTS.md` for the full methodology and caveats.

---

## 1. Install

```bash
make setup
# equivalent to:
#   pip install -r requirements.txt
#   pip install -e ".[test,waterfall]"
```

`requirements.txt` and `pyproject.toml` already include `pyyaml`, `blimpy`,
`h5py`, `Pillow`, `numpy`, `scikit-learn`, etc. No C compiler is required:
the optional `axiom/lib/axiom_core.so` accelerates DSP but the code falls back
to pure Python automatically.

```bash
# optional: build the C accelerator (needs gcc)
bash axiom/lib/recompile.sh
```

---

## 2. Run the production pipeline

Trains the ensemble + per-class density estimator on HTRU2 and runs the
real-world OOD anomaly audit (real Voyager 1 carrier, Breakthrough Listen
GUPPI spectrograms, CHIME/FRB, HTRU2 RFI). Offline-only components fall
back to seeded synthetic controls, so the pipeline always runs.

```bash
python3 run_axiom.py                       # uses configs/pipeline_config.yaml
python3 run_axiom.py --config configs/pipeline_config.yaml
```

First run downloads HTRU2 (~1.5 MB) and caches it under `data/` (git-ignored).
Real OOD sources download on demand and are checksum-verified; disable with
`AXIOM_REAL_OOD=0` for a fast, network-light run.

---

## 3. Run the validation suite

```bash
make benchmark          # python3 benchmark.py
```

Seven suites (deterministic, seed 42):

1. In-distribution 5-fold CV on HTRU2
2. Ablation study
3. Out-of-distribution anomaly detection (real controls)
4. Baseline comparison vs sklearn classifiers
5. McNemar significance test
6. Real-waterfall manifold OOD (Lane 1)
7. Population-scale catalog manifold OOD (Lane 2)

Expected result (with real data cached): Suites 1, 3, 6, 7 **PASS**.
The production pipeline (`run_axiom.py`) reports **100% TPR / 0% FPR** on the real
anomaly controls (Voyager 1 detected, no false positives). The benchmark Suite 3
uses a different evaluation protocol — see `BENCHMARKS.md` for honest numbers.
The final verdict is `validated = True`.

---

## 4. Run the tests

```bash
make test               # python3 -m pytest tests/ -v
```

Real-waterfall / Voyager tests self-skip when the large pinned filterbanks are
not cached. HTRU2 auto-downloads once for the in-distribution tests.

---

## 5. Generate the research-grade report

```bash
make report            # python3 scripts/generate_reports.py
```

Regenerates 22 scientific figures under `benchmarks/charts/` and the markdown
+ JSON reports under `benchmarks/reports/` (verdict summary, per-suite reports,
`methodology.md`).

```bash
make all              # test + benchmark + report
```

---

## 6. Configuration

`configs/pipeline_config.yaml` is dot-path-overridable. Run:

```bash
python3 run_axiom.py --set data.cache_dir=/tmp/axiom_cache
```

All randomness flows from `split_seed=42` / `random_state=42` — keep it
deterministic; do not introduce unseeded randomness into reported metrics.

---

## 7. Repository map

- `axiom/` — package
  - `data/` — HTRU2 load/cache/catalogs, real-OOD ingestion
  - `dsp/` — physical features, synthesis, waterfall + native descriptors, C bindings
  - `ml/` — ensemble, per-class density, CNN
  - `stats/` — calibration, arbitration (composite + FDR), manifold OOD, chaos
- `run_axiom.py` — production pipeline (train + OOD audit)
- `benchmark.py` — validation suite (7 suites)
- `scripts/` — reporting + plots
- `tests/` — pytest
- `configs/pipeline_config.yaml` — overridable config

Git-ignored (regenerated, never committed): `data/`, `benchmarks/`, `*.pkl`,
`*.so`, `*.csv`, `*.h5`.

---

## 8. Lint

```bash
ruff check axiom scripts benchmark.py run_axiom.py
```

Checks E9/F/B/I. CI gates on it (non-blocking style, blocking on errors).
There is no mypy/typecheck configured.

---

*axiom-astrophysics v2 — real data, self-consistent manifolds, honest verdicts.*
