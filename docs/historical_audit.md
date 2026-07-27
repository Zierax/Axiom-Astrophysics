# Historical Anomaly Audit — Methodology & How to Run

The historical audit answers: *"What are the weirdest signals in the
observational record, and can we rank them reproducibly?"* It ships as committed,
runnable code so anyone can reproduce the result.

- Logic: `axiom/historical/__init__.py`
- CLI: `scripts/historical_audit.py`
- Make target: `make historical`

## Quick start

```bash
pip install numpy scipy scikit-learn joblib pandas   # core deps
python3 scripts/historical_audit.py                  # both lanes
python3 scripts/historical_audit.py --curated-only   # famous signals only
python3 scripts/historical_audit.py --report benchmarks/reports/historical_audit.md
```

First run fetches ATNF (`B/psr/psr`) and CHIME/FRB Catalog 1
(`J/ApJS/257/59/table2`) via VizieR and caches them under
`data/historical_cache/`. Subsequent runs are offline. The CLI also writes a
markdown report if `--report` is given.

## Two lanes

### Lane A — Curated historical signals (illustrative)
Scores the 25 famous signals in `data/historical_anomalies.csv` (Wow!, BLC1, FRB
121102, perytons, Voyager/Pioneer telemetry, …) through the HTRU2 per-class
density model. **Honesty note:** several rows (narrowband tones, human telemetry)
carry *placeholder* HTRU2-moment features and are not measured from raw data.
This lane demonstrates where documented classes fall — a teaching/sanity-check
illustration only. Do not cite its numbers as measurements.

### Lane B — Real catalogs (measurement)
Fetches real catalogs and ranks objects by the **minimum per-class split-conformal
off-manifold p-value**:

1. Featurize each object on the commensurate physical manifold
   (`axiom.dsp.physical_features`): log DM, DM excess over the catalog's own
   Galactic model, Galactic latitude, pulse width, S/N, period, duty cycle, plus
   presence indicators. Fold-local median imputation (no leakage).
2. Fit a per-class GMM density on a 70% in-fold (stratified by class).
3. Score the held-out 30% calibration split; build per-class empirical CDFs of
   negative log-likelihood.
4. Map every object to a per-class conformal p-value
   `( #{cal in class with loglik <= this} + 1 ) / (n_cal + 1)`.
5. Off-manifold p-value = **minimum over classes** (flagged if rare under *any*
   natural population — a sensitive triage choice).

The top-ranked objects are the most statistically unusual members of the *known*
populations (extreme-DM pulsars, RRATs) — a prioritised candidate list for human
review, **not** a claim of artificial origin or a new discovery.

## Extending to other catalogs

`run_real_catalog_audit(keys=(...))` accepts any registered catalog keys from
`axiom.data.catalogs.REGISTRY`. To add a new real catalog, register a
`CatalogSpec` (VizieR id / columns / constraints) in `axiom/data/catalogs.py`
and normalise it to the common `SCHEMA`; it is then featurised and scored
automatically. Always verify the upstream content before passing `--repin` to
update the digest lock.

## Reproducibility

Fixed seed (`42`), deterministic featurizer, checksum-pinned catalogs
(`configs/catalog_locks.json`). Re-running yields identical rankings.
