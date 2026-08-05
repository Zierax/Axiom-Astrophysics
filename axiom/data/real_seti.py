"""Real SETI / Breakthrough Listen ingestion from the Kaggle release.

This module turns the **real, unlabeled** Breakthrough Listen observations
shipped in ``tentotheminus9/breakthrough-listen-search-for-advanced-life``
(Kaggle) into OOD records for the anomaly audit. Unlike the synthetic
narrowband-tone controls, these are genuine radio-astronomy spectrograms of
nearby stars, with measured signal-to-noise — used as anomaly controls to
verify the engine surfaces off-manifold telescope signals.

Every signal is placed on the HTRU2 survey manifold via
physics_map_htru2_features (real DM=0, measured S/N) so it flows through the
same per-class density estimator / conformal path as the rest of the audit, and
its raw waveform feeds the DSP complexity + chaos + CNN branches.

All functions are defensive: if the (large) archive is absent or its on-disk
layout differs, they return empty results and the caller falls back to the other
real sources (Voyager, BL candidates) or, last resort, synthetic tones.
"""
from __future__ import annotations

import logging
import os
import zipfile
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np

if TYPE_CHECKING:  # avoid circular import at runtime; used only for annotation
    from .real_loaders import RealOODManifest

log = logging.getLogger("axiom.data.real_seti")

_KAGGLE_DATASET = "tentotheminus9/breakthrough-listen-search-for-advanced-life"
SIGNAL_LEN = 256  # CNN/chaos branch expects 256-sample 1-D waveforms.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Prefer the compact real Breakthrough Listen release; fall back to the larger
# tentotheminus9/seti-data archive if it has been downloaded instead.
DEFAULT_LOCAL = os.path.join(
    _REPO, "data", "kaggle", "seti_advanced",
    "breakthrough-listen-search-for-advanced-life.zip",
)


def _find_archive(local: Optional[str] = None) -> Optional[str]:
    from .real_loaders import DEFAULT_CACHE

    candidates = [
        local,
        DEFAULT_LOCAL,
        os.path.join(_REPO, "data", "kaggle", "seti", "seti-data.zip"),
        os.path.join(DEFAULT_CACHE, "seti-data.zip"),
        os.path.join(DEFAULT_CACHE, "seti", "seti-data.zip"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    return None


def _unpack(archive: str, dest: str) -> str:
    if not os.path.isfile(archive):
        raise FileNotFoundError(archive)
    os.makedirs(dest, exist_ok=True)
    if not os.path.isdir(os.path.join(dest, "seti-data")):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    # Normalise: the zip may contain a top-level "seti-data/" folder or flat files.
    if os.path.isdir(os.path.join(dest, "seti-data")):
        return os.path.join(dest, "seti-data")
    return dest


def _as_waveform(arr: np.ndarray, return_spec: bool = False) -> Tuple[np.ndarray, float, Optional[np.ndarray]]:
    """Reduce an arbitrary real array to a 1-D waveform + measured S/N.

    Accepts 1-D time series, 2-D waterfalls (freq x time) and 3-D cadences.
    For spectrograms we isolate the frequency channel with the largest temporal
    variance (the narrowband hit) and take its time series; S/N is
    peak-over-background of that channel. When ``return_spec`` is True the
    full 2-D spectrogram (freq x time) is also returned for native
    frequency-resolved feature extraction.
    """
    from .real_loaders import _resample

    arr = np.asarray(arr, dtype=np.float64)
    # Collapse degenerate leading/batch dimensions (e.g. GUPPI (16, 1, T) cubes).
    arr = np.squeeze(arr)
    if arr.ndim >= 3:
        # Keep the two largest trailing dims as (channels, time).
        arr = arr.reshape(arr.shape[-2], arr.shape[-1])
    if arr.ndim == 2:
        var = arr.var(axis=1)
        hit = int(np.argmax(var)) if var.size else 0
        ts = arr[hit]
        spec = arr
    else:
        ts = arr
        # 1-D arrays carry no frequency axis; a single-channel spectrogram is valid.
        spec = arr.reshape(1, -1)
    bg = float(np.median(ts))
    noise = max(float(np.std(ts - bg)), 1e-9)
    snr = float(np.clip((float(ts.max()) - bg) / noise, 1.0, 1000.0))
    return _resample(ts, SIGNAL_LEN), snr, spec


def _load_array(path: str) -> np.ndarray:
    """Read a signal file into a numpy array, format-dispatch.

    Supports ``.npy`` arrays, HDF5 (Breakthrough Listen GUPPI ``.gpuspec`` /
    ``.h5`` spectrograms — the ``data`` dataset is used, never the ``mask``),
    and ``.png`` spectrogram images. Raises on unsupported/opaque files.
    """
    low = path.lower()
    if low.endswith(".npy"):
        return np.load(path)
    if low.endswith((".h5", ".hdf5")):
        import h5py  # local import: optional dependency

        with h5py.File(path, "r") as hf:

            def _first_data(grp):
                for key in grp.keys():
                    obj = grp[key]
                    if isinstance(obj, h5py.Dataset) and key.lower() != "mask":
                        return np.asarray(obj)
                for key in grp.keys():  # fall back to any dataset
                    obj = grp[key]
                    if isinstance(obj, h5py.Dataset):
                        return np.asarray(obj)
                raise ValueError("no dataset in h5 file")

            return _first_data(hf)
    if low.endswith(".png"):
        from PIL import Image  # local import: optional dependency

        return np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    raise ValueError(f"unsupported signal extension: {path}")


def parse_seti_archive(archive: Optional[str] = None,
                        dest: Optional[str] = None
                        ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract real signals from the Kaggle SETI / Breakthrough Listen archive.

    Handles the real Breakthrough Listen GUPPI ``.gpuspec.h5`` spectrograms
    (ON/OFF observations of nearby stars) as well as ``.npy`` / ``.png``
    waterfalls. Returns a mapping ``name -> (wave, spec)`` for every parseable
    signal file, where ``wave`` is the 1-D (256-sample) waveform used by the
    CNN/chaos branch and ``spec`` is the **2-D (nchan x ntime) spectrogram** used
    by the native frequency-resolved feature extractor. Preserving ``spec`` is
    essential: flattening to 1-D silently collapses ``compute_waterfall_features``
    to all-zero descriptors. Robust to format: opaque/unreadable files are
    skipped, not raised.
    """
    archive = _find_archive(archive)
    if archive is None:
        log.warning("[real_seti] archive not found; skipping real SETI ingestion.")
        return {}
    dest = dest or os.path.join(os.path.dirname(archive), "_extracted")
    try:
        root = _unpack(archive, dest)
    except (FileNotFoundError, zipfile.BadZipFile, OSError) as exc:
        log.warning("[real_seti] could not open archive %s: %s", archive, exc)
        return {}

    signals: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    exts = (".npy", ".h5", ".hdf5", ".png")
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if not fn.lower().endswith(exts):
                continue
            path = os.path.join(dirpath, fn)
            try:
                arr = _load_array(path)
            except Exception as exc:  # pragma: no cover - opaque binary
                log.debug("[real_seti] skip %s: %s", fn, exc)
                continue
            try:
                wave, _, spec = _as_waveform(arr, return_spec=True)
                rel = os.path.relpath(path, root).replace(os.sep, "__")
                signals[rel] = (wave, spec)
            except Exception as exc:  # pragma: no cover - malformed array
                log.debug("[real_seti] reshape skip %s: %s", fn, exc)
    log.info("[real_seti] parsed %d real signals from archive.", len(signals))
    return signals


def build_real_seti_ood(n_anom: int = 25, archive: Optional[str] = None
                        ) -> Tuple[list, Dict[str, np.ndarray], Dict[str, dict], Optional[RealOODManifest]]:
    """Build real technosignature OOD records from the Kaggle SETI archive.

    Returns (records, real_waves, waterfall_features, manifest). records is a
    list of (name, display_type, role, dm, snr, verdict) tuples with
    role == "Anomaly" for the genuine candidate signals (measured S/N, DM=0),
    so they are scored by the same manifold/density path as synthetic carriers
    but are *real* observations. real_waves maps each name to its 1-D waveform
    for the DSP/chaos/CNN branch; waterfall_features maps each name to its
    native frequency-resolved spectrogram descriptors (occupancy, drift,
    spectral kurtosis, ...), computed directly from the 2-D GUPPI spectrogram.

    Empty outputs are returned (with manifest.retrieved_ok = False) if the
    archive is unavailable, so callers can fall back gracefully.
    """
    from axiom.dsp.waterfall_features import compute_waterfall_features

    from .real_loaders import RealOODManifest

    signals = parse_seti_archive(_find_archive(archive))
    if not signals:
        return [], {}, {}, RealOODManifest(
            source=("Kaggle tentotheminus9/breakthrough-listen-search-for-advanced-life "
                    "(real Breakthrough Listen)"),
            doi="", url=("https://www.kaggle.com/datasets/tentotheminus9/"
                         "breakthrough-listen-search-for-advanced-life"),
            retrieved_ok=False, n_real_frb=0,
            notes="archive absent or unparseable; caller should fall back.")

    names = list(signals.keys())
    rng = np.random.default_rng(42)
    if len(names) > n_anom:
        names = list(rng.choice(names, size=n_anom, replace=False))

    records, real_waves, waterfall_features = [], {}, {}
    for name in names:
        wave, spec = signals[name]
        # Recompute the measured S/N from the stored waveform so the record stays
        # consistent with the 1-D branch; the 2-D spec is passed untouched to the
        # native frequency-resolved feature extractor (never re-flattened).
        _, snr, _ = _as_waveform(wave)
        display = "SETI-Candidate"
        # Prefix with SETI__ so downstream counting/notes can identify genuine
        # Breakthrough Listen observations unambiguously.
        key = f"SETI__{name}"
        rec = (key, display, "Narrowband", 0.0, float(snr), "Anomaly")
        records.append(rec)
        real_waves[key] = wave
        try:
            waterfall_features[key] = compute_waterfall_features(spec)
        except Exception as exc:  # pragma: no cover - defensive
            log.debug("[real_seti] waterfall features failed for %s: %s", name, exc)
            waterfall_features[key] = {}

    man = RealOODManifest(
        source=("Real Breakthrough Listen observations (Kaggle "
                "tentotheminus9/breakthrough-listen-search-for-advanced-life): "
                "nearby-star GUPPI spectrograms with measured S/N, placed on the "
                "HTRU2 manifold (DM=0); their native spectrogram morphology is "
                "characterised independently by frequency-resolved descriptors."),
        doi="", url=("https://www.kaggle.com/datasets/tentotheminus9/"
                     "breakthrough-listen-search-for-advanced-life"),
        retrieved_ok=True, n_real_frb=len(records),
        notes=f"{len(records)} real candidate signals (no synthetic tones).")
    return records, real_waves, waterfall_features, man


if __name__ == "__main__":
    recs, waves, m = build_real_seti_ood()
    print("manifest:", m.to_dict())
    print("records:", len(recs), "waves:", len(waves))
