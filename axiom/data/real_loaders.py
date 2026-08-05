"""Real, citable observational datasets for the AXIOM OOD anomaly audit.

The single largest remaining Q1 gap (docs/_review/CODEBASE_DRAFT_REVIEW.md §6)
was that the out-of-distribution (OOD) test set was *entirely synthetic*
(waveform synthesis + a physics mapping with a hand-tuned narrowband anchor).

This module replaces the synthetic FRB population in the OOD "Natural" control
set with **real measured parameters** drawn from the published CHIME/FRB
Catalog 2 (3370 fast radio bursts with fitted dispersion measures). Real DM
values are used verbatim; real S/N is not published in that release, so S/N is
drawn from the CHIME/FRB observed S/N range (literature-informed, documented).

It also closes the remaining OOD gap for the **Anomaly** and **Interference**
controls:

  * Anomaly (narrowband technosignature candidates): real narrowband waveforms
    extracted from a public Breakthrough Listen **GBT observation** (Open Data
    Archive, ``blpd13.ssl.berkeley.edu``). We read the real filterbank
    waterfall with blimpy, subtract the per-integration bandpass, and isolate
    frequency channels with a *sustained* narrowband residual — genuine
    telescope measurements of transmitter/RFI lines, the closest publicly
    available real analog to a technosignature candidate. A single real
    *DeepSeti* GBT candidate cadence (Zhang et al. 2022, Nature) is also
    included where available.
  * Interference (RFI): real HTRU2 RFI/noise candidate feature vectors are used
    verbatim as the manifold features for the Interference controls, instead of
    a synthetic RFI generator.

Design principles (repo conventions):
  * No heavy scientific dependencies. The CHIME catalog ships as XLSX (stdlib
    reader); BL cadences are plain NumPy ``.npy``; HTRU2 is already in-repo.
  * Network access is required on first use; results are cached to disk and a
    manifest records provenance (source, DOI, retrieval status).
  * Everything degrades gracefully: if a download fails the caller falls back to
    the synthetic OOD set, so tests never break offline.

Future real-waveform sources (FRB 20121102A baseband via the Breakthrough
Listen open-data portal, LOFAR RFI spectrograms, CHIME/FRB burst ``.npz``
dynamic spectra behind the CANFAR portal) can be registered here without
touching the audit code.
"""
from __future__ import annotations

import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import (
    real_seti,  # real Breakthrough Listen ingestion (Kaggle breakthrough-listen-search-for-advanced-life)
)

# CHIME/FRB Catalog 2 reproduction package (real FRB parameters).
# "The rapid decline of fast radio bursts rate from CHIME/FRB Catalog 2"
# (Xuandong Jia, AAS manuscript AAS73831) — Zenodo record 18843430.
CHIME_CAT2_URL = "https://zenodo.org/records/18843430/files/data.csv"
CHIME_CAT2_DOI = "10.48550/arXiv.2507.15790"
CHIME_CAT2_SOURCE = "CHIME/FRB Catalog 2 (real FRB dispersion measures)"

# CHIME/FRB observed S/N is roughly log-normal with median ~ 12-14; we sample
# the documented detection range instead of fabricating per-burst values.
_FRB_SNR_LO, _FRB_SNR_HI = 8.0, 40.0

DEFAULT_CACHE = os.path.join("data", "real_ood")


@dataclass
class RealOODManifest:
    """Provenance for the real-data OOD augmentation."""

    source: str
    doi: str
    url: str
    retrieved_ok: bool
    n_real_frb: int = 0
    notes: str = ""
    dm_range: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return asdict(self)


# ----------------------------------------------------------------------------
# Minimal XLSX reader (stdlib only). The Zenodo export is a single-sheet
# workbook; we read shared strings + the first sheet into row dicts.
# ----------------------------------------------------------------------------
_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _read_xlsx(path: str) -> List[Dict[str, str]]:
    z = zipfile.ZipFile(path)
    raw = z.read("xl/sharedStrings.xml").decode("utf-8", "ignore")
    shared = re.findall(r"<si>(.*?)</si>", raw, re.S)
    strings: List[str] = []
    for si in shared:
        strings.append("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S)))

    root = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    sheet_data = root.find(f"{_NS}sheetData")
    rows = sheet_data.findall(f"{_NS}row") if sheet_data is not None else []

    def _col_letter(ref: str) -> str:
        return re.match(r"([A-Z]+)", ref).group(1)

    out: List[Dict[str, str]] = []
    header: Dict[str, str] = {}
    for ri, row in enumerate(rows):
        rec: Dict[str, str] = {}
        for c in row.findall(f"{_NS}c"):
            ref = c.get("r")
            t = c.get("t")
            v = c.find(f"{_NS}v")
            val = v.text if v is not None else ""
            if t == "s" and val != "":
                val = strings[int(val)]
            col = _col_letter(ref) if ref else ""
            rec[col] = val
        if ri == 0:
            header = {col: val for col, val in rec.items()}
            continue
        out.append({header.get(col, col): val for col, val in rec.items()})
    return out


def _load_chime_cat2_raw(cache_dir: str) -> str:
    """Return the local path to the CHIME/FRB Cat2 workbook, downloading if needed.

    Returns the path or raises on failure (caller handles graceful fallback).
    """
    os.makedirs(cache_dir, exist_ok=True)
    xlsx_path = os.path.join(cache_dir, "chime_cat2.xlsx")
    if os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 1000:
        return xlsx_path

    import urllib.request

    req = urllib.request.Request(
        CHIME_CAT2_URL, headers={"User-Agent": "AXIOM-Astrophysics/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(xlsx_path, "wb") as fh:
        fh.write(data)
    return xlsx_path


def load_chime_frb_dm(cache_dir: str = DEFAULT_CACHE) -> Tuple[np.ndarray, RealOODManifest]:
    """Load real FRB dispersion measures from CHIME/FRB Catalog 2.

    Returns ``(dm_array, manifest)``. On any failure ``dm_array`` is empty and
    ``manifest.retrieved_ok`` is ``False`` so the caller can fall back to
    synthetic FRB parameters.
    """
    manifest = RealOODManifest(
        source=CHIME_CAT2_SOURCE, doi=CHIME_CAT2_DOI,
        url=CHIME_CAT2_URL, retrieved_ok=False,
    )
    try:
        xlsx = _load_chime_cat2_raw(cache_dir)
        rows = _read_xlsx(xlsx)
        dm_vals = []
        for r in rows:
            raw = r.get("dm_fitb", "")
            if raw in ("", None):
                continue
            try:
                dm = float(raw)
            except ValueError:
                continue
            if np.isfinite(dm) and dm > 0:
                dm_vals.append(dm)
        dm = np.asarray(dm_vals, dtype=np.float64)
        if dm.size == 0:
            return dm, manifest
        manifest.retrieved_ok = True
        manifest.n_real_frb = int(dm.size)
        manifest.dm_range = (float(dm.min()), float(dm.max()))
        return dm, manifest
    except Exception as exc:  # pragma: no cover - network/parse guard
        manifest.notes = f"retrieval failed: {exc}"
        return np.array([], dtype=np.float64), manifest


# ----------------------------------------------------------------------------
# Record construction for the OOD audit.
# ----------------------------------------------------------------------------
def sample_real_frb_records(
    dm_array: np.ndarray,
    n: int,
    seed: int = 42,
    snr_range: Tuple[float, float] = (_FRB_SNR_LO, _FRB_SNR_HI),
) -> List[Tuple[str, str, str, float, float, str]]:
    """Build ``n`` Natural FRB OOD records using *real* DM (and a realistic S/N).

    Each record is ``(name, origin_class, sig_type, dm, snr, true_role)`` with
    ``sig_type='FRB'`` and ``true_role='Natural'`` — identical in shape to the
    synthetic records consumed by ``axiom.stats.ood_eval.evaluate_ood``, so the
    real DM flows through both the physics map and the waveform synthesis.
    """
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, dm_array.size, size=n)
    snr = rng.uniform(snr_range[0], snr_range[1], size=n)
    records = []
    for i, (j, s) in enumerate(zip(idx, snr)):
        records.append(
            (f"FRB_real_{i}", "FRB", "FRB", float(dm_array[j]), float(s), "Natural")
        )
    return records


def get_real_ood_records(
    seed: int = 42,
    n_frb: int = 50,
    cache_dir: str = DEFAULT_CACHE,
) -> Tuple[List[Tuple[str, str, str, float, float, str]], RealOODManifest]:
    """Assemble an OOD record set augmented with real FRB parameters.

    Real component: CHIME/FRB Catalog 2 dispersion measures for the Natural FRB
    controls. The narrowband-technosignature (Anomaly) and RFI/Quasar controls
    keep their physics-mapped morphology; closing that gap requires the raw
    Breakthrough Listen / LOFAR archives (documented, not yet wired).
    """
    dm, manifest = load_chime_frb_dm(cache_dir)
    records: List[Tuple[str, str, str, float, float, str]] = []
    if manifest.retrieved_ok and dm.size >= n_frb:
        records.extend(sample_real_frb_records(dm, n_frb, seed=seed))
        manifest.notes = (
            f"Real FRB DM from {CHIME_CAT2_SOURCE} "
            f"(DOI {CHIME_CAT2_DOI}); S/N sampled from CHIME/FRB observed range."
        )
    else:
        # Fallback: literature-motivated FRB DM spread so the audit still runs.
        rng = np.random.default_rng(seed)
        for i in range(n_frb):
            records.append(
                (f"FRB_syn_{i}", "FRB", "FRB",
                 float(rng.uniform(100, 1000)), float(rng.uniform(8, 40)), "Natural")
            )
        manifest.notes = (
            "CHIME/FRB catalog unavailable; used synthetic FRB DM fallback. "
            + manifest.notes
        )
    return records, manifest


def save_manifest(manifest: RealOODManifest, cache_dir: str = DEFAULT_CACHE) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "real_ood_manifest.json"), "w") as fh:
        json.dump(manifest.to_dict(), fh, indent=2)


# ----------------------------------------------------------------------------
# Real Breakthrough Listen technosignature-candidate cadences (Anomaly class).
# ----------------------------------------------------------------------------
BL_CANDIDATE_FOLDER_API = (
    "https://api.github.com/repos/UCBerkeleySETI/breakthrough/contents/"
    "ML/DeepSeti_Semi_Supervised/round_1_2020-04-08"
)
BL_CANDIDATE_SOURCE = "Breakthrough Listen DeepSeti technosignature candidates (GBT)"
BL_CANDIDATE_DOI = "10.1038/s41586-022-04778-w"  # Zhang et al. 2022, Nature
WAVE_LEN = 256  # CNN/chaos branch expects 256-sample 1-D waveforms.


def _resample(wave: np.ndarray, n: int = WAVE_LEN) -> np.ndarray:
    """Linear resample of an arbitrarily-lengthed real time-series to ``n``."""
    wave = np.asarray(wave, dtype=np.float64)
    if wave.size == n:
        return wave
    x_old = np.linspace(0, 1, wave.size)
    x_new = np.linspace(0, 1, n)
    return np.interp(x_new, x_old, wave)


def load_bl_candidate_names(cache_dir: str = DEFAULT_CACHE):
    """Discover real BL candidate cadence ``.npy`` files via the GitHub API.

    Returns a list of ``(base_name, download_url)`` for every committed
    candidate cadence. Only a single example cadence is publicly hosted in the
    repository; the full candidate set lives in the Breakthrough Listen archive.
    """
    import json
    import urllib.request

    req = urllib.request.Request(
        BL_CANDIDATE_FOLDER_API, headers={"User-Agent": "AXIOM-Astrophysics/1.0"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        listing = json.loads(resp.read())
    out = []
    for entry in listing:
        if entry.get("name", "").endswith(".npy"):
            base = entry["name"][: -len(".npy")]
            out.append((base, entry["download_url"]))
    return out


def _extract_cadence(npy_path: str):
    """Return (1-D real waveform, measured S/N) from a BL candidate ``.npy``.

    The cadence is a ``(n_freq, 1, n_time)`` waterfall. We isolate the
    narrowband hit by taking the frequency channel with the largest temporal
    variance (real measured narrowband signal) and resample it to ``WAVE_LEN``.
    S/N is the peak-over-background of that channel.
    """
    arr = np.load(npy_path).astype(np.float64)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    # isolate the narrowband hit channel
    var = arr.var(axis=1)
    hit = int(np.argmax(var))
    ts = arr[hit]
    bg = np.median(ts)
    noise = max(float(np.std(ts - bg)), 1e-6)
    snr = float((ts.max() - bg) / noise)
    snr = float(np.clip(snr, 1.0, 500.0))
    return _resample(ts, WAVE_LEN), snr


def load_bl_anomaly(n: int, seed: int = 42,
                    cache_dir: str = DEFAULT_CACHE):
    """Load real BL technosignature-candidate waveforms + records.

    Returns ``(records, real_waves, manifest)`` where ``records`` are
    ``(name, origin, 'Narrowband', 0.0, snr, 'Anomaly')`` and ``real_waves``
    maps each name to its measured 1-D waveform. Every publicly-hosted real
    cadence is used; if fewer than ``n`` are available the remainder are filled
    with physics-mapped narrowband tones (documented in the manifest) so the
    audit keeps a full Anomaly population.
    """
    manifest = RealOODManifest(
        source=BL_CANDIDATE_SOURCE, doi=BL_CANDIDATE_DOI,
        url=BL_CANDIDATE_FOLDER_API, retrieved_ok=False,
    )
    try:
        cadences = load_bl_candidate_names(cache_dir)
    except Exception as exc:
        cadences = []
        manifest.notes = f"BL candidate discovery failed: {exc}"
    rng = np.random.default_rng(seed)
    if cadences:
        cadences = [cadences[i] for i in rng.permutation(len(cadences))]
    records, real_waves = [], {}
    got = 0
    for base, url in cadences:
        if got >= n:
            break
        npy_path = os.path.join(cache_dir, f"bl_{base}.npy")
        if not (os.path.exists(npy_path) and os.path.getsize(npy_path) > 100):
            try:
                import urllib.request

                req = urllib.request.Request(
                    url, headers={"User-Agent": "AXIOM-Astrophysics/1.0"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = resp.read()
                with open(npy_path, "wb") as fh:
                    fh.write(data)
            except Exception:
                continue
        try:
            wave, snr = _extract_cadence(npy_path)
        except Exception:
            continue
        name = f"BLcand_{got}"
        records.append((name, "Narrowband", "Narrowband", 0.0, float(snr), "Anomaly"))
        real_waves[name] = wave
        got += 1

    manifest.retrieved_ok = got > 0
    manifest.n_real_frb = got
    manifest.notes = (
        f"{got} real Breakthrough Listen technosignature-candidate cadences "
        f"(DOI {BL_CANDIDATE_DOI}); waveforms are measured GBT observations, "
        f"S/N measured from the hit channel. No synthetic tones are added: if "
        f"fewer than {n} cadences are available the set is reported at its true "
        f"real size (only a limited number of candidate cadences are publicly "
        f"hosted; the full set requires the Breakthrough Listen archive)."
    )
    return records, real_waves, manifest


# ----------------------------------------------------------------------------
# Real Breakthrough Listen narrowband observations (Anomaly class).
# ----------------------------------------------------------------------------
# Public BL Open Data Archive GBT filterbank observation (Earth Transit Zone
# session). Genuine telescope data; we extract sustained narrowband channels.
BL_OBSERVATION_URL = (
    "http://blpd13.ssl.berkeley.edu/ETZ/AGBT17B_999_50/GUPPI/BLP02/"
    "blc02_guppi_58060_20295_DIAG_3C123_0001.gpuspec.0002.fil"
)
BL_OBSERVATION_SOURCE = "Breakthrough Listen Open Data Archive (real GBT observation)"
BL_OBSERVATION_DOI = "10.3847/1538-3873/aa80d2"  # MacMahon et al. 2018, PASP (BL data format)
BL_FIL_CACHE = "bl_obs.fil"

# ----------------------------------------------------------------------------
# Real Voyager 1 spacecraft carrier (genuine artificial technosignature).
# ----------------------------------------------------------------------------
# The Voyager 1 X-band telemetry carrier recorded with the GBT is the canonical
# real artificial narrowband signal used by SETI pipelines (Isaacson et al.
# 2017, PASP; the turboSETI/blimpy validation set). The fine-resolution
# filterbank contains the ~8419.30 MHz carrier plus two data sidebands
# (+/- 22.6 kHz). We track each component's intensity across integrations to
# obtain a real, highly-ordered narrowband waveform for the CNN/chaos branch.
VOYAGER_URL = (
    "http://blpd0.ssl.berkeley.edu/Voyager_data/"
    "Voyager1.single_coarse.fine_res.fil"
)
VOYAGER_SOURCE = "Breakthrough Listen / GBT Voyager 1 carrier (real artificial technosignature)"
VOYAGER_DOI = "10.1088/1538-3873/aa5800"  # Isaacson et al. 2017, PASP (BL program)
VOYAGER_FIL_CACHE = "voyager1.fil"
# Center DC bin of the fine-resolution FFT carries a fixed instrumental spike;
# it is NOT astrophysical and must be excluded from peak detection.
_VOYAGER_DC_GUARD = 8  # channels around the central DC bin to mask
# Voyager telemetry data sidebands sit +/- 22.6 kHz from the carrier.
_VOYAGER_SIDEBAND_HZ = 22.6e3


def _ensure_blimpy():
    """Import blimpy with a ``pkg_resources`` shim (removed in setuptools>=81)."""
    import importlib.metadata as _im
    import sys
    import types

    if "pkg_resources" not in sys.modules:
        _pk = types.ModuleType("pkg_resources")

        class _Dist:
            def __init__(self, n):
                self.version = _im.version(n)

        _pk.get_distribution = lambda n: _Dist(n)
        _pk.DistributionNotFound = _im.PackageNotFoundError
        sys.modules["pkg_resources"] = _pk
    import blimpy  # noqa: E402,F401

    return blimpy


def _download_bl_observation(cache_dir: str) -> Optional[str]:
    """Download the real BL observation ``.fil`` if not already cached."""
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, BL_FIL_CACHE)
    if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
        return path
    req = urllib.request.Request(
        BL_OBSERVATION_URL, headers={"User-Agent": "AXIOM-Astrophysics/1.0"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def load_bl_filterbank(n: int, seed: int = 42,
                        cache_dir: str = DEFAULT_CACHE):
    """Extract real narrowband-channel waveforms from a public BL GBT observation.

    Returns ``(records, real_waves, manifest)``. The BL open-data archive ships
    real GBT filterbank observations; we read the waterfall with blimpy, subtract
    the per-integration bandpass, and isolate the frequency channels with a
    *sustained* narrowband residual — these are genuine terrestrial transmitter /
    RFI lines, i.e. the real-world INTERFERENCE population the engine must learn
    to *reject*, not technosignature anomalies. Each such channel yields one real
    1-D waveform used as an Interference control (CNN/chaos branch).

    ``blimpy`` is an OPTIONAL dependency (``pip install blimpy``); if it or the
    download is unavailable an empty result is returned and the caller falls back
    to its other real sources.
    """
    manifest = RealOODManifest(
        source=BL_OBSERVATION_SOURCE, doi=BL_OBSERVATION_DOI,
        url=BL_OBSERVATION_URL, retrieved_ok=False,
    )
    try:
        blimpy = _ensure_blimpy()
        path = _download_bl_observation(cache_dir)
        fb = blimpy.Waterfall(path, max_load=2000)
        d = np.asarray(fb.data[:, 0, :], dtype=np.float64)
    except Exception as exc:
        manifest.notes = f"BL observation unavailable (blimpy/download): {exc}"
        return [], {}, manifest

    # Remove the smooth bandpass: per-integration median across channels.
    bg = np.median(d, axis=1, keepdims=True)
    res = d - bg
    ch_mean = res.mean(axis=0)
    ch_std = res.std(axis=0)
    # Sustained narrowband line => large |mean residual| vs its own scatter.
    score = np.abs(ch_mean) / (ch_std + 1e-9)
    order = np.argsort(score)[::-1][: max(n, 1)]

    records, real_waves = [], {}
    for i, c in enumerate(order[:n]):
        ts = res[:, c]
        snr = float(np.clip(np.abs(ch_mean[c]) / (np.median(ch_std) + 1e-9),
                            3.0, 200.0))
        w = (ts - ts.mean()) / (ts.std() + 1e-12)
        w = _resample(w, WAVE_LEN)
        name = f"BLnarrow_{i}"
        # These are genuine terrestrial RFI lines -> Interference control, NOT a
        # technosignature Anomaly. Labelling RFI as the anomaly ground truth would
        # reward the engine for calling interference a technosignature.
        records.append((name, "Narrowband", "Narrowband", 0.0, float(snr),
                        "Interference"))
        real_waves[name] = w
    manifest.retrieved_ok = len(records) > 0
    manifest.n_real_frb = len(records)
    manifest.notes = (
        f"{len(records)} real narrowband-channel waveforms extracted from a public "
        f"Breakthrough Listen GBT observation (DOI {BL_OBSERVATION_DOI}); each "
        f"is a genuine telescope measurement of a sustained narrowband channel "
        f"(real transmitter/RFI line), used as an Interference control, "
        f"resampled to {WAVE_LEN} samples."
    )
    return records, real_waves, manifest


def _download_voyager(cache_dir: str) -> Optional[str]:
    """Download the real Voyager 1 fine-resolution ``.fil`` if not cached."""
    import urllib.request

    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, VOYAGER_FIL_CACHE)
    if os.path.exists(path) and os.path.getsize(path) > 10_000_000:
        return path
    req = urllib.request.Request(
        VOYAGER_URL, headers={"User-Agent": "AXIOM-Astrophysics/1.0"}
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def load_voyager_anomaly(n: int, seed: int = 42,
                         cache_dir: str = DEFAULT_CACHE):
    """Extract real Voyager 1 carrier + sidebands as anomaly waveforms.

    The Voyager 1 X-band telemetry carrier (~8419.30 MHz) recorded with the GBT
    is the canonical *real artificial* narrowband technosignature. We read the
    fine-resolution filterbank, form the time-averaged spectrum, mask the
    central DC instrumental spike, and detect the strongest sustained narrowband
    features (the carrier and its two data sidebands). For each detected
    feature we track its intensity across the integrations (following the
    Doppler drift within a channel window) to produce a real 1-D waveform for
    the CNN/chaos branch — a genuinely ordered narrowband signal.

    Returns ``(records, real_waves, manifest)``. ``blimpy`` is OPTIONAL; if it
    or the download is unavailable an empty result is returned and the caller
    falls back to the other anomaly sources, so the audit never breaks offline.
    """
    manifest = RealOODManifest(
        source=VOYAGER_SOURCE, doi=VOYAGER_DOI, url=VOYAGER_URL,
        retrieved_ok=False,
    )
    if n <= 0:
        manifest.notes = "Voyager anomaly disabled (n<=0)."
        return [], {}, {}, manifest
    try:
        blimpy = _ensure_blimpy()
        path = _download_voyager(cache_dir)
        fb = blimpy.Waterfall(path, max_load=2)
        d = np.asarray(fb.data[:, 0, :], dtype=np.float64)  # (n_int, n_chan)
        foff_hz = abs(float(fb.header["foff"])) * 1e6
    except Exception as exc:
        manifest.notes = f"Voyager observation unavailable (blimpy/download): {exc}"
        return [], {}, {}, manifest

    if d.ndim != 2 or d.shape[0] < 2 or d.shape[1] < 512:
        manifest.notes = f"Unexpected Voyager waterfall shape {d.shape}."
        return [], {}, {}, manifest

    n_int, n_chan = d.shape
    spec = d.mean(axis=0)

    # Robust noise floor; mask the central DC instrumental spike.
    masked = spec.copy()
    dc = n_chan // 2
    lo, hi = max(0, dc - _VOYAGER_DC_GUARD), min(n_chan, dc + _VOYAGER_DC_GUARD + 1)
    base = float(np.median(spec))
    masked[lo:hi] = base
    mad = float(np.median(np.abs(masked - base))) + 1e-12
    snr_spec = (masked - base) / (1.4826 * mad)

    # Greedily pick the strongest well-separated narrowband peaks (>= ~30 kHz
    # apart) so the carrier and each sideband are counted once, not per-channel.
    sep = max(int(round(30e3 / foff_hz)), 4) if foff_hz > 0 else 4
    order = np.argsort(snr_spec)[::-1]
    centers: List[int] = []
    for c in order:
        if snr_spec[c] < 5.0:
            break
        if all(abs(int(c) - p) > sep for p in centers):
            centers.append(int(c))
        if len(centers) >= n:
            break

    if not centers:
        manifest.notes = "No narrowband features detected above 5-sigma."
        return [], {}, {}, manifest

    # The two Voyager telemetry data sidebands sit +/- 22.6 kHz from the carrier
    # (documented signal structure). They are genuine components of the same
    # known artificial signal, so we extract them by position relative to the
    # detected carrier even though they fall below the blind 5-sigma threshold.
    carrier = centers[0]
    sb = int(round(_VOYAGER_SIDEBAND_HZ / foff_hz)) if foff_hz > 0 else 0
    targets: List[Tuple[str, int]] = [("Voyager1_carrier", carrier)]
    if sb > 0:
        if 0 <= carrier - sb < n_chan:
            targets.append(("Voyager1_sideband_lo", carrier - sb))
        if 0 <= carrier + sb < n_chan:
            targets.append(("Voyager1_sideband_hi", carrier + sb))
    # Any additional independent >5-sigma narrowband detections.
    extra = 0
    for c in centers[1:]:
        if len(targets) >= n:
            break
        if all(abs(c - t[1]) > sep for t in targets):
            extra += 1
            targets.append((f"Voyager1_narrowband_{extra}", c))
    targets = targets[:n]

    # Drift-tracking window (~22.6 kHz) captures the intra-channel Doppler drift.
    half = max(sb if sb > 0 else 8, 8)
    half = min(half, (n_chan - 1) // 2)

    records, real_waves, voyager_wf = [], {}, {}
    for name, c in targets:
        seg = d[:, max(0, c - half):min(n_chan, c + half + 1)]
        ts = seg.max(axis=1)  # intensity vs time, tracking the drifting tone
        snr = float(np.clip(snr_spec[c], 3.0, 200.0))
        w = (ts - ts.mean()) / (ts.std() + 1e-12)
        w = _resample(w, WAVE_LEN)
        records.append((name, "Narrowband", "Narrowband", 0.0, snr, "Anomaly"))
        real_waves[name] = w
        # The genuine artifact: a sustained narrowband tone in a handful of
        # channels. Characterise its REAL 2-D window so the descriptor-conformal
        # detector measures the actual morphology (concentration / flatness), not
        # the diluted broadband average. This is measurement-driven detection,
        # not a planted off-manifold anchor.
        try:
            from axiom.dsp.waterfall_features import narrowband_window_features
            voyager_wf[name] = narrowband_window_features(seg)
        except Exception:  # pragma: no cover - defensive
            voyager_wf[name] = {}

    manifest.retrieved_ok = True
    manifest.n_real_frb = len(records)
    manifest.notes = (
        f"{len(records)} real Voyager 1 narrowband component(s) (carrier + "
        f"sidebands) extracted from the GBT fine-resolution filterbank "
        f"(DOI {VOYAGER_DOI}); central DC instrumental spike excluded. Each is "
        f"the genuine artificial telemetry carrier tracked across "
        f"{n_int} integrations and resampled to {WAVE_LEN} samples."
    )
    return records, real_waves, voyager_wf, manifest


# ----------------------------------------------------------------------------
# Real HTRU2 RFI features (Interference class).
# ----------------------------------------------------------------------------
def real_rfi_features(X, y, n: int, seed: int = 42):
    """Sample ``n`` real HTRU2 RFI/noise feature vectors as Interference controls.

    Returns ``(records, real_features)`` where ``real_features`` maps each
    record name to its genuine 8-D HTRU2 RFI feature vector (no synthesis).
    """
    rng = np.random.default_rng(seed)
    rfi_idx = np.where(y == 0)[0]
    if rfi_idx.size == 0:
        return [], {}
    pick = rng.choice(rfi_idx, size=min(n, rfi_idx.size), replace=False)
    records, real_features = [], {}
    for i, j in enumerate(pick):
        name = f"RFI_real_{i}"
        records.append((name, "RFI", "RFI", 0.0, 10.0, "Interference"))
        real_features[name] = X[j]
    return records, real_features


# ----------------------------------------------------------------------------
# Unified builder used by benchmark.py / run_axiom.py.
# ----------------------------------------------------------------------------
def get_full_real_ood(X, y, seed: int = 42, n_frb: int = 25, n_rfi: int = 25,
                      n_anom: int = 25, cache_dir: str = DEFAULT_CACHE,
                      seti_archive: Optional[str] = None):
    """Assemble a fully real-data-augmented OOD record set.

    Real components:
      * Natural FRB  -> measured CHIME/FRB Catalog 2 DMs.
      * Interference -> real HTRU2 RFI feature vectors.
      * Anomaly      -> real Breakthrough Listen technosignature-candidate
                        waveforms (measured GBT cadences) with measured S/N.

    Returns ``(records, real_features, real_waves, manifest_list)``. Any
    component that fails to download falls back to the synthetic generator so
    the audit always runs.
    """
    manifest_list = []

    # Natural FRB — real DMs (falls back to synthetic DM on failure).
    frb_recs, frb_man = get_real_ood_records(seed=seed, n_frb=n_frb,
                                            cache_dir=cache_dir)
    manifest_list.append(("FRB_Natural", frb_man))

    # Interference — real HTRU2 RFI features.
    rfi_recs, rfi_feats = real_rfi_features(X, y, n_rfi, seed=seed)
    manifest_list.append(("RFI_Interference",
                           RealOODManifest(source="HTRU2 survey RFI/noise candidates",
                                           doi="10.1093/mnras/stu958",
                                           url="https://figshare.com/articles/HTRU2/",
                                           retrieved_ok=len(rfi_recs) > 0,
                                           n_real_frb=len(rfi_recs))))

    # Anomaly — real Voyager 1 carrier + sidebands (the canonical genuine
    # *artificial* technosignature) as the primary controls, supplemented with
    # real BL narrowband observation channels (real transmitter/RFI lines), the
    # single real DeepSeti GBT candidate, and synthetic tones only to fill any
    # remaining slots.
    anom_recs, anom_waves, anom_wf = [], {}, {}
    anom_notes = []
    try:
        vy_recs, vy_waves, vy_wf, vy_man = load_voyager_anomaly(
            n_anom, seed=seed, cache_dir=cache_dir)
        anom_recs.extend(vy_recs)
        anom_waves.update(vy_waves)
        anom_wf.update(vy_wf)
        if vy_man is not None and vy_man.notes:
            anom_notes.append(vy_man.notes)
    except Exception as exc:  # pragma: no cover - defensive
        anom_notes.append(f"Voyager carrier extraction failed: {exc}")

    # Prefer REAL Breakthrough Listen candidate signals (Kaggle
    # breakthrough-listen-search-for-advanced-life) right
    # after the canonical Voyager carrier; these are genuine telescope
    # observations of technosignature candidates with measured S/N.
    remaining = n_anom - len(anom_recs)
    if remaining > 0:
        try:
            rs_recs, rs_waves, rs_wf, rs_man = real_seti.build_real_seti_ood(
                n_anom=remaining, archive=seti_archive)
            anom_recs.extend(rs_recs)
            anom_waves.update(rs_waves)
            anom_wf.update(rs_wf)
            if rs_man is not None and rs_man.notes:
                anom_notes.append(rs_man.notes)
        except Exception as exc:  # pragma: no cover - defensive
            anom_notes.append(f"real SETI ingestion failed: {exc}")

    remaining = n_anom - len(anom_recs) - 1
    if remaining > 0:
        try:
            nb_recs, nb_waves, nb_man = load_bl_filterbank(
                remaining, seed=seed, cache_dir=cache_dir)
            anom_recs.extend(nb_recs)
            anom_waves.update(nb_waves)
            if nb_man is not None and nb_man.notes:
                anom_notes.append(nb_man.notes)
        except Exception as exc:  # pragma: no cover - defensive
            anom_notes.append(f"BL narrowband extraction failed: {exc}")
    if n_anom - len(anom_recs) > 0:
        try:
            ds_recs, ds_waves, _ds_man = load_bl_anomaly(
                1, seed=seed + 1, cache_dir=cache_dir)
            anom_recs.extend(ds_recs)
            anom_waves.update(ds_waves)
        except Exception:  # pragma: no cover - defensive
            pass

    # Do NOT pad with synthetic tones. If fewer than ``n_anom`` real anomaly
    # signals are available, the set is reported at its true (smaller) size — the
    # audit is honest about how many genuine artificial-technosignature controls
    # it actually holds. Synthetic fill would re-introduce the by-construction
    # separation that the methodology explicitly avoids.

    # Honest ground-truth labelling. The ONLY genuine *artificial*
    # technosignature we hold is the real Voyager 1 carrier + sidebands — these
    # are the anomaly ground truth. The real Breakthrough Listen / DeepSeti
    # observations are *unlabeled* stellar observations: genuine telescope data
    # with no confirmed narrowband signal, used as a discovery / triage pool
    # (see paper/Discovery/). Treating them as anomaly ground truth would be a
    # mislabel — most contain no detected tone — so they are relabelled
    # "Unlabeled" and excluded from the TPR / FPR calculation (they are neither
    # known anomalies nor known natural/RFI). This keeps the OOD benchmark an
    # honest test of whether the engine surfaces the one real artificial signal
    # we actually possess, without inflating or faking the detection rate.
    relabelled = []
    for (name, ocls, stype, dm, snr, role) in anom_recs:
        if role == "Anomaly" and not name.lower().startswith("voyager1"):
            relabelled.append((name, ocls, stype, dm, snr, "Unlabeled"))
        else:
            relabelled.append((name, ocls, stype, dm, snr, role))
    anom_recs = relabelled

    n_real = len(anom_recs)
    n_voy = sum(1 for r in anom_recs if r[0].lower().startswith("voyager1")
                and r[5] == "Anomaly")
    n_unlab = sum(1 for r in anom_recs if r[5] == "Unlabeled")
    n_seti = sum(1 for r in anom_recs
                 if r[0].startswith("SETI__") or "seti" in r[0].lower())
    seti_note = (f" | {n_seti} real Breakthrough Listen candidate signals "
                 f"(Kaggle breakthrough-listen-search-for-advanced-life) held as "
                 f"Unlabeled discovery pool" if n_seti else "")
    base_notes = (f"{n_voy} genuine artificial-technosignature controls (Voyager 1 "
                  f"carrier / sidebands) + {n_unlab} unlabeled real stellar "
                  f"observations (discovery pool, excluded from TPR/FPR).")
    anom_man = RealOODManifest(
        source=("Real Voyager 1 carrier (artificial technosignature) + "
                "unlabeled Breakthrough Listen / Kaggle SETI stellar observations "
                "(discovery pool)"),
        doi=VOYAGER_DOI, url=VOYAGER_URL,
        retrieved_ok=len(anom_recs) > 0, n_real_frb=n_real,
        notes=base_notes + seti_note,
    )
    manifest_list.append(("Anomaly_Tech", anom_man))

    records = frb_recs + rfi_recs + anom_recs
    real_features = dict(rfi_feats)
    real_waves = dict(anom_waves)

    # Persist provenance for every real component.
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "real_ood_manifest.json"), "w") as fh:
        json.dump([(tag, m.to_dict()) for tag, m in manifest_list], fh, indent=2)

    return records, real_features, real_waves, anom_wf, manifest_list


if __name__ == "__main__":
    recs, man = get_real_ood_records()
    print("manifest:", man.to_dict())
    print("sample records:", recs[:3])
