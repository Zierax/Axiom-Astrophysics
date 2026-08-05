"""Assemble the self-consistent real-waterfall manifold.

Every class in the AXIOM manifold — astrophysical pulsars, fast radio bursts,
terrestrial radio-frequency interference (RFI) and the artificial Voyager 1
carrier — is derived from a *real, provenance-pinned, checksum-verified*
dynamic spectrum (see :mod:`axiom.data.provenance`) and passed through the
**single** deterministic featurizer :func:`axiom.dsp.waterfall.extract_features`.
The feature space is therefore commensurate by construction: no signal is ever
projected onto a foreign (HTRU2) manifold via a hand-tuned anchor point, which
was the central by-construction artifact flagged in the codebase review.

Sampling strategy & honest limitations
--------------------------------------
Each source class is currently backed by a *single* public observation. We draw
multiple feature vectors from each by deterministically **windowing** the
dynamic spectrum:

* pulsar / FRB: overlapping windows along the *time* axis (each window is
  independently de-dispersed);
* RFI: contiguous blocks along the *frequency* axis (the observation is short in
  time but spans tens of thousands of channels);
* artificial (Voyager): frequency blocks centred on the telemetry carrier
  (located deterministically as the strongest channel).

These windows are **not** statistically independent detections — they are
segments of one observation per class. This is a known limitation (documented in
``docs/_review`` §6.7): expanding each class to many independent observations is
required for a fully population-level claim. The manifold nonetheless
establishes a *physically self-consistent* feature space in which the four
regimes are separable by measured (not assumed) properties.

All operations are deterministic and the assembled matrix is cached to disk,
keyed by a hash of the full windowing/feature configuration, so it is computed
once and replayed thereafter (featurization is expensive: a single full pulsar
pass is ~30 s).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from axiom.data.provenance import FetchResult, fetch, resolve_cache_dir
from axiom.data.real_loaders import _ensure_blimpy
from axiom.dsp.waterfall import (
    N_WATERFALL_FEATURES,
    WATERFALL_FEATURE_NAMES,
    WaterfallError,
    extract_features,
)

log = logging.getLogger(__name__)

#: Manifold class codes. The "normal" manifold comprises the astrophysical and
#: terrestrial classes; ARTIFICIAL is the held-out technosignature ground truth.
CLASS_PULSAR = 0
CLASS_FRB = 1
CLASS_RFI = 2
CLASS_ARTIFICIAL = 3

CLASS_NAMES: Tuple[str, ...] = ("PULSAR", "FRB", "RFI", "ARTIFICIAL")

#: Classes that define the "normal" (in-manifold) density.
NORMAL_CLASSES: Tuple[int, ...] = (CLASS_PULSAR, CLASS_FRB, CLASS_RFI)
#: Classes treated as out-of-distribution / anomalous.
ANOMALY_CLASSES: Tuple[int, ...] = (CLASS_ARTIFICIAL,)

#: Schema version; bump to invalidate on-disk caches when logic changes.
MANIFOLD_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WindowPlan:
    """Deterministic windowing + featurization plan for one source class.

    Parameters
    ----------
    dataset : str
        Registry key in :data:`axiom.data.provenance.REGISTRY`.
    label : int
        Manifold class code (see ``CLASS_*``).
    axis : str
        ``"time"`` to slide windows along the time axis, ``"freq"`` to tile the
        frequency axis into contiguous channel blocks.
    window : int
        Window length in samples along ``axis``.
    stride : int
        Step between successive windows along ``axis``.
    max_windows : int
        Upper bound on the number of windows produced (keeps runtime bounded and
        the classes roughly balanced).
    dm_max, n_dm, n_subbands : featurizer parameters (see
        :func:`axiom.dsp.waterfall.extract_features`).
    center_on_peak : bool
        If True (used for the Voyager carrier), windows are ordered by descending
        in-window peak power so the carrier-bearing blocks are retained first.
    max_load_gb : float
        ``blimpy`` memory ceiling when reading the filterbank.
    """

    dataset: str
    label: int
    axis: str
    window: int
    stride: int
    max_windows: int
    dm_max: float
    n_dm: int
    n_subbands: int
    center_on_peak: bool = False
    max_load_gb: float = 2.0


#: The canonical assembly plan. Windowing parameters are tuned to each
#: observation's geometry (verified via header inspection) so every window is
#: long enough to resolve the relevant dispersive delay.
MANIFOLD_PLAN: Tuple[WindowPlan, ...] = (
    WindowPlan(  # 8192 time x 4096 chan, tsamp 256 us, 960 MHz band, DM~26.8
        dataset="pulsar_b0329", label=CLASS_PULSAR, axis="time",
        window=2048, stride=1024, max_windows=24,
        dm_max=200.0, n_dm=96, n_subbands=256,
    ),
    WindowPlan(  # 5120 time x 336 chan, tsamp 1.27 ms, 336 MHz band, DM~474
        dataset="frb_180417", label=CLASS_FRB, axis="time",
        window=1280, stride=512, max_windows=24,
        dm_max=1000.0, n_dm=128, n_subbands=336,
    ),
    WindowPlan(  # 279 time x 65536 chan: tile the frequency axis
        dataset="rfi_bl_gbt", label=CLASS_RFI, axis="freq",
        window=4096, stride=4096, max_windows=24,
        dm_max=400.0, n_dm=96, n_subbands=256,
    ),
    WindowPlan(  # 16 time x 1048576 chan: carrier-centred frequency blocks
        dataset="voyager1_carrier", label=CLASS_ARTIFICIAL, axis="freq",
        window=8192, stride=8192, max_windows=12,
        dm_max=400.0, n_dm=64, n_subbands=256, center_on_peak=True,
    ),
)


@dataclass
class ManifoldData:
    """Assembled feature matrix and full provenance for the manifold."""

    X: np.ndarray                    # (n_samples, N_WATERFALL_FEATURES)
    y: np.ndarray                    # (n_samples,) class codes
    feature_names: Tuple[str, ...]
    class_names: Tuple[str, ...]
    provenance: List[Dict[str, object]] = field(default_factory=list)
    config_hash: str = ""

    def normal_mask(self) -> np.ndarray:
        return np.isin(self.y, np.asarray(NORMAL_CLASSES))

    def anomaly_mask(self) -> np.ndarray:
        return np.isin(self.y, np.asarray(ANOMALY_CLASSES))

    def counts(self) -> Dict[str, int]:
        return {
            CLASS_NAMES[c]: int(np.count_nonzero(self.y == c))
            for c in range(len(CLASS_NAMES))
        }


# ---------------------------------------------------------------------------
# Filterbank loading.
# ---------------------------------------------------------------------------
def _load_waterfall(plan: WindowPlan) -> Tuple[np.ndarray, np.ndarray, float, FetchResult]:
    """Fetch (checksum-verified) and read a filterbank into (data, freqs, tsamp)."""
    result = fetch(plan.dataset)
    blimpy = _ensure_blimpy()
    fb = blimpy.Waterfall(result.path, max_load=plan.max_load_gb)
    header = fb.header
    data = np.asarray(fb.data[:, 0, :], dtype=np.float64)
    n_chan = data.shape[1]
    fch1 = float(header["fch1"])
    foff = float(header["foff"])
    tsamp = float(header["tsamp"])
    freqs = fch1 + np.arange(n_chan, dtype=np.float64) * foff
    if not np.all(np.isfinite(data)):
        log.warning("[populations] %s contains non-finite samples (repaired "
                    "downstream by the featurizer).", plan.dataset)
    return data, freqs, tsamp, result


# ---------------------------------------------------------------------------
# Deterministic windowing.
# ---------------------------------------------------------------------------
def _iter_windows(
    data: np.ndarray, freqs: np.ndarray, plan: WindowPlan
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Yield (window_data, window_freqs) slices per the plan (deterministic)."""
    n_time, n_chan = data.shape
    if plan.axis == "time":
        axis_len, win, stride = n_time, plan.window, plan.stride
    elif plan.axis == "freq":
        axis_len, win, stride = n_chan, plan.window, plan.stride
    else:  # pragma: no cover - guarded by WindowPlan construction
        raise ValueError(f"unknown axis {plan.axis!r}")
    if win <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive.")
    if win > axis_len:
        raise ValueError(
            f"window {win} exceeds {plan.axis} extent {axis_len} for "
            f"{plan.dataset}."
        )

    starts = list(range(0, axis_len - win + 1, stride))
    slices: List[Tuple[np.ndarray, np.ndarray, float]] = []
    for s in starts:
        if plan.axis == "time":
            wdata = data[s:s + win, :]
            wfreqs = freqs
        else:
            wdata = data[:, s:s + win]
            wfreqs = freqs[s:s + win]
        peak = float(np.max(wdata.mean(axis=0) - np.median(wdata.mean(axis=0))))
        slices.append((wdata, wfreqs, peak))

    if plan.center_on_peak:
        # Retain the highest-power blocks first (carrier-bearing for Voyager).
        slices.sort(key=lambda t: t[2], reverse=True)

    for wdata, wfreqs, _ in slices[: plan.max_windows]:
        yield np.ascontiguousarray(wdata), np.ascontiguousarray(wfreqs)


def _featurize_class(plan: WindowPlan) -> Tuple[np.ndarray, Dict[str, object]]:
    """Return the (n_windows, N_FEATURES) matrix and provenance for one class."""
    data, freqs, tsamp, result = _load_waterfall(plan)
    vectors: List[np.ndarray] = []
    n_failed = 0
    for wdata, wfreqs in _iter_windows(data, freqs, plan):
        try:
            vec = extract_features(
                wdata, wfreqs, tsamp,
                dm_max=plan.dm_max, n_dm=plan.n_dm, n_subbands=plan.n_subbands,
            )
        except WaterfallError as exc:
            n_failed += 1
            log.warning("[populations] window skipped for %s: %s",
                        plan.dataset, exc)
            continue
        vectors.append(vec)
    if not vectors:
        raise RuntimeError(
            f"no valid feature windows produced for {plan.dataset}; check the "
            f"WindowPlan against the observation geometry."
        )
    matrix = np.vstack(vectors)
    prov: Dict[str, object] = {
        "dataset": plan.dataset,
        "label": plan.label,
        "class_name": CLASS_NAMES[plan.label],
        "n_windows": int(matrix.shape[0]),
        "n_failed_windows": int(n_failed),
        "doi": result.doi,
        "description": result.description,
        "digest": f"{result.digest_algo}:{result.digest_value}",
        "source_path": os.path.basename(result.path),
        "plan": asdict(plan),
    }
    log.info("[populations] %s -> %d windows (label=%s).",
             plan.dataset, matrix.shape[0], CLASS_NAMES[plan.label])
    return matrix, prov


# ---------------------------------------------------------------------------
# Cache management.
# ---------------------------------------------------------------------------
def _config_hash(plan: Tuple[WindowPlan, ...]) -> str:
    """Stable hash of the schema version + full windowing plan + feature names."""
    payload = {
        "schema": MANIFOLD_SCHEMA_VERSION,
        "features": list(WATERFALL_FEATURE_NAMES),
        "plan": [asdict(p) for p in plan],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: str, config_hash: str) -> str:
    return os.path.join(cache_dir, f"manifold_{config_hash}.npz")


def _load_cache(path: str, config_hash: str) -> Optional[ManifoldData]:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as npz:
            if str(npz["config_hash"]) != config_hash:
                return None
            X = np.asarray(npz["X"], dtype=np.float64)
            y = np.asarray(npz["y"], dtype=np.int64)
            provenance = json.loads(str(npz["provenance_json"]))
    except Exception as exc:  # pragma: no cover - corrupt cache
        log.warning("[populations] ignoring unreadable cache %s: %s", path, exc)
        return None
    if X.shape[0] != y.shape[0] or X.shape[1] != N_WATERFALL_FEATURES:
        log.warning("[populations] cache shape mismatch; recomputing.")
        return None
    return ManifoldData(
        X=X, y=y,
        feature_names=WATERFALL_FEATURE_NAMES,
        class_names=CLASS_NAMES,
        provenance=provenance,
        config_hash=config_hash,
    )


def _save_cache(path: str, manifold: ManifoldData) -> None:
    # NOTE: np.savez appends ".npz" if the name lacks it; keep the suffix on the
    # temp file so the atomic os.replace target actually exists.
    tmp = path + ".tmp.npz"
    try:
        np.savez(
            tmp,
            X=manifold.X,
            y=manifold.y,
            config_hash=manifold.config_hash,
            provenance_json=json.dumps(manifold.provenance),
        )
        os.replace(tmp, path)
    except Exception as exc:  # pragma: no cover - disk failure
        log.warning("[populations] failed to write cache %s: %s", path, exc)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def build_manifold(
    plan: Tuple[WindowPlan, ...] = MANIFOLD_PLAN,
    *,
    cache: bool = True,
    cache_dir: Optional[str] = None,
) -> ManifoldData:
    """Assemble (or load from cache) the real-waterfall feature manifold.

    Parameters
    ----------
    plan : tuple of WindowPlan
        Per-class windowing/featurization plan (defaults to :data:`MANIFOLD_PLAN`).
    cache : bool
        If True, read/write a deterministic on-disk cache keyed by the config
        hash. Set False to force recomputation.
    cache_dir : str, optional
        Cache directory override (defaults to the provenance cache dir).

    Returns
    -------
    ManifoldData
        Feature matrix, labels, feature/class names and full provenance.
    """
    directory = resolve_cache_dir(cache_dir)
    os.makedirs(directory, exist_ok=True)
    config_hash = _config_hash(plan)
    path = _cache_path(directory, config_hash)

    if cache:
        cached = _load_cache(path, config_hash)
        if cached is not None:
            log.info("[populations] manifold cache hit (%s): %s",
                     config_hash, cached.counts())
            return cached

    matrices: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    provenance: List[Dict[str, object]] = []
    for p in plan:
        matrix, prov = _featurize_class(p)
        matrices.append(matrix)
        labels.append(np.full(matrix.shape[0], p.label, dtype=np.int64))
        provenance.append(prov)

    X = np.vstack(matrices)
    y = np.concatenate(labels)
    if not np.all(np.isfinite(X)):  # pragma: no cover - featurizer guards this
        raise RuntimeError("assembled manifold contains non-finite features.")

    manifold = ManifoldData(
        X=X, y=y,
        feature_names=WATERFALL_FEATURE_NAMES,
        class_names=CLASS_NAMES,
        provenance=provenance,
        config_hash=config_hash,
    )
    if cache:
        _save_cache(path, manifold)
    log.info("[populations] manifold assembled (%s): %s",
             config_hash, manifold.counts())
    return manifold
