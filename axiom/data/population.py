"""Assemble the population-scale, per-object catalog manifold.

This module unifies several verified real catalogs (:mod:`axiom.data.catalogs`)
into a single feature manifold of **thousands of independent astronomical
objects**, each represented once, and featurised by the single commensurate
physical map :func:`axiom.dsp.physical_features.featurize_frame`.

Why this supersedes the windowed-waterfall manifold for population-level claims
-------------------------------------------------------------------------------
The waterfall manifold (:mod:`axiom.data.populations`) draws many overlapping
*windows* from a *single observation per class* — statistically dependent
segments, not independent detections. Here every row is a distinct object (one
pulsar, one FRB, one RFI candidate) with a stable ``group_id`` equal to its
catalog ``object_id``. Cross-validation keyed on ``group_id`` therefore contains
**no within-observation leakage by construction**: the pseudo-replication
limitation is removed, not merely mitigated.

The assembled matrix, labels, group ids and provenance are cached deterministically
to disk, keyed by a hash of the catalog set + schema version + feature names, so
the (network-dependent) assembly runs once and replays offline thereafter.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from axiom.data.catalogs import CLASS_CODES, load_many
from axiom.data.downloader import resolve_cache_dir
from axiom.dsp.physical_features import (
    N_PHYSICAL_FEATURES,
    PHYSICAL_FEATURE_NAMES,
    featurize_frame,
)

log = logging.getLogger(__name__)

#: Schema version; bump to invalidate on-disk caches when assembly logic changes.
POPULATION_SCHEMA_VERSION = 1

#: Default catalog set for the canonical population manifold. Chosen for verified
#: live availability and independent-object semantics.
DEFAULT_CATALOG_KEYS: Tuple[str, ...] = (
    "atnf_pulsars",
    "chime_frb_cat1",
    "htru2_local",
)

#: Inverse of CLASS_CODES for readable reporting.
CODE_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_CODES.items()}


class PopulationError(RuntimeError):
    """Raised on malformed population assembly."""


@dataclass
class PopulationData:
    """Assembled population manifold with per-object group ids and provenance."""

    X: np.ndarray                       # (n_objects, N_PHYSICAL_FEATURES)
    y: np.ndarray                       # (n_objects,) integer class codes
    group_ids: np.ndarray              # (n_objects,) unique object identifiers
    class_names: np.ndarray            # (n_objects,) string class labels
    feature_names: Tuple[str, ...]
    catalog_keys: Tuple[str, ...]
    provenance: List[Dict[str, object]] = field(default_factory=list)
    config_hash: str = ""

    def n_objects(self) -> int:
        return int(self.X.shape[0])

    def class_counts(self) -> Dict[str, int]:
        codes, counts = np.unique(self.y, return_counts=True)
        return {CODE_TO_CLASS.get(int(c), str(int(c))): int(n)
                for c, n in zip(codes, counts)}

    def code_of(self, class_name: str) -> int:
        if class_name not in CLASS_CODES:
            raise PopulationError(
                f"unknown class {class_name!r}; known: {sorted(CLASS_CODES)}."
            )
        return CLASS_CODES[class_name]

    def mask_for(self, class_names: Sequence[str]) -> np.ndarray:
        codes = np.asarray([self.code_of(c) for c in class_names])
        return np.isin(self.y, codes)


def _config_hash(catalog_keys: Sequence[str]) -> str:
    payload = {
        "schema": POPULATION_SCHEMA_VERSION,
        "catalogs": sorted(catalog_keys),
        "features": list(PHYSICAL_FEATURE_NAMES),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _cache_path(cache_dir: str, config_hash: str) -> str:
    return os.path.join(cache_dir, f"population_{config_hash}.npz")


def _load_cache(path: str, config_hash: str) -> Optional[PopulationData]:
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=True) as npz:
            if str(npz["config_hash"]) != config_hash:
                return None
            X = np.asarray(npz["X"], dtype=np.float64)
            y = np.asarray(npz["y"], dtype=np.int64)
            group_ids = np.asarray(npz["group_ids"], dtype=object).astype(str)
            class_names = np.asarray(npz["class_names"], dtype=object).astype(str)
            catalog_keys = tuple(str(k) for k in npz["catalog_keys"])
            provenance = json.loads(str(npz["provenance_json"]))
    except Exception as exc:  # pragma: no cover - corrupt cache
        log.warning("[population] ignoring unreadable cache %s: %s", path, exc)
        return None
    if X.shape[0] != y.shape[0] or X.shape[1] != N_PHYSICAL_FEATURES:
        log.warning("[population] cache shape mismatch; recomputing.")
        return None
    return PopulationData(
        X=X, y=y, group_ids=group_ids, class_names=class_names,
        feature_names=PHYSICAL_FEATURE_NAMES, catalog_keys=catalog_keys,
        provenance=provenance, config_hash=config_hash,
    )


def _save_cache(path: str, pop: PopulationData) -> None:
    tmp = path + ".tmp.npz"
    try:
        np.savez(
            tmp,
            X=pop.X,
            y=pop.y,
            group_ids=np.asarray(pop.group_ids, dtype=object),
            class_names=np.asarray(pop.class_names, dtype=object),
            catalog_keys=np.asarray(pop.catalog_keys, dtype=object),
            config_hash=pop.config_hash,
            provenance_json=json.dumps(pop.provenance),
        )
        os.replace(tmp, path)
    except Exception as exc:  # pragma: no cover - disk failure
        log.warning("[population] failed to write cache %s: %s", path, exc)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def build_population(
    catalog_keys: Sequence[str] = DEFAULT_CATALOG_KEYS,
    *,
    cache: bool = True,
    cache_dir: Optional[str] = None,
    ignore_errors: bool = False,
    min_class_size: int = 2,
    class_aliases: Optional[Dict[str, str]] = None,
) -> PopulationData:
    """Assemble (or load from cache) the population-scale catalog manifold.

    Parameters
    ----------
    catalog_keys : sequence of str
        Registry keys to merge (defaults to :data:`DEFAULT_CATALOG_KEYS`).
    cache : bool
        Read/write a deterministic on-disk cache keyed by the config hash.
    cache_dir : str, optional
        Cache directory override (defaults to the downloader catalog cache dir).
    ignore_errors : bool
        Passed to :func:`axiom.data.catalogs.load_many`; skip failing catalogs.
    min_class_size : int
        Drop classes with fewer than this many objects (keeps CV well-posed).
    class_aliases : dict, optional
        Map fine-grained class names to a coarser label (e.g. merge the rare
        rotating-neutron-star subtypes RRAT and MAGNETAR into "RARE_PULSAR").
        Merging genuinely-related rare variants keeps the multiclass problem
        well-posed instead of reporting a meaningless ~0 F1 for a 4-sample class.

    Returns
    -------
    PopulationData
    """
    if not catalog_keys:
        raise ValueError("catalog_keys must be non-empty.")
    directory = resolve_cache_dir(cache_dir)
    os.makedirs(directory, exist_ok=True)
    config_hash = _config_hash(catalog_keys)
    path = _cache_path(directory, config_hash)

    if cache:
        cached = _load_cache(path, config_hash)
        if cached is not None:
            log.info("[population] cache hit (%s): %d objects %s",
                     config_hash, cached.n_objects(), cached.class_counts())
            return cached

    frame: pd.DataFrame = load_many(
        catalog_keys, cache_dir=cache_dir, ignore_errors=ignore_errors
    )
    if frame.empty:
        raise PopulationError("assembled catalog frame is empty.")

    # Merge fine-grained variants into coarser, learnable classes (e.g. the rare
    # rotating-neutron-star subtypes RRAT + MAGNETAR -> RARE_PULSAR). This is a
    # scientifically legitimate grouping (both are neutron-star rotation
    # variants) and prevents reporting a meaningless ~0 F1 for a 4-sample class.
    if class_aliases:
        frame = frame.copy()
        frame["class_name"] = frame["class_name"].map(
            lambda c: class_aliases.get(c, c))

    # Drop under-populated classes so downstream stratified/group CV is valid.
    counts = frame["class_name"].value_counts()
    keep_classes = counts[counts >= int(min_class_size)].index.tolist()
    dropped = sorted(set(counts.index) - set(keep_classes))
    if dropped:
        log.info("[population] dropping classes below min_class_size=%d: %s",
                 min_class_size, {c: int(counts[c]) for c in dropped})
    frame = frame[frame["class_name"].isin(keep_classes)].reset_index(drop=True)
    if frame.empty:
        raise PopulationError(
            f"no class met min_class_size={min_class_size}; counts={counts.to_dict()}."
        )

    X, names = featurize_frame(frame)
    y = frame["class_name"].map(CLASS_CODES).to_numpy(dtype=np.int64)
    group_ids = frame["object_id"].to_numpy(dtype=object).astype(str)
    class_names = frame["class_name"].to_numpy(dtype=object).astype(str)

    if np.unique(group_ids).size != group_ids.size:  # pragma: no cover - invariant
        raise PopulationError("group_ids are not unique; catalog dedup failed.")
    if not np.all(np.isfinite(X[:, len(names) - 4:])):  # indicators must be finite
        raise PopulationError("indicator features contain non-finite values.")

    provenance: List[Dict[str, object]] = []
    for src, sub in frame.groupby("source"):
        provenance.append({
            "source": str(src),
            "n_objects": int(len(sub)),
            "classes": {k: int(v) for k, v in sub["class_name"].value_counts().items()},
            "dm_min": float(np.nanmin(sub["dm"])),
            "dm_median": float(np.nanmedian(sub["dm"])),
            "dm_max": float(np.nanmax(sub["dm"])),
        })

    pop = PopulationData(
        X=X, y=y, group_ids=group_ids, class_names=class_names,
        feature_names=names, catalog_keys=tuple(catalog_keys),
        provenance=provenance, config_hash=config_hash,
    )
    if cache:
        _save_cache(path, pop)
    log.info("[population] assembled (%s): %d objects %s",
             config_hash, pop.n_objects(), pop.class_counts())
    return pop
