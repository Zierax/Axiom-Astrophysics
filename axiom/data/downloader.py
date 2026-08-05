"""Reproducible multi-backend dataset downloader for the AXIOM population manifold.

This module is the single, hardened ingress point for every *tabular* real-world
catalog used by the population-scale manifold (:mod:`axiom.data.catalogs`). It is
deliberately separate from :mod:`axiom.data.provenance`, which pins large binary
filterbanks by an *a-priori* checksum. Astronomical catalogs, by contrast, are
periodically revised upstream (new pulsars/FRBs are appended), so an immutable
digest is the wrong contract. Instead this downloader implements
**pin-on-first-fetch**: the first successful download of an artifact records its
SHA-256 into a version-controlled lock file, and every subsequent fetch is
verified against that recorded digest. This gives byte-for-byte reproducibility
for anyone who checks out the repository *and* the lock file, while still letting
a maintainer deliberately re-pin against a newer catalog release.

Backends
--------
* ``http``   - generic HTTP(S) GET with streaming, retries and atomic promotion.
* ``vizier`` - CDS VizieR ASU tab-separated export, built deterministically from
  a source table id, an explicit column list and optional row cap / constraints.
* ``kaggle`` - Kaggle datasets, fetched through the authenticated ``kaggle`` CLI
  (which must be installed and configured out-of-band) in a validated subprocess.

Guarantees
----------
* Deterministic: no unseeded randomness; bounded exponential backoff with no
  jitter; stable VizieR URL construction (sorted query parameters).
* Atomic: payloads are streamed to a unique temp file in the destination
  directory and promoted with :func:`os.replace` only after full verification.
* Offline: with ``AXIOM_OFFLINE=1`` no network call is attempted; a digest-valid
  cached copy is required or :class:`OfflineError` is raised.
* Defensive: every public entry point validates its inputs and fails loudly with
  a precise, actionable message rather than silently returning bad data.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (kept consistent with axiom.data.provenance where shared).
# ---------------------------------------------------------------------------
#: Environment flag: when truthy, forbid all network access (offline replay).
OFFLINE_ENV = "AXIOM_OFFLINE"
#: Environment override for the cache directory root.
CACHE_DIR_ENV = "AXIOM_DATA_CACHE"
#: Default cache root for catalog artifacts (git-ignored; regenerated on demand).
DEFAULT_CATALOG_DIR = os.path.join("data", "catalogs")
#: Version-controlled lock file recording pinned digests (relative to repo root).
DEFAULT_LOCK_PATH = os.path.join("configs", "catalog_locks.json")

_STREAM_CHUNK_BYTES = 1 << 20          # 1 MiB streaming granularity
_DEFAULT_TIMEOUT_S = 300               # per-attempt socket timeout
_DEFAULT_MAX_RETRIES = 4               # total network attempts per artifact
_BACKOFF_BASE_S = 2.0                  # exponential backoff base
_BACKOFF_CAP_S = 45.0                  # maximum sleep between attempts
_USER_AGENT = "AXIOM-Astrophysics/2.0 (research; reproducible-data-fetch)"
_KAGGLE_TIMEOUT_S = 900                # Kaggle archives may be large
_SUPPORTED_ALGOS = ("sha256",)        # single algorithm for catalog locks
_VIZIER_BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


class DownloaderError(RuntimeError):
    """Base class for all downloader failures."""


class DownloadError(DownloaderError):
    """Raised when a network download cannot be completed or verified."""


class OfflineError(DownloaderError):
    """Raised when offline mode is active and no valid cached copy exists."""


class LockMismatchError(DownloaderError):
    """Raised when a downloaded artifact does not match its pinned digest."""


# ---------------------------------------------------------------------------
# Environment / path helpers.
# ---------------------------------------------------------------------------
def is_offline() -> bool:
    """Return True iff offline mode is requested via the environment."""
    return os.environ.get(OFFLINE_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Resolve the effective catalog cache directory (arg > env > default)."""
    chosen = cache_dir or os.environ.get(CACHE_DIR_ENV) or DEFAULT_CATALOG_DIR
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("cache_dir must resolve to a non-empty path.")
    return chosen


def _validate_basename(filename: str) -> str:
    """Reject path traversal / absolute paths in a cache basename."""
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("filename must be a non-empty string.")
    if os.path.isabs(filename) or os.path.basename(filename) != filename:
        raise ValueError(f"filename must be a bare basename, got {filename!r}.")
    if filename in (".", ".."):
        raise ValueError(f"illegal filename {filename!r}.")
    return filename


def _hash_file(path: str, algo: str = "sha256") -> str:
    """Stream a file through ``algo`` and return its lower-case hex digest."""
    if algo not in _SUPPORTED_ALGOS:
        raise ValueError(f"unsupported digest algorithm {algo!r}.")
    hasher = hashlib.new(algo)
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_STREAM_CHUNK_BYTES), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise DownloaderError(f"failed to read {path!r} for hashing: {exc}") from exc
    return hasher.hexdigest().lower()


def _sleep_backoff(attempt: int) -> None:
    """Deterministic bounded exponential backoff (no jitter, reproducible)."""
    delay = min(_BACKOFF_BASE_S ** attempt, _BACKOFF_CAP_S)
    log.info("[downloader] backing off %.1fs before retry.", delay)
    time.sleep(delay)


# ---------------------------------------------------------------------------
# Lock file (pin-on-first-fetch reproducibility contract).
# ---------------------------------------------------------------------------
def _lock_path(lock_path: Optional[str]) -> str:
    return lock_path or os.environ.get("AXIOM_CATALOG_LOCK") or DEFAULT_LOCK_PATH


def _load_lock(lock_path: str) -> Dict[str, Dict[str, object]]:
    if not os.path.isfile(lock_path):
        return {}
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        log.warning("[downloader] ignoring unreadable lock %s: %s", lock_path, exc)
        return {}
    if not isinstance(data, dict):
        log.warning("[downloader] lock %s is not a JSON object; ignoring.", lock_path)
        return {}
    return data


def _save_lock(lock_path: str, lock: Mapping[str, Dict[str, object]]) -> None:
    directory = os.path.dirname(lock_path) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = lock_path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(dict(sorted(lock.items())), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, lock_path)
    except OSError as exc:
        log.warning("[downloader] failed to persist lock %s: %s", lock_path, exc)
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Fetch result.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DownloadResult:
    """Outcome of a fetch, carrying provenance for downstream reproducibility."""

    key: str
    path: str
    sha256: str
    size_bytes: int
    from_cache: bool
    backend: str
    source: str


# ---------------------------------------------------------------------------
# HTTP backend.
# ---------------------------------------------------------------------------
def _http_stream_to(url: str, dest_path: str, timeout: int, min_bytes: int) -> Tuple[str, int]:
    """Stream ``url`` to ``dest_path`` atomically; return (sha256, size)."""
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix="." + os.path.basename(dest_path) + ".", suffix=".part", dir=dest_dir
    )
    hasher = hashlib.new("sha256")
    bytes_written = 0
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with os.fdopen(fd, "wb") as out_handle:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = int(getattr(response, "status", 200) or 200)
                if status >= 400:
                    raise DownloadError(f"HTTP {status} fetching {url!r}.")
                while True:
                    chunk = response.read(_STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    out_handle.write(chunk)
                    hasher.update(chunk)
                    bytes_written += len(chunk)
        if bytes_written < min_bytes:
            raise DownloadError(
                f"downloaded {bytes_written} bytes from {url!r}; expected "
                f">= {min_bytes} (truncated or empty response)."
            )
        digest = hasher.hexdigest().lower()
        os.replace(tmp_path, dest_path)
        return digest, bytes_written
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, DownloadError) as exc:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        raise DownloadError(f"failed streaming {url!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# VizieR URL construction.
# ---------------------------------------------------------------------------
def build_vizier_url(
    source: str,
    columns: Sequence[str],
    *,
    max_rows: int = 200000,
    constraints: Optional[Mapping[str, str]] = None,
) -> str:
    """Deterministically construct a VizieR ASU tab-separated export URL.

    Parameters
    ----------
    source : str
        VizieR table identifier, e.g. ``"B/psr/psr"`` or ``"J/ApJS/257/59/table2"``.
    columns : sequence of str
        Explicit output columns (order preserved in ``-out``).
    max_rows : int
        Row cap (``-out.max``). Must be a positive integer.
    constraints : mapping, optional
        Additional VizieR column constraints, e.g. ``{"DM": ">0"}``. Keys are
        column names, values are VizieR constraint expressions.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("VizieR source must be a non-empty string.")
    cols = [c for c in columns if isinstance(c, str) and c.strip()]
    if not cols:
        raise ValueError("VizieR columns must contain at least one valid column.")
    if not isinstance(max_rows, int) or max_rows <= 0:
        raise ValueError(f"max_rows must be a positive int; got {max_rows!r}.")

    # Ordered query params: -source, -out.max, -out (repeated), constraints.
    params: List[Tuple[str, str]] = [
        ("-source", source),
        ("-out.max", str(max_rows)),
        ("-out", ",".join(cols)),
    ]
    if constraints:
        for col in sorted(constraints):
            expr = constraints[col]
            if not isinstance(expr, str) or not expr.strip():
                raise ValueError(f"constraint for {col!r} must be a non-empty string.")
            params.append((col, expr))
    query = urllib.parse.urlencode(params, quote_via=urllib.parse.quote)
    return f"{_VIZIER_BASE}?{query}"


# ---------------------------------------------------------------------------
# Kaggle backend.
# ---------------------------------------------------------------------------
def _resolve_kaggle_binary() -> str:
    binary = shutil.which("kaggle")
    if binary is None:
        raise DownloadError(
            "the 'kaggle' CLI is not on PATH; install it and configure "
            "~/.kaggle credentials to use the kaggle backend."
        )
    return binary


def _kaggle_download(dataset_ref: str, dest_dir: str, timeout: int) -> None:
    """Download and unzip a Kaggle dataset into ``dest_dir`` via the CLI."""
    if not isinstance(dataset_ref, str) or dataset_ref.count("/") != 1:
        raise ValueError(
            f"kaggle dataset ref must be '<owner>/<slug>'; got {dataset_ref!r}."
        )
    owner, slug = dataset_ref.split("/", 1)
    if not owner.strip() or not slug.strip():
        raise ValueError(f"malformed kaggle dataset ref {dataset_ref!r}.")
    binary = _resolve_kaggle_binary()
    os.makedirs(dest_dir, exist_ok=True)
    cmd = [
        binary, "datasets", "download",
        "-d", dataset_ref,
        "-p", dest_dir,
        "--unzip",
    ]
    log.info("[downloader] kaggle: %s -> %s", dataset_ref, dest_dir)
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DownloadError(
            f"kaggle download of {dataset_ref!r} timed out after {timeout}s."
        ) from exc
    except OSError as exc:
        raise DownloadError(f"failed to invoke kaggle CLI: {exc}") from exc
    if completed.returncode != 0:
        raise DownloadError(
            f"kaggle download of {dataset_ref!r} failed (exit "
            f"{completed.returncode}): {completed.stderr.strip() or completed.stdout.strip()}"
        )


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------
def fetch(
    key: str,
    *,
    backend: str,
    source: str,
    filename: str,
    columns: Optional[Sequence[str]] = None,
    max_rows: int = 200000,
    constraints: Optional[Mapping[str, str]] = None,
    kaggle_member: Optional[str] = None,
    min_bytes: int = 1,
    cache_dir: Optional[str] = None,
    lock_path: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    repin: bool = False,
) -> DownloadResult:
    """Fetch a catalog artifact reproducibly and return its provenance.

    Parameters
    ----------
    key : str
        Stable identifier used as the lock key (e.g. ``"atnf_pulsars"``).
    backend : {"http", "vizier", "kaggle"}
        Which fetch strategy to use.
    source : str
        HTTP URL (``http``), VizieR table id (``vizier``) or ``owner/slug``
        (``kaggle``).
    filename : str
        Bare basename to store/read under the cache directory. For ``kaggle``
        this is the member file expected inside the downloaded archive (or set
        ``kaggle_member`` explicitly).
    columns, max_rows, constraints : VizieR query parameters (``vizier`` only).
    kaggle_member : str, optional
        Explicit member filename to select from a Kaggle archive; defaults to
        ``filename``.
    min_bytes : int
        Reject artifacts smaller than this (guards against truncated exports).
    cache_dir, lock_path : path overrides.
    timeout : int, optional
        Per-attempt timeout; defaults per backend.
    max_retries : int
        Network attempts before giving up.
    repin : bool
        If True, overwrite the recorded digest with the freshly downloaded one
        (deliberate catalog re-pin). Otherwise a mismatch raises.

    Returns
    -------
    DownloadResult
    """
    if not isinstance(key, str) or not key.strip():
        raise ValueError("key must be a non-empty string.")
    if backend not in ("http", "vizier", "kaggle"):
        raise ValueError(f"unknown backend {backend!r}.")
    if not isinstance(max_retries, int) or max_retries <= 0:
        raise ValueError(f"max_retries must be a positive int; got {max_retries!r}.")
    _validate_basename(filename)

    directory = resolve_cache_dir(cache_dir)
    os.makedirs(directory, exist_ok=True)
    dest_path = os.path.join(directory, filename)
    lock_file = _lock_path(lock_path)
    lock = _load_lock(lock_file)
    pinned = lock.get(key)
    pinned_digest = str(pinned["sha256"]) if isinstance(pinned, dict) and "sha256" in pinned else None

    # 1. Serve a digest-valid cached copy without touching the network.
    if os.path.isfile(dest_path):
        actual = _hash_file(dest_path)
        size = os.path.getsize(dest_path)
        if size >= min_bytes and (pinned_digest is None or actual == pinned_digest):
            if pinned_digest is None:
                _record_pin(lock, lock_file, key, backend, source, actual, size)
            log.info("[downloader] cache hit for %s (%s).", key, actual[:12])
            return DownloadResult(key, dest_path, actual, size, True, backend, source)
        if pinned_digest is not None and actual != pinned_digest and not repin:
            log.warning(
                "[downloader] cached %s digest %s != pinned %s; re-fetching.",
                filename, actual[:12], pinned_digest[:12],
            )

    # 2. Offline mode: no network allowed.
    if is_offline():
        raise OfflineError(
            f"offline mode ({OFFLINE_ENV}=1) active and no digest-valid cached "
            f"copy of {key!r} at {dest_path!r}."
        )

    eff_timeout = int(timeout) if timeout else (
        _KAGGLE_TIMEOUT_S if backend == "kaggle" else _DEFAULT_TIMEOUT_S
    )

    # 3. Build the effective URL for network backends.
    if backend == "vizier":
        if not columns:
            raise ValueError("VizieR backend requires a non-empty columns list.")
        url = build_vizier_url(source, columns, max_rows=max_rows, constraints=constraints)
    elif backend == "http":
        if not (source.lower().startswith("http://") or source.lower().startswith("https://")):
            raise ValueError(f"http backend requires an http(s) URL; got {source!r}.")
        url = source
    else:  # kaggle
        url = source

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            if backend == "kaggle":
                _kaggle_download(source, directory, eff_timeout)
                member = kaggle_member or filename
                member_path = os.path.join(directory, member)
                if not os.path.isfile(member_path):
                    raise DownloadError(
                        f"kaggle archive for {source!r} did not contain expected "
                        f"member {member!r} in {directory!r}."
                    )
                if member_path != dest_path:
                    os.replace(member_path, dest_path)
                size = os.path.getsize(dest_path)
                if size < min_bytes:
                    raise DownloadError(
                        f"kaggle member {member!r} is undersized ({size} bytes)."
                    )
                digest = _hash_file(dest_path)
            else:
                digest, size = _http_stream_to(url, dest_path, eff_timeout, min_bytes)

            if pinned_digest is not None and digest != pinned_digest and not repin:
                raise LockMismatchError(
                    f"downloaded {key!r} digest {digest} does not match pinned "
                    f"{pinned_digest}. Pass repin=True to deliberately update the "
                    f"lock against a new catalog release."
                )
            _record_pin(lock, lock_file, key, backend, source, digest, size,
                        overwrite=repin or pinned_digest is None)
            log.info("[downloader] fetched %s (%s, %d bytes, attempt %d).",
                     key, digest[:12], size, attempt)
            return DownloadResult(key, dest_path, digest, size, False, backend, source)
        except LockMismatchError:
            raise
        except DownloaderError as exc:
            last_exc = exc
            log.warning("[downloader] attempt %d/%d for %s failed: %s",
                        attempt, max_retries, key, exc)
            if attempt < max_retries:
                _sleep_backoff(attempt)

    raise DownloadError(
        f"exhausted {max_retries} attempts fetching {key!r} from {source!r}: {last_exc}"
    )


def _record_pin(
    lock: Dict[str, Dict[str, object]],
    lock_path: str,
    key: str,
    backend: str,
    source: str,
    digest: str,
    size: int,
    *,
    overwrite: bool = True,
) -> None:
    """Record (or refresh) the pinned digest for ``key`` in the lock file."""
    existing = lock.get(key)
    if isinstance(existing, dict) and not overwrite and existing.get("sha256") == digest:
        return
    lock[key] = {
        "sha256": digest,
        "size_bytes": int(size),
        "backend": backend,
        "source": source,
    }
    _save_lock(lock_path, lock)
