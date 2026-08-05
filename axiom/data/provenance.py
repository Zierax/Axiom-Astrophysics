"""Deterministic, integrity-checked acquisition of real observational data.

This module is the single trust boundary between the AXIOM pipeline and the
public radio-astronomy data archives. Every byte that enters the
self-consistent real-waterfall manifold (``axiom.data.populations``) is routed
through :func:`fetch`, which guarantees:

* **Determinism** — a dataset is identified by a *pinned cryptographic digest*
  (SHA-256 or, where the archive only publishes it, MD5). A file is only ever
  accepted if its digest matches the value recorded in :data:`REGISTRY`. Two
  runs on two machines therefore consume byte-identical inputs or fail loudly.
* **Content addressing** — the verified payload is cached on disk. Repeated
  runs never re-download, and a corrupted cache entry is detected (digest
  mismatch) and transparently repaired unless the process is offline.
* **Offline replay** — setting ``AXIOM_OFFLINE=1`` forbids all network access;
  the cache is used if (and only if) it already contains a digest-valid file,
  otherwise a precise :class:`OfflineDataUnavailable` error is raised. This
  makes CI and air-gapped reproduction first-class, not an afterthought.
* **Graceful failure** — network faults are retried with bounded exponential
  backoff; partial downloads are written to a temporary file and atomically
  renamed only after digest verification, so an interrupted run can never leave
  a half-written file masquerading as valid data.

The registry pins four real, citable telescope observations spanning the four
signal classes used by the manifold:

======================  ===========================================  =========
Class                   Source                                       Digest
======================  ===========================================  =========
Pulsar (astrophysical)  PSR B0329+54, GBT/Greenburst single pulses    MD5
FRB (astrophysical)     FRB 180417, ASKAP incoherent sum             MD5
RFI / background        Breakthrough Listen GBT ETZ observation      SHA-256
Artificial (ground      Voyager 1 X-band carrier, GBT fine-res       SHA-256
truth technosignature)
======================  ===========================================  =========

References
----------
Agarwal, D. et al. (2019) MNRAS 490, 1 — DOI 10.1093/mnras/stz2574
Agarwal, D. et al. (2020) MNRAS 497, 1661 — DOI 10.1093/mnras/staa1927
    (the ``B0329+54.fil`` / ``FRB180417.fil`` test set; Zenodo 10.5281/zenodo.3905426)
MacMahon, D. H. E. et al. (2018) PASP 130, 044502 — DOI 10.3847/1538-3873/aa80d2
Isaacson, H. et al. (2017) PASP 129, 054501 — DOI 10.1088/1538-3873/aa5800
"""
from __future__ import annotations

import errno
import hashlib
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment toggles (documented, deterministic).
# ---------------------------------------------------------------------------
#: If set to "1"/"true"/"yes", no network access is permitted; the cache is the
#: only allowed source. Missing/invalid cache entries raise OfflineDataUnavailable.
OFFLINE_ENV = "AXIOM_OFFLINE"
#: Overrides the on-disk cache directory (default: ``data/real_ood``).
CACHE_DIR_ENV = "AXIOM_DATA_CACHE"

DEFAULT_CACHE_DIR = os.path.join("data", "real_ood")

_SUPPORTED_ALGOS = ("sha256", "sha1", "md5")
_STREAM_CHUNK_BYTES = 1 << 20          # 1 MiB streaming granularity
_DEFAULT_TIMEOUT_S = 600               # per-attempt socket timeout
_DEFAULT_MAX_RETRIES = 4               # total network attempts
_BACKOFF_BASE_S = 2.0                  # exponential backoff base
_BACKOFF_CAP_S = 60.0                  # maximum sleep between attempts
_USER_AGENT = "AXIOM-Astrophysics/2.0 (research; reproducible-data-fetch)"


# ---------------------------------------------------------------------------
# Exceptions — precise, catchable failure modes.
# ---------------------------------------------------------------------------
class ProvenanceError(RuntimeError):
    """Base class for all data-provenance failures."""


class UnknownDatasetError(ProvenanceError):
    """Raised when a dataset name is not present in the registry."""


class IntegrityError(ProvenanceError):
    """Raised when a downloaded/cached file fails digest verification."""


class OfflineDataUnavailable(ProvenanceError):
    """Raised when offline mode is active but no valid cached file exists."""


class DownloadError(ProvenanceError):
    """Raised when a download fails after exhausting all retries."""


# ---------------------------------------------------------------------------
# Dataset specification and registry.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """Immutable, self-validating description of a pinned remote dataset.

    Parameters
    ----------
    name : str
        Stable identifier used by callers (e.g. ``"pulsar_b0329"``).
    url : str
        Fully-qualified ``http(s)`` download URL.
    filename : str
        Local basename under the cache directory.
    digest_algo : str
        One of ``"sha256"``, ``"sha1"`` or ``"md5"`` (lower-case).
    digest_value : str
        Expected lower-case hex digest of the *entire* file.
    doi : str
        Digital Object Identifier for citation/provenance.
    description : str
        Human-readable provenance note.
    min_bytes : int
        Lower bound on the accepted file size; guards against truncated or
        error-page responses being cached as data.
    """

    name: str
    url: str
    filename: str
    digest_algo: str
    digest_value: str
    doi: str
    description: str
    min_bytes: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DatasetSpec.name must be a non-empty string.")
        if not isinstance(self.url, str) or not self.url.lower().startswith(
            ("http://", "https://")
        ):
            raise ValueError(
                f"DatasetSpec.url must be an http(s) URL, got {self.url!r}."
            )
        if not isinstance(self.filename, str) or not self.filename.strip():
            raise ValueError("DatasetSpec.filename must be a non-empty string.")
        # Prevent path traversal / absolute paths in the cache basename.
        if (
            os.path.isabs(self.filename)
            or os.path.basename(self.filename) != self.filename
        ):
            raise ValueError(
                f"DatasetSpec.filename must be a bare basename, got "
                f"{self.filename!r}."
            )
        algo = self.digest_algo.lower()
        if algo not in _SUPPORTED_ALGOS:
            raise ValueError(
                f"Unsupported digest_algo {self.digest_algo!r}; "
                f"expected one of {_SUPPORTED_ALGOS}."
            )
        value = self.digest_value.lower()
        expected_len = hashlib.new(algo).digest_size * 2
        if len(value) != expected_len or any(
            c not in "0123456789abcdef" for c in value
        ):
            raise ValueError(
                f"digest_value for {self.name!r} is not a valid {algo} hex "
                f"digest (expected {expected_len} hex chars)."
            )
        if not isinstance(self.min_bytes, int) or self.min_bytes < 1:
            raise ValueError("DatasetSpec.min_bytes must be a positive int.")

    @property
    def normalized_algo(self) -> str:
        return self.digest_algo.lower()

    @property
    def normalized_digest(self) -> str:
        return self.digest_value.lower()


#: The authoritative, pinned catalogue of real observational inputs.
REGISTRY: Dict[str, DatasetSpec] = {
    "pulsar_b0329": DatasetSpec(
        name="pulsar_b0329",
        url="https://zenodo.org/records/3905426/files/B0329%2B54.fil?download=1",
        filename="B0329+54.fil",
        digest_algo="md5",
        digest_value="5c46f38f573c3f36258630fea1dc3052",
        doi="10.5281/zenodo.3905426",
        description=(
            "PSR B0329+54 single pulses, GBT/Greenburst backend "
            "(Agarwal et al. 2020, MNRAS 497, 1661)."
        ),
        min_bytes=10_000_000,
    ),
    "frb_180417": DatasetSpec(
        name="frb_180417",
        url="https://zenodo.org/records/3905426/files/FRB180417.fil?download=1",
        filename="FRB180417.fil",
        digest_algo="md5",
        digest_value="a6cae8f71fc0e188229d156c38f5308f",
        doi="10.5281/zenodo.3905426",
        description=(
            "FRB 180417, ASKAP incoherent-sum dynamic spectrum "
            "(Agarwal et al. 2020, MNRAS 497, 1661)."
        ),
        min_bytes=1_000_000,
    ),
    "rfi_bl_gbt": DatasetSpec(
        name="rfi_bl_gbt",
        url=(
            "http://blpd13.ssl.berkeley.edu/ETZ/AGBT17B_999_50/GUPPI/BLP02/"
            "blc02_guppi_58060_20295_DIAG_3C123_0001.gpuspec.0002.fil"
        ),
        filename="bl_obs.fil",
        digest_algo="sha256",
        digest_value=(
            "7c6a2af01b2c930ebc72d8c8faa8c61d061344c39e9d5f378aa037f67bf8dbc8"
        ),
        doi="10.3847/1538-3873/aa80d2",
        description=(
            "Breakthrough Listen GBT Earth-Transit-Zone observation "
            "(MacMahon et al. 2018, PASP 130, 044502); RFI/background source."
        ),
        min_bytes=10_000_000,
    ),
    "voyager1_carrier": DatasetSpec(
        name="voyager1_carrier",
        url=(
            "http://blpd0.ssl.berkeley.edu/Voyager_data/"
            "Voyager1.single_coarse.fine_res.fil"
        ),
        filename="voyager1.fil",
        digest_algo="sha256",
        digest_value=(
            "49af50577e136d6e1184e1709fda54d1f48562290df34aa5ed5df34e96e2baf7"
        ),
        doi="10.1088/1538-3873/aa5800",
        description=(
            "Voyager 1 X-band telemetry carrier, GBT fine-resolution filterbank "
            "(Isaacson et al. 2017, PASP 129, 054501); ground-truth artificial "
            "technosignature."
        ),
        min_bytes=10_000_000,
    ),
}


# ---------------------------------------------------------------------------
# Provenance record returned to callers.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FetchResult:
    """Outcome of a :func:`fetch` call (all fields are provenance metadata)."""

    name: str
    path: str
    digest_algo: str
    digest_value: str
    doi: str
    description: str
    size_bytes: int
    from_cache: bool


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------
def _is_offline() -> bool:
    return os.environ.get(OFFLINE_ENV, "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def resolve_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Return the effective cache directory (arg > env > default)."""
    chosen = cache_dir or os.environ.get(CACHE_DIR_ENV) or DEFAULT_CACHE_DIR
    if not isinstance(chosen, str) or not chosen.strip():
        raise ValueError("cache_dir must resolve to a non-empty path.")
    return chosen


def _hash_file(path: str, algo: str) -> str:
    """Stream a file through ``algo`` and return its lower-case hex digest."""
    if algo not in _SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported digest algorithm {algo!r}.")
    hasher = hashlib.new(algo)
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_STREAM_CHUNK_BYTES), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise ProvenanceError(f"Failed to read {path!r} for hashing: {exc}") from exc
    return hasher.hexdigest().lower()


def _cached_file_is_valid(path: str, spec: DatasetSpec) -> bool:
    """Return True iff ``path`` exists, is large enough, and matches the digest."""
    if not os.path.isfile(path):
        return False
    try:
        size = os.path.getsize(path)
    except OSError:
        return False
    if size < spec.min_bytes:
        log.warning(
            "[provenance] cached %s is undersized (%d < %d bytes); rejecting.",
            spec.filename,
            size,
            spec.min_bytes,
        )
        return False
    actual = _hash_file(path, spec.normalized_algo)
    if actual != spec.normalized_digest:
        log.warning(
            "[provenance] cached %s failed %s check (have %s, want %s).",
            spec.filename,
            spec.normalized_algo,
            actual,
            spec.normalized_digest,
        )
        return False
    return True


def _sleep_backoff(attempt: int) -> None:
    """Deterministic bounded exponential backoff (no jitter, reproducible)."""
    delay = min(_BACKOFF_BASE_S ** attempt, _BACKOFF_CAP_S)
    log.info("[provenance] backing off %.1fs before retry.", delay)
    time.sleep(delay)


def _stream_download(spec: DatasetSpec, dest_path: str, timeout: int) -> None:
    """Download ``spec`` to ``dest_path`` atomically, verifying the digest.

    The payload is written to a uniquely-named temporary file in the same
    directory (so the final ``os.replace`` is atomic on POSIX), hashed, and only
    promoted to ``dest_path`` if the digest matches. Any failure removes the
    temporary file.
    """
    dest_dir = os.path.dirname(dest_path) or "."
    os.makedirs(dest_dir, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{spec.filename}.", suffix=".part", dir=dest_dir
    )
    hasher = hashlib.new(spec.normalized_algo)
    bytes_written = 0
    request = urllib.request.Request(spec.url, headers={"User-Agent": _USER_AGENT})
    try:
        with os.fdopen(fd, "wb") as out_handle:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status = getattr(response, "status", 200) or 200
                if int(status) >= 400:
                    raise DownloadError(
                        f"HTTP {status} while fetching {spec.url!r}."
                    )
                while True:
                    chunk = response.read(_STREAM_CHUNK_BYTES)
                    if not chunk:
                        break
                    out_handle.write(chunk)
                    hasher.update(chunk)
                    bytes_written += len(chunk)
        if bytes_written < spec.min_bytes:
            raise IntegrityError(
                f"{spec.filename}: received {bytes_written} bytes "
                f"(< min {spec.min_bytes}); refusing to cache."
            )
        actual = hasher.hexdigest().lower()
        if actual != spec.normalized_digest:
            raise IntegrityError(
                f"{spec.filename}: {spec.normalized_algo} mismatch after "
                f"download (have {actual}, want {spec.normalized_digest})."
            )
        os.replace(tmp_path, dest_path)
        log.info(
            "[provenance] verified download of %s (%d bytes, %s ok).",
            spec.filename,
            bytes_written,
            spec.normalized_algo,
        )
    except BaseException:
        # Never leave a partial/corrupt temp file behind on any failure.
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError as cleanup_exc:  # pragma: no cover - best-effort cleanup
            if cleanup_exc.errno != errno.ENOENT:
                log.warning(
                    "[provenance] could not remove temp file %s: %s",
                    tmp_path,
                    cleanup_exc,
                )
        raise


def _download_with_retries(
    spec: DatasetSpec, dest_path: str, timeout: int, max_retries: int
) -> None:
    """Attempt :func:`_stream_download` up to ``max_retries`` times."""
    last_error: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            _stream_download(spec, dest_path, timeout)
            return
        except IntegrityError:
            # A digest mismatch is deterministic corruption, not transient;
            # retrying an identical URL is worthwhile only if it was a partial
            # transfer, so we retry but surface the last error precisely.
            last_error = IntegrityError(
                f"Integrity check failed for {spec.filename} on attempt "
                f"{attempt + 1}/{max_retries}."
            )
            log.warning("[provenance] %s", last_error)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError,
                DownloadError) as exc:
            last_error = exc
            log.warning(
                "[provenance] download attempt %d/%d for %s failed: %s",
                attempt + 1,
                max_retries,
                spec.filename,
                exc,
            )
        if attempt < max_retries - 1:
            _sleep_backoff(attempt + 1)
    raise DownloadError(
        f"Exhausted {max_retries} attempts fetching {spec.filename} "
        f"from {spec.url!r}: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------
def get_spec(name: str) -> DatasetSpec:
    """Return the :class:`DatasetSpec` for ``name`` or raise UnknownDatasetError."""
    if not isinstance(name, str):
        raise TypeError(f"dataset name must be a string, got {type(name)!r}.")
    try:
        return REGISTRY[name]
    except KeyError:
        raise UnknownDatasetError(
            f"Unknown dataset {name!r}. Known datasets: "
            f"{sorted(REGISTRY)}."
        ) from None


def fetch(
    name: str,
    cache_dir: Optional[str] = None,
    *,
    timeout: int = _DEFAULT_TIMEOUT_S,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> FetchResult:
    """Return a digest-verified local path to the registered dataset ``name``.

    Resolution order:

    1. If a digest-valid file already exists in the cache, return it (no
       network access).
    2. Otherwise, if offline mode is active, raise :class:`OfflineDataUnavailable`.
    3. Otherwise, download with bounded retries, verifying the digest before the
       file is atomically committed to the cache, then return it.

    Parameters
    ----------
    name : str
        Registry key (see :data:`REGISTRY`).
    cache_dir : str, optional
        Cache directory override (else ``AXIOM_DATA_CACHE`` env, else default).
    timeout : int
        Per-attempt socket timeout in seconds.
    max_retries : int
        Maximum number of network attempts.

    Returns
    -------
    FetchResult
        Provenance record including the verified path and digest.

    Raises
    ------
    UnknownDatasetError, OfflineDataUnavailable, DownloadError, IntegrityError
    """
    spec = get_spec(name)
    if not isinstance(timeout, int) or timeout <= 0:
        raise ValueError("timeout must be a positive integer (seconds).")
    if not isinstance(max_retries, int) or max_retries <= 0:
        raise ValueError("max_retries must be a positive integer.")

    directory = resolve_cache_dir(cache_dir)
    dest_path = os.path.join(directory, spec.filename)

    if _cached_file_is_valid(dest_path, spec):
        log.info("[provenance] cache hit for %s (%s verified).",
                 spec.name, spec.normalized_algo)
        return FetchResult(
            name=spec.name,
            path=dest_path,
            digest_algo=spec.normalized_algo,
            digest_value=spec.normalized_digest,
            doi=spec.doi,
            description=spec.description,
            size_bytes=os.path.getsize(dest_path),
            from_cache=True,
        )

    if _is_offline():
        raise OfflineDataUnavailable(
            f"Offline mode ({OFFLINE_ENV}=1) is active and no digest-valid "
            f"cached copy of {spec.name!r} exists at {dest_path!r}. "
            f"Run once online to populate the cache."
        )

    log.info("[provenance] fetching %s from %s", spec.name, spec.url)
    _download_with_retries(spec, dest_path, timeout, max_retries)

    if not _cached_file_is_valid(dest_path, spec):  # pragma: no cover - defensive
        raise IntegrityError(
            f"{spec.name}: file failed verification immediately after a "
            f"reportedly successful download; aborting to avoid poisoning "
            f"the manifold."
        )

    return FetchResult(
        name=spec.name,
        path=dest_path,
        digest_algo=spec.normalized_algo,
        digest_value=spec.normalized_digest,
        doi=spec.doi,
        description=spec.description,
        size_bytes=os.path.getsize(dest_path),
        from_cache=False,
    )


def verify_cached(name: str, cache_dir: Optional[str] = None) -> bool:
    """Return True iff a digest-valid cached copy of ``name`` exists (no network)."""
    spec = get_spec(name)
    dest_path = os.path.join(resolve_cache_dir(cache_dir), spec.filename)
    return _cached_file_is_valid(dest_path, spec)
