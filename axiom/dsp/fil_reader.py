"""Self-contained SIGPROC `.fil` filterbank reader.

This module intentionally avoids the heavy `blimpy`/`sigproc` dependencies so
the engine can ingest real telescope waterfalls on a clean Python install. It
parses the SIGPROC header convention (length-prefixed keyword/value pairs) and
decodes 8-bit unsigned filterbank data into a (nchans, nsamples) dynamic
spectrum (float, normalised to per-channel relative intensity).
"""

from __future__ import annotations

import logging
import os
import struct
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

log = logging.getLogger(__name__)

# SIGPROC header fields that carry a machine-readable scalar value.
_SCALAR_KEYS = {
    "machine_id": "i",
    "barycentric": "i",
    "pulsarcentric": "i",
    "telescope_id": "i",
    "data_type": "i",
    "nchans": "i",
    "nbits": "i",
    "nbeams": "i",
    "ibeam": "i",
    "nifs": "i",
    "src_raj": "d",
    "src_dej": "d",
    "az_start": "d",
    "za_start": "d",
    "fch1": "d",
    "foff": "d",
    "fchannel": "i",
    "tsamp": "d",
    "tstart": "d",
    "nbins": "i",
    "period": "d",
    "decimated": "i",
}
_STRING_KEYS = {"rawdatafile", "source_name", " telescope_id", "telescope"}


@dataclass
class FilHeader:
    """Parsed SIGPROC filterbank header."""

    source_name: str = "UNKNOWN"
    nchans: int = 0
    nbits: int = 8
    nifs: int = 1
    tsamp: float = 1.0
    tstart: float = 0.0
    fch1: float = 0.0
    foff: float = 1.0
    telescope_id: int = 0
    rawdatafile: str = ""
    nbytes_data: int = 0
    header_bytes: int = 0
    extra: Dict[str, object] = field(default_factory=dict)

    @property
    def freqs_hz(self) -> np.ndarray:
        """Centre frequency of each channel, in Hz (descending if foff < 0)."""
        return self.fch1 * 1e6 + self.foff * 1e6 * np.arange(self.nchans, dtype=np.float64)


def _read_keyword(fh):
    """Read one SIGPROC keyword/value pair. Returns (key, value, done).

    SIGPROC layout: 4-byte little-endian keyword length, keyword name, then an
    8-byte scalar for numeric keys or a 4-byte length + value for string keys.
    """
    len_bytes = fh.read(4)
    if len(len_bytes) < 4:
        return None, None, True
    (klen,) = struct.unpack("<i", len_bytes)
    key = fh.read(klen).decode("latin-1", errors="replace")
    if key == "HEADER_END":
        return key, None, False
    if key in _SCALAR_KEYS:
        fmt = _SCALAR_KEYS[key]
        nbytes = 8 if fmt == "d" else 4
        raw = fh.read(nbytes)
        if len(raw) < nbytes:
            return key, None, True
        (val,) = struct.unpack("<" + fmt, raw)
        return key, val, False
    # Default: string value (4-byte length prefix).
    vlen_bytes = fh.read(4)
    if len(vlen_bytes) < 4:
        return key, None, True
    (vlen,) = struct.unpack("<i", vlen_bytes)
    val = fh.read(vlen).decode("latin-1", errors="replace")
    return key, val, False


def parse_fil_header(path: str) -> Tuple[FilHeader, int]:
    """Parse a SIGPROC `.fil` header. Returns (header, data_offset)."""
    size = os.path.getsize(path)
    header = FilHeader()
    with open(path, "rb") as fh:
        # SIGPROC prefixes every keyword (including HEADER_START) with a 4-byte
        # little-endian length, so the stream begins with b"\x0c\x00\x00\x00HEADER_START".
        magic = fh.read(4 + 12)
        if magic != b"\x0c\x00\x00\x00HEADER_START":
            raise ValueError(f"{path}: not a SIGPROC filterbank (bad magic {magic!r})")
        while True:
            key, val, done = _read_keyword(fh)
            if done:
                break
            if key == "HEADER_END":
                break
            if key == "source_name":
                header.source_name = val or "UNKNOWN"
            elif key == "rawdatafile":
                header.rawdatafile = val or ""
            elif key in ("telescope_id", "telescope"):
                header.telescope_id = int(val) if val is not None else 0
            elif hasattr(header, key) and key in _SCALAR_KEYS:
                setattr(header, key, val)
            else:
                header.extra[key] = val
        offset = fh.tell()
    header.header_bytes = offset
    header.nbytes_data = size - offset
    if header.nchans <= 0 or header.nbits <= 0:
        raise ValueError(f"{path}: malformed header (nchans={header.nchans}, nbits={header.nbits})")
    return header, offset


def read_fil_spectrum(path: str, max_samples: int | None = None) -> Tuple[np.ndarray, FilHeader]:
    """Read a `.fil` file into a (nchans, nsamples) float dynamic spectrum.

    Channels are returned in file order (frequency descending when foff<0).
    Each channel is normalised by its own median to relative intensity so the
    dynamic spectrum is robust to absolute flux calibration differences.
    """
    header, offset = parse_fil_header(path)
    dtype_map = {8: np.uint8, 16: np.uint16, 32: np.float32}
    if header.nbits not in dtype_map:
        raise NotImplementedError(
            f"{path}: unsupported nbits={header.nbits} (supported: 8/16/32)"
        )
    np_dtype = dtype_map[header.nbits]
    with open(path, "rb") as fh:
        fh.seek(offset)
        raw = np.frombuffer(fh.read(), dtype=np_dtype)
    if raw.size == 0:
        raise ValueError(f"{path}: empty filterbank data")
    n_samples = raw.size // (header.nchans * header.nifs)
    if max_samples is not None and n_samples > max_samples:
        n_samples = max_samples
        raw = raw[: n_samples * header.nchans * header.nifs]
    spectrum = raw.reshape(n_samples, header.nifs, header.nchans).astype(np.float64)
    spectrum = spectrum[:, 0, :].T  # (nchans, nsamples)
    # Per-channel relative intensity (median-normalised) for robustness.
    med = np.median(spectrum, axis=1, keepdims=True)
    med = np.where(med == 0, 1.0, med)
    spectrum = spectrum / med
    return spectrum, header
